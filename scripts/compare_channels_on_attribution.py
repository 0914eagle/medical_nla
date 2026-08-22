"""Which channel can tell that the referring note is what changed the answer?

The comparison the paper turns on, and the one that decides whether an internal
readout earns its place beside a chain of thought.

**The question, asked identically of both channels.** Each sees one arm -- the
one carrying a wrong referring note -- and nothing else. From that it must say
whether the note changed this case's answer. The ground truth comes from the
no-note arm, which neither channel is shown. That is also the deployed
situation: there is no counterfactual to consult at inference time.

**Why the chain is expected to fail.** It mentions the note at 0.9833 where the
note changed the answer and 0.9875 where it did not. Mentioning is not
attribution when everything gets mentioned; a constant carries no signal.

**Why the readout might not.** The cue positions cannot help -- the note sits
after them, so under causal attention they are bit-identical between the arms
and say the same thing twice. Anything here has to come from the note's own
position or from the last token, where the layer sweep found an internal
conclusion rather than evidence.

**The refutation condition, written before the run.** If the readout's AUROC
does not clear the chain's by a margin the case count supports, the claim is
not that the readout attributes causation. It falls back to localization: the
evidence readout is invariant while the conclusion readout is not, which is a
description of where the note acts and not a prediction of when it acted.

Reported within diagnosis as well as pooled. Errors in this corpus cluster by
label hard enough that a pooled AUROC of 0.93 is reachable knowing only the
diagnosis, and the same confound is available to a feature that merely tracks
which diagnoses are fragile.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_hint_effect import (
    Case,
    annotations_by_id,
    answer_names,
    group_by_case,
    lost_the_gold,
    took_the_hint,
)
from scripts.analyze_hint_mention import cites_referral, mentions_diagnosis
from scripts.predict_error_from_readouts import auroc
from scripts.score_cue_position_readouts import readout_body
from src.answer_matching import is_correct
from src.cue_readout_scoring import content_words, overlap_f1
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl

Readouts = dict[tuple[str, str, str], str]


def load_readouts(paths: list[str], manifests: list[str] | None = None) -> Readouts:
    """Keyed by (case, arm, position). One text per key; later files win.

    `manifests` are the extraction manifests the readouts were generated from,
    joined back by id. Needed for runs made before run_nla carried
    hint_variant: 5,241 readouts came back with target_role but no arm, and
    the arm was decided when the extraction row was written, so this costs no
    GPU time. The readout row's own values win where both carry a key.
    """
    annotations: dict[str, dict[str, Any]] = {}
    for path in manifests or []:
        annotations.update(annotations_by_id(path))
    out: Readouts = {}
    joined = missing = 0
    for path in paths:
        for row in read_jsonl(path):
            fallback = annotations.get(str(row.get("id")), {})
            row = {**fallback, **{k: v for k, v in row.items() if v is not None}}
            if fallback:
                joined += 1
            if row.get("hint_variant") is None:
                missing += 1
                continue
            key = (
                str(row.get("base_id")),
                str(row.get("hint_variant") or ""),
                str(row.get("target_role") or ""),
            )
            out[key] = readout_body(row)
    if missing:
        print(
            f"[!] {missing:,} readout rows carry no hint_variant and were skipped; "
            "pass the extraction manifests with --readout-manifests to join them."
        )
    return out


def moved(case: Case) -> bool:
    return took_the_hint(case, "wrong") or lost_the_gold(case, "wrong")


def chain_features(case: Case) -> dict[str, float]:
    """What the chain of thought offers, from the wrong-note arm alone."""
    chain = str(case["wrong"].get("response") or "")
    hint = str(case["wrong"].get("hint_diagnosis_name") or "")
    return {
        "chain cites the referral": float(cites_referral(chain)),
        "chain names the suspicion": float(mentions_diagnosis(chain, hint)),
        # How much of the chain is spent on the suggestion: a chain that argued
        # itself into the note should dwell on it more than one that dismissed it.
        "chain dwells on the suspicion": float(
            len(content_words(chain) & content_words(hint))
        ),
    }


def readout_answer(text: str) -> str:
    """The <answer> field of a structured readout, or the whole body."""
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>", 1)[1].split("</answer>", 1)[0]
    return text


def readout_cues(text: str) -> str:
    """The <supporting_cues> field of a structured readout, or empty."""
    if "<supporting_cues>" in text and "</supporting_cues>" in text:
        return text.split("<supporting_cues>", 1)[1].split("</supporting_cues>", 1)[0].strip()
    return ""


def readout_features(case: Case, readouts: Readouts) -> dict[str, float] | None:
    """What the internal readout offers, from the same arm alone.

    The `final`-position pair is the one measurement here that uses both arms,
    and it is reported separately for that reason: it is not available at
    inference time and stands as an upper bound, not as a deployable signal.
    """
    base = str(case["none"].get("base_id") or "")
    hint_read = readouts.get((base, "wrong", "hint"))
    final_wrong = readouts.get((base, "wrong", "final"))
    if hint_read is None or final_wrong is None:
        return None
    hint = str(case["wrong"].get("hint_diagnosis_name") or "")
    features = {
        "readout at the note names it": float(mentions_diagnosis(hint_read, hint)),
        "readout before the answer names the suspicion": float(
            mentions_diagnosis(final_wrong, hint)
        ),
        # The "why" fields: does the readout's *grounds* slot carry the note?
        # A conclusion is what the state points at; the supporting_cues field
        # is what the state treats as evidence. If pulled cases cite the note
        # there, the readout is verbalizing the cause per case -- the one
        # thing no classifier head emits.
        "readout cues cite the referral": float(cites_referral(readout_cues(final_wrong))),
        "readout cues name the suspicion": float(
            mentions_diagnosis(readout_cues(final_wrong), hint)
        ),
        # The project's original signal, repurposed: v1 flagged errors when the
        # internal conclusion disagreed with the emitted answer. Here it asks
        # whether a pulled answer leaves the conclusion state still pointing
        # elsewhere -- readable from one run, no counterfactual needed.
        "internal conclusion contradicts the answer": 1.0
        - overlap_f1(readout_answer(final_wrong), str(case["wrong"].get("answer") or "")),
        # The length-robust form of the same signal. Token overlap falls as the
        # answer grows, and pulled answers are wordier (length alone reaches
        # 0.67 within diagnosis), so the f1 version part-measures verbosity.
        # Containment with aliases does not: "Anemia of CKD" contains the
        # conclusion "Anemia" however long it gets, and a drifted answer fails
        # to contain it for reasons of content, not style.
        "answer omits the internal conclusion (containment)": 1.0
        - float(
            is_correct(
                case["wrong"].get("answer"),
                readout_answer(final_wrong).strip() or "-",
                aliases_for(readout_answer(final_wrong).strip()),
            )
        ),
    }
    final_none = readouts.get((base, "none", "final"))
    if final_none is not None:
        # Paired, so not deployable: an upper bound on what the position holds.
        features["[paired] conclusion readouts diverge"] = 1.0 - overlap_f1(
            final_wrong, final_none
        )
    return features


def report(
    name: str,
    features: dict[str, list[float]],
    labels: list[bool],
    diagnosis: list[str],
) -> None:
    print(f"\n{name}")
    for label, values in features.items():
        within, pairs = stratified_auroc(values, labels, diagnosis)
        print(
            f"  {label:<46} AUROC {auroc(values, labels):.4f}"
            f"   within diagnosis {within:.4f}  ({pairs:,} pairs)"
        )


def stratified_auroc(
    values: list[float], labels: list[bool], diagnosis: list[str]
) -> tuple[float, int]:
    by_label: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    for value, label, name in zip(values, labels, diagnosis, strict=True):
        by_label[name].append((value, label))
    concordant = ties = pairs = 0
    for rows in by_label.values():
        for a in [v for v, is_pos in rows if is_pos]:
            for b in [v for v, is_pos in rows if not is_pos]:
                pairs += 1
                concordant += a > b
                ties += a == b
    if not pairs:
        return float("nan"), 0
    return (concordant + ties / 2) / pairs, pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--answers",
        nargs="+",
        required=True,
        help="Direct answers on the hint cases: the ground truth for what moved.",
    )
    parser.add_argument("--cases", help="Hint case file, if the run predates carried arms.")
    parser.add_argument(
        "--cot-answers", nargs="+", help="Chain answers on the same cases, for the chain channel."
    )
    parser.add_argument(
        "--readouts", nargs="+", default=[], help="run_nla outputs over the hint position rows."
    )
    parser.add_argument(
        "--restrict-diagnoses",
        help="File of diagnosis names (one per line): keep only these cases. "
        "For reading a conclusion adapter only on the classes it was trained on.",
    )
    parser.add_argument(
        "--readout-manifests",
        nargs="+",
        default=[],
        help="Extraction manifests, to restore the arm on readouts from runs "
        "that predate run_nla carrying it.",
    )
    args = parser.parse_args()

    cases = group_by_case(args.answers, args.cases)
    if args.restrict_diagnoses:
        wanted = {
            line.strip().lower()
            for line in Path(args.restrict_diagnoses).read_text().splitlines()
            if line.strip()
        }
        before = len(cases)
        cases = {
            c: arms
            for c, arms in cases.items()
            if str(arms["none"].get("diagnosis_name") or "").lower() in wanted
        }
        print(f"[restrict] {len(cases):,} of {before:,} cases in {len(wanted)} diagnoses")
    if "wrong" not in {v for arms in cases.values() for v in arms}:
        raise SystemExit("the wrong-note arm is what attribution is asked about")

    labels = [moved(case) for case in cases.values()]
    diagnosis = [str(case["none"].get("diagnosis_name") or "") for case in cases.values()]
    print(f"cases {len(cases):,}   the note moved the answer in {sum(labels):,}")
    if not any(labels) or all(labels):
        raise SystemExit("no contrast to measure: every case moved, or none did")

    # The honesty bar. The answer itself and the note are both visible at
    # inference time, and "the answer is what the note suspected" identifies
    # every anchored case for free. A channel earns its place only above this.
    trivial = {
        "answer states the suspicion (output-only)": [
            float(answer_names(case["wrong"], case["wrong"].get("hint_diagnosis_name")))
            for case in cases.values()
        ],
        # The confound check for the disagreement feature: token-overlap
        # disagreement rises with verbose answers, and a pulled answer might
        # simply be wordier. Whatever this row predicts is not internal signal.
        "answer length in words (output-only)": [
            float(len(str(case["wrong"].get("answer") or "").split()))
            for case in cases.values()
        ],
    }
    report(f"ANSWER ALONE  ({len(cases):,} cases)", trivial, labels, diagnosis)

    if args.cot_answers:
        chains = group_by_case(args.cot_answers, args.cases)
        shared = [c for c in cases if c in chains]
        if shared:
            built: dict[str, list[float]] = defaultdict(list)
            for case in shared:
                for key, value in chain_features(chains[case]).items():
                    built[key].append(value)
            report(
                f"CHAIN OF THOUGHT  ({len(shared):,} cases)",
                built,
                [moved(cases[c]) for c in shared],
                [str(cases[c]["none"].get("diagnosis_name") or "") for c in shared],
            )

    if args.readouts:
        readouts = load_readouts(args.readouts, args.readout_manifests)
        built = defaultdict(list)
        kept: list[str] = []
        for case_id, case in cases.items():
            values = readout_features(case, readouts)
            if values is None:
                continue
            kept.append(case_id)
            for key, value in values.items():
                built[key].append(value)
        if not kept:
            raise SystemExit(
                "no case joined a readout to an answer. The readout rows carry "
                "base_id and hint_variant from the extraction manifest; check that "
                "the run kept them."
            )
        report(
            f"INTERNAL READOUT  ({len(kept):,} cases)",
            built,
            [moved(cases[c]) for c in kept],
            [str(cases[c]["none"].get("diagnosis_name") or "") for c in kept],
        )

    # The decisive cut. "The answer states the suspicion" identifies every
    # anchored case for free, so a channel matters only where that baseline is
    # blind: the cases the note moved somewhere OTHER than its own suspicion.
    # In this subset the answer looks unremarkable -- it names some plausible
    # diagnosis that simply is not what the model would have said unhinted --
    # and anything that flags it is information the output does not carry.
    if args.readouts:
        silent = {
            case_id: case
            for case_id, case in cases.items()
            if not answer_names(case["wrong"], case["wrong"].get("hint_diagnosis_name"))
        }
        silent_kept = [c for c in kept if c in silent]
        moved_count = sum(moved(cases[c]) for c in silent_kept)
        if silent_kept and 0 < moved_count < len(silent_kept):
            report(
                f"ANSWER ALONE, same subset  ({len(silent_kept):,} cases)",
                {
                    "answer length in words (output-only)": [
                        float(len(str(cases[c]["wrong"].get("answer") or "").split()))
                        for c in silent_kept
                    ]
                },
                [moved(cases[c]) for c in silent_kept],
                [str(cases[c]["none"].get("diagnosis_name") or "") for c in silent_kept],
            )
            print(
                f"\nWHERE THE ANSWER SAYS NOTHING -- moved, but not onto the suspicion"
                f"\n({len(silent_kept):,} cases, {moved_count:,} moved; the answer-alone"
                " baseline is constant here, AUROC 0.5 by construction)"
            )
            index = {c: i for i, c in enumerate(kept)}
            report(
                "",
                {
                    name: [values[index[c]] for c in silent_kept]
                    for name, values in built.items()
                },
                [moved(cases[c]) for c in silent_kept],
                [str(cases[c]["none"].get("diagnosis_name") or "") for c in silent_kept],
            )

    print(
        "\n  0.5 is no information. The chain's row is expected to sit there --\n"
        "  it mentions the note whether or not the note mattered. A readout row\n"
        "  that also sits there means the claim is localization, not attribution:\n"
        "  the evidence readout is invariant and the conclusion readout is not,\n"
        "  which says where the note acts without saying when it acted.\n"
        "  Rows marked [paired] see both arms and are upper bounds, not signals\n"
        "  available at inference time."
    )


if __name__ == "__main__":
    main()
