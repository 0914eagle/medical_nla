"""Does the internal record say the model is about to be wrong, where the chain does not?

The chain of thought names 92% of a case's findings and names 0.937 of them
when it turns out right against 0.917 when it turns out wrong. Its
thoroughness carries almost nothing about its reliability. This asks the same
question of the readout: on cases the model answers wrongly, were the findings
read out worse?

Two things this must not do.

**Train rows are memorized.** A cue string the adapter was supervised on is
reproduced verbatim 91% of the time, so a case built partly from those rows
would look internalized for a reason that has nothing to do with this case.
Only test-pool rows are used, and the script refuses if it is handed a
manifest that overlaps the training split.

**A signal can be about the diagnosis rather than about the error.** Errors are
concentrated: some DDXPlus labels are answered right almost always and others
almost never, so a predictor that knows only the label already reaches 0.850
accuracy. Any AUROC here is reported beside the same quantity computed from the
diagnosis's own error rate, which is the part that is not about this case.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_cue_position_readouts import gold_cue, readout_body
from src.cue_readout_scoring import score_readout
from src.jsonl import read_jsonl

READ = 0.5


def auroc(scores: list[float], labels: list[bool]) -> float:
    """Probability a positive outranks a negative, ties counted as half.

    Rank-based rather than by threshold sweep, so it is exact and needs no
    dependency. Positive here means "the model got this case wrong", so 0.5 is
    no information and above 0.5 means a higher score goes with an error.
    """
    pairs = sorted(zip(scores, labels, strict=True))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return float("nan")
    # Average ranks over ties.
    ranks: list[float] = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        average = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = average
        i = j + 1
    rank_sum = sum(rank for rank, (_, label) in zip(ranks, pairs, strict=True) if label)
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--readouts", nargs="+", required=True, help="run_nla outputs over test pools."
    )
    parser.add_argument("--answers", required=True, help="Source answers with source_correct.")
    parser.add_argument(
        "--train-manifest",
        help="Training split, so cases sharing rows with it can be refused.",
    )
    parser.add_argument("--cot-answers", help="Chain answers, for the coverage baseline.")
    args = parser.parse_args()

    trained_ids: set[str] = set()
    if args.train_manifest:
        trained_ids = {str(row.get("id")) for row in read_jsonl(args.train_manifest)}

    per_case: dict[str, list[float]] = defaultdict(list)
    overlapping = 0
    for path in args.readouts:
        for row in read_jsonl(path):
            if str(row.get("id")) in trained_ids:
                overlapping += 1
                continue
            gold = gold_cue(row)
            if not gold:
                continue
            score = score_readout(readout_body(row), gold)
            per_case[str(row.get("base_id"))].append(float(score["f1"]))
    if overlapping:
        raise SystemExit(
            f"[!] {overlapping:,} readout rows are training rows, whose cue strings the\n"
            "    adapter reproduces verbatim 91% of the time. Score the test pools only."
        )
    if not per_case:
        raise SystemExit("no readout rows")

    outcome: dict[str, bool] = {}
    diagnosis: dict[str, str] = {}
    for row in read_jsonl(args.answers):
        key = str(row.get("base_id", row.get("id")))
        outcome[key] = bool(row.get("source_correct"))
        diagnosis[key] = str(row.get("diagnosis_name") or "")

    cases = [c for c in per_case if c in outcome]
    if not cases:
        raise SystemExit("no case joined a readout to an answer; check base_id")
    wrong = [not outcome[c] for c in cases]
    print(f"cases {len(cases):,}  ({sum(wrong):,} answered wrongly)")
    print(f"cue readouts per case  {mean(len(per_case[c]) for c in cases):.1f}")

    # Higher score must mean "more likely wrong", so the readout features are
    # negated: a well-read case should be the easy one.
    features = {
        "mean readout f1 (negated)": [-mean(per_case[c]) for c in cases],
        "worst cue f1 (negated)": [-min(per_case[c]) for c in cases],
        "share of cues unread": [
            sum(f < READ for f in per_case[c]) / len(per_case[c]) for c in cases
        ],
    }

    if args.cot_answers:
        # The baseline this exists to beat: how much of the chart the chain
        # recited, which was 0.937 on right answers and 0.917 on wrong ones.
        from scripts.analyze_cot_cue_recall import mentions
        from src.cue_readout_scoring import content_words, strip_parentheticals

        cue_text: dict[str, list[str]] = defaultdict(list)
        for path in args.readouts:
            for row in read_jsonl(path):
                cue_text[str(row.get("base_id"))].append(gold_cue(row))
        coverage: dict[str, float] = {}
        for row in read_jsonl(args.cot_answers):
            key = str(row.get("base_id", row.get("id")))
            cues = cue_text.get(key)
            if not cues:
                continue
            chain = str(row.get("response") or "")
            words = content_words(chain) | content_words(strip_parentheticals(chain))
            coverage[key] = mean(float(mentions(words, cue)) for cue in cues)
        shared = [c for c in cases if c in coverage]
        if shared:
            print(f"\nchain coverage available for {len(shared):,} of these cases")
            print(
                f"  cot coverage (negated)     AUROC "
                f"{auroc([-coverage[c] for c in shared], [not outcome[c] for c in shared]):.4f}"
            )

    # The circularity check. A case's diagnosis alone already predicts the
    # outcome well, because errors cluster by label; anything a readout adds
    # has to be visible against this.
    by_diagnosis: dict[str, list[bool]] = defaultdict(list)
    for case in cases:
        by_diagnosis[diagnosis[case]].append(not outcome[case])
    label_rate = {name: mean(v) for name, v in by_diagnosis.items()}
    print("\nAUROC for predicting a wrong answer (0.5 = no information)")
    print(f"  diagnosis error rate       {auroc([label_rate[diagnosis[c]] for c in cases], wrong):.4f}"
          "   <- knows only the label")
    for name, values in features.items():
        print(f"  {name:<26} {auroc(values, wrong):.4f}")
    print(
        "\n  A readout feature only means something if it beats the label row,\n"
        "  which is not a prediction about this case at all."
    )


if __name__ == "__main__":
    main()
