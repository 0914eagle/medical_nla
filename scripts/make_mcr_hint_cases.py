"""The referring-note intervention on real case reports (MCR).

DDXPlus hands us the plausible wrong suggestion for free -- the differential's
top non-gold entry. MedCaseReasoning has no differential field, so the
plausibility has to come from somewhere else. Two sources, in order:

1. **The model's own confusions**: across the corpus's direct answers, the
   diagnoses the model actually confused with a given gold are suggestions
   it cannot dismiss out of hand. But MCR's label space is open -- most
   golds appear once -- so a gold the model answered correctly usually has
   no confusion record at all (81 of 113 on the test split).
2. **The nearest neighbor's gold**: for those, the suggestion is the
   diagnosis of the most cue-similar *other* case in the corpus. Same
   plausibility logic -- a diagnosis whose presentation overlaps this one's
   is clinically confusable -- sourced from case similarity instead of
   observed error.

A case with neither (no confusion, no overlapping neighbor) is skipped
loudly rather than handed a strawman.

Everything else mirrors the DDXPlus builder: same four arms, same sentence
templates, same placement (after the presentation, before the instruction),
same leak flag, same carried fields -- so analyze_hint_effect and the
comparison harness run unchanged on the output.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_hint_injection_cases import (
    NEUTRAL_SENTENCE,
    gold_is_written_in,
    hint_sentence,
    presentation_of,
)
from src.answer_matching import is_correct
from src.case_prompts import build_prompt
from src.jsonl import read_jsonl, write_jsonl


def confusions(answers: list[dict[str, Any]]) -> dict[str, Counter]:
    """gold diagnosis -> what the model wrongly answered instead, with counts."""
    out: dict[str, Counter] = defaultdict(Counter)
    for row in answers:
        if row.get("source_correct"):
            continue
        gold = str(row.get("diagnosis_name") or "").strip()
        answer = str(row.get("answer") or "").strip()
        if gold and answer:
            out[gold][answer] += 1
    return out


def plausible_wrong(gold: str, aliases: list[str], pool: Counter | None) -> str | None:
    for name, _ in (pool or Counter()).most_common():
        if not is_correct(name, gold, aliases):
            return name
    return None


def cue_words(case: dict[str, Any]) -> set[str]:
    words: set[str] = set()
    for cue in case.get("cue_targets") or []:
        words.update(w for w in str(cue).lower().split() if len(w) > 3)
    return words


def neighbor_gold(
    case: dict[str, Any], gold: str, aliases: list[str], corpus: list[tuple[set[str], str]]
) -> tuple[str | None, float]:
    """The diagnosis of the most cue-similar other case -- Jaccard on cue words.

    Returns the score too, because the fallback's whole risk is here. DDXPlus
    hands us a ranked differential, so its wrong arm carries a condition the
    model itself considers second-most likely. MCR has no such field, and a
    nearest neighbour found on one or two shared words is not a differential:
    it produced 'bullous dermatomyositis' for a hypothalamic hamartoma. A
    suggestion the model dismisses on sight moves nothing and measures
    nothing, so the score travels with the case and the analysis splits on it
    rather than averaging a real intervention together with a nonsensical one.
    """
    own = cue_words(case)
    if not own:
        return None, 0.0
    best, best_score = None, 0.0
    for words, other_gold in corpus:
        if not words or is_correct(other_gold, gold, aliases):
            continue
        score = len(own & words) / len(own | words)
        if score > best_score:
            best, best_score = other_gold, score
    return best, best_score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        nargs="+",
        required=True,
        help="MCR case files. Both splits are accepted: the split is theirs, "
        "made for fine-tuning a reasoner, and nothing here trains on it -- "
        "widening the pool only widens what the intervention can be measured "
        "on. (If the readout adapter is ever trained, it trains on train and "
        "is read out on test; that rule is not affected by this.)",
    )
    parser.add_argument(
        "--answers",
        nargs="+",
        required=True,
        help="MCR direct source answers; extra files (e.g. the train split) "
        "thicken the confusion dictionary.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--arms", nargs="+", default=["none", "neutral", "wrong", "correct"])
    args = parser.parse_args()

    answers = [row for path in args.answers for row in read_jsonl(path)]
    confused = confusions(answers)
    correct_ids = {
        str(r.get("base_id", r.get("id")))
        for r in answers
        if r.get("source_correct")
    }
    print(f"direct-correct cases: {len(correct_ids):,}   golds with confusions: {len(confused):,}")

    all_cases = [row for path in args.cases for row in read_jsonl(path)]
    corpus = [
        (cue_words(c), str(c.get("diagnosis_name") or "").strip())
        for c in all_cases
        if str(c.get("diagnosis_name") or "").strip()
    ]

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    sourced = Counter()
    n_cases = 0
    for case in all_cases:
        base_id = str(case.get("base_id") or case["id"])
        if base_id not in correct_ids:
            continue
        raw = str(case.get("prompt") or "")
        presentation = presentation_of(raw) or raw.strip()
        if not presentation:
            skipped["empty prompt"] += 1
            continue
        gold = str(case.get("diagnosis_name") or "").strip()
        aliases = [str(a) for a in (case.get("diagnosis_aliases") or [])]
        if not gold:
            skipped["no gold diagnosis"] += 1
            continue
        wrong = plausible_wrong(gold, aliases, confused.get(gold))
        source, score = "confusion", 1.0
        if wrong:
            sourced["confusion"] += 1
        else:
            wrong, score = neighbor_gold(case, gold, aliases, corpus)
            source = "neighbor"
            if wrong:
                sourced["neighbor"] += 1
        if not wrong:
            skipped["no confused diagnosis and no overlapping neighbor"] += 1
            continue

        n_cases += 1
        leaked = gold_is_written_in(presentation, case)
        carry = {
            key: case.get(key)
            for key in (
                "diagnosis_id",
                "diagnosis_name",
                "diagnosis_aliases",
                "cue_targets",
                "source",
                "case_id",
            )
        }
        arms = (
            ("none", None, None),
            ("neutral", None, NEUTRAL_SENTENCE),
            ("wrong", wrong, hint_sentence(wrong)),
            ("correct", gold, hint_sentence(gold)),
        )
        for variant, hinted, sentence in arms:
            if variant not in args.arms:
                continue
            prefix = presentation if sentence is None else f"{presentation}\n\n{sentence}"
            row = dict(carry)
            row.update(
                {
                    "id": f"{base_id}__hint_{variant}",
                    "base_id": base_id,
                    "variant": f"hint_{variant}",
                    "hint_variant": variant,
                    "hint_wording": "referral",
                    "hint_diagnosis_name": hinted,
                    "gold_in_prompt": leaked,
                    "suggestion_source": source,
                    "suggestion_score": round(score, 4),
                    "prompt": build_prompt(prefix, "direct"),
                    "prompt_cot": build_prompt(prefix, "cot"),
                }
            )
            rows.append(row)

    if not rows:
        raise SystemExit("no rows built; check --cases/--answers")
    write_jsonl(Path(args.output), rows)
    for reason, count in skipped.items():
        print(f"skipped {count:,}: {reason}")
    print(
        f"suggestion source: confusion {sourced['confusion']:,} / "
        f"neighbor {sourced['neighbor']:,}"
    )
    print(f"wrote {len(rows):,} rows over {n_cases:,} cases to {args.output}")

    # Printed per source and worst-first within the fallback: the suggestions
    # that need eyeballing are the low-overlap ones, and one cherry-picked
    # example hid exactly those.
    wrongs = [r for r in rows if r["hint_variant"] == "wrong"]
    for source in ("confusion", "neighbor"):
        sample = [r for r in wrongs if r["suggestion_source"] == source]
        if not sample:
            continue
        sample.sort(key=lambda r: r["suggestion_score"])
        print(f"\n--- {source} ({len(sample):,}) ---")
        for row in sample[:5]:
            score = "" if source == "confusion" else f"  overlap {row['suggestion_score']:.3f}"
            print(f"  gold {row['diagnosis_name']!r}"
                  f"  suggestion {row['hint_diagnosis_name']!r}{score}")
    fallback = [r["suggestion_score"] for r in wrongs if r["suggestion_source"] == "neighbor"]
    if fallback:
        fallback.sort()
        print(f"\nneighbour overlap: median {fallback[len(fallback) // 2]:.3f}, "
              f"min {fallback[0]:.3f}, max {fallback[-1]:.3f}")
        print("  Split the effect on suggestion_source before averaging: a "
              "suggestion the model dismisses on sight is not a weaker "
              "intervention, it is a different one.")


if __name__ == "__main__":
    main()
