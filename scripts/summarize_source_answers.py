"""Break the source model's accuracy down by diagnosis, and bound the free lunch.

The error-prediction axis asks whether an activation says the model is about to
be wrong. It only asks that if being wrong is not already predictable from
something cheaper. On DDXPlus the running accuracy swings from 0.11 to 0.40
across a file ordered by diagnosis, which means errors are concentrated in
particular diagnoses -- and a probe that learns "which diagnosis is this"
would score well on error prediction without carrying any information about
error at all.

The number that settles it is `diagnosis_only_accuracy`: the accuracy of a
predictor that sees nothing but the diagnosis label and answers with that
diagnosis's majority outcome. Any activation probe must beat it to have shown
anything, and the gap between it and the majority-class baseline is how much of
the error signal is diagnosis identity.

It is computed on the same rows the probe would train on, so it is a bound, not
an analogy. Where the corpus has one case per diagnosis -- MedCaseReasoning has
6,934 labels over 12,766 cases -- the bound is 1.0 by construction and is
reported as inapplicable rather than as a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

# Below this many cases a diagnosis's majority outcome is memorization of a
# handful of rows rather than a property of the diagnosis.
MIN_CASES_FOR_A_BOUND = 5


def diagnosis_only_accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How well the diagnosis label alone predicts correct/incorrect."""
    by_diagnosis: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        key = str(row.get("diagnosis_id") or row.get("diagnosis_name") or "")
        by_diagnosis[key].append(bool(row.get("source_correct")))

    total = sum(len(v) for v in by_diagnosis.values())
    if not total:
        return {}
    # The majority outcome within each diagnosis: the best any predictor seeing
    # only the label can do.
    hits = sum(max(sum(v), len(v) - sum(v)) for v in by_diagnosis.values())
    overall = sum(sum(v) for v in by_diagnosis.values())
    majority = max(overall, total - overall) / total
    sizes = [len(v) for v in by_diagnosis.values()]
    return {
        "n_diagnoses": len(by_diagnosis),
        "cases_per_diagnosis_median": sorted(sizes)[len(sizes) // 2],
        "majority_class_accuracy": round(majority, 4),
        "diagnosis_only_accuracy": round(hits / total, 4),
        "headroom_over_diagnosis_only": round(1.0 - hits / total, 4),
        "applicable": bool(sorted(sizes)[len(sizes) // 2] >= MIN_CASES_FOR_A_BOUND),
    }


def per_diagnosis_table(rows: list[dict[str, Any]], min_cases: int) -> list[dict[str, Any]]:
    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_diagnosis[str(row.get("diagnosis_id") or row.get("diagnosis_name") or "")].append(row)
    table = []
    for name, group in by_diagnosis.items():
        if len(group) < min_cases:
            continue
        correct = sum(1 for row in group if row.get("source_correct"))
        table.append(
            {
                "diagnosis": name,
                "n": len(group),
                "accuracy": round(correct / len(group), 4),
                "n_errors": len(group) - correct,
            }
        )
    return sorted(table, key=lambda entry: (entry["accuracy"], -entry["n"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--min-cases", type=int, default=5)
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    if not rows:
        raise SystemExit(f"no rows in {args.answers}")

    n = len(rows)
    correct = sum(1 for row in rows if row.get("source_correct"))
    forced = sum(1 for row in rows if row.get("answer_forced"))
    bound = diagnosis_only_accuracy(rows)

    summary = {
        "answers": args.answers,
        "n": n,
        "accuracy": round(correct / n, 4),
        "n_errors": n - correct,
        "answer_parse_rate": round(sum(1 for r in rows if r.get("answer_parsed")) / n, 4),
        "answer_forced_rate": round(forced / n, 4),
        **bound,
    }
    if forced:
        # A forced answer comes from a different procedure; if those rows are
        # much worse, the accuracy is partly an artifact of the token budget.
        forced_correct = sum(1 for r in rows if r.get("answer_forced") and r.get("source_correct"))
        summary["accuracy_forced_rows"] = round(forced_correct / forced, 4)
        summary["accuracy_unforced_rows"] = round((correct - forced_correct) / (n - forced), 4)
    print(json.dumps(summary, indent=2))

    if not bound.get("applicable", False):
        print(
            "\n[note] Most diagnoses appear fewer than "
            f"{MIN_CASES_FOR_A_BOUND} times, so diagnosis_only_accuracy is near 1.0 "
            "by construction and bounds nothing. Read it only on a corpus with a "
            "closed label set."
        )
    else:
        print(
            f"\nA probe seeing only the diagnosis label scores "
            f"{bound['diagnosis_only_accuracy']:.3f} on error prediction, against "
            f"{bound['majority_class_accuracy']:.3f} for always guessing the "
            "majority outcome. An activation probe has to beat the former, not "
            "the latter, and the errors must be stratified by diagnosis when the "
            "splits are made."
        )

    table = per_diagnosis_table(rows, args.min_cases)
    if table:
        print(f"\nHardest diagnoses (n >= {args.min_cases}):")
        for entry in table[: args.show]:
            print(f"  {entry['accuracy']:.3f}  n={entry['n']:>4}  {entry['diagnosis']}")
        print("\nEasiest:")
        for entry in reversed(table[-args.show :]):
            print(f"  {entry['accuracy']:.3f}  n={entry['n']:>4}  {entry['diagnosis']}")
        solved = [e for e in table if e["n_errors"] == 0]
        never = [e for e in table if e["accuracy"] == 0.0]
        print(
            f"\n{len(never)} diagnoses are never answered correctly and "
            f"{len(solved)} are always answered correctly, of {len(table)} with "
            f"n >= {args.min_cases}. Those two groups carry no error signal to "
            "learn beyond their own identity."
        )

    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({**summary, "per_diagnosis": table}, indent=2), encoding="utf-8"
        )
        print(f"\n[done] wrote {path}")


if __name__ == "__main__":
    main()
