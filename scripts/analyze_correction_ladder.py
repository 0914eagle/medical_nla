"""Read the correction ladder: what did each second pass buy, and where.

Accuracy alone hides the two motions that matter, so both are reported:
**recovered** (first pass wrong, second pass right) and **broken** (first pass
right, second pass wrong). A rung that recovers 40 cases and breaks 39 did
nothing worth deploying.

Reported over three populations:
- all cases -- intervene everywhere, the naive deployment;
- flagged -- only where the disagreement signal fired, the selective
  deployment the attribution result makes possible;
- moved -- the causally-pulled cases, the ceiling on what correction can fix.

The comparison that decides the readout's contribution is r5 against r4 on
the flagged population: r4 re-shows the chart's findings, so r5 only claims
something if the *internal conclusion* -- the one thing r4 lacks -- recovers
cases the findings alone do not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct
from src.jsonl import read_jsonl


def load(path: str) -> dict[str, dict[str, Any]]:
    return {str(r["base_id"]): r for r in read_jsonl(path)}


def conclusion_correct(row: dict[str, Any]) -> bool:
    conclusion = str(row.get("readout_conclusion") or "").strip()
    return bool(conclusion) and is_correct(
        conclusion,
        str(row.get("diagnosis_name") or ""),
        row.get("diagnosis_aliases") or [],
    )


def replacement_policy(rows: list[dict[str, Any]]) -> None:
    """The scalpel without the second pass: swap in the conclusion where flagged.

    Every rung shows re-asking breaks far more than it fixes -- the model
    treats "reconsider" as "change your answer", flagged or not. But the rungs
    also show the readout conclusion recovers most moved cases. This policy
    keeps the first answer everywhere the flag is silent and *replaces* the
    answer with the readout conclusion where it fired: no second pass, no
    flip incentive, cost limited to flagged-but-correct cases whose
    conclusion is wrong.

    Computed from the carried fields alone, so it needs no GPU and no new
    run -- which also makes it the honest comparison: any second-pass rung
    that cannot beat this free policy has no deployment case.
    """
    n = len(rows)
    first = sum(bool(r.get("first_correct")) for r in rows)
    swapped = [
        conclusion_correct(r) if r.get("correction_flag") else bool(r.get("first_correct"))
        for r in rows
    ]
    print("\nPOLICY: replace the answer with the conclusion where flagged (no second pass)")
    print(f"  keep first everywhere          {first / n:.4f}")
    print(f"  swap conclusion where flagged  {sum(swapped) / n:.4f}")
    for name, population in (
        ("flagged", [r for r in rows if r.get("correction_flag")]),
        ("moved", [r for r in rows if r.get("moved")]),
    ):
        if not population:
            continue
        acc = sum(conclusion_correct(r) for r in population) / len(population)
        print(f"  conclusion accuracy on {name:<8} {acc:.4f}  (n={len(population):,})")
    flagged = [r for r in rows if r.get("correction_flag")]
    moved = [r for r in rows if r.get("moved")]
    hit = sum(1 for r in moved if r.get("correction_flag"))
    print(
        f"  flag vs moved: recall {hit}/{len(moved)} = {hit / len(moved):.4f}"
        f"   precision {hit}/{len(flagged)} = {hit / len(flagged):.4f}"
        if moved and flagged
        else "  flag vs moved: no contrast"
    )


def show_broken(rows: list[dict[str, Any]], count: int) -> None:
    """Eyeball guard: a uniform collapse could be scoring, not flipping."""
    broken = [
        r for r in rows if r.get("first_correct") and not r.get("source_correct")
    ][:count]
    for r in broken:
        print(
            f"    gold {r.get('diagnosis_name')!r}  first {r.get('first_answer')!r}"
            f"  second {str(r.get('answer') or '')[:80]!r}"
        )


def block(name: str, rows: list[dict[str, Any]]) -> None:
    n = len(rows)
    if not n:
        print(f"  {name}: no cases")
        return
    first = sum(bool(r.get("first_correct")) for r in rows)
    second = sum(bool(r.get("source_correct")) for r in rows)
    recovered = sum(
        bool(r.get("source_correct")) and not r.get("first_correct") for r in rows
    )
    broken = sum(
        bool(r.get("first_correct")) and not r.get("source_correct") for r in rows
    )
    print(
        f"  {name:<28} n={n:<6} first {first / n:.4f} -> second {second / n:.4f}"
        f"   recovered {recovered:,}  broken {broken:,}  net {recovered - broken:+,}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rungs", nargs="+", required=True, help="run_source_answers outputs, one per rung."
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=0,
        help="Print this many broken (first right, second wrong) samples per rung.",
    )
    args = parser.parse_args()

    first_rows: list[dict[str, Any]] = []
    for path in args.rungs:
        rows = list(read_jsonl(path))
        if not rows:
            print(f"{path}: empty")
            continue
        first_rows = first_rows or rows
        rung = rows[0].get("ladder_rung")
        print(f"\nRUNG {rung}  ({path})")
        block("all cases", rows)
        block("flagged (disagreement)", [r for r in rows if r.get("correction_flag")])
        block("not flagged", [r for r in rows if not r.get("correction_flag")])
        block("moved (causal ceiling)", [r for r in rows if r.get("moved")])
        if args.examples:
            print(f"  broken samples (first {args.examples}):")
            show_broken(rows, args.examples)

    if first_rows:
        replacement_policy(first_rows)
    print(
        "\n  The deployable comparison is r5 vs r4 on the flagged rows: r4 already"
        "\n  re-shows the findings, so r5's margin there is the internal conclusion's"
        "\n  contribution and nothing else. `net` on the not-flagged rows is the cost"
        "\n  of intervening where the signal said not to."
    )


if __name__ == "__main__":
    main()
