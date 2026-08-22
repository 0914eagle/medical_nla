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

from src.jsonl import read_jsonl


def load(path: str) -> dict[str, dict[str, Any]]:
    return {str(r["base_id"]): r for r in read_jsonl(path)}


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
    args = parser.parse_args()

    for path in args.rungs:
        rows = list(read_jsonl(path))
        if not rows:
            print(f"{path}: empty")
            continue
        rung = rows[0].get("ladder_rung")
        print(f"\nRUNG {rung}  ({path})")
        block("all cases", rows)
        block("flagged (disagreement)", [r for r in rows if r.get("correction_flag")])
        block("not flagged", [r for r in rows if not r.get("correction_flag")])
        block("moved (causal ceiling)", [r for r in rows if r.get("moved")])
    print(
        "\n  The deployable comparison is r5 vs r4 on the flagged rows: r4 already"
        "\n  re-shows the findings, so r5's margin there is the internal conclusion's"
        "\n  contribution and nothing else. `net` on the not-flagged rows is the cost"
        "\n  of intervening where the signal said not to."
    )


if __name__ == "__main__":
    main()
