"""How many scored rows change when the matcher stops matching inside words.

`is_correct` used plain substring containment, so "pe" -- an alias of pulmonary
embolism -- matched inside "pericarditis", "stable angina" matched inside
"unstable angina", and "bronchitis" matched inside
"laryngotracheobronchitis". Every one of those is the other disease, and this
function scores every answer in the paper. It now requires word boundaries.

Changing a scorer moves every rate computed with it, so the size of the move
has to be visible before any table is re-quoted. This replays both rules over
the finished answer files and reports, per file and per arm:

  - how many rows change verdict, and in which direction
  - the distinct (gold, answer) pairs behind the change, with counts
  - the accuracy under each rule

Nothing here decides anything. It says how much of the paper's numbers turn on
the fix, so that re-running the analyses is a known cost rather than a
surprise.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, normalize
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl


def is_correct_substring(answer: str | None, gold: str, aliases: list[str]) -> bool:
    """The rule as it stood before the boundary fix, kept here to compare."""
    if not answer:
        return False
    got = normalize(answer)
    for candidate in [gold, *aliases]:
        want = normalize(candidate)
        if want and (want in got or got in want):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--gold-field", default="diagnosis_name")
    parser.add_argument("--show", type=int, default=15)
    args = parser.parse_args()

    for path in args.answers:
        rows = list(read_jsonl(path))
        by_arm: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
        changed: Counter[tuple[str, str, str]] = Counter()
        scored = 0
        for row in rows:
            gold = str(row.get(args.gold_field) or "").strip()
            answer = str(row.get("answer") or "").strip()
            if not gold:
                continue
            scored += 1
            aliases = aliases_for(gold)
            old = is_correct_substring(answer, gold, aliases)
            new = is_correct(answer, gold, aliases)
            arm = str(row.get("hint_variant") or "-")
            by_arm[arm].append((old, new))
            if old != new:
                changed[("lost" if old else "gained", gold, answer)] += 1

        print(f"\n=== {Path(path).name} ===")
        print(f"rows {len(rows):,}   scored {scored:,}   verdict changed "
              f"{sum(changed.values()):,} ({sum(changed.values()) / max(scored, 1):.4f})")
        print(f"  {'arm':<10}{'n':>8}{'old':>9}{'new':>9}{'delta':>9}")
        for arm in sorted(by_arm):
            pairs = by_arm[arm]
            old_rate = sum(o for o, _ in pairs) / len(pairs)
            new_rate = sum(n for _, n in pairs) / len(pairs)
            print(f"  {arm:<10}{len(pairs):>8,}{old_rate:>9.4f}"
                  f"{new_rate:>9.4f}{new_rate - old_rate:>+9.4f}")

        if not changed:
            print("  no row changes verdict in this file")
            continue
        # Grouped by pair: DDXPlus has 49 names, so one collision recurs and a
        # flat listing would print the same disagreement dozens of times.
        print(f"\n  pairs whose verdict changed ({len(changed):,} distinct):")
        for (direction, gold, answer), n in changed.most_common(args.show):
            mark = "was correct, now wrong" if direction == "lost" else "was wrong, now correct"
            print(f"    x{n:<4} [{mark}]")
            print(f"          gold:   {gold}")
            print(f"          answer: {answer}")
        if len(changed) > args.show:
            print(f"    ... {len(changed) - args.show:,} more pairs")

    print("\n  'lost' rows were being counted correct because one name sat "
          "inside another.\n  Read a few and confirm they are different "
          "diseases before re-running the analyses.")


if __name__ == "__main__":
    main()
