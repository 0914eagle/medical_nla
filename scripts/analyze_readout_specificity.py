"""Does a readout name findings other than the one its vector sits on?

Precision over content words says an output is padded; it does not say what it
was padded with. Two kinds of padding matter differently here.

**Context leakage** -- naming another cue from the same case. Under causal
attention the vector at cue k has seen cues 1..k-1, so this is not
hallucination, but it means the readout describes the case rather than the
position, and the cue-position design exists precisely to separate those.

**Confabulation** -- naming a cue that is nowhere in this case. This is the one
that damages a faithfulness claim. The adapter does it: a numbness cue reading
"upper gum, labia majora ..., bottom lip" is picking body parts out of
DDXPlus's vocabulary, not out of the vector.

Both are counted against a matched chance rate. For every case cue the readout
is also scored against the same number of cues drawn from other cases, so
"names 12% of foreign cues" can be read against "names 11% of any cues at all"
rather than against zero. Without that the confabulation rate is unreadable:
these cues share a clinical vocabulary, and a readout that says "pain" collides
with every pain cue in the corpus.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_cue_position_readouts import gold_cue, readout_body
from src.cue_readout_scoring import observed_items, overlap_f1
from src.jsonl import read_jsonl

# The same threshold the read rate uses, so "named" means one thing in this
# project rather than two.
NAMED = 0.5


def names_cue(items: list[str], cue: str) -> bool:
    return any(overlap_f1(item, cue) >= NAMED for item in items)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", required=True)
    parser.add_argument("--cases", required=True, help="Case file with full cue lists.")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--show", type=int, default=8, help="Confabulation examples.")
    args = parser.parse_args()

    cues_by_case: dict[str, list[str]] = {}
    for case in read_jsonl(args.cases):
        cues = [" ".join(str(c).split()) for c in (case.get("cue_targets") or []) if str(c).strip()]
        for key in (case.get("base_id"), case.get("id")):
            if key:
                cues_by_case[str(key)] = cues
    vocabulary = sorted({cue for cues in cues_by_case.values() for cue in cues})
    print(f"cases {len(cues_by_case):,} | cue vocabulary {len(vocabulary):,}")

    rng = random.Random(args.seed)
    stats = {"gold": 0, "other": 0, "other_n": 0, "foreign": 0, "foreign_n": 0}
    rows = 0
    unjoined = 0
    examples: list[tuple[str, str, str]] = []

    for row in read_jsonl(args.readouts):
        gold = " ".join(gold_cue(row).split())
        case_cues = cues_by_case.get(str(row.get("base_id")))
        if not gold or case_cues is None:
            unjoined += 1
            continue
        rows += 1
        items = observed_items(readout_body(row))
        stats["gold"] += names_cue(items, gold)

        others = [cue for cue in case_cues if cue != gold]
        # Matched draw: as many foreign cues as there are other cues in the
        # case, so the two rates are comparable per row rather than in bulk.
        foreign = rng.sample(
            [cue for cue in vocabulary if cue not in case_cues],
            min(len(others), max(len(vocabulary) - len(case_cues), 0)),
        )
        for cue in others:
            stats["other_n"] += 1
            stats["other"] += names_cue(items, cue)
        for cue in foreign:
            stats["foreign_n"] += 1
            hit = names_cue(items, cue)
            stats["foreign"] += hit
            if hit and len(examples) < args.show:
                best = max(items, key=lambda item: overlap_f1(item, cue), default="")
                examples.append((gold, cue, best))

    if not rows:
        raise SystemExit("no readout rows joined to a case; check --cases")
    if unjoined:
        print(f"[!] {unjoined:,} rows had no matching case and were skipped")

    gold_rate = stats["gold"] / rows
    other_rate = stats["other"] / max(stats["other_n"], 1)
    foreign_rate = stats["foreign"] / max(stats["foreign_n"], 1)
    print(f"\nrows {rows:,}")
    print(f"  names its own cue        {gold_rate:.4f}")
    print(f"  names another case cue   {other_rate:.4f}  ({stats['other_n']:,} chances)")
    print(f"  names a cue not in case  {foreign_rate:.4f}  ({stats['foreign_n']:,} chances)")
    print(f"\n  leakage above chance     {other_rate - foreign_rate:+.4f}")
    print(
        "  Leakage is the second rate minus the third: naming another cue from\n"
        "  the same case more often than an unrelated one means the readout is\n"
        "  describing the case, not the position. The third rate on its own is\n"
        "  the confabulation floor -- these cues share a vocabulary, so it is not\n"
        "  expected to be zero."
    )

    if examples:
        print(f"\nreadouts matching a cue that is not in their case ({args.show} shown):")
        for gold, cue, item in examples:
            print(f"  gold    {gold}")
            print(f"  foreign {cue}")
            print(f"  said    {item}\n")


if __name__ == "__main__":
    main()
