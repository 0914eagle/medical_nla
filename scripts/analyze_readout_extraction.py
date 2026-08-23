"""Score the judge's extractions: does the tuned readout read more than the
untuned checkpoint, or only write more neatly?

Both channels' texts went through one judge asking one question, so the
comparison here is between two extracted claims scored by the same rule the
answers already use. Reported per channel, split by what the model did with
the case:

- **conclusion = gold** on cases that lost the gold is the rift claim. If the
  untuned checkpoint reaches it too, the rift is a property of the activation
  and our adapter only made it legible -- which is the honest and still
  publishable version.
- **NONE / MULTIPLE rates** are the other half of the same question. A channel
  that says nothing extractable on most cases has not read the state, however
  well it scores on the cases where it did say something, so the rates are
  printed beside the accuracy rather than folded into the denominator.
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

from scripts.analyze_trajectory_readouts import group_of
from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl

ABSTAIN = {"none", "multiple"}


def verdict_of(response: str) -> str:
    """The judge's answer, reduced to a claim.

    Taken from the last non-empty line: a judge told to answer with a name
    alone mostly does, but when it prefaces ("Based on the text, ...") the
    name is what it ends on.
    """
    lines = [line.strip(" .*`") for line in str(response or "").splitlines()]
    for line in reversed(lines):
        if line:
            return line
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extractions", nargs="+", required=True,
                        help="run_source_answers outputs over the extraction prompts.")
    parser.add_argument("--ladder", required=True,
                        help="Any ladder rung file: gold, suggestion, moved, first answer.")
    parser.add_argument("--examples", type=int, default=0,
                        help="Print this many disagreements between channels.")
    args = parser.parse_args()

    ladder = {str(r["base_id"]): r for r in read_jsonl(args.ladder)}

    # (channel, role, group) -> list of (hits_gold, hits_suggestion, abstained)
    stats: dict[tuple[str, str, str], list[tuple[bool, bool, bool]]] = defaultdict(list)
    verdicts: dict[tuple[str, str], str] = {}
    channels: list[str] = []
    for path in args.extractions:
        for row in read_jsonl(path):
            base_id = str(row.get("base_id") or "")
            channel = str(row.get("readout_channel") or "?")
            role = str(row.get("target_role") or "final")
            if base_id not in ladder:
                continue
            if channel not in channels:
                channels.append(channel)
            claim = verdict_of(row.get("response"))
            verdicts[(base_id, channel)] = claim
            lrow = ladder[base_id]
            gold = str(lrow.get("diagnosis_name") or "")
            hint = str(lrow.get("hint_diagnosis_name") or "")
            abstained = claim.lower() in ABSTAIN or not claim
            stats[(channel, role, group_of(lrow))].append(
                (
                    not abstained
                    and is_correct(claim, gold, list(lrow.get("diagnosis_aliases") or [])),
                    not abstained
                    and bool(hint)
                    and is_correct(claim, hint, aliases_for(hint)),
                    abstained,
                )
            )

    roles = sorted({role for _, role, _ in stats})
    for role in roles:
        print(f"\n=== {role.upper()} ===")
        for group in ("kept", "moved-onto-hint", "moved-lost-gold"):
            present = [c for c in channels if stats.get((c, role, group))]
            if not present:
                continue
            print(f"\n  [{group}]")
            for channel in present:
                rows = stats[(channel, role, group)]
                n = len(rows)
                print(
                    f"    {channel:<10} n={n:<5}"
                    f" claim=gold {sum(r[0] for r in rows) / n:.3f}"
                    f"   claim=suggestion {sum(r[1] for r in rows) / n:.3f}"
                    f"   no extractable claim {sum(r[2] for r in rows) / n:.3f}"
                )

    if args.examples and len(channels) >= 2:
        a, b = channels[0], channels[1]
        shown = 0
        print(f"\ndisagreements ({a} vs {b}):")
        for base_id, lrow in ladder.items():
            va, vb = verdicts.get((base_id, a)), verdicts.get((base_id, b))
            if not va or not vb or va.lower() == vb.lower():
                continue
            print(f"  {base_id}  gold {lrow.get('diagnosis_name')!r}"
                  f"  answered {lrow.get('first_answer')!r}")
            print(f"    {a:<10} {va!r}")
            print(f"    {b:<10} {vb!r}")
            shown += 1
            if shown >= args.examples:
                break

    counts = Counter(c for c, _, _ in stats)
    print(f"\nchannels scored: {dict(counts)}")


if __name__ == "__main__":
    main()
