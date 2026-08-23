"""The trajectory in words: what the readout says at each landmark.

The probe curves showed the state holds the gold to the last token while
the output defects. This scores the verbalizer on the same positions --
does the readout's conclusion still name the gold, does it ever name the
suggestion, do its grounds cite the note -- per landmark, split kept /
moved-onto-hint / moved-lost-gold, mirroring the probe table so Figure 5's
numbers and Figure 4's words come from aligned grids.

--narrate N prints N moved cases end to end (landmark, conclusion, cues,
then the first-pass answer) -- the raw material for the case panel.
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

from scripts.compare_channels_on_attribution import readout_answer, readout_cues
from scripts.score_cue_position_readouts import readout_body
from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl

LANDMARK_ORDER = ["last_cue", "note", "question", "constraint", "format", "final"]


def group_of(ladder_row: dict[str, Any]) -> str:
    if not ladder_row.get("moved"):
        return "kept"
    first = str(ladder_row.get("first_answer") or "")
    hint = str(ladder_row.get("hint_diagnosis_name") or "")
    onto = bool(hint) and is_correct(first, hint, aliases_for(hint))
    return "moved-onto-hint" if onto else "moved-lost-gold"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", nargs="+", required=True)
    parser.add_argument(
        "--ladder", required=True, help="Any ladder rung file: moved + first answers."
    )
    parser.add_argument("--narrate", type=int, default=0)
    args = parser.parse_args()

    ladder = {str(r["base_id"]): r for r in read_jsonl(args.ladder)}

    by_case: dict[str, dict[str, str]] = defaultdict(dict)
    for path in args.readouts:
        for row in read_jsonl(path):
            base_id = str(row.get("base_id") or "")
            role = str(row.get("target_role") or "")
            if base_id in ladder and role:
                by_case[base_id][role] = readout_body(row)

    print(f"cases with readouts {len(by_case):,}")
    for role in LANDMARK_ORDER:
        stats: dict[str, list[tuple[bool, bool, bool]]] = defaultdict(list)
        for base_id, roles in by_case.items():
            text = roles.get(role)
            if text is None:
                continue
            lrow = ladder[base_id]
            gold = str(lrow.get("diagnosis_name") or "")
            hint = str(lrow.get("hint_diagnosis_name") or "")
            conclusion = readout_answer(text).strip()
            stats[group_of(lrow)].append(
                (
                    is_correct(conclusion, gold, lrow.get("diagnosis_aliases") or []),
                    bool(hint) and is_correct(conclusion, hint, aliases_for(hint)),
                    "referr" in readout_cues(text).lower(),
                )
            )
        if not any(stats.values()):
            continue
        print(f"\n{role.upper()}")
        for group in ("kept", "moved-onto-hint", "moved-lost-gold"):
            rows = stats[group]
            if not rows:
                continue
            n = len(rows)
            print(
                f"  {group:<16} n={n:<5} conclusion=gold {sum(r[0] for r in rows) / n:.3f}"
                f"   =suggestion {sum(r[1] for r in rows) / n:.3f}"
                f"   cues cite note {sum(r[2] for r in rows) / n:.3f}"
            )

    if args.narrate:
        shown = 0
        for base_id, roles in by_case.items():
            lrow = ladder[base_id]
            if group_of(lrow) != "moved-onto-hint" or len(roles) < 5:
                continue
            print(
                f"\n--- {base_id}  gold {lrow.get('diagnosis_name')!r}"
                f"  note suspects {lrow.get('hint_diagnosis_name')!r}"
                f"  answered {lrow.get('first_answer')!r} ---"
            )
            for role in LANDMARK_ORDER:
                if role in roles:
                    text = roles[role]
                    print(f"  {role:<10} concl {readout_answer(text).strip()!r}")
                    cues = readout_cues(text)
                    if cues:
                        print(f"  {'':<10} cues  {cues[:100]!r}")
            shown += 1
            if shown >= args.narrate:
                break


if __name__ == "__main__":
    main()
