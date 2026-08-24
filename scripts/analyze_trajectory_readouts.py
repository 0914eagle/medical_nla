"""The trajectory in words: what the readout says at each landmark.

The probe curves showed the state holds the gold to the last token while
the output defects. This scores the verbalizer on the same positions --
does the readout's conclusion still name the gold, does it ever name the
suggestion, do its grounds cite the note -- per landmark, split kept /
moved-onto-hint / moved-lost-gold, mirroring the probe table so Figure 5's
numbers and Figure 4's words come from aligned grids.

--narrate N prints N moved cases end to end (landmark, conclusion, cues,
then the first-pass answer) -- the raw material for the case panel.

--lenient scores by containment anywhere in the output instead of inside the
<answer> tag, which is the only way to put the untuned checkpoint on the same
grid: it emits no schema, so a tag-based rule scores it zero by construction
and proves nothing. The generosity cuts both ways, so lenient mode also counts
how many distinct diagnoses each readout names. A channel that names six
conditions per case and is therefore "right" about the gold has not read the
state, and that number is what says so.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.compare_channels_on_attribution import readout_answer, readout_cues
from scripts.score_cue_position_readouts import readout_body
from src.answer_matching import is_correct, normalize
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl

LANDMARK_ORDER = ["last_cue", "note", "question", "constraint", "format", "final"]


def mentions_name(text: str, name: str, aliases: list[str]) -> bool:
    """Whether free text names a diagnosis, matched on word boundaries.

    `is_correct` is bare containment, which is right for a short answer field
    and wrong here: "PE", an alias of pulmonary embolism, matches inside
    "appears", and the untuned checkpoint writes hundreds of words per
    readout. Scored that way it would look like a superb reader of every
    vector it is handed. The same collision cost thirty-four false hits in the
    gold-in-chart filter, which is where this rule comes from.
    """
    haystack = normalize(text)
    for candidate in [name, *aliases]:
        needle = normalize(candidate)
        if needle and re.search(
            rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", haystack
        ):
            return True
    return False


def names_mentioned(text: str, vocabulary: list[str]) -> int:
    """How many distinct diagnoses this readout names.

    The precision half of lenient scoring. Containment is generous by design;
    without this counter a rambling channel that lists half the differential
    looks as accurate as one that names a single condition.
    """
    return sum(1 for name in vocabulary if mentions_name(text, name, aliases_for(name)))


_WARNED: list[bool] = []


def group_of(ladder_row: dict[str, Any]) -> str:
    """kept / moved-onto-hint / moved-lost-gold, by the same rule as everywhere.

    The onto-hint split used to be "the answer equals the suspicion", which
    counts cases whose no-note arm already answered that suspicion -- cases
    with nothing for the note to move. `took_the_hint` excludes them, and the
    two rules disagreed on exactly 12 of 324 cases (107 vs 95), a gap that was
    filed for weeks as an alias-matching discrepancy. Both rules call the same
    alias-aware matcher; only this clause differed.

    Rows built before the builder carried the field fall back to the old rule
    and say so, because silently reporting 107 as 95 would be worse than
    either number.
    """
    if not ladder_row.get("moved"):
        return "kept"
    if "took_the_hint" in ladder_row:
        return "moved-onto-hint" if ladder_row["took_the_hint"] else "moved-lost-gold"
    if not _WARNED:
        _WARNED.append(True)
        print("[warn] ladder rows predate took_the_hint; falling back to "
              "answer==suggestion, which over-counts onto-hint. Rebuild with "
              "make_correction_ladder_cases.py to match the other analyses.",
              file=sys.stderr)
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
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Score by containment in the whole output rather than the <answer> "
        "tag, so an untuned checkpoint that emits no schema can be compared. "
        "Also reports diagnoses named per readout -- run it on the tuned "
        "outputs too, or the comparison is between two different rules.",
    )
    parser.add_argument(
        "--narrate-group",
        default="moved-onto-hint",
        choices=["kept", "moved-onto-hint", "moved-lost-gold"],
        help="Which trajectory group to print end to end. The lost-gold group "
        "is where the rift shows: conclusion=gold, output elsewhere.",
    )
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
    vocabulary = sorted(
        {str(r.get("diagnosis_name") or "") for r in ladder.values()} - {""}
    )
    if args.lenient:
        print(f"lenient scoring: containment over {len(vocabulary)} diagnosis names")

    for role in LANDMARK_ORDER:
        stats: dict[str, list[tuple[bool, bool, bool, int]]] = defaultdict(list)
        for base_id, roles in by_case.items():
            text = roles.get(role)
            if text is None:
                continue
            lrow = ladder[base_id]
            gold = str(lrow.get("diagnosis_name") or "")
            hint = str(lrow.get("hint_diagnosis_name") or "")
            # Lenient reads the whole output; strict reads only the slot the
            # tuned readout is supposed to fill.
            cues = text if args.lenient else readout_cues(text)
            if args.lenient:
                # Whole output, word-boundary matched: a tag rule scores the
                # untuned checkpoint zero by construction, bare containment
                # scores it near one.
                hits_gold = mentions_name(
                    text, gold, list(lrow.get("diagnosis_aliases") or [])
                )
                hits_hint = bool(hint) and mentions_name(text, hint, aliases_for(hint))
            else:
                conclusion = readout_answer(text).strip()
                hits_gold = is_correct(
                    conclusion, gold, lrow.get("diagnosis_aliases") or []
                )
                hits_hint = bool(hint) and is_correct(
                    conclusion, hint, aliases_for(hint)
                )
            stats[group_of(lrow)].append(
                (
                    hits_gold,
                    hits_hint,
                    "referr" in cues.lower(),
                    names_mentioned(text, vocabulary) if args.lenient else 0,
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
            line = (
                f"  {group:<16} n={n:<5} conclusion=gold {sum(r[0] for r in rows) / n:.3f}"
                f"   =suggestion {sum(r[1] for r in rows) / n:.3f}"
                f"   cues cite note {sum(r[2] for r in rows) / n:.3f}"
            )
            if args.lenient:
                line += f"   diagnoses named {sum(r[3] for r in rows) / n:.2f}"
            print(line)

    if args.narrate:
        shown = 0
        for base_id, roles in by_case.items():
            lrow = ladder[base_id]
            if group_of(lrow) != args.narrate_group or len(roles) < 5:
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
