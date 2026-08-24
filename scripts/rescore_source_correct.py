"""Recompute the stored `source_correct` field with the current matcher.

`run_source_answers.py` writes `source_correct` when the answers are generated,
and every analyser downstream reads that field rather than re-scoring:
`analyze_hint_effect` takes accuracy straight from it, and `lost_the_gold` --
therefore `moved`, therefore the population of every attribution and trajectory
analysis -- is defined by it. `took_the_hint` is the exception; it calls
`is_correct` live.

So a fix to the matcher reaches half the pipeline and not the other half. After
word boundaries went in, the adoption count moved (95 -> 91, live) while the
accuracies did not (stored). Re-running the analysers changes nothing until the
field is rewritten.

This rewrites it, reports how many rows flip in each arm, and names the cases
whose **no-note arm** is no longer correct. Those matter more than the count:
the hint experiment selected its cases for being answered correctly without the
note, and a case that no longer qualifies was selected under a rule that no
longer holds. It cannot be un-run, but it can be seen -- `lost_the_gold`
already refuses to fire on it, so it silently leaves the moved population, and
the size of that exit belongs in the record rather than in a diff.

Writes to --output. Rewriting a result file in place would leave no way to tell
which matcher produced a number, which is the situation this exists to end.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold-field", default="diagnosis_name")
    parser.add_argument("--show", type=int, default=10)
    parser.add_argument(
        "--also", nargs="*", default=[], metavar="FLAG:ANSWER",
        help="Extra stored verdicts to recompute, as flag_field:answer_field. "
        "The correction ladder carries first_correct beside first_answer, and "
        "its analyser reads both -- leaving them behind would rescore the "
        "second pass under the new matcher and the first under the old one.",
    )
    parser.add_argument(
        "--carry-moved-from", nargs=2, default=None, metavar=("ANSWERS", "CASES"),
        help="Recompute `moved` and `took_the_hint` from a rescored answers "
        "file and its case file, joined by base_id. A ladder row cannot derive "
        "them itself -- both need the no-note arm, which lives there.",
    )
    args = parser.parse_args()

    carried: dict[str, dict[str, bool]] = {}
    if args.carry_moved_from:
        from scripts.analyze_hint_effect import group_by_case, lost_the_gold, took_the_hint
        src, case_file = args.carry_moved_from
        for base_id, case in group_by_case([src], case_file).items():
            if "wrong" not in case or "none" not in case:
                continue
            hint = took_the_hint(case, "wrong")
            carried[base_id] = {
                "took_the_hint": bool(hint),
                "moved": bool(hint or lost_the_gold(case, "wrong")),
            }
        print(f"carried moved/took_the_hint for {len(carried):,} cases")

    rows = list(read_jsonl(args.answers))
    flips: dict[str, Counter[str]] = defaultdict(Counter)
    by_arm: dict[str, list[bool]] = defaultdict(list)
    no_note_lost: list[tuple[str, str, str]] = []
    for row in rows:
        gold = str(row.get(args.gold_field) or "").strip()
        if not gold:
            continue
        arm = str(row.get("hint_variant") or "-")
        old = bool(row.get("source_correct"))
        new = is_correct(row.get("answer"), gold, aliases_for(gold))
        row["source_correct"] = new
        row["source_correct_matcher"] = "word_boundary_2026_08_24"
        by_arm[arm].append(new)
        for spec in args.also:
            flag_field, answer_field = spec.split(":", 1)
            if answer_field in row:
                was = bool(row.get(flag_field))
                now = is_correct(row.get(answer_field), gold, aliases_for(gold))
                row[flag_field] = now
                if was != now:
                    flips[f"{arm} [{flag_field}]"]["gained" if now else "lost"] += 1
        if carried:
            extra = carried.get(str(row.get("base_id") or ""))
            if extra:
                for key, value in extra.items():
                    if bool(row.get(key)) != value:
                        flips[f"{arm} [{key}]"]["gained" if value else "lost"] += 1
                    row[key] = value
        if old != new:
            flips[arm]["gained" if new else "lost"] += 1
            if arm == "none" and not new:
                no_note_lost.append((str(row.get("base_id") or row.get("id")),
                                     gold, str(row.get("answer") or "")))

    write_jsonl(Path(args.output), rows)
    total = sum(sum(c.values()) for c in flips.values())
    print(f"rows {len(rows):,}   flipped {total:,}   -> {args.output}")
    print(f"\n  {'arm':<10}{'n':>8}{'accuracy':>11}{'lost':>7}{'gained':>8}")
    for arm in sorted(by_arm):
        vals = by_arm[arm]
        print(f"  {arm:<10}{len(vals):>8,}{sum(vals) / len(vals):>11.4f}"
              f"{flips[arm]['lost']:>7,}{flips[arm]['gained']:>8,}")

    extra_flips = {k: v for k, v in flips.items() if "[" in k}
    if extra_flips:
        print("\n  carried/derived fields (not part of the arm table above):")
        for key in sorted(extra_flips):
            c = extra_flips[key]
            print(f"    {key:<28} lost {c['lost']:,}   gained {c['gained']:,}")

    if no_note_lost:
        print(f"\n  ⚠ {len(no_note_lost):,} cases no longer answer correctly WITHOUT "
              "the note.")
        print("    The experiment selected its cases for exactly that, so these no "
              "longer meet\n    the design condition. lost_the_gold cannot fire on "
              "them, so they leave the\n    moved population on their own -- record "
              "the number, do not re-select.")
        for base_id, gold, answer in no_note_lost[: args.show]:
            print(f"      {base_id}")
            print(f"        gold:   {gold}")
            print(f"        answer: {answer}")
        if len(no_note_lost) > args.show:
            print(f"      ... {len(no_note_lost) - args.show:,} more")
    else:
        print("\n  every case still answers correctly without the note")

    print("\n  Re-run the analysers against --output, not the original file.")


if __name__ == "__main__":
    main()
