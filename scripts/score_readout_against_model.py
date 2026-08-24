"""Score a readout against what the model said, not only against the gold.

The MCR conclusion readout matches the gold on 114 of 821 held-out rows,
13.89%. The source model matches the gold on 113 of the same 821, 13.8%. Those
two numbers being the same is either a coincidence or the whole story, and one
join settles which.

A readout reads a hidden state. What that state contains is the model's own
conclusion, which on this corpus is the gold only 12% of the time. So a
faithful readout **must** be wrong against the gold wherever the model is
wrong -- scoring it against the gold measures the model's accuracy with extra
steps, and a readout that beat it would be the alarming result, not the good
one. The quantity the instrument is for is agreement with the model.

Three populations, because they answer different questions:

  source-correct rows   the model reached the gold, so gold and conclusion
                        coincide and either target scores the same. This is
                        the only place the old number meant what it looked
                        like.
  source-wrong rows     the model concluded something other than the gold.
                        Agreement with the model here is faithfulness;
                        agreement with the gold here would be confabulation
                        that happens to land right.
  all rows              reported so the headline is not assembled from the
                        halves.

The chance floor is printed beside every rate. On an open vocabulary where
most labels occur once, a readout that always emitted the corpus's most common
diagnosis would score near zero, so the floor is low -- but it is stated
rather than assumed, because "beats chance" is the claim being made.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct
from src.jsonl import read_jsonl

ANSWER = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def readout_answer(text: str) -> str:
    match = ANSWER.search(text or "")
    return match.group(1).strip() if match else ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", required=True, help="run_nla output.")
    parser.add_argument("--answers", nargs="+", required=True,
                        help="Source answers carrying the model's own answer.")
    parser.add_argument("--show", type=int, default=6)
    args = parser.parse_args()

    said: dict[str, dict] = {}
    for path in args.answers:
        for row in read_jsonl(path):
            base_id = str(row.get("base_id") or row.get("id") or "")
            if base_id:
                said[base_id] = row

    rows = []
    unparsed = unjoined = 0
    for row in read_jsonl(args.readouts):
        read = readout_answer(str(row.get("nla_output") or ""))
        if not read:
            unparsed += 1
            continue
        source = said.get(str(row.get("base_id") or ""))
        if source is None:
            unjoined += 1
            continue
        gold = str(row.get("diagnosis_name") or "")
        aliases = [str(a) for a in (row.get("diagnosis_aliases") or [])]
        model = str(source.get("answer") or "").strip()
        rows.append({
            "read": read,
            "gold_match": is_correct(read, gold, aliases),
            # Symmetric by construction: what the model said is the target, so
            # its own string is the reference and it carries no alias list.
            "model_match": is_correct(read, model, []) if model else False,
            "source_correct": bool(source.get("source_correct")),
            "gold": gold,
            "model": model,
        })

    if not rows:
        raise SystemExit("nothing joined -- do the readouts and answers share base_id?")

    # What a readout scores by emitting the single most common answer.
    floor_gold = Counter(r["gold"] for r in rows).most_common(1)[0][1] / len(rows)
    floor_model = Counter(r["model"] for r in rows).most_common(1)[0][1] / len(rows)

    print(f"rows {len(rows):,}   no <answer> {unparsed:,}   unjoined {unjoined:,}")
    print(f"\n  {'population':<22}{'n':>7}{'vs gold':>10}{'vs model':>11}")
    groups = [
        ("all", rows),
        ("source-correct", [r for r in rows if r["source_correct"]]),
        ("source-wrong", [r for r in rows if not r["source_correct"]]),
    ]
    for name, group in groups:
        if not group:
            continue
        g = sum(r["gold_match"] for r in group) / len(group)
        m = sum(r["model_match"] for r in group) / len(group)
        print(f"  {name:<22}{len(group):>7,}{g:>10.4f}{m:>11.4f}")
    print(f"  {'(most-common floor)':<22}{'':>7}{floor_gold:>10.4f}{floor_model:>11.4f}")

    wrong = [r for r in rows if not r["source_correct"]]
    if wrong:
        agree = sum(r["model_match"] for r in wrong)
        print(f"\n  On the {len(wrong):,} rows where the model did NOT reach the "
              f"gold, the readout\n  named what the model said in {agree:,} "
              f"({agree / len(wrong):.4f}) and the gold in "
              f"{sum(r['gold_match'] for r in wrong)} "
              f"({sum(r['gold_match'] for r in wrong) / len(wrong):.4f}).")
        print("  The first is faithfulness. The second, if it were high, would be "
              "a readout\n  guessing the right answer from context rather than "
              "reading the state.")
        for r in wrong[: args.show]:
            if r["model_match"]:
                print(f"\n    gold:  {r['gold']}")
                print(f"    model: {r['model']}")
                print(f"    read:  {r['read']}")


if __name__ == "__main__":
    main()
