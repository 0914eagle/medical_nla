"""Turn a pool of readouts into numbered judging requests for an LLM judge.

The judge being a different model from the one under study is the point --
judging Gemma's readouts with Gemma invites the obvious objection.

Two ways out. `--requests` writes {id, prompt} JSONL for run_judge.py, one
pair per request, and is what to use: an external judge is reachable now, a
skipped or renumbered line cannot shift every verdict after it, and a killed
run resumes per pair. The chunk files remain for pasting into a chat when no
judge is reachable; `--from-judge` converts a run_judge output back into the
verdict format `apply_readout_judgements.py` already reads, so both paths meet
at the same place.

**Distinct pairs, not rows.** DDXPlus renders its cues from a fixed
questionnaire, so the same (gold, readout) pair recurs: eleven of one sixty-row
sample were a single fever cue. Judging pairs cuts the volume several-fold and
makes the result self-consistent -- the same pair cannot be judged A in one row
and B in another.

The rubric is written into every chunk rather than kept in someone's head, so
the file that produced a number carries the definition the number was made
under.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_cue_position_readouts import gold_cue, readout_body
from src.cue_readout_scoring import score_readout
from src.jsonl import read_jsonl, write_jsonl

RUBRIC = """\
You are scoring whether a readout describes the same clinical finding as the
gold text. The readout was produced by a model reading an internal activation
vector; the gold is the finding that vector was taken from. Judge meaning, not
wording.

  A  the same finding, however differently worded
     ("undergo dialysis" / "have to dialyze" is A)
     ("a fever (either felt or measured with a thermometer)" / "had a fever
      (defined as 100F or higher)" is A -- the gloss is the questionnaire's,
      not part of the finding)
  B  the finding is right but an attribute is wrong, missing or invented, or
     only one clause of a compound gold is read
     ("the itching rated 10 for severity" / "rated 3 for severity" is B)
     ("involuntary weight loss over 3 months" / "lost 2 pounds in the last 3
      months" is B -- the quantity is not in the gold)
  C  a different finding
  D  empty, refused, or no clinical content

When A and B are both arguable, answer B. When B and C are both arguable,
answer C. The rate being measured is a claim about faithfulness, so err
against it.

Reply with one verdict per line and nothing else, in the form

  1=A
  2=C
  3=B

covering every number below exactly once."""

# The same rubric asked one pair at a time, for `--requests`. Batching 120
# pairs into one prompt was a constraint of pasting, and it carries a failure
# the paste era could not avoid: a judge that skips or renumbers one line
# shifts every verdict after it, silently. One pair per request cannot
# misalign, resumes per pair, and costs only the rubric repeated.
SINGLE_RUBRIC = RUBRIC.split("Reply with one verdict")[0] + """\
Reply with exactly one character and nothing else: A, B, C or D.

GOLD: {gold}
READ: {read}"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", required=True, help="A run_nla output JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=120,
        help="Pairs per chunk. Small enough to paste, large enough to be few.",
    )
    parser.add_argument(
        "--requests",
        help="Also write {id, prompt} JSONL for run_judge.py -- one request "
        "per pair, ids matching judge_index.jsonl's n. Use this instead of "
        "pasting chunks now that an external judge is reachable.",
    )
    parser.add_argument(
        "--from-judge",
        help="Convert a run_judge.py output back into the verdicts_NN.txt "
        "format apply_readout_judgements.py reads, and exit. Needs --out-dir.",
    )
    parser.add_argument(
        "--skip-judged",
        help=(
            "TSV of pairs already judged (verdict/pool/gold/read), whose pairs "
            "are left out of the request and merged back in afterwards."
        ),
    )
    args = parser.parse_args()

    if args.from_judge:
        out_dir = Path(args.out_dir)
        lines, unparsed = [], 0
        for row in read_jsonl(args.from_judge):
            verdict = str(row.get("response") or "").strip().upper()
            letter = next((c for c in verdict if c in "ABCD"), "")
            if not letter:
                unparsed += 1
                continue
            lines.append(f"{row.get('id')}={letter}")
        path = out_dir / "verdicts_01.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"verdicts {len(lines):,} -> {path}   unparseable {unparsed:,}")
        if unparsed:
            print("  unparseable verdicts are dropped, not defaulted -- a judge "
                  "that did not answer is missing data, and any default would "
                  "move the rate it is measuring.")
        return

    already: set[tuple[str, str]] = set()
    if args.skip_judged:
        for line in Path(args.skip_judged).read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip() or line.startswith("verdict\t"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                already.add((parts[2], parts[3]))

    pairs: dict[tuple[str, str], list[str]] = {}
    for row in read_jsonl(args.readouts):
        gold = gold_cue(row)
        if not gold:
            continue
        read = " ".join(str(score_readout(readout_body(row), gold)["best_item"]).split())
        gold = " ".join(gold.split())
        pairs.setdefault((gold, read), []).append(str(row.get("id")))

    fresh = [pair for pair in pairs if pair not in already]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The numbering is the join key, so it is written down rather than implied
    # by the order of anything.
    index = [
        {"n": n, "gold": gold, "read": read, "ids": pairs[(gold, read)]}
        for n, (gold, read) in enumerate(fresh, start=1)
    ]
    write_jsonl(out_dir / "judge_index.jsonl", index)

    if args.requests:
        write_jsonl(Path(args.requests), [
            {"id": str(e["n"]),
             "prompt": SINGLE_RUBRIC.format(gold=e["gold"], read=e["read"])}
            for e in index
        ])
        print(f"requests        {len(index):,} -> {args.requests}")

    chunks = 0
    for start in range(0, len(fresh), args.chunk_size):
        chunks += 1
        block = index[start : start + args.chunk_size]
        lines = [RUBRIC, ""]
        for entry in block:
            lines.append(f"{entry['n']}. GOLD: {entry['gold']}")
            lines.append(f"   READ: {entry['read']}")
        path = out_dir / f"judge_request_{chunks:02d}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    covered = sum(len(pairs[p]) for p in fresh)
    total_rows = sum(len(ids) for ids in pairs.values())
    print(f"rows            {total_rows:,}")
    print(f"distinct pairs  {len(pairs):,}")
    if already:
        print(f"already judged  {len(pairs) - len(fresh):,} pairs (skipped)")
    print(f"to judge        {len(fresh):,} pairs covering {covered:,} rows")
    print(f"chunks          {chunks} in {out_dir}")
    print(
        "\nPaste each judge_request_NN.txt into a chat with a model that is not "
        "the backbone,\nsave the reply as verdicts_NN.txt beside it, then run "
        "apply_readout_judgements.py."
    )


if __name__ == "__main__":
    main()
