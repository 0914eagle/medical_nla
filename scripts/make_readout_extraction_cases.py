"""Judge prompts that turn a free-form readout into a scoreable claim.

Rule-based scoring cannot compare a tuned readout against an untuned one, and
the layer sweep already measured why: on the same vanilla outputs the lexical
rule scored 0.04-0.14 and the soft rule 0.56-0.66, because vanilla wraps real
content in confabulated frames ("the player's inventory", "a medical forum
question"). One rule undercounts, the other overcounts, and the truth sits
between two numbers we cannot defend. Tightening the regex does not fix this;
the object being scored is prose with a frame around it, and deciding what a
piece of prose claims is a reading task.

So the judge does the reading, and only the reading. It is asked one question
-- which diagnosis, if any, does this text put forward as what the state
represents -- and answers with a name or NONE. Everything after that is the
same scoring code both channels already go through, so the tuned and untuned
readouts are compared on one rule that neither was built to satisfy.

Two properties make this fair rather than generous. The judge never sees the
gold, the referral's suspicion, or which system wrote the text, so it cannot
grade toward an answer. And a text that names several conditions is a
`multiple` verdict, not a lucky hit: a channel that lists half the
differential must be visible as doing that, or it scores well for saying
everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_cue_position_readouts import readout_body
from src.jsonl import read_jsonl, write_jsonl

EXTRACT_TEMPLATE = """Below is one system's description of an internal state \
taken from a medical diagnosis model as it read a patient case.

The description may be terse or rambling, and may be wrapped in commentary \
about formats, tasks, or unrelated topics. Ignore the framing and read only \
what it claims about the patient.

--- description ---
{text}
--- end ---

Which single diagnosis does this description put forward as the condition the \
state represents?

Rules:
- If it puts forward exactly one condition, answer with that condition's name.
- If it puts forward several with none clearly leading, answer: MULTIPLE
- If it names no condition at all, answer: NONE
- Do not guess from your own medical knowledge. Report only what the text \
claims.

Answer with the diagnosis name alone, or MULTIPLE, or NONE. No explanation."""


def extraction_prompt(text: str) -> str:
    return EXTRACT_TEMPLATE.format(text=text.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readouts", nargs="+", required=True,
                        help="Readout files to score; pass tuned and vanilla separately.")
    parser.add_argument("--channel", required=True,
                        help="Label carried onto every row, e.g. tuned / vanilla.")
    parser.add_argument("--roles", nargs="+", default=["final"],
                        help="Landmarks to score. Default is the answer position, "
                        "where the rift claim lives.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    empty = 0
    for path in args.readouts:
        for row in read_jsonl(path):
            role = str(row.get("target_role") or "")
            base_id = str(row.get("base_id") or "")
            if not base_id or role not in args.roles:
                continue
            text = readout_body(row).strip()
            if not text:
                empty += 1
                continue
            rows.append(
                {
                    "id": f"{base_id}__extract_{args.channel}_{role}",
                    "base_id": base_id,
                    "readout_channel": args.channel,
                    "target_role": role,
                    "readout_text": text,
                    "prompt": extraction_prompt(text),
                }
            )

    if not rows:
        raise SystemExit("no rows built; check --readouts and --roles")
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows):,} extraction prompts ({args.channel}) -> {args.output}")
    lengths = sorted(len(r["readout_text"]) for r in rows)
    print(f"readout length: median {lengths[len(lengths) // 2]:,} chars, "
          f"max {lengths[-1]:,}")
    if empty:
        print(f"skipped {empty:,} rows with empty readout text")


if __name__ == "__main__":
    main()
