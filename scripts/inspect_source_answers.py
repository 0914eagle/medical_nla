"""Say why answers failed to parse, without re-running the model.

A parse rate of zero has three quite different causes and they need opposite
fixes, so guessing is expensive:

- **truncation** -- the model is still reasoning when `max_new_tokens` runs
  out, so the closing string never appears. Fix: raise the budget.
- **non-compliance** -- the model answers, but not in the demanded form.
  Fix: the prompt.
- **a parser miss** -- the closing string is there and the regex does not
  match it. Fix: the regex.

The three are told apart by two facts per response: whether "answer is"
appears at all, and whether the response ends mid-sentence.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.case_prompts import parse_answer
from src.jsonl import read_jsonl

# A response that stops mid-sentence was cut off; one that ends on punctuation
# finished on its own terms and simply did not comply.
FINISHED_ENDINGS = (".", "!", "?", '"', "*", ")", ":")


def looks_truncated(text: str) -> bool:
    stripped = str(text or "").rstrip()
    return bool(stripped) and not stripped.endswith(FINISHED_ENDINGS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--show", type=int, default=3, help="Responses to print in full.")
    parser.add_argument("--head-chars", type=int, default=400)
    parser.add_argument("--tail-chars", type=int, default=300)
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    if not rows:
        raise SystemExit(f"no rows in {args.answers}")

    n = len(rows)
    lengths = sorted(len(str(row.get("response") or "")) for row in rows)
    has_phrase = sum(1 for row in rows if "answer is" in str(row.get("response") or "").lower())
    parsed = sum(1 for row in rows if parse_answer(row.get("response") or ""))
    truncated = sum(1 for row in rows if looks_truncated(row.get("response")))
    empty = sum(1 for row in rows if not str(row.get("response") or "").strip())

    print(f"answers: {args.answers}")
    print(f"n: {n:,}")
    print(f"condition: {Counter(str(row.get('condition')) for row in rows).most_common()}")
    print(f"empty_response: {empty / n:.3f}")
    print(f"contains_'answer is': {has_phrase / n:.3f}")
    print(f"parses: {parsed / n:.3f}")
    print(f"ends_mid_sentence: {truncated / n:.3f}")
    print(
        "response_chars: min {} / p50 {} / p90 {} / max {}".format(
            lengths[0], lengths[n // 2], lengths[int(n * 0.9)], lengths[-1]
        )
    )

    # The phrase present but unparsed is the only cause that points at the
    # regex; call it out rather than leaving it to be read off the rates.
    if has_phrase > parsed:
        print(f"\n[!] {has_phrase - parsed} responses contain the phrase but do not parse.")
        for row in rows:
            text = str(row.get("response") or "")
            if "answer is" in text.lower() and not parse_answer(text):
                index = text.lower().rfind("answer is")
                print(f"    ...{text[max(0, index - 60) : index + 120]!r}")
                break
    if truncated > n * 0.5 and has_phrase < n * 0.5:
        print("\n[!] Most responses end mid-sentence and never reach the closing string:")
        print("    this is a max_new_tokens problem, not a prompt problem.")

    for row in rows[: args.show]:
        text = str(row.get("response") or "")
        print(f"\n{'=' * 72}\nid: {row.get('id')}  chars: {len(text)}")
        print(f"--- response head ---\n{text[: args.head_chars]}")
        if len(text) > args.head_chars + args.tail_chars:
            print(f"--- response tail ---\n{text[-args.tail_chars :]}")


if __name__ == "__main__":
    main()
