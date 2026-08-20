"""Ask how much of a low accuracy is the model and how much is the scorer.

`is_correct` requires the answer to contain the gold name or be contained by
it. On DDXPlus's 49 fixed pathology names that is exactly right. On MCR's
6,934 free-text labels it fails in a direction that costs accuracy: one word
between the shared terms -- "phototoxic **drug** reaction" against
"phototoxic reaction" -- and a correct answer scores wrong.

This prints the strict rate beside a token-overlap rate and, more usefully,
the pairs that separate them, so the decision about which metric the paper
quotes is made from the actual disagreements rather than from a rate. A large
gap made of genuine synonyms means the strict rule is undercounting; a large
gap made of near-misses ("carcinoma" against "renal cell carcinoma") means the
loose rule would be inflating.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, token_f1
from src.jsonl import read_jsonl

THRESHOLDS = (0.5, 0.67, 0.8, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", required=True)
    parser.add_argument("--show", type=int, default=25, help="Disagreements to print.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Token-F1 above which a strict miss is shown as a candidate match.",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.answers))
    if not rows:
        raise SystemExit(f"no rows in {args.answers}")

    scored = []
    for row in rows:
        answer = row.get("answer")
        gold = str(row.get("diagnosis_name") or "")
        aliases = [str(a) for a in (row.get("diagnosis_aliases") or [])]
        scored.append(
            {
                "id": row.get("id"),
                "answer": answer,
                "gold": gold,
                "strict": is_correct(answer, gold, aliases),
                "f1": token_f1(answer, gold, aliases),
            }
        )

    n = len(scored)
    strict = sum(1 for s in scored if s["strict"])
    print(f"answers: {args.answers}")
    print(f"n: {n:,}")
    print(f"strict_accuracy: {strict / n:.4f}  ({strict:,} rows)")
    for threshold in THRESHOLDS:
        hits = sum(1 for s in scored if s["f1"] >= threshold)
        print(f"token_f1 >= {threshold:.2f}: {hits / n:.4f}  ({hits:,} rows)")
    # A strict hit that the loose rule misses would mean the two disagree in
    # both directions, which would make the gap uninterpretable.
    both_ways = sum(1 for s in scored if s["strict"] and s["f1"] < args.threshold)
    if both_ways:
        print(f"[!] {both_ways} rows pass the strict rule but fall below the threshold.")

    candidates = [
        s for s in scored if not s["strict"] and s["f1"] >= args.threshold
    ]
    candidates.sort(key=lambda s: -s["f1"])
    print(
        f"\nStrict misses with token_f1 >= {args.threshold}: {len(candidates):,} "
        f"({len(candidates) / n:.4f} of all rows). Read these -- they are the "
        "difference between the two metrics:"
    )
    for entry in candidates[: args.show]:
        print(f"  f1={entry['f1']:.2f}  gold={entry['gold']!r}  answer={entry['answer']!r}")

    misses = [s for s in scored if not s["strict"] and s["f1"] < args.threshold]
    print(f"\nStrict misses with low overlap: {len(misses):,}. A sample:")
    for entry in misses[: args.show]:
        print(f"  f1={entry['f1']:.2f}  gold={entry['gold']!r}  answer={entry['answer']!r}")


if __name__ == "__main__":
    main()
