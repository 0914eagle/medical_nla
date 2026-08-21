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

Disagreements are printed once per distinct (gold, answer) pair with a count.
DDXPlus has a closed vocabulary of 49 pathology names, so the same pair recurs
hundreds of times and a flat listing shows one pair sixty times over; MCR's
6,934 free-text labels make nearly every pair unique and the grouping is then a
no-op. The point is that the review is by pair, and a pair's count is how much
of the corpus turns on it.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
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
    parser.add_argument(
        "--cases",
        help=(
            "Case file to take diagnosis_aliases from, for answer files written "
            "before the run recorded them. Without the aliases the recomputation "
            "here is stricter than the run was."
        ),
    )
    parser.add_argument(
        "--show", type=int, default=25, help="Distinct (gold, answer) pairs to print."
    )
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

    alias_by_id: dict[str, list[str]] = {}
    if args.cases:
        for case in read_jsonl(args.cases):
            aliases = [str(a) for a in (case.get("diagnosis_aliases") or [])]
            for key in (case.get("id"), case.get("base_id")):
                if key:
                    alias_by_id[str(key)] = aliases

    scored = []
    n_aliased = 0
    for row in rows:
        answer = row.get("answer")
        gold = str(row.get("diagnosis_name") or "")
        aliases = [str(a) for a in (row.get("diagnosis_aliases") or [])]
        if not aliases:
            for key in (row.get("base_id"), row.get("id")):
                if key and str(key) in alias_by_id:
                    aliases = alias_by_id[str(key)]
                    break
        n_aliased += bool(aliases)
        scored.append(
            {
                "id": row.get("id"),
                "answer": answer,
                "gold": gold,
                "strict": is_correct(answer, gold, aliases),
                "f1": token_f1(answer, gold, aliases),
                "recorded": row.get("source_correct"),
            }
        )

    n = len(scored)
    strict = sum(1 for s in scored if s["strict"])
    print(f"answers: {args.answers}")
    print(f"n: {n:,}")
    print(f"rows with an alias list: {n_aliased:,} / {n:,}")
    print(f"strict_accuracy: {strict / n:.4f}  ({strict:,} rows)")

    # The run that wrote this file scored every row as it went. If this script
    # disagrees, one of the two is using a different rule, and until that is
    # settled nothing below means anything -- so it stops here rather than
    # printing a plausible second number. This is not hypothetical: the answer
    # rows did not carry diagnosis_aliases, so the audit read DDXPlus as 0.2920
    # against the 0.3724 the run recorded, and the gap looked like a finding.
    recorded = [s for s in scored if s["recorded"] is not None]
    drift = [s for s in recorded if bool(s["recorded"]) != s["strict"]]
    if drift:
        example = drift[0]
        raise SystemExit(
            f"[!] {len(drift):,} of {len(recorded):,} rows score differently here "
            f"than the run recorded ({sum(bool(s['recorded']) for s in recorded) / len(recorded):.4f} "
            f"recorded vs {strict / n:.4f} here).\n"
            f"    e.g. id={example['id']} gold={example['gold']!r} "
            f"answer={example['answer']!r} recorded={bool(example['recorded'])} here={example['strict']}\n"
            "    Usually the answer file predates diagnosis_aliases being written "
            "into it: pass --cases with the case file it was generated from."
        )
    for threshold in THRESHOLDS:
        hits = sum(1 for s in scored if s["f1"] >= threshold)
        print(f"token_f1 >= {threshold:.2f}: {hits / n:.4f}  ({hits:,} rows)")
    # A strict hit that the loose rule misses would mean the two disagree in
    # both directions, which would make the gap uninterpretable.
    both_ways = [s for s in scored if s["strict"] and s["f1"] < args.threshold]
    if both_ways:
        print(
            f"\n[!] {len(both_ways):,} rows pass the strict rule but fall below the "
            "threshold. These are where containment counts a match the word overlap "
            "does not -- a short gold name swallowed by a long answer -- so they are "
            "the accuracy's false positives, not its undercount:"
        )
        print_pairs(both_ways, args.show, n)

    candidates = [s for s in scored if not s["strict"] and s["f1"] >= args.threshold]
    print(
        f"\nStrict misses with token_f1 >= {args.threshold}: {len(candidates):,} "
        f"({len(candidates) / n:.4f} of all rows). Read these -- they are the "
        "difference between the two metrics:"
    )
    print_pairs(candidates, args.show, n)

    misses = [s for s in scored if not s["strict"] and s["f1"] < args.threshold]
    print(f"\nStrict misses with low overlap: {len(misses):,}.")
    print_pairs(misses, args.show, n)


def print_pairs(entries: list[dict], show: int, n_rows: int) -> None:
    """Distinct (gold, answer) pairs, commonest first, with their row counts.

    Sorted by count rather than by F1 because the question a reviewer is
    answering is how much of the corpus a decision moves, and one pair
    accounting for a hundred rows outranks twenty pairs accounting for one
    each however close their overlap scores are.
    """
    if not entries:
        return
    counts: Counter[tuple[str, str]] = Counter()
    f1_of: dict[tuple[str, str], float] = {}
    for entry in entries:
        key = (entry["gold"], str(entry["answer"]))
        counts[key] += 1
        f1_of[key] = entry["f1"]
    print(f"  {len(counts):,} distinct (gold, answer) pairs:")
    shown = 0
    for (gold, answer), count in counts.most_common(show):
        share = count / n_rows
        print(
            f"  {count:5,} rows ({share:.3f})  f1={f1_of[(gold, answer)]:.2f}  "
            f"gold={gold!r}  answer={answer!r}"
        )
        shown += count
    remainder = len(counts) - min(show, len(counts))
    if remainder:
        print(f"  ... {remainder:,} further pairs covering {len(entries) - shown:,} rows")


if __name__ == "__main__":
    main()
