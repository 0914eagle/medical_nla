"""Read pasted verdicts back, merge them onto the rows, and report the rates.

The judge is reached by pasting, so its reply is loose text rather than a
structured response. This parses "12=A" in whatever surrounding prose arrives,
then refuses on anything that would make the resulting rate wrong: a number
that was not asked about, a number answered twice with different letters, or
any request left unanswered. A partially judged pool reported as a rate is the
failure this exists to prevent.

Where a human verdict exists for the same pair, agreement is reported with
Cohen's kappa. That is what makes the judge quotable: the number in the paper
is the judge's, and the sentence next to it says how far it sits from the
hand labelling on the overlapping pairs.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

VERDICTS = ("A", "B", "C", "D")
_LINE = re.compile(r"^\s*(\d+)\s*[=:.\)]\s*([ABCD])\b", re.MULTILINE)


def parse_verdicts(text: str) -> dict[int, str]:
    out: dict[int, str] = {}
    for number, verdict in _LINE.findall(text):
        n = int(number)
        if n in out and out[n] != verdict:
            raise SystemExit(f"[!] number {n} answered twice, as {out[n]} and {verdict}")
        out[n] = verdict
    return out


def read_hand_labels(path: str | None) -> dict[tuple[str, str], str]:
    labels: dict[tuple[str, str], str] = {}
    if not path:
        return labels
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip() or line.startswith("verdict\t"):
            continue
        parts = line.split("\t")
        if len(parts) >= 4:
            labels[(parts[2], parts[3])] = parts[0]
    return labels


def cohens_kappa(pairs: list[tuple[str, str]]) -> float:
    """Agreement above what two raters with these marginals would reach by chance."""
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    left, right = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    expected = sum(left[v] / n * right[v] / n for v in VERDICTS)
    if expected >= 1.0:
        return float("nan")
    return (observed - expected) / (1 - expected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True, help="judge_index.jsonl")
    parser.add_argument("--verdicts", nargs="+", required=True, help="Pasted reply files.")
    parser.add_argument("--output", help="Per-row verdicts as JSONL.")
    parser.add_argument("--hand-labels", help="TSV of human verdicts, for kappa.")
    args = parser.parse_args()

    index = list(read_jsonl(args.index))
    asked = {int(entry["n"]) for entry in index}

    given: dict[int, str] = {}
    for path in args.verdicts:
        for n, verdict in parse_verdicts(Path(path).read_text(encoding="utf-8")).items():
            if n in given and given[n] != verdict:
                raise SystemExit(f"[!] number {n} answered differently in two files")
            given[n] = verdict

    unknown = sorted(set(given) - asked)
    missing = sorted(asked - set(given))
    if unknown:
        raise SystemExit(
            f"[!] {len(unknown)} verdicts for numbers that were not asked: {unknown[:10]}"
        )
    if missing:
        raise SystemExit(
            f"[!] {len(missing)} of {len(asked)} requests unanswered: {missing[:10]}\n"
            "    A rate over part of a pool is not the pool's rate; judge the rest first."
        )

    rows: list[dict[str, Any]] = []
    by_pair: dict[tuple[str, str], str] = {}
    for entry in index:
        verdict = given[int(entry["n"])]
        by_pair[(entry["gold"], entry["read"])] = verdict
        for row_id in entry["ids"]:
            rows.append(
                {
                    "id": row_id,
                    "gold": entry["gold"],
                    "read": entry["read"],
                    "verdict": verdict,
                }
            )

    counts = Counter(row["verdict"] for row in rows)
    n = len(rows)
    print(f"rows judged     {n:,} over {len(index):,} distinct pairs")
    for verdict in VERDICTS:
        print(f"  {verdict}  {counts[verdict]:5,}  {counts[verdict] / n:.4f}")
    print(f"\nA        {counts['A'] / n:.4f}")
    print(f"A + B    {(counts['A'] + counts['B']) / n:.4f}")
    print(f"C        {counts['C'] / n:.4f}   <- named a different finding")

    hand = read_hand_labels(args.hand_labels)
    if hand:
        overlap = [(hand[p], by_pair[p]) for p in by_pair if p in hand]
        if overlap:
            agree = sum(a == b for a, b in overlap) / len(overlap)
            print(f"\nagainst {len(overlap)} hand-labelled pairs:")
            print(f"  exact agreement  {agree:.4f}")
            print(f"  cohen's kappa    {cohens_kappa(overlap):.4f}")
            harsher = sum(VERDICTS.index(b) > VERDICTS.index(a) for a, b in overlap)
            softer = sum(VERDICTS.index(b) < VERDICTS.index(a) for a, b in overlap)
            print(f"  judge harsher    {harsher}   judge softer {softer}")
            print(
                "  A judge softer than the human inflates exactly the rate this\n"
                "  paper claims, so that count belongs beside the rate."
            )
        else:
            print("\n[!] no hand-labelled pair appears in this pool; kappa not computed")

    if args.output:
        write_jsonl(Path(args.output), rows)
        print(f"\nwrote {len(rows):,} rows to {args.output}")


if __name__ == "__main__":
    main()
