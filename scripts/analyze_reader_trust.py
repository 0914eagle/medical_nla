"""Per-channel AUROC on the reader-trust task.

Each row is one channel's account of one case, judged alone, against the
hidden label of whether the referring note actually moved that answer. AUROC
over the signed confidence (doubt at confidence 4 ranks above doubt at
confidence 2, which ranks above no-doubt at confidence 2) so a channel that
doubts everything lands at 0.5 instead of looking vigilant.

Reported beside it: the raw doubt rate per group. A channel can carry real
signal and still be useless if it doubts 90% of the correct answers, and the
rate is what says so.

The DeLong-style significance test is deliberately absent: with one judge and
one backbone the interval that matters is between judges, not within one.
Bootstrap CIs over cases are printed so the ordering can be read with its
uncertainty.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl


def parse_verdict(response: str) -> tuple[bool, float] | None:
    """The judge's doubt call and confidence, from the first complete object."""
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", str(response or "")):
        try:
            obj, _ = decoder.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "doubt" in obj:
            conf = obj.get("confidence")
            conf = float(conf) if isinstance(conf, (int, float)) else 3.0
            return bool(obj["doubt"]), min(5.0, max(1.0, conf))
    return None


def score_of(doubt: bool, confidence: float) -> float:
    """One ranking axis: doubt pushes up, certainty pushes further."""
    return confidence if doubt else -confidence


def auroc(pairs: list[tuple[float, bool]]) -> float:
    """Rank-based AUROC with ties averaged."""
    positives = sum(1 for _, y in pairs if y)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return float("nan")
    ordered = sorted(pairs, key=lambda p: p[0])
    ranks: dict[int, float] = {}
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[k] = shared
        i = j + 1
    rank_sum = sum(ranks[k] for k, (_, y) in enumerate(ordered) if y)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def bootstrap(pairs: list[tuple[float, bool]], seed: int = 17, n: int = 1000) -> tuple[float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(n):
        sample = [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]
        value = auroc(sample)
        if value == value:
            values.append(value)
    if not values:
        return float("nan"), float("nan")
    values.sort()
    return values[int(0.025 * len(values))], values[int(0.975 * len(values))]


BASELINE = "no_account"
ITERATIONS = 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", required=True)
    parser.add_argument(
        "--cases",
        help="The case file the judgements were built from. Required when the "
        "judging went through run_judge.py, which carries only {id, response} "
        "-- the channel and the label live here and there is nothing to "
        "compare without them.",
    )
    args = parser.parse_args()

    rows = list(read_jsonl(args.judgements))
    if args.cases:
        meta = {str(r["id"]): r for r in read_jsonl(args.cases)}
        joined = 0
        for row in rows:
            extra = meta.get(str(row.get("id")))
            if extra:
                joined += 1
                for key in ("readout_channel", "label_moved", "group",
                            "base_id", "diagnosis_name"):
                    row.setdefault(key, extra.get(key))
        print(f"[join] {joined:,} of {len(rows):,} judgements matched a case")
        if joined < len(rows):
            print("  unmatched judgements carry no channel and fall into '?'")
    elif not any(r.get("readout_channel") for r in rows):
        raise SystemExit(
            "no judgement carries readout_channel, so every row would be "
            "pooled into one unlabelled bucket and the AUROC would be "
            "undefined. Pass --cases to join the channel and the label back."
        )
    by_channel: dict[str, list[tuple[float, bool]]] = defaultdict(list)
    by_case: dict[str, dict[str, tuple[float, bool]]] = defaultdict(dict)
    doubt_rate: dict[tuple[str, bool], list[bool]] = defaultdict(list)
    unparsed = 0
    for row in rows:
        verdict = parse_verdict(row.get("response"))
        if verdict is None:
            unparsed += 1
            continue
        doubt, confidence = verdict
        channel = str(row.get("readout_channel") or "?")
        label = bool(row.get("label_moved"))
        by_channel[channel].append((score_of(doubt, confidence), label))
        doubt_rate[(channel, label)].append(doubt)
        by_case[channel][str(row.get("base_id") or row.get("id"))] = (
            score_of(doubt, confidence), label)

    print(f"rows {len(rows):,} | unparseable {unparsed:,}")
    # A partial run can hold only positives, and then every AUROC is nan while
    # the doubt rates still print. Said plainly, because three nans beside a
    # 0.897 reads as a channel that won.
    for channel, pairs in by_channel.items():
        n_pos = sum(1 for _, y in pairs if y)
        if n_pos in (0, len(pairs)):
            print(f"  ⚠ '{channel}' has only {'moved' if n_pos else 'kept'} "
                  f"cases ({len(pairs):,}); AUROC is undefined and the doubt "
                  "rate below is not a score -- a channel that doubts "
                  "everything would show the same number")
            break
    print(f"\n  {'channel':<10}{'AUROC':>8}{'95% CI':>18}"
          f"{'doubts moved':>14}{'doubts kept':>13}{'n':>7}")
    for channel in sorted(by_channel, key=lambda c: -auroc(by_channel[c])):
        pairs = by_channel[channel]
        low, high = bootstrap(pairs)
        moved = doubt_rate[(channel, True)]
        kept = doubt_rate[(channel, False)]
        rate = lambda xs: sum(xs) / len(xs) if xs else float("nan")
        print(f"  {channel:<10}{auroc(pairs):>8.4f}"
              f"{f'[{low:.3f}, {high:.3f}]':>18}"
              f"{rate(moved):>14.3f}{rate(kept):>13.3f}{len(pairs):>7,}")
    print("\n  A channel that doubts everything scores 0.5 here, which is the")
    print("  point: the question is what a reader gets from the words, not")
    print("  whether the words sound cautious.")

    # The number the claim actually rests on. The judge sees the presentation
    # and the answer in every arm, so it can work the case itself; whatever
    # that is worth is inside all three channel AUROCs. Only the increment over
    # the no-account arm is attributable to the words, and it is paired on the
    # case because the same patients are scored under both.
    base = by_case.get(BASELINE)
    if not base:
        print(f"\n  ⚠ no '{BASELINE}' rows. Absolute AUROCs above are NOT "
              "attributable to the accounts -- rebuild the cases with "
              "--controls none and judge those rows too.")
        return
    print(f"\n  increment over '{BASELINE}' (paired on the case, {ITERATIONS:,} draws)")
    print(f"  {'channel':<16}{'delta':>9}{'95% CI':>20}{'n paired':>10}")
    for channel in sorted(by_case):
        if channel == BASELINE:
            continue
        shared = sorted(set(by_case[channel]) & set(base))
        if len(shared) < 2:
            continue
        arm = [by_case[channel][c] for c in shared]
        ref = [base[c] for c in shared]
        delta = auroc(arm) - auroc(ref)
        rng = random.Random(17)
        deltas = []
        for _ in range(ITERATIONS):
            idx = [rng.randrange(len(shared)) for _ in range(len(shared))]
            d = auroc([arm[i] for i in idx]) - auroc([ref[i] for i in idx])
            if d == d:
                deltas.append(d)
        if not deltas:
            continue
        deltas.sort()
        low = deltas[int(0.025 * len(deltas))]
        high = deltas[min(int(0.975 * len(deltas)), len(deltas) - 1)]
        mark = "" if low > 0 else "   (CI includes 0)"
        print(f"  {channel:<16}{delta:>+9.4f}"
              f"{f'[{low:+.3f}, {high:+.3f}]':>20}{len(shared):>10,}{mark}")
    print("\n  Only a delta whose interval excludes zero says the account "
          "helped the reader.")


if __name__ == "__main__":
    main()
