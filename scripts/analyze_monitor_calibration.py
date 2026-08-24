"""Is the CoT monitor's probability a probability, or only a ranking?

The monitor scores 0.687 within-diagnosis AUROC, which is a statement about
order: cases it scores higher moved more often than cases it scores lower.
Nothing in an AUROC says the number 0.30 means "thirty percent of these
moved". A sentence like "it knows how often this happens but not to whom"
is a calibration claim and needs calibration evidence -- the mean probability
landing near the base rate is not that, because a monitor that answers 0.185
to everything would match the mean exactly and rank at chance.

Three things, all on the same rows the AUROC was computed from:

- **Brier score**, with its skill against the base-rate constant predictor.
  A monitor that beats "always answer the prevalence" has done something a
  constant cannot; one that does not has not, whatever its AUROC.
- **ECE**, on equal-width bins, plus the reliability table those bins come
  from -- the summary hides which direction the error runs, and the table is
  what a reader needs to see it.
- **Prevalence estimate**: the mean forecast against the true rate. Reported
  as its own line and never as evidence of calibration, since it is the
  quantity a constant predictor gets right for free.

Bins with too few cases are printed rather than merged, because a reliability
table whose tails hold three cases each should not read like one whose tails
hold three hundred.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cot_monitor import parse_probability
from src.jsonl import read_jsonl

MIN_BIN = 20


def brier(probs: list[float], labels: list[bool]) -> float:
    return sum((p - float(y)) ** 2 for p, y in zip(probs, labels)) / len(probs)


def ece(probs: list[float], labels: list[bool], bins: int) -> tuple[float, list[tuple]]:
    """Equal-width binning, and the reliability rows behind the number."""
    rows = []
    total = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs)
               if (p >= lo and p < hi) or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        mean_p = sum(probs[i] for i in idx) / len(idx)
        observed = sum(labels[i] for i in idx) / len(idx)
        total += len(idx) / len(probs) * abs(mean_p - observed)
        rows.append((lo, hi, len(idx), mean_p, observed))
    return total, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verdicts", nargs="+", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--silent-only", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Restrict to cases whose answer differs from the "
                        "suggestion. Off by default: calibration is a claim "
                        "about the monitor's numbers over the population it "
                        "was asked about, which is every case.")
    args = parser.parse_args()

    labels_by_id = {str(r["id"]): r for r in read_jsonl(args.labels)}
    probs: list[float] = []
    labels: list[bool] = []
    unparsed = missing = 0
    for path in args.verdicts:
        for row in read_jsonl(path):
            meta = labels_by_id.get(str(row.get("id")))
            if meta is None:
                missing += 1
                continue
            if args.silent_only and meta.get("answer_is_suggestion"):
                continue
            value = parse_probability(str(row.get("response") or ""))
            if value is None:
                unparsed += 1
                continue
            probs.append(value)
            labels.append(bool(meta["moved"]))

    if not probs:
        raise SystemExit("nothing scored")
    base = sum(labels) / len(labels)
    subset = "silent" if args.silent_only else "all"
    print(f"n {len(probs):,} ({subset})   moved {sum(labels):,}   "
          f"base rate {base:.4f}   unparseable {unparsed:,}   "
          f"no label {missing:,}")

    b = brier(probs, labels)
    b_const = brier([base] * len(probs), labels)
    skill = 1 - b / b_const if b_const else float("nan")
    print(f"\nBrier                 {b:.4f}")
    print(f"  constant predictor  {b_const:.4f}   (always answer the base rate)")
    print(f"  skill score         {skill:+.4f}   "
          f"{'beats the constant' if skill > 0 else 'DOES NOT beat the constant'}")

    value, rows = ece(probs, labels, args.bins)
    rng = random.Random(args.seed)
    draws = []
    for _ in range(args.iterations):
        idx = [rng.randrange(len(probs)) for _ in range(len(probs))]
        draws.append(ece([probs[i] for i in idx], [labels[i] for i in idx], args.bins)[0])
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(int(0.975 * len(draws)), len(draws) - 1)]
    print(f"\nECE ({args.bins} equal-width bins)  {value:.4f}   "
          f"95% CI [{lo:.4f}, {hi:.4f}]")

    print(f"\n  {'bin':<14}{'n':>7}{'mean P':>10}{'observed':>11}{'gap':>9}")
    for lo_b, hi_b, n, mean_p, observed in rows:
        flag = "   (thin)" if n < MIN_BIN else ""
        print(f"  [{lo_b:.1f}, {hi_b:.1f})  {n:>7,}{mean_p:>10.3f}"
              f"{observed:>11.3f}{observed - mean_p:>+9.3f}{flag}")

    mean_p = sum(probs) / len(probs)
    print(f"\nPrevalence: mean forecast {mean_p:.4f} against a true rate of "
          f"{base:.4f} ({mean_p - base:+.4f}).")
    print("  This line is not evidence of calibration -- a monitor answering "
          f"{base:.3f} to every case would match it exactly and rank at chance.")
    print("  Read the skill score and the reliability table for that.")


if __name__ == "__main__":
    main()
