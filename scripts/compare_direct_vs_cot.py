"""Does reasoning help, on the same cases?

The direct arm ran on all 4,900 cases and the chain-of-thought arm on a sample,
so the two accuracies are not comparable as printed: the difference between
0.3724 and 0.3107 is inside the sampling error of the smaller one. Joining on
case id removes that entirely -- the same cases under both conditions, and the
interesting quantity is not the two rates but the two counts underneath them.

A condition that rescues as many cases as it loses has the same accuracy and is
not the same behaviour, and if reasoning loses more than it rescues then the
chain is not merely unhelpful but harmful. Those are different claims and only
the paired counts separate them.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl


def correctness(path: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in read_jsonl(path):
        row_id = str(row.get("id"))
        if row_id in out:
            raise SystemExit(f"duplicate id in {path}: {row_id}")
        out[row_id] = bool(row.get("source_correct"))
    return out


def mcnemar_exact(gain: int, loss: int) -> float:
    """Two-sided exact binomial p for the discordant pairs.

    The concordant cases carry no information about a difference between the
    conditions, so the test is over the cases where exactly one condition was
    right -- and with these counts an exact test is cheap and avoids the
    chi-square approximation being asked for numbers it is poor at.
    """
    n = gain + loss
    if n == 0:
        return 1.0
    from math import comb

    tail = sum(comb(n, k) for k in range(0, min(gain, loss) + 1))
    return min(1.0, 2 * tail / (2**n))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--direct", required=True)
    parser.add_argument("--cot", required=True)
    args = parser.parse_args()

    direct, cot = correctness(args.direct), correctness(args.cot)
    shared = sorted(set(direct) & set(cot))
    if not shared:
        raise SystemExit("no case ids in common")
    n = len(shared)

    pairs = Counter((direct[i], cot[i]) for i in shared)
    both = pairs[(True, True)]
    gain = pairs[(False, True)]
    loss = pairs[(True, False)]
    neither = pairs[(False, False)]

    d_acc = (both + loss) / n
    c_acc = (both + gain) / n
    print(f"cases in both arms  {n:,}  (direct file {len(direct):,}, cot file {len(cot):,})")
    print(f"  direct accuracy   {d_acc:.4f}")
    print(f"  cot accuracy      {c_acc:.4f}")
    print(f"  difference        {c_acc - d_acc:+.4f}")
    print("\n  both right        {:5,}".format(both))
    print("  cot rescued       {:5,}   direct wrong, cot right".format(gain))
    print("  cot broke         {:5,}   direct right, cot wrong".format(loss))
    print("  both wrong        {:5,}".format(neither))
    print(f"\n  exact p (discordant {gain + loss:,})  {mcnemar_exact(gain, loss):.4g}")
    print(
        "\n  Accuracy hides which of these moved. Equal counts mean reasoning\n"
        "  changed which cases are right without changing how many, and losing\n"
        "  more than it rescues is a stronger claim than being unhelpful."
    )


if __name__ == "__main__":
    main()
