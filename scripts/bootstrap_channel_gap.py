"""Put an interval on the gap between two attribution channels.

The paper's §4.3 sentence is that reading the state beats reading the chain by
14.7 points on the silent subset -- monitor 0.695 against readout 0.842. Those
are point estimates on 1,652 cases and the sentence is not a claim until the
difference has an interval. A reviewer asked to accept "14.7 points" without
one is being asked to take our word for the sampling noise.

**Paired on the case.** Both channels score the same patients, so a resample
must draw cases, not scores, and recompute both channels on the drawn set. An
unpaired interval would be wider than the comparison actually is: the two
channels rise and fall together with which diagnoses land in the sample, and
pairing removes exactly that shared variation.

**Resampled within diagnosis, because that is how the statistic is computed.**
Every AUROC in the paper is stratified by diagnosis -- on this corpus the
diagnosis name alone scores 0.93, so a pooled figure part-measures "guess the
diagnosis". Bootstrapping cases and then stratifying reproduces the estimator
rather than a pooled cousin of it.

Cases a channel did not score are dropped from the comparison rather than
imputed, and the count is printed. A channel that answered fewer cases is not
better for it, and a missing verdict scored as 0.5 would drag a real signal
towards chance.
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cot_monitor import parse_probability
from scripts.compare_channels_on_attribution import stratified_auroc
from src.jsonl import read_jsonl


def load_dump(path: str) -> dict[str, dict]:
    return {str(r["base_id"]): r for r in read_jsonl(path) if r.get("base_id")}


def load_monitor(paths: list[str]) -> dict[str, float]:
    """Judge verdicts as a channel: one probability per case."""
    out: dict[str, float] = {}
    for path in paths:
        for row in read_jsonl(path):
            value = parse_probability(str(row.get("response") or ""))
            if value is not None:
                out[str(row.get("id"))] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True,
                        help="Per-case channel scores from "
                        "compare_channels_on_attribution.py --dump.")
    parser.add_argument("--monitor", nargs="+", default=[],
                        help="run_judge.py output(s) to add as a channel named "
                        "'llm monitor over the chain'.")
    parser.add_argument("--a", required=True, help="First channel's column name.")
    parser.add_argument("--b", required=True, help="Second channel's column name.")
    parser.add_argument("--silent-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Restrict to cases whose answer differs from the "
                        "suggestion -- where output-only signals are blind by "
                        "construction, and the subset the deployment claim "
                        "rests on.")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--list-channels", action="store_true")
    args = parser.parse_args()

    rows = load_dump(args.dump)
    for case_id, value in load_monitor(args.monitor).items():
        if case_id in rows:
            rows[case_id]["llm monitor over the chain"] = value

    if args.list_channels:
        keys = sorted({k for r in rows.values() for k in r
                       if k not in {"base_id", "diagnosis_name", "moved",
                                    "answer_is_suggestion"}})
        print(f"{len(rows):,} cases. channels:")
        for k in keys:
            n = sum(1 for r in rows.values() if k in r)
            print(f"  {n:6,}  {k}")
        return

    usable = [r for r in rows.values() if args.a in r and args.b in r]
    if args.silent_only:
        usable = [r for r in usable if not r.get("answer_is_suggestion")]
    dropped = len(rows) - len(usable)
    if len(usable) < 2:
        raise SystemExit(f"only {len(usable)} cases carry both channels")
    moved = sum(1 for r in usable if r["moved"])
    if not 0 < moved < len(usable):
        raise SystemExit("no contrast: every case moved, or none did")

    def auroc_of(sample: list[dict], column: str) -> float:
        return stratified_auroc(
            [float(r[column]) for r in sample],
            [bool(r["moved"]) for r in sample],
            [str(r["diagnosis_name"]) for r in sample],
        )[0]

    a0, b0 = auroc_of(usable, args.a), auroc_of(usable, args.b)
    if a0 != a0 or b0 != b0:
        raise SystemExit(
            "within-diagnosis AUROC is undefined on these cases: no diagnosis "
            "holds both a moved and an unmoved case, so there is no pair to "
            "rank. Widen the subset (--no-silent-only) or check the dump."
        )
    print(f"cases scored by both {len(usable):,}"
          f"   moved {moved:,}   dropped {dropped:,}"
          f"   subset {'silent' if args.silent_only else 'all'}")
    print(f"  A  {args.a:<44} {a0:.4f}")
    print(f"  B  {args.b:<44} {b0:.4f}")
    print(f"  B - A {b0 - a0:+.4f}")

    rng = random.Random(args.seed)
    gaps, degenerate = [], 0
    for _ in range(args.iterations):
        sample = [usable[rng.randrange(len(usable))] for _ in range(len(usable))]
        labels = [r["moved"] for r in sample]
        if not any(labels) or all(labels):
            degenerate += 1
            continue
        gap = auroc_of(sample, args.b) - auroc_of(sample, args.a)
        if gap == gap:  # NaN when no within-diagnosis pair survives a draw
            gaps.append(gap)
        else:
            degenerate += 1
    if not gaps:
        raise SystemExit(
            f"all {args.iterations:,} draws were degenerate -- no interval can "
            "be formed. This means the within-diagnosis pairs are too few to "
            "survive resampling, not that the gap is zero."
        )
    if len(gaps) < args.iterations * 0.9:
        print(f"  ⚠ {degenerate:,} of {args.iterations:,} draws had no usable "
              "contrast -- the interval below rests on the rest")
    gaps.sort()
    lo = gaps[int(0.025 * len(gaps))]
    hi = gaps[min(int(0.975 * len(gaps)), len(gaps) - 1)]
    crosses = lo <= 0 <= hi
    print(f"\npaired bootstrap, {len(gaps):,} draws, stratified within diagnosis")
    print(f"  gap (B - A)   {b0 - a0:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]"
          f"   median {statistics.median(gaps):+.4f}")
    print(f"  the interval {'INCLUDES' if crosses else 'excludes'} zero"
          f" -> {'not separable at 95%' if crosses else 'B is above A'}")


if __name__ == "__main__":
    main()
