"""One verdict per id, and what the duplicates say about the judge.

Two judge processes wrote to the same output before the lock existed, so 535
of 2,256 ids were judged twice. The obvious response is to drop the extras.
The useful one is to read them first: two independent verdicts on the same
prompt, from the same model at temperature it chose itself, are a
self-consistency measurement nobody planned to buy.

That number matters for how the reader-trust table is read. If the judge
disagrees with itself on a third of repeated prompts, a channel gap of five
points is inside the noise; if it disagrees on a twentieth, the gap is real.
The table cannot say which without this, and the accident supplies it.

Agreement is reported two ways because the task has two parts: whether the
doubt call flips, and how far the confidence moves when it does not. A judge
that answers "doubt, 5" and "doubt, 2" agreed on the verdict and not on the
strength, and the AUROC ranks on the strength.

Keeps the first verdict per id. Which copy is kept does not matter for a
rate -- both were judged under the same rubric -- but it has to be decided
once and stated, or two analyses of the same file disagree.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_reader_trust import parse_verdict
from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    by_id: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(args.judgements):
        by_id[str(row.get("id"))].append(row)

    repeated = {k: v for k, v in by_id.items() if len(v) > 1}
    print(f"rows {sum(len(v) for v in by_id.values()):,}   "
          f"distinct ids {len(by_id):,}   judged more than once {len(repeated):,}")

    if repeated:
        same_verdict = 0
        scorable = 0
        conf_gap: list[float] = []
        unparsed = 0
        for rows in repeated.values():
            verdicts = [parse_verdict(r.get("response")) for r in rows[:2]]
            if any(v is None for v in verdicts):
                unparsed += 1
                continue
            scorable += 1
            (d1, c1), (d2, c2) = verdicts
            if d1 == d2:
                same_verdict += 1
                conf_gap.append(abs(c1 - c2))
        if scorable:
            print(f"\n  JUDGE SELF-CONSISTENCY on {scorable:,} repeated prompts")
            print(f"    same doubt verdict          {same_verdict:,}/{scorable:,}"
                  f" = {same_verdict / scorable:.4f}")
            if conf_gap:
                print(f"    |confidence gap| when it agreed   mean "
                      f"{statistics.mean(conf_gap):.2f}   "
                      f"identical {sum(1 for g in conf_gap if g == 0):,}"
                      f"/{len(conf_gap):,}")
            print("    A channel gap smaller than this disagreement is inside "
                  "the judge's own noise.")
        if unparsed:
            print(f"    {unparsed:,} pairs had an unparseable member and were skipped")

    kept = [rows[0] for rows in by_id.values()]
    write_jsonl(Path(args.output), kept)
    print(f"\n  wrote {len(kept):,} rows (first verdict per id) -> {args.output}")


if __name__ == "__main__":
    main()
