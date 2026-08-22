"""Select trajectory rows worth verbalizing: every moved case, a kept sample.

The trajectory probes showed the state holds the gold to the last token
while the output defects. The readout's job on the same positions is the
"why": what the state says at each landmark, in words, on the cases where
the defection happened -- plus enough kept cases to show the contrast.

Reads the ladder file for the moved labels (it carries them per base_id),
filters the extraction manifest to wrong-arm rows of moved cases plus a
seeded sample of kept cases, and writes a manifest run_nla can consume
directly. Roughly (324 + sample) x 6 landmarks -- an hour of generation,
not a night.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifests", nargs="+", required=True, help="Trajectory extraction manifests."
    )
    parser.add_argument(
        "--ladder", required=True, help="Any ladder rung file: carries moved per base_id."
    )
    parser.add_argument("--kept-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    moved_by_id = {
        str(r["base_id"]): bool(r.get("moved")) for r in read_jsonl(args.ladder)
    }
    moved_ids = {b for b, m in moved_by_id.items() if m}
    kept_ids = sorted(b for b, m in moved_by_id.items() if not m)
    keep = moved_ids | set(random.Random(args.seed).sample(kept_ids, args.kept_sample))

    rows = []
    for manifest in args.manifests:
        for row in read_jsonl(manifest):
            if (
                str(row.get("hint_variant") or "") == "wrong"
                and str(row.get("base_id") or "") in keep
            ):
                rows.append(row)
    if not rows:
        raise SystemExit("no rows selected; check --manifests and --ladder")
    write_jsonl(Path(args.output), rows)
    n_moved = sum(1 for r in rows if str(r["base_id"]) in moved_ids)
    print(
        f"wrote {len(rows):,} rows ({n_moved:,} from moved cases, "
        f"{len(rows) - n_moved:,} from kept sample) to {args.output}"
    )


if __name__ == "__main__":
    main()
