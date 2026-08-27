"""Merge JSONL files while rejecting missing or duplicate row IDs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-rows", type=int)
    args = parser.parse_args()

    rows = []
    seen: set[str] = set()
    for path in args.input:
        for row in read_jsonl(path):
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                raise ValueError(f"Missing row ID in {path}")
            if row_id in seen:
                raise ValueError(f"Duplicate row ID across inputs: {row_id}")
            seen.add(row_id)
            rows.append(row)
    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(
            f"Merged {len(rows)} rows; expected {args.expected_rows}"
        )
    if not rows:
        raise ValueError("No input rows")

    write_jsonl(args.output, rows)
    print(
        f"[merge-jsonl] inputs={len(args.input)} rows={len(rows)} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
