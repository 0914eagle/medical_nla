"""Filter JSONL by an exact field value while enforcing the expected count."""

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
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--field", required=True)
    parser.add_argument("--value", required=True)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows = [row for row in read_jsonl(args.input) if str(row.get(args.field)) == args.value]
    if len(rows) != args.expected_rows:
        raise ValueError(
            f"Selected {len(rows)} rows where {args.field}={args.value!r}; "
            f"expected {args.expected_rows}"
        )
    ids = [str(row.get("id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Filtered rows have missing or duplicate IDs")
    write_jsonl(args.output, rows)
    print(f"[filter] rows={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
