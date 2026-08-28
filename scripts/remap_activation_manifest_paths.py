"""Rewrite activation-path prefixes with row and file-existence checks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


PATH_FIELDS = ("activation_path", "original_activation_path", "donor_activation_path")


def parse_path_map(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected OLD=NEW")
    old, new = value.split("=", 1)
    if not old or not new:
        raise argparse.ArgumentTypeError("Expected non-empty OLD=NEW")
    return old.rstrip("/"), new.rstrip("/")


def remap(value: Any, mappings: list[tuple[str, str]]) -> str:
    path = str(value or "")
    for old, new in mappings:
        if path == old or path.startswith(f"{old}/"):
            return f"{new}{path[len(old):]}"
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--path-map", action="append", required=True, type=parse_path_map)
    parser.add_argument("--expected-rows", type=int)
    parser.add_argument(
        "--verify-paths", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    rows = []
    seen: set[str] = set()
    changed = 0
    verified = 0
    for row in read_jsonl(args.input):
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in seen:
            raise ValueError(f"Missing or duplicate row ID: {row_id!r}")
        seen.add(row_id)
        out = dict(row)
        for field in PATH_FIELDS:
            if field not in out or not out.get(field):
                continue
            original = str(out[field])
            mapped = remap(original, args.path_map)
            out[field] = mapped
            changed += mapped != original
            if args.verify_paths:
                if not Path(mapped).is_file():
                    raise FileNotFoundError(mapped)
                verified += 1
        rows.append(out)

    if args.expected_rows is not None and len(rows) != args.expected_rows:
        raise ValueError(f"Read {len(rows)} rows; expected {args.expected_rows}")
    if not rows:
        raise ValueError("No input rows")
    write_jsonl(args.output, rows)
    print(
        f"[remap] rows={len(rows)} changed_paths={changed} "
        f"verified_paths={verified} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
