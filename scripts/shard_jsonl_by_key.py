"""Deterministically shard JSONL rows while keeping each key in one shard."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def shard_for(value: str, num_shards: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % num_shards


def row_key(row: dict[str, Any], field: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise ValueError(f"Row {row.get('id')!r} has no non-empty {field!r}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--key", default="base_id")
    parser.add_argument("--prefix", default="shard")
    args = parser.parse_args()
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")

    rows = list(read_jsonl(args.input))
    if not rows:
        raise ValueError(f"No rows in {args.input}")
    ids: set[str] = set()
    shards: list[list[dict[str, Any]]] = [[] for _ in range(args.num_shards)]
    keys_by_shard: list[set[str]] = [set() for _ in range(args.num_shards)]
    variants: list[Counter[str]] = [Counter() for _ in range(args.num_shards)]
    for row in rows:
        row_id = str(row.get("id") or "")
        if not row_id or row_id in ids:
            raise ValueError(f"Missing or duplicate row id: {row_id!r}")
        ids.add(row_id)
        key = row_key(row, args.key)
        index = shard_for(key, args.num_shards)
        shards[index].append(row)
        keys_by_shard[index].add(key)
        variants[index][str(row.get("variant") or "<missing>")] += 1

    args.out_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for index, shard_rows in enumerate(shards):
        path = args.out_dir / f"{args.prefix}_{index:03d}_of_{args.num_shards:03d}.jsonl"
        write_jsonl(path, shard_rows)
        outputs.append(
            {
                "index": index,
                "path": str(path),
                "rows": len(shard_rows),
                "keys": len(keys_by_shard[index]),
                "variants": dict(sorted(variants[index].items())),
            }
        )
        print(
            f"[shard] {index}/{args.num_shards} rows={len(shard_rows)} "
            f"keys={len(keys_by_shard[index])} path={path}",
            flush=True,
        )
    report = {
        "input": str(args.input),
        "key": args.key,
        "num_shards": args.num_shards,
        "rows": len(rows),
        "unique_keys": len(set().union(*keys_by_shard)),
        "outputs": outputs,
    }
    (args.out_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
