"""Merge disjoint activation-extraction shards with grid and path checks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Path map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError(f"Empty source prefix in path map {value!r}")
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    return mappings


def remap_path(value: str, mappings: list[tuple[str, str]]) -> str:
    for old, new in mappings:
        if value == old or value.startswith(f"{old}/"):
            return f"{new}{value[len(old):]}"
    return value


def layer_from_manifest(path: Path) -> int:
    match = re.fullmatch(r"layer(\d+)", path.parent.parent.name)
    if match is None:
        raise ValueError(f"Cannot infer layer from {path}")
    if path.parent.name != "last_token":
        raise ValueError(f"Expected last_token manifest, got {path}")
    return int(match.group(1))


def merge_rows(
    roots: list[Path],
    mappings: list[tuple[str, str]],
    *,
    verify_paths: bool,
) -> dict[int, list[dict[str, Any]]]:
    by_layer: dict[int, list[dict[str, Any]]] = {}
    seen: dict[int, set[str]] = {}
    manifests = [
        manifest
        for root in roots
        for manifest in sorted(root.glob("layer*/last_token/manifest.jsonl"))
    ]
    if not manifests:
        raise ValueError("No layer*/last_token/manifest.jsonl files found")
    for manifest in manifests:
        layer = layer_from_manifest(manifest)
        by_layer.setdefault(layer, [])
        seen.setdefault(layer, set())
        for row in read_jsonl(manifest):
            row_id = str(row.get("id") or "")
            if not row_id or row_id in seen[layer]:
                raise ValueError(f"Missing or duplicate HS{layer} row id: {row_id!r}")
            seen[layer].add(row_id)
            out = dict(row)
            path = remap_path(str(out.get("activation_path") or ""), mappings)
            if not path:
                raise ValueError(f"Missing activation path for HS{layer} {row_id}")
            if verify_paths and not Path(path).is_file():
                raise FileNotFoundError(path)
            out["activation_path"] = path
            out["layer"] = layer
            by_layer[layer].append(out)
    return by_layer


def validate_grid(by_layer: dict[int, list[dict[str, Any]]], layers: list[int]) -> None:
    if sorted(by_layer) != sorted(layers):
        raise ValueError(
            f"Layer mismatch: expected {sorted(layers)}, found {sorted(by_layer)}"
        )
    reference = {str(row["id"]) for row in by_layer[layers[0]]}
    for layer in layers[1:]:
        current = {str(row["id"]) for row in by_layer[layer]}
        if current != reference:
            raise ValueError(
                f"HS{layer} row grid differs: missing={len(reference-current)} "
                f"extra={len(current-reference)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard-roots", nargs="+", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--expected-layers", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument(
        "--verify-paths", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    mappings = parse_path_maps(args.path_map)
    by_layer = merge_rows(
        args.shard_roots, mappings, verify_paths=args.verify_paths
    )
    validate_grid(by_layer, args.expected_layers)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "shard_roots": [str(path) for path in args.shard_roots],
        "layers": args.expected_layers,
        "rows_per_layer": {},
    }
    for layer in args.expected_layers:
        rows = sorted(by_layer[layer], key=lambda row: str(row["id"]))
        output = args.out_dir / f"layer{layer}" / "last_token" / "manifest.jsonl"
        write_jsonl(output, rows)
        summary["rows_per_layer"][str(layer)] = len(rows)
        print(f"[merge] HS{layer} rows={len(rows)} output={output}", flush=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
