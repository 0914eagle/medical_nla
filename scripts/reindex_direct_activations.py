"""Merge DiReCT activation manifests and assign rows to a frozen split.

The source forward pass is independent of the downstream train/test assignment.
This script therefore reuses already extracted tensors after a split revision,
while enforcing a complete one-to-one join over case, layer, and position.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


SPLITS = ("train", "val_seen", "test_seen", "test_pdd_heldout")
POSITION_SELECTION = {
    "P0": "last_token",
    "P1": "last_subtoken",
    "P2": "last_subtoken",
}


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    mappings: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Path map must be OLD=NEW, got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError(f"Path map has an empty source prefix: {value!r}")
        mappings.append((old.rstrip("/"), new.rstrip("/")))
    return mappings


def remap_path(value: str, mappings: list[tuple[str, str]]) -> str:
    for old, new in mappings:
        if value == old or value.startswith(f"{old}/"):
            return f"{new}{value[len(old):]}"
    return value


def load_assignments(split_dir: Path) -> tuple[dict[str, str], dict[str, int]]:
    assignments: dict[str, str] = {}
    counts: dict[str, int] = {}
    for split in SPLITS:
        rows = list(read_jsonl(split_dir / f"{split}.jsonl"))
        counts[split] = len(rows)
        for row in rows:
            case_id = str(row.get("base_id") or row.get("id") or "")
            if not case_id:
                raise ValueError(f"Missing id in {split}.jsonl")
            previous = assignments.setdefault(case_id, split)
            if previous != split:
                raise ValueError(f"Case {case_id!r} appears in {previous} and {split}")
    if not assignments:
        raise ValueError(f"No split assignments found under {split_dir}")
    return assignments, counts


def manifest_coordinates(path: Path) -> tuple[int, str]:
    layer_match = re.fullmatch(r"layer(\d+)", path.parent.parent.name)
    if layer_match is None:
        raise ValueError(f"Cannot infer layer from {path}")
    return int(layer_match.group(1)), path.parent.name


def merge_rows(
    manifest_roots: list[Path],
    assignments: dict[str, str],
    path_maps: list[tuple[str, str]],
    verify_paths: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    merged: list[dict[str, Any]] = []
    seen: dict[tuple[str, int, str], Path] = {}
    stats: Counter[str] = Counter()

    manifests: list[Path] = []
    for root in manifest_roots:
        manifests.extend(sorted(root.glob("layer*/last_*/manifest.jsonl")))
    if not manifests:
        raise ValueError("No layer*/last_*/manifest.jsonl files were found")

    for manifest in manifests:
        file_layer, selection = manifest_coordinates(manifest)
        for row in read_jsonl(manifest):
            case_id = str(row.get("base_id") or row.get("id") or "")
            if case_id not in assignments:
                stats["unassigned"] += 1
                continue
            layer = int(row.get("layer", file_layer))
            if layer != file_layer:
                raise ValueError(f"Layer mismatch in {manifest}: {layer} != {file_layer}")
            family = str(row.get("position_family") or "")
            if family not in POSITION_SELECTION:
                raise ValueError(f"Unknown position_family {family!r} in {manifest}")
            if POSITION_SELECTION[family] != selection:
                raise ValueError(
                    f"Selection mismatch for {case_id}: {family} is under {selection}"
                )

            key = (case_id, layer, family)
            if key in seen:
                raise ValueError(
                    f"Duplicate activation {key}; seen in {seen[key]} and {manifest}"
                )
            seen[key] = manifest

            out = dict(row)
            out["base_id"] = case_id
            out["split"] = assignments[case_id]
            out["layer"] = layer
            activation_path = remap_path(str(out.get("activation_path") or ""), path_maps)
            if not activation_path:
                raise ValueError(f"Missing activation_path for {key}")
            if verify_paths and not Path(activation_path).is_file():
                raise FileNotFoundError(f"Activation path does not exist: {activation_path}")
            out["activation_path"] = activation_path
            merged.append(out)

    return merged, stats


def validate_grid(
    rows: list[dict[str, Any]],
    assignments: dict[str, str],
    expected_layers: list[int],
) -> None:
    actual = {
        (str(row["base_id"]), int(row["layer"]), str(row["position_family"]))
        for row in rows
    }
    expected = {
        (case_id, layer, family)
        for case_id in assignments
        for layer in expected_layers
        for family in POSITION_SELECTION
    }
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        sample_missing = sorted(missing)[:3]
        sample_extra = sorted(extra)[:3]
        raise ValueError(
            "Activation grid is incomplete: "
            f"expected={len(expected)} actual={len(actual)} missing={len(missing)} "
            f"extra={len(extra)} missing_sample={sample_missing} extra_sample={sample_extra}"
        )


def write_outputs(
    out_dir: Path,
    rows: list[dict[str, Any]],
    split_counts: dict[str, int],
    expected_layers: list[int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        selection = POSITION_SELECTION[str(row["position_family"])]
        grouped[(int(row["layer"]), selection, str(row["split"]))].append(row)

    for (layer, selection, split), split_rows in sorted(grouped.items()):
        output = out_dir / f"layer{layer}" / selection / f"manifest_{split}.jsonl"
        write_jsonl(output, sorted(split_rows, key=lambda row: (row["base_id"], row["position_family"])))

    summary = out_dir / "summary.md"
    with summary.open("w", encoding="utf-8") as handle:
        handle.write("# DiReCT Activation Reindex Summary\n\n")
        handle.write("Existing tensors assigned to the frozen downstream split.\n\n")
        handle.write(f"- cases: **{sum(split_counts.values())}**\n")
        handle.write(f"- activation rows: **{len(rows)}**\n")
        handle.write(f"- layers: `{expected_layers}`\n")
        handle.write("- positions: `P0`, `P1`, `P2`\n")
        handle.write("- complete case x layer x position grid: **yes**\n\n")
        handle.write("| split | cases | activation rows |\n")
        handle.write("|---|---:|---:|\n")
        for split in SPLITS:
            row_count = sum(1 for row in rows if row["split"] == split)
            handle.write(f"| {split} | {split_counts[split]} | {row_count} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-roots", nargs="+", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--expected-layers", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument(
        "--verify-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    assignments, split_counts = load_assignments(args.split_dir)
    path_maps = parse_path_maps(args.path_map)
    rows, stats = merge_rows(
        args.manifest_roots,
        assignments,
        path_maps,
        args.verify_paths,
    )
    validate_grid(rows, assignments, args.expected_layers)
    write_outputs(args.out_dir, rows, split_counts, args.expected_layers)
    print(
        f"[reindex] cases={len(assignments)} rows={len(rows)} "
        f"unassigned={stats['unassigned']} out_dir={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
