"""Fit and gate the frozen D16 source-balanced PCA bottleneck initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.nla_bottleneck import (
    BOTTLENECK_FILENAME,
    IMPLEMENTATION_VERSION,
    NlaBottleneckProjector,
    fit_source_balanced_pca,
    reconstruction_metrics,
    save_bottleneck,
    sha256_file,
)


def identifier(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def hash_ids(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(identifier(row) for row in rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OLD=NEW path map, got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError("Path-map OLD prefix cannot be empty")
        result.append((old, new))
    return result


def mapped_path(value: str, path_maps: list[tuple[str, str]]) -> Path:
    for old, new in path_maps:
        if value.startswith(old):
            value = new + value[len(old) :]
            break
    return Path(value)


def load_population(
    path: Path,
    *,
    path_maps: list[tuple[str, str]],
    ddxplus_original_only: bool,
) -> tuple[list[dict[str, Any]], torch.Tensor]:
    rows = []
    vectors = []
    seen = set()
    for row in read_jsonl(path):
        if ddxplus_original_only and str(row.get("variant") or "original") != "original":
            continue
        if str(row.get("position_family")) != "P0" or int(row.get("layer") or -1) != 32:
            raise ValueError(f"Non-CoT-P0/HS32 row in {path}: {identifier(row)}")
        key = identifier(row)
        if not key or key in seen:
            raise ValueError(f"Missing or duplicate ID in {path}: {key!r}")
        seen.add(key)
        activation_path = mapped_path(str(row.get("activation_path") or ""), path_maps)
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        vector = (
            torch.load(activation_path, map_location="cpu", weights_only=True)
            .float()
            .flatten()
        )
        rows.append(row)
        vectors.append(vector)
    if not rows:
        raise ValueError(f"No eligible rows in {path}")
    matrix = torch.stack(vectors)
    if matrix.ndim != 2 or matrix.shape[1] != 3840:
        raise ValueError(f"Unexpected activation matrix from {path}: {tuple(matrix.shape)}")
    return rows, matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ddxplus-train", required=True, type=Path)
    parser.add_argument("--direct-train", required=True, type=Path)
    parser.add_argument("--ddxplus-validation", required=True, type=Path)
    parser.add_argument("--direct-validation", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--d-z", type=int, default=256)
    parser.add_argument("--cosine-gate", type=float, default=0.95)
    parser.add_argument("--expected-ddxplus-train", type=int, default=4655)
    parser.add_argument("--expected-direct-train", type=int, default=248)
    parser.add_argument("--expected-ddxplus-validation", type=int, default=4525)
    parser.add_argument("--expected-direct-validation", type=int, default=50)
    parser.add_argument("--path-map", action="append", default=[])
    args = parser.parse_args()
    if args.d_z != 256 or args.cosine_gate != 0.95:
        raise ValueError("D16 freezes d_z=256 and cosine gate=.95")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")

    path_maps = parse_path_maps(args.path_map)
    populations = {}
    for name, path, ddx, expected in (
        ("ddxplus_train", args.ddxplus_train, True, args.expected_ddxplus_train),
        ("direct_train", args.direct_train, False, args.expected_direct_train),
        ("ddxplus_validation", args.ddxplus_validation, True, args.expected_ddxplus_validation),
        ("direct_validation", args.direct_validation, False, args.expected_direct_validation),
    ):
        rows, matrix = load_population(
            path, path_maps=path_maps, ddxplus_original_only=ddx
        )
        if len(rows) != expected:
            raise ValueError(f"{name}: {len(rows)} rows, expected {expected}")
        populations[name] = (rows, matrix)
        print(f"[load] {name} rows={len(rows)} dim={matrix.shape[1]}", flush=True)

    mean, basis, eigenvalues = fit_source_balanced_pca(
        populations["ddxplus_train"][1],
        populations["direct_train"][1],
        d_z=args.d_z,
    )
    projector = NlaBottleneckProjector(mean, basis)
    metrics = {
        name: reconstruction_metrics(projector, matrix)
        for name, (_rows, matrix) in populations.items()
    }
    gate_passed = all(
        metrics[name]["mean_reconstruction_cosine"] >= args.cosine_gate
        for name in ("ddxplus_validation", "direct_validation")
    )
    inputs = {
        name: {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(populations[name][0]),
            "id_sha256": hash_ids(populations[name][0]),
        }
        for name, path in (
            ("ddxplus_train", args.ddxplus_train),
            ("direct_train", args.direct_train),
            ("ddxplus_validation", args.ddxplus_validation),
            ("direct_validation", args.direct_validation),
        )
    }
    metadata = {
        "decision": "D16",
        "implementation_version": IMPLEMENTATION_VERSION,
        "d_model": 3840,
        "d_z": args.d_z,
        "source_weights": {"ddxplus": 0.5, "direct": 0.5},
        "input_transform": "unit_l2_normalize_then_subtract_weighted_mixture_mean",
        "basis_algorithm": "float64_cpu_torch.linalg.eigh_top256_canonical_sign",
        "pca_sanity_cosine_gate": args.cosine_gate,
        "pca_sanity_gate_passed": gate_passed,
        "metrics": metrics,
        "top_eigenvalues": [float(value) for value in eigenvalues],
        "inputs": inputs,
        "locked_test_read": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=False)
    artifact = args.out_dir / BOTTLENECK_FILENAME
    save_bottleneck(artifact, projector, metadata=metadata)
    report = {**metadata, "artifact": str(artifact), "artifact_sha256": sha256_file(artifact)}
    (args.out_dir / "protocol.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# D16 PCA Bottleneck Sanity Gate",
        "",
        f"- d_model / d_z: **3840 / {args.d_z}**",
        "- PCA fit: **train only, source weights .5/.5**",
        f"- validation cosine gate: **{args.cosine_gate:.2f} per source**",
        f"- overall gate: **{'PASS' if gate_passed else 'FAIL'}**",
        "- locked test read: **no**",
        "",
        "| population | n | mean cosine | min cosine | retained variance |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {int(values['n'])} | "
            f"{values['mean_reconstruction_cosine']:.6f} | "
            f"{values['min_reconstruction_cosine']:.6f} | "
            f"{values['retained_variance']:.6f} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)
    if not gate_passed:
        raise SystemExit("[stop] D16 PCA sanity gate failed; no dimension sweep is allowed")


if __name__ == "__main__":
    main()
