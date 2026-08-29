"""Materialize frozen D16 z vectors for fresh train-to-validation probe audits."""

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

from scripts.train_medical_nla_soft_bottleneck import parse_path_maps
from src.jsonl import read_jsonl
from src.nla_bottleneck import load_bottleneck, sha256_file


def mapped_path(value: str, mappings: list[tuple[str, str]]) -> Path:
    for old, new in mappings:
        if value.startswith(old):
            return Path(new + value[len(old) :])
    return Path(value)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("base_id") or "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bottleneck-projector", required=True, type=Path)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    output_manifest = args.out_root / "layer32" / "last_token" / "manifest.jsonl"
    if args.out_root.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_root}")
    mappings = parse_path_maps(args.path_map)
    rows = list(read_jsonl(args.manifest))
    if not rows:
        raise ValueError("No latent rows")
    seen = set()
    for row in rows:
        identifier = row_id(row)
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate row ID: {identifier!r}")
        seen.add(identifier)
        if str(row.get("position_family") or "P0") != "P0" or int(
            row.get("layer") or 32
        ) != 32:
            raise ValueError(f"D16 only accepts CoT-P0/HS32: {identifier}")

    device = torch.device(args.device)
    projector, metadata = load_bottleneck(
        args.bottleneck_projector, device=device, require_gate_passed=True
    )
    projector.eval()
    output_manifest.parent.mkdir(parents=True, exist_ok=False)
    output_rows = []
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        vectors = []
        for row in batch_rows:
            source = mapped_path(str(row.get("activation_path") or ""), mappings)
            if not source.is_file():
                raise FileNotFoundError(source)
            vectors.append(
                torch.load(source, map_location="cpu", weights_only=True).float().flatten()
            )
        with torch.inference_mode():
            latent = projector.encode(torch.stack(vectors).to(device)).cpu()
        for row, value in zip(batch_rows, latent, strict=True):
            identifier = row_id(row)
            shard = hashlib.sha256(identifier.encode()).hexdigest()[:2]
            filename = hashlib.sha256(identifier.encode()).hexdigest()
            path = (
                args.out_root
                / "layer32"
                / "last_token"
                / f"shard_{shard}"
                / f"{filename}.pt"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(value, path)
            output = dict(row)
            output["activation_path"] = str(path)
            output["raw_activation_path"] = str(row.get("activation_path") or "")
            output["latent_dim"] = projector.d_z
            output["bottleneck_sha256"] = sha256_file(args.bottleneck_projector)
            output_rows.append(output)
        print(f"[z] {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "decision": "D16",
        "rows": len(output_rows),
        "latent_dim": projector.d_z,
        "input_manifest": str(args.manifest),
        "input_manifest_sha256": sha256_file(args.manifest),
        "bottleneck_projector": str(args.bottleneck_projector),
        "bottleneck_sha256": sha256_file(args.bottleneck_projector),
        "pca_sanity_gate_passed": bool(metadata.get("pca_sanity_gate_passed")),
        "locked_test_read": False,
    }
    (args.out_root / "protocol.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[done] {output_manifest} rows={len(output_rows)}", flush=True)


if __name__ == "__main__":
    main()
