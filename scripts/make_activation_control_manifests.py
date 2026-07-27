"""Create activation-grounding control manifests.

Controls:
- mismatched: keep each row's prompt/gold metadata, but replace activation_path
  with an activation from a different diagnosis.
- random: keep each row's prompt/gold metadata, but replace activation_path with
  a random direction vector matched to the original activation norm.

If Medical-NLA is truly activation-grounded, mismatched outputs should move
toward the donor diagnosis and random controls should lose diagnosis accuracy.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def diagnosis_id(row: dict[str, Any]) -> str:
    return str(row.get("diagnosis_id") or row.get("gold_diagnosis_id") or "")


def load_activation(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    if tensor.ndim != 1:
        raise ValueError(f"Expected 1D activation at {path}, got shape {tuple(tensor.shape)}")
    return tensor


def copy_control_fields(row: dict[str, Any], *, control_type: str) -> dict[str, Any]:
    out = dict(row)
    out["control_type"] = control_type
    out["original_id"] = row.get("id")
    out["original_activation_path"] = row.get("activation_path")
    return out


def mismatched_rows(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dx[diagnosis_id(row)].append(row)
    all_rows = list(rows)
    rng = random.Random(seed)
    controls = []
    for row in rows:
        gold_dx = diagnosis_id(row)
        candidates = [candidate for candidate in all_rows if diagnosis_id(candidate) != gold_dx]
        if not candidates:
            raise ValueError("Cannot create mismatched controls: only one diagnosis present.")
        donor = rng.choice(candidates)
        out = copy_control_fields(row, control_type="mismatched_activation")
        out["id"] = f"{row['id']}__mismatched"
        out["activation_path"] = donor["activation_path"]
        out["donor_id"] = donor.get("id")
        out["donor_activation_path"] = donor.get("activation_path")
        out["donor_diagnosis_id"] = donor.get("diagnosis_id")
        out["donor_diagnosis_name"] = donor.get("diagnosis_name")
        out["donor_diagnosis_aliases"] = donor.get("diagnosis_aliases")
        controls.append(out)
    return controls


def random_rows(rows: list[dict[str, Any]], out_dir: Path, seed: int) -> list[dict[str, Any]]:
    random_dir = out_dir / "random_activations"
    random_dir.mkdir(parents=True, exist_ok=True)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    controls = []
    for idx, row in enumerate(rows):
        original = load_activation(str(row["activation_path"]))
        random_vec = torch.randn(
            original.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        random_vec = random_vec / random_vec.float().norm().clamp_min(1e-12)
        random_vec = random_vec * original.float().norm().clamp_min(1e-12)
        random_vec = random_vec.to(dtype=original.dtype)
        random_path = random_dir / f"{row['id']}__random.pt"
        torch.save(random_vec, random_path)

        out = copy_control_fields(row, control_type="random_activation")
        out["id"] = f"{row['id']}__random"
        out["activation_path"] = str(random_path)
        out["random_seed"] = seed + idx
        out["random_strategy"] = "gaussian_direction_matched_original_norm"
        controls.append(out)
    return controls


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["multi_format"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--make-mismatched", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--make-random", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if row.get("activation_path") and (not args.variants or row.get("variant") in args.variants)
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No manifest rows selected.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    if args.make_mismatched:
        outputs["mismatched"] = mismatched_rows(rows, args.seed)
    if args.make_random:
        outputs["random"] = random_rows(rows, out_dir, args.seed)
    combined = [row for items in outputs.values() for row in items]

    for name, items in outputs.items():
        write_jsonl(out_dir / f"manifest_{name}.jsonl", items)
    write_jsonl(out_dir / "manifest_controls.jsonl", combined)
    write_jsonl(out_dir / "manifest_original_subset.jsonl", rows)

    summary = out_dir / "summary.md"
    with summary.open("w", encoding="utf-8") as f:
        f.write("# Activation Control Manifests\n\n")
        f.write(f"- source_manifest: `{args.manifest}`\n")
        f.write(f"- selected_original_rows: {len(rows)}\n")
        f.write(f"- variants: {', '.join(args.variants)}\n")
        for name, items in outputs.items():
            f.write(f"- {name}: {len(items)}\n")
        f.write("\n## Files\n\n")
        f.write(f"- original_subset: `{out_dir / 'manifest_original_subset.jsonl'}`\n")
        for name in outputs:
            f.write(f"- {name}: `{out_dir / f'manifest_{name}.jsonl'}`\n")
        f.write(f"- combined_controls: `{out_dir / 'manifest_controls.jsonl'}`\n")
    print(f"[done] wrote controls to {out_dir}")


if __name__ == "__main__":
    main()
