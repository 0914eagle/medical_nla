"""Compare activation shifts in the specificity experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


COMPARISONS = [
    (
        "same_nonspecific_cue_full_vs_alone",
        "specific_full_nonspecific_cue",
        "nonspecific_alone_cue",
    ),
    ("format_full_vs_alone", "specific_full_format", "nonspecific_alone_format"),
    ("specific_cue_1_vs_nonspecific_full", "specific_full_specific_cue_1", "specific_full_nonspecific_cue"),
    ("specific_cue_2_vs_nonspecific_full", "specific_full_specific_cue_2", "specific_full_nonspecific_cue"),
    ("specific_cue_3_vs_nonspecific_full", "specific_full_specific_cue_3", "specific_full_nonspecific_cue"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def variant(row: dict) -> str:
    return row.get("variant") or row["id"].split("__", 1)[1]


def load_vector(path: str) -> torch.Tensor:
    tensor = torch.load(path, map_location="cpu")
    return tensor.float().reshape(-1)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(F.cosine_similarity(a, b, dim=0).item())


def l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a - b).item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    grouped: dict[str, dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("base_id", row["id"]), {})[variant(row)] = row

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as f:
        for base_id, variants in sorted(grouped.items()):
            for name, left_key, right_key in COMPARISONS:
                if left_key not in variants or right_key not in variants:
                    continue
                left = variants[left_key]
                right = variants[right_key]
                left_vec = load_vector(left["activation_path"])
                right_vec = load_vector(right["activation_path"])
                row = {
                    "base_id": base_id,
                    "comparison": name,
                    "left_id": left["id"],
                    "right_id": right["id"],
                    "left_variant": left_key,
                    "right_variant": right_key,
                    "cosine": cosine(left_vec, right_vec),
                    "l2": l2(left_vec, right_vec),
                    "left_norm": float(torch.linalg.vector_norm(left_vec).item()),
                    "right_norm": float(torch.linalg.vector_norm(right_vec).item()),
                    "nonspecific_target": left.get("nonspecific_target")
                    or right.get("nonspecific_target"),
                    "specific_target": left.get("specific_target") or right.get("specific_target"),
                    "nonspecific_expected": left.get("nonspecific_expected")
                    or right.get("nonspecific_expected"),
                    "specific_expected": left.get("specific_expected")
                    or right.get("specific_expected"),
                    "diagnostic_shift": left.get("diagnostic_shift")
                    or right.get("diagnostic_shift"),
                }
                f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                count += 1
    print(f"wrote {count} vector comparisons to {out_path}")


if __name__ == "__main__":
    main()
