"""Compare activation-vector shifts across distractor order-control variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F


COMPARISONS = [
    ("after_primary_vs_original", "after_distractor_primary", "original_primary"),
    ("before_primary_vs_original", "before_distractor_primary", "original_primary"),
    ("neutral_primary_vs_original", "before_neutral_primary", "original_primary"),
    ("before_distractor_vs_neutral_primary", "before_distractor_primary", "before_neutral_primary"),
    ("before_format_vs_after_format", "before_distractor_format", "after_distractor_format"),
    ("neutral_format_vs_after_format", "before_neutral_format", "after_distractor_format"),
]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
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
    with out_path.open("w") as f:
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
                    "cosine": cosine(left_vec, right_vec),
                    "l2": l2(left_vec, right_vec),
                    "left_norm": float(torch.linalg.vector_norm(left_vec).item()),
                    "right_norm": float(torch.linalg.vector_norm(right_vec).item()),
                    "correct_dx": left.get("correct_dx") or right.get("correct_dx"),
                    "distractor_dx": left.get("distractor_dx") or right.get("distractor_dx"),
                    "primary_target": left.get("primary_target") or right.get("primary_target"),
                    "distractor_target": left.get("distractor_target") or right.get("distractor_target"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote vector comparisons to {out_path}")


if __name__ == "__main__":
    main()
