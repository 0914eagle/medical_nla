"""Create leakage-safe train/val/test files for Medical-NLA SFT.

The split is stratified by diagnosis_id and grouped by base_id, so the same
patient/case never appears in more than one split. The script writes both:

- manifest_{split}.jsonl: original activation rows for evaluation
- sft_{split}.jsonl: training rows with target_text for LoRA SFT
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_medical_nla_sft_dataset import diagnosis_name, target_text
from src.jsonl import read_jsonl, write_jsonl


def split_cases(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("--train-frac must be > 0 and train_frac + val_frac must be < 1")

    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        base_id = str(row.get("base_id", row["id"]))
        diagnosis = str(row["diagnosis_id"])
        if base_id in by_case and by_case[base_id]["diagnosis_id"] != diagnosis:
            raise ValueError(f"Case {base_id} has inconsistent diagnosis labels.")
        by_case[base_id] = {"base_id": base_id, "diagnosis_id": diagnosis}

    by_diagnosis: dict[str, list[str]] = defaultdict(list)
    for case in by_case.values():
        by_diagnosis[case["diagnosis_id"]].append(case["base_id"])

    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    for diagnosis, case_ids in sorted(by_diagnosis.items()):
        case_ids = list(case_ids)
        rng.shuffle(case_ids)
        n = len(case_ids)
        if n < 3:
            raise ValueError(f"Diagnosis {diagnosis!r} has fewer than 3 cases: {n}")
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        for case_id in case_ids[:n_train]:
            split_map[case_id] = "train"
        for case_id in case_ids[n_train : n_train + n_val]:
            split_map[case_id] = "val"
        for case_id in case_ids[n_train + n_val :]:
            split_map[case_id] = "test"
    return split_map


def make_sft_row(row: dict[str, Any], *, style: str, max_cues: int, split: str) -> dict[str, Any]:
    return {
        "id": row["id"],
        "base_id": row.get("base_id", row["id"]),
        "split": split,
        "activation_path": row["activation_path"],
        "prompt": row.get("prompt"),
        "variant": row.get("variant"),
        "diagnosis_id": row.get("diagnosis_id"),
        "diagnosis_name": diagnosis_name(row),
        "diagnosis_aliases": row.get("diagnosis_aliases"),
        "target_text": target_text(row, style=style, max_cues=max_cues),
        "target_style": style,
        "cue_targets": row.get("cue_targets"),
        "cue_types": row.get("cue_types"),
        "cue_evidence_ids": row.get("cue_evidence_ids"),
        "source": row.get("source"),
        "patient_id": row.get("patient_id"),
        "layer": row.get("layer"),
        "position": row.get("position"),
        "position_family": row.get("position_family"),
        "position_mode": row.get("position_mode"),
    }


def write_summary(path: Path, *, rows: list[dict[str, Any]], split_map: dict[str, str]) -> None:
    split_counts = Counter(split_map.values())
    row_split_counts = Counter(str(row["split"]) for row in rows)
    by_split_dx: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_split_dx[str(row["split"])][str(row["diagnosis_id"])] += 1

    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA SFT Split Summary\n\n")
        f.write("Split is grouped by `base_id` and stratified by `diagnosis_id`.\n\n")
        f.write(f"- cases: {len(split_map)}\n")
        f.write(f"- rows: {len(rows)}\n")
        for split in ("train", "val", "test"):
            f.write(f"- {split}_cases: {split_counts[split]}\n")
            f.write(f"- {split}_rows: {row_split_counts[split]}\n")
        f.write("\n## Diagnosis Counts By Split\n\n")
        f.write("| diagnosis_id | train | val | test |\n")
        f.write("|---|---:|---:|---:|\n")
        diagnoses = sorted({str(row["diagnosis_id"]) for row in rows})
        for diagnosis in diagnoses:
            f.write(
                f"| {diagnosis} | "
                f"{by_split_dx['train'][diagnosis]} | "
                f"{by_split_dx['val'][diagnosis]} | "
                f"{by_split_dx['test'][diagnosis]} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["multi_format"])
    parser.add_argument(
        "--style",
        choices=["diagnosis_first", "cue_then_diagnosis"],
        default="diagnosis_first",
    )
    parser.add_argument("--max-cues", type=int, default=3)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    variants = set(args.variants)
    rows = []
    for row in read_jsonl(args.manifest):
        if variants and row.get("variant") not in variants:
            continue
        if not row.get("activation_path"):
            continue
        if not row.get("diagnosis_id") and not row.get("diagnosis_name"):
            continue
        rows.append(row)
        if args.limit is not None and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("No rows selected. Check --manifest and --variants.")

    split_map = split_cases(
        rows,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = [
        {
            "base_id": base_id,
            "split": split,
            "diagnosis_id": next(
                str(row["diagnosis_id"])
                for row in rows
                if str(row.get("base_id", row["id"])) == base_id
            ),
        }
        for base_id, split in sorted(split_map.items())
    ]
    write_jsonl(out_dir / "splits.jsonl", split_rows)

    selected_manifest_rows: list[dict[str, Any]] = []
    selected_sft_rows: list[dict[str, Any]] = []
    for row in rows:
        base_id = str(row.get("base_id", row["id"]))
        split = split_map[base_id]
        manifest_row = dict(row)
        manifest_row["split"] = split
        selected_manifest_rows.append(manifest_row)
        selected_sft_rows.append(
            make_sft_row(row, style=args.style, max_cues=args.max_cues, split=split)
        )

    for split in ("train", "val", "test"):
        manifest_out = [row for row in selected_manifest_rows if row["split"] == split]
        sft_out = [row for row in selected_sft_rows if row["split"] == split]
        write_jsonl(out_dir / f"manifest_{split}.jsonl", manifest_out)
        write_jsonl(out_dir / f"sft_{split}.jsonl", sft_out)

    metadata = {
        "manifest": args.manifest,
        "variants": args.variants,
        "style": args.style,
        "max_cues": args.max_cues,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "test_frac": 1.0 - args.train_frac - args.val_frac,
        "seed": args.seed,
        "n_rows": len(selected_sft_rows),
        "n_cases": len(split_map),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(out_dir / "summary.md", rows=selected_sft_rows, split_map=split_map)
    print(f"[done] wrote SFT split files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
