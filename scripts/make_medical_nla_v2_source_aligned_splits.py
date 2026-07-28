"""Create Medical-NLA SFT splits using source-correct rows for training.

This is for a faithful-readout variant: train/val rows are restricted to cases
where the source model selected the DDXPlus gold label. The test manifest keeps
all remaining rows from the selected diagnoses so source-correct and
source-wrong behavior can be analyzed separately.
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

from scripts.make_medical_nla_v2_sft_splits import make_sft_row
from src.jsonl import read_jsonl, write_jsonl


def source_answer_map(path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        row_id = str(row["id"])
        if row_id in out:
            raise ValueError(f"Duplicate source-answer id: {row_id}")
        out[row_id] = row
    return out


def attach_source(row: dict[str, Any], source_row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    out["source_selected_diagnosis_id"] = source_row.get("selected_diagnosis_id")
    out["source_selected_diagnosis_name"] = source_row.get("selected_diagnosis_name")
    out["source_diagnosis_hit"] = bool(source_row.get("diagnosis_hit"))
    out["source_answer"] = source_row.get("answer")
    out["source_parse_method"] = source_row.get("parse_method")
    return out


def split_source_correct(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
    min_source_correct_per_diagnosis: int,
) -> tuple[dict[str, str], set[str]]:
    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dx[str(row["diagnosis_id"])].append(row)

    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    selected_dx: set[str] = set()
    for diagnosis_id, dx_rows in sorted(by_dx.items()):
        correct = [row for row in dx_rows if row["source_diagnosis_hit"]]
        if len(correct) < min_source_correct_per_diagnosis:
            continue
        selected_dx.add(diagnosis_id)
        correct = list(correct)
        rng.shuffle(correct)
        n = len(correct)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac))) if val_frac > 0 and n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 3 else 0
        for row in correct[:n_train]:
            split_map[str(row.get("base_id", row["id"]))] = "train"
        for row in correct[n_train : n_train + n_val]:
            split_map[str(row.get("base_id", row["id"]))] = "val"

    for row in rows:
        diagnosis_id = str(row["diagnosis_id"])
        if diagnosis_id not in selected_dx:
            continue
        base_id = str(row.get("base_id", row["id"]))
        split_map.setdefault(base_id, "test")
    return split_map, selected_dx


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    split_map: dict[str, str],
    selected_dx: set[str],
    target_style: str,
) -> None:
    selected_rows = [
        row
        for row in rows
        if str(row["diagnosis_id"]) in selected_dx
        and str(row.get("base_id", row["id"])) in split_map
    ]
    split_counts = Counter(split_map.values())
    by_split_source = Counter(
        (split_map[str(row.get("base_id", row["id"]))], bool(row["source_diagnosis_hit"]))
        for row in selected_rows
    )
    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_dx[str(row["diagnosis_id"])].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA v2 Source-Aligned SFT Split Summary\n\n")
        f.write("Train/val rows are restricted to source-correct cases. Test keeps all remaining selected-diagnosis rows.\n\n")
        f.write(f"- target_style: {target_style}\n")
        f.write(f"- selected_diagnoses: {len(selected_dx)}\n")
        f.write(f"- selected_rows: {len(selected_rows)}\n")
        for split in ("train", "val", "test"):
            f.write(f"- {split}_rows: {split_counts[split]}\n")
            f.write(f"- {split}_source_correct: {by_split_source[(split, True)]}\n")
            f.write(f"- {split}_source_wrong: {by_split_source[(split, False)]}\n")
        f.write("\n## Diagnosis Counts\n\n")
        f.write("| diagnosis_id | total | source_correct | train | val | test |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for diagnosis_id in sorted(selected_dx):
            items = by_dx[diagnosis_id]
            counts = Counter(split_map[str(row.get("base_id", row["id"]))] for row in items)
            correct = sum(bool(row["source_diagnosis_hit"]) for row in items)
            f.write(
                f"| {diagnosis_id} | {len(items)} | {correct} | "
                f"{counts['train']} | {counts['val']} | {counts['test']} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--source-answers", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["cue_count_all"])
    parser.add_argument("--max-cues", type=int, default=12)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-source-correct-per-diagnosis", type=int, default=10)
    args = parser.parse_args()

    variants = set(args.variants)
    source_by_id = source_answer_map(args.source_answers)
    rows: list[dict[str, Any]] = []
    missing_source = 0
    for row in read_jsonl(args.manifest):
        if variants and row.get("variant") not in variants:
            continue
        if not row.get("activation_path"):
            continue
        source_row = source_by_id.get(str(row["id"]))
        if source_row is None:
            missing_source += 1
            continue
        rows.append(attach_source(row, source_row))
    if not rows:
        raise ValueError("No rows selected. Check --manifest, --variants, and --source-answers.")

    split_map, selected_dx = split_source_correct(
        rows,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        min_source_correct_per_diagnosis=args.min_source_correct_per_diagnosis,
    )
    selected_rows = [
        row
        for row in rows
        if str(row["diagnosis_id"]) in selected_dx
        and str(row.get("base_id", row["id"])) in split_map
    ]
    if not selected_rows:
        raise ValueError("No selected rows after source-correct diagnosis filtering.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_style = "structured_readout_v2_source_aligned"

    split_rows = []
    for row in selected_rows:
        base_id = str(row.get("base_id", row["id"]))
        split = split_map[base_id]
        manifest_row = dict(row)
        manifest_row["split"] = split
        split_rows.append(manifest_row)

    for split in ("train", "val", "test"):
        manifest_out = [row for row in split_rows if row["split"] == split]
        write_jsonl(out_dir / f"manifest_{split}.jsonl", manifest_out)
        sft_out = [
            {
                **make_sft_row(row, max_cues=args.max_cues, split=split),
                "target_style": target_style,
                "source_selected_diagnosis_id": row.get("source_selected_diagnosis_id"),
                "source_selected_diagnosis_name": row.get("source_selected_diagnosis_name"),
                "source_diagnosis_hit": row.get("source_diagnosis_hit"),
            }
            for row in manifest_out
        ]
        write_jsonl(out_dir / f"sft_{split}.jsonl", sft_out)

    metadata = {
        "manifest": args.manifest,
        "source_answers": args.source_answers,
        "variants": args.variants,
        "target_style": target_style,
        "max_cues": args.max_cues,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "min_source_correct_per_diagnosis": args.min_source_correct_per_diagnosis,
        "missing_source_rows": missing_source,
        "n_input_rows": len(rows),
        "n_selected_rows": len(selected_rows),
        "n_selected_diagnoses": len(selected_dx),
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.md",
        rows=rows,
        split_map=split_map,
        selected_dx=selected_dx,
        target_style=target_style,
    )
    print(f"[done] wrote source-aligned SFT files to {out_dir}")
    print(f"[done] selected_diagnoses={len(selected_dx)} selected_rows={len(selected_rows)}")


if __name__ == "__main__":
    main()
