"""Create diagnosis-heldout (true OOD) Medical-NLA SFT splits.

The source-aligned v2 splits share the same diagnosis classes across
train/val/test, so a high test answer_hit cannot distinguish a semantic
activation readout from a seen-class classifier. This script splits at the
diagnosis-class level instead:

- train/heldout diagnosis classes are disjoint.
- train/val rows come from train classes only (source-correct only by
  default, matching the source-aligned v2 recipe).
- `test_seen` keeps the remaining train-class rows for an in-distribution
  reference under the identical adapter.
- `test_heldout` keeps every row of the heldout classes; the adapter never
  sees these diagnosis names or their activations during SFT.

Interpretation follows the handoff doc: low answer_hit with high cue_recall
on `test_heldout` means the model reads cue semantics but does not
generalize diagnosis names; low answer_hit with low cue_recall suggests a
seen-class classifier; high/high is a strong OOD readout.
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
from scripts.make_medical_nla_v2_source_aligned_splits import attach_source, source_answer_map
from src.jsonl import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test_seen", "test_heldout")
TARGET_STYLE = "structured_readout_v2_diagnosis_heldout"


def eligible_diagnoses(
    rows: list[dict[str, Any]],
    *,
    min_source_correct_per_diagnosis: int,
) -> list[str]:
    """Diagnoses with enough source-correct rows to be usable in either pool.

    The same eligibility rule is applied to train and heldout candidates so
    that the two pools stay comparable in per-class support.
    """
    correct_counts: Counter[str] = Counter()
    for row in rows:
        if row["source_diagnosis_hit"]:
            correct_counts[str(row["diagnosis_id"])] += 1
    return sorted(
        dx for dx, count in correct_counts.items() if count >= min_source_correct_per_diagnosis
    )


def split_diagnosis_classes(
    diagnoses: list[str],
    *,
    seed: int,
    heldout_frac: float,
    num_heldout: int | None,
    heldout_diagnoses: list[str] | None,
) -> tuple[list[str], list[str]]:
    """Return (train_diagnoses, heldout_diagnoses), disjoint by construction."""
    pool = sorted(diagnoses)
    if heldout_diagnoses:
        heldout = sorted(set(heldout_diagnoses))
        unknown = [dx for dx in heldout if dx not in pool]
        if unknown:
            raise ValueError(f"Requested heldout diagnoses not eligible: {unknown}")
    else:
        n_heldout = num_heldout if num_heldout is not None else int(round(len(pool) * heldout_frac))
        if not 0 < n_heldout < len(pool):
            raise ValueError(
                f"Heldout class count {n_heldout} must be in (0, {len(pool)}). "
                "Adjust --heldout-frac or --num-heldout."
            )
        shuffled = list(pool)
        random.Random(seed).shuffle(shuffled)
        heldout = sorted(shuffled[:n_heldout])
    train = [dx for dx in pool if dx not in set(heldout)]
    if not train:
        raise ValueError("No train diagnoses left after heldout selection.")
    return train, heldout


def assign_splits(
    rows: list[dict[str, Any]],
    *,
    train_diagnoses: list[str],
    heldout_diagnoses: list[str],
    seed: int,
    train_frac: float,
    val_frac: float,
    train_source_correct_only: bool,
) -> dict[str, str]:
    """Map base_id -> split name for every row of a selected diagnosis.

    Source-correct cases of train classes are split train/val/test_seen by
    `train_frac`/`val_frac` so that `test_seen` keeps some source-correct
    rows and stays comparable to `test_heldout` (which has both kinds).
    """
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1 so test_seen keeps fit-pool rows.")
    train_set = set(train_diagnoses)
    heldout_set = set(heldout_diagnoses)
    if train_set & heldout_set:
        raise ValueError(f"Train/heldout class overlap: {sorted(train_set & heldout_set)}")

    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dx[str(row["diagnosis_id"])].append(row)

    rng = random.Random(seed)
    split_map: dict[str, str] = {}
    for diagnosis_id in sorted(train_set):
        dx_rows = by_dx.get(diagnosis_id, [])
        if train_source_correct_only:
            fit_rows = [row for row in dx_rows if row["source_diagnosis_hit"]]
        else:
            fit_rows = list(dx_rows)
        base_ids = sorted({str(row.get("base_id", row["id"])) for row in fit_rows})
        rng.shuffle(base_ids)
        n = len(base_ids)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1 if n >= 3 else 0
        for base_id in base_ids[:n_train]:
            split_map[base_id] = "train"
        for base_id in base_ids[n_train : n_train + n_val]:
            split_map[base_id] = "val"
        for row in dx_rows:
            split_map.setdefault(str(row.get("base_id", row["id"])), "test_seen")

    for diagnosis_id in sorted(heldout_set):
        for row in by_dx.get(diagnosis_id, []):
            base_id = str(row.get("base_id", row["id"]))
            if base_id in split_map:
                raise ValueError(
                    f"Leakage: base_id {base_id} of heldout diagnosis {diagnosis_id} "
                    "already assigned to a train-class split."
                )
            split_map[base_id] = "test_heldout"
    return split_map


def write_summary(
    path: Path,
    *,
    rows: list[dict[str, Any]],
    split_map: dict[str, str],
    train_diagnoses: list[str],
    heldout_diagnoses: list[str],
) -> None:
    split_of = lambda row: split_map.get(str(row.get("base_id", row["id"])))  # noqa: E731
    selected_rows = [row for row in rows if split_of(row)]
    split_counts = Counter(split_of(row) for row in selected_rows)
    by_split_source = Counter(
        (split_of(row), bool(row["source_diagnosis_hit"])) for row in selected_rows
    )
    by_dx: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in selected_rows:
        by_dx[str(row["diagnosis_id"])].append(row)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA Diagnosis-Heldout SFT Split Summary\n\n")
        f.write(
            "Diagnosis classes are disjoint between train/val/test_seen and "
            "test_heldout. The adapter never sees heldout diagnosis names.\n\n"
        )
        f.write(f"- target_style: {TARGET_STYLE}\n")
        f.write(f"- train_diagnoses: {len(train_diagnoses)}\n")
        f.write(f"- heldout_diagnoses: {len(heldout_diagnoses)}\n")
        f.write(f"- selected_rows: {len(selected_rows)}\n")
        for split in SPLITS:
            f.write(f"- {split}_rows: {split_counts[split]}\n")
            f.write(f"- {split}_source_correct: {by_split_source[(split, True)]}\n")
            f.write(f"- {split}_source_wrong: {by_split_source[(split, False)]}\n")
        f.write("\n## Heldout Diagnoses\n\n")
        for diagnosis_id in heldout_diagnoses:
            f.write(f"- {diagnosis_id}\n")
        f.write("\n## Diagnosis Counts\n\n")
        f.write(
            "| diagnosis_id | pool | total | source_correct | train | val "
            "| test_seen | test_heldout |\n"
        )
        f.write("|---|---|---:|---:|---:|---:|---:|---:|\n")
        for diagnosis_id in sorted(by_dx):
            items = by_dx[diagnosis_id]
            pool = "heldout" if diagnosis_id in set(heldout_diagnoses) else "train"
            counts = Counter(split_of(row) for row in items)
            correct = sum(bool(row["source_diagnosis_hit"]) for row in items)
            f.write(
                f"| {diagnosis_id} | {pool} | {len(items)} | {correct} | "
                f"{counts['train']} | {counts['val']} | "
                f"{counts['test_seen']} | {counts['test_heldout']} |\n"
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
    parser.add_argument("--heldout-frac", type=float, default=0.30)
    parser.add_argument("--num-heldout", type=int, default=None)
    parser.add_argument(
        "--heldout-diagnoses",
        nargs="+",
        default=None,
        help="Explicit heldout diagnosis_id list; overrides --heldout-frac/--num-heldout.",
    )
    parser.add_argument(
        "--allow-source-wrong-train",
        action="store_true",
        help="Include source-wrong rows in train/val (default keeps the source-aligned recipe).",
    )
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

    eligible = eligible_diagnoses(
        rows, min_source_correct_per_diagnosis=args.min_source_correct_per_diagnosis
    )
    if len(eligible) < 2:
        raise ValueError(f"Need at least 2 eligible diagnoses, got {len(eligible)}.")
    train_dx, heldout_dx = split_diagnosis_classes(
        eligible,
        seed=args.seed,
        heldout_frac=args.heldout_frac,
        num_heldout=args.num_heldout,
        heldout_diagnoses=args.heldout_diagnoses,
    )
    kept_dx = set(train_dx) | set(heldout_dx)
    selected_rows = [row for row in rows if str(row["diagnosis_id"]) in kept_dx]
    split_map = assign_splits(
        selected_rows,
        train_diagnoses=train_dx,
        heldout_diagnoses=heldout_dx,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        train_source_correct_only=not args.allow_source_wrong_train,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_rows = []
    for row in selected_rows:
        base_id = str(row.get("base_id", row["id"]))
        split = split_map.get(base_id)
        if split is None:
            continue
        manifest_row = dict(row)
        manifest_row["split"] = split
        manifest_row["diagnosis_pool"] = (
            "heldout" if str(row["diagnosis_id"]) in set(heldout_dx) else "train"
        )
        split_rows.append(manifest_row)

    for split in SPLITS:
        manifest_out = [row for row in split_rows if row["split"] == split]
        write_jsonl(out_dir / f"manifest_{split}.jsonl", manifest_out)
        sft_out = [
            {
                **make_sft_row(row, max_cues=args.max_cues, split=split),
                "target_style": TARGET_STYLE,
                "diagnosis_pool": row["diagnosis_pool"],
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
        "target_style": TARGET_STYLE,
        "max_cues": args.max_cues,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "min_source_correct_per_diagnosis": args.min_source_correct_per_diagnosis,
        "heldout_frac": args.heldout_frac,
        "num_heldout": args.num_heldout,
        "train_source_correct_only": not args.allow_source_wrong_train,
        "missing_source_rows": missing_source,
        "n_input_rows": len(rows),
        "n_selected_rows": len(split_rows),
        "train_diagnoses": train_dx,
        "heldout_diagnoses": heldout_dx,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.md",
        rows=selected_rows,
        split_map=split_map,
        train_diagnoses=train_dx,
        heldout_diagnoses=heldout_dx,
    )
    print(f"[done] wrote diagnosis-heldout SFT files to {out_dir}")
    print(
        f"[done] train_diagnoses={len(train_dx)} "
        f"heldout_diagnoses={len(heldout_dx)} rows={len(split_rows)}"
    )


if __name__ == "__main__":
    main()
