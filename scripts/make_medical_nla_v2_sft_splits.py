"""Create Medical-NLA v2-alpha SFT rows with structured XML readouts.

This keeps the leakage-safe split logic from v1, but changes the target from a
free-form diagnosis sentence to a small schema:

<explanation>
<readout>
  <task_type>diagnosis</task_type>
  <answer>...</answer>
  <supporting_cues>...</supporting_cues>
</readout>
</explanation>

The first v2-alpha target is intentionally DDXPlus-only and format-position
focused. It tests whether a structured target improves answer/cue readout
without adding dataset diversity yet.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_medical_nla_sft_dataset import clean_text, diagnosis_name
from scripts.make_medical_nla_sft_splits import split_cases
from src.jsonl import read_jsonl, write_jsonl


def cue_list(row: dict[str, Any], *, max_cues: int) -> list[str]:
    cues = row.get("cue_targets") or row.get("specific_targets") or []
    if isinstance(cues, str):
        cues = [cues]
    cleaned = []
    for cue in cues:
        text = clean_text(cue)
        if text and text not in cleaned:
            cleaned.append(text)
    if row.get("target_text"):
        target = clean_text(row.get("target_text"))
        if target and target not in cleaned:
            cleaned.insert(0, target)
    return cleaned[:max_cues]


def xml_text(text: str) -> str:
    return html.escape(clean_text(text), quote=False)


def structured_target_text(row: dict[str, Any], *, max_cues: int) -> str:
    dx = xml_text(diagnosis_name(row))
    cues = cue_list(row, max_cues=max_cues)
    cue_text = "; ".join(xml_text(cue) for cue in cues) if cues else "the clinical cues in the prompt"
    return (
        "<explanation>\n"
        "<readout>\n"
        "  <task_type>diagnosis</task_type>\n"
        f"  <answer>{dx}</answer>\n"
        f"  <supporting_cues>{cue_text}</supporting_cues>\n"
        "</readout>\n"
        "</explanation>"
    )


def make_sft_row(row: dict[str, Any], *, max_cues: int, split: str) -> dict[str, Any]:
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
        "target_text": structured_target_text(row, max_cues=max_cues),
        "target_style": "structured_readout_v2_alpha",
        "task_type": "diagnosis",
        "cue_targets": row.get("cue_targets"),
        "cue_types": row.get("cue_types"),
        "cue_evidence_ids": row.get("cue_evidence_ids"),
        "source": row.get("source"),
        "patient_id": row.get("patient_id"),
        "layer": row.get("layer"),
        "position": row.get("position"),
        "position_family": row.get("position_family"),
        "position_mode": row.get("position_mode"),
        "target_role": row.get("target_role"),
    }


def write_summary(path: Path, *, rows: list[dict[str, Any]], split_map: dict[str, str]) -> None:
    split_counts = Counter(split_map.values())
    row_split_counts = Counter(str(row["split"]) for row in rows)
    by_split_dx: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_split_dx[str(row["split"])][str(row["diagnosis_id"])] += 1

    diagnoses = sorted({str(row["diagnosis_id"]) for row in rows})
    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA v2-alpha SFT Split Summary\n\n")
        f.write("Structured XML readout target. Split is grouped by `base_id` and stratified by `diagnosis_id`.\n\n")
        f.write(f"- cases: {len(split_map)}\n")
        f.write(f"- rows: {len(rows)}\n")
        f.write("- target_style: structured_readout_v2_alpha\n")
        for split in ("train", "val", "test"):
            f.write(f"- {split}_cases: {split_counts[split]}\n")
            f.write(f"- {split}_rows: {row_split_counts[split]}\n")
        f.write("\n## Diagnosis Counts By Split\n\n")
        f.write("| diagnosis_id | train | val | test |\n")
        f.write("|---|---:|---:|---:|\n")
        for diagnosis in diagnoses:
            f.write(
                f"| {diagnosis} | "
                f"{by_split_dx['train'][diagnosis]} | "
                f"{by_split_dx['val'][diagnosis]} | "
                f"{by_split_dx['test'][diagnosis]} |\n"
            )
        f.write("\n## Target Example\n\n")
        if rows:
            f.write("```xml\n")
            f.write(str(rows[0]["target_text"]))
            f.write("\n```\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--variants", nargs="+", default=["multi_format"])
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
        selected_sft_rows.append(make_sft_row(row, max_cues=args.max_cues, split=split))

    for split in ("train", "val", "test"):
        write_jsonl(
            out_dir / f"manifest_{split}.jsonl",
            [row for row in selected_manifest_rows if row["split"] == split],
        )
        write_jsonl(
            out_dir / f"sft_{split}.jsonl",
            [row for row in selected_sft_rows if row["split"] == split],
        )

    metadata = {
        "manifest": args.manifest,
        "variants": args.variants,
        "target_style": "structured_readout_v2_alpha",
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
    print(f"[done] wrote v2-alpha SFT split files to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
