"""Build private DiReCT source-run cases for CoT and activation extraction.

The output repeats restricted clinical note text and must remain under the
private data root. The summary is aggregate-only and safe to share.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.case_prompts import build_prompt, prose_prefix


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val_seen", "test_seen", "test_pdd_heldout"],
    )
    args = parser.parse_args()

    cases: list[dict[str, Any]] = []
    for split in args.splits:
        path = args.split_dir / f"{split}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for source in read_jsonl(path):
            note_text = str(source.get("note_text") or "").strip()
            diagnosis = str(source.get("canonical_pdd") or "").strip()
            if not note_text or not diagnosis:
                raise ValueError(f"Row {source.get('id')} lacks note_text or canonical_pdd")
            aliases = sorted(
                {
                    diagnosis,
                    str(source.get("annotation_root_diagnosis") or "").strip(),
                    str(source.get("folder_pdd") or "").strip(),
                }
                - {""}
            )
            prefix = prose_prefix(note_text)
            cases.append(
                {
                    "id": source["id"],
                    "base_id": source["id"],
                    "split": split,
                    "source": "direct",
                    "patient_group": source["patient_group"],
                    "disease_category": source["disease_category"],
                    "canonical_pdd": diagnosis,
                    "diagnosis_name": diagnosis,
                    "diagnosis_aliases": aliases,
                    "prompt": build_prompt(prefix, "direct"),
                    "prompt_cot": build_prompt(prefix, "cot"),
                }
            )

    if len({row["id"] for row in cases}) != len(cases):
        raise ValueError("Duplicate case IDs across requested splits")
    write_jsonl(args.output_jsonl, cases)
    split_counts = Counter(row["split"] for row in cases)
    lines = [
        "# DiReCT E1 Private Case Summary",
        "",
        "Aggregate-only summary. The case JSONL contains restricted note text.",
        "",
        f"- rows: **{len(cases)}**",
        f"- patient groups: **{len({row['patient_group'] for row in cases})}**",
        f"- canonical PDDs: **{len({row['canonical_pdd'] for row in cases})}**",
        "",
        "| split | n |",
        "|---|---:|",
        *[f"| {split} | {split_counts[split]} |" for split in args.splits],
    ]
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[cases] rows={len(cases)} output={args.output_jsonl}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
