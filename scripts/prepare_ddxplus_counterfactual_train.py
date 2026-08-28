"""Build train-only DDXPlus counterfactual families for Medical-NLA SFT.

The official DDXPlus train population is already frozen by
``prepare_ddxplus_probe_train.py``.  This script derives one cue-deletion arm
per base case and, where the ontology supplies a native alternate value, one
value-edit arm.  It emits only derived activation rows so existing original
CoT-P0 activations are never recomputed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_ddxplus_probe_dataset import read_json
from scripts.prepare_ddxplus_e5 import (
    counterfactual_cases,
    make_activation_row,
    sha256_file,
    sha256_values,
)
from src.jsonl import read_jsonl, write_jsonl


def normalized_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def validate_family(original: dict[str, Any], derived: list[dict[str, Any]]) -> None:
    base_id = str(original.get("base_id") or "")
    variants = Counter(str(row.get("variant") or "") for row in derived)
    if variants["cue_deleted"] != 1 or variants["value_edited"] > 1:
        raise ValueError(f"Invalid counterfactual family for {base_id}: {variants}")
    original_cues = [normalized_text(value) for value in original.get("cue_targets") or []]
    if len(original_cues) < 3:
        raise ValueError(f"Base case {base_id} has fewer than three cues")

    deleted = next(row for row in derived if row["variant"] == "cue_deleted")
    deleted_cues = [normalized_text(value) for value in deleted.get("cue_targets") or []]
    removed = normalized_text(deleted.get("cf_original_cue"))
    if len(deleted_cues) != len(original_cues) - 1 or removed in deleted_cues:
        raise ValueError(f"Deletion arm does not remove exactly one cue for {base_id}")

    edited_rows = [row for row in derived if row["variant"] == "value_edited"]
    if edited_rows:
        edited = edited_rows[0]
        edited_cues = [normalized_text(value) for value in edited.get("cue_targets") or []]
        replacement = normalized_text(edited.get("cf_replacement_cue"))
        if len(edited_cues) != len(original_cues) or replacement not in edited_cues:
            raise ValueError(f"Value-edit arm is malformed for {base_id}")
        if removed in edited_cues:
            raise ValueError(f"Old value persists verbatim in value-edit arm for {base_id}")


def write_summary(path: Path, protocol: dict[str, Any]) -> None:
    lines = [
        "# DDXPlus Counterfactual SFT Training Population",
        "",
        "Official-train-only derived population. No validation or test row is used as training data.",
        "",
        f"- base cases: **{protocol['base_cases']}**",
        f"- cue-deletion arms: **{protocol['cue_deleted']}**",
        f"- native value-edit arms: **{protocol['value_edited']}**",
        f"- derived CoT-P0 activation rows: **{protocol['activation_rows']}**",
        f"- complete original/deletion families: **{protocol['complete_deletion_families']}**",
        "- hidden state: **CoT-P0/HS32/last_token**",
        "- diagnosis text in target: **forbidden**",
        "",
        "Original activations are reused from the train-only probe population. Only the derived prompts require extraction.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-train", required=True, type=Path)
    parser.add_argument("--evidences", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    evidence_meta = read_json(args.evidences)
    if not isinstance(evidence_meta, dict):
        raise ValueError("--evidences must contain a JSON object")
    cases = list(read_jsonl(args.cases_train))
    if not cases:
        raise ValueError("--cases-train is empty")

    seen_bases: set[str] = set()
    counterfactuals: list[dict[str, Any]] = []
    family_rows: list[dict[str, Any]] = []
    for case in cases:
        base_id = str(case.get("base_id") or "")
        if not base_id or base_id in seen_bases:
            raise ValueError(f"Missing or duplicate train base_id: {base_id!r}")
        seen_bases.add(base_id)
        if str(case.get("official_split")) != "train":
            raise ValueError(f"Non-train case in training population: {base_id}")
        derived = counterfactual_cases(case, evidence_meta, seed=args.seed)
        validate_family(case, derived)
        counterfactuals.extend(derived)
        family_rows.append(
            {
                "base_id": base_id,
                "diagnosis_id": case.get("diagnosis_id"),
                "original_id": case.get("id"),
                "cue_deleted_id": next(
                    row["id"] for row in derived if row["variant"] == "cue_deleted"
                ),
                "value_edited_id": next(
                    (row["id"] for row in derived if row["variant"] == "value_edited"),
                    None,
                ),
            }
        )

    activation_rows = [make_activation_row(row, condition="cot") for row in counterfactuals]
    variant_counts = Counter(str(row["variant"]) for row in counterfactuals)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "counterfactual_cases_train.jsonl", counterfactuals)
    write_jsonl(args.out_dir / "activation_rows_counterfactual_train.jsonl", activation_rows)
    write_jsonl(args.out_dir / "families_train.jsonl", family_rows)

    protocol = {
        "schema_version": 1,
        "purpose": "DDXPlus train-only counterfactual Medical-NLA supervision",
        "seed": args.seed,
        "cases_train_path": str(args.cases_train),
        "cases_train_sha256": sha256_file(args.cases_train),
        "evidences_path": str(args.evidences),
        "evidences_sha256": sha256_file(args.evidences),
        "base_cases": len(cases),
        "base_id_sha256": sha256_values(seen_bases),
        "cue_deleted": variant_counts["cue_deleted"],
        "value_edited": variant_counts["value_edited"],
        "activation_rows": len(activation_rows),
        "complete_deletion_families": len(family_rows),
        "primary_hidden_state": "CoT-P0/HS32/last_token",
        "training_sources": ["DDXPlus official train"],
        "forbidden_training_sources": ["DDXPlus validation", "DDXPlus test"],
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", protocol)
    print(
        f"[counterfactual] bases={len(cases)} deletion={variant_counts['cue_deleted']} "
        f"value_edit={variant_counts['value_edited']} derived={len(activation_rows)}",
        flush=True,
    )
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
