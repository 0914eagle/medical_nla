"""Freeze the DDXPlus semantic mapper ontology, aliases, prompt, and protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ddxplus_semantic_mapping import (
    build_alias_and_ontology,
    canonical_json,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-protocol", required=True, type=Path)
    parser.add_argument("--evidences", required=True, type=Path)
    parser.add_argument(
        "--prompt", default=Path("prompts/ddxplus_semantic_mapper_v1.txt"), type=Path
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    structured = json.loads(args.structured_protocol.read_text(encoding="utf-8"))
    evidence_meta = json.loads(args.evidences.read_text(encoding="utf-8"))
    if not isinstance(evidence_meta, dict):
        raise ValueError("--evidences must contain an object")
    alias_table, ontology = build_alias_and_ontology(structured, evidence_meta)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    alias_path = args.out_dir / "alias_table.json"
    ontology_path = args.out_dir / "ontology.json"
    prompt_path = args.out_dir / "mapper_prompt.txt"
    alias_path.write_text(canonical_json(alias_table) + "\n", encoding="utf-8")
    ontology_path.write_text(canonical_json(ontology) + "\n", encoding="utf-8")
    prompt_path.write_text(args.prompt.read_text(encoding="utf-8"), encoding="utf-8")
    protocol = {
        "schema_version": 1,
        "name": "ddxplus_open_text_semantic_mapper_v1",
        "method_blind": True,
        "claim_to_multiple_evidence": True,
        "assertion_and_value_validation_separate": True,
        "manual_aliases": False,
        "batch_size": args.batch_size,
        "finding_labels": structured["finding_labels"],
        "values_by_evidence": structured["values_by_evidence"],
        "structured_protocol": {
            "path": str(args.structured_protocol),
            "sha256": sha256_file(args.structured_protocol),
        },
        "release_evidences": {
            "path": str(args.evidences),
            "sha256": sha256_file(args.evidences),
        },
        "alias_table": {"path": str(alias_path), "sha256": sha256_file(alias_path)},
        "ontology": {"path": str(ontology_path), "sha256": sha256_file(ontology_path)},
        "mapper_prompt": {"path": str(prompt_path), "sha256": sha256_file(prompt_path)},
        "stage0": "observed bullets; otherwise deterministic sentence split v1",
        "stage1": "boundary-safe unambiguous train/release aliases; assertive claims only",
        "stage2": "blind quote-constrained batched AI mapping",
        "gates": {
            "G1_finding_micro_f1_min": 0.98,
            "G1_native_value_accuracy_min": 0.98,
            "G2_false_map_max": 0.05,
            "G3_replay_byte_identical": True,
            "G4_evidence_disagreement_max": 0.05,
            "G4_value_disagreement_max": 0.05,
            "G4_value_denominator_min": 20,
        },
        "validation_only_until_receipt": True,
        "locked_test_read": False,
    }
    protocol_path = args.out_dir / "semantic_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = [
        "# DDXPlus Semantic Mapper Freeze",
        "",
        f"- evidence labels: **{len(protocol['finding_labels'])}**",
        f"- value tasks: **{len(protocol['values_by_evidence'])}**",
        "- ambiguous aliases excluded: "
        f"**{len(alias_table['ambiguous_normalized_aliases_excluded'])}**",
        f"- Stage-2 batch size: **{args.batch_size}**",
        "- manual aliases: **no**",
        "- validation / locked test read: **no / no**",
        "",
    ]
    (args.out_dir / "freeze_summary.md").write_text("\n".join(summary), encoding="utf-8")
    print(f"[freeze] {protocol_path}")


if __name__ == "__main__":
    main()
