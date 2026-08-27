"""Build a train-only DDXPlus population for finding and value probes.

The frozen E5 validation/test protocol remains unchanged.  This script reads
its diagnosis support, applies the same eligibility rules to the official
training CSV, and emits original-case CoT-P0 rows only.  Counterfactual and
locked-test rows are deliberately outside this interface.
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
    make_activation_row,
    sample_split,
    sha256_file,
    sha256_values,
)
from src.jsonl import read_jsonl, write_jsonl


def cue_statistics(cases: list[dict[str, Any]]) -> dict[str, Any]:
    evidence_counts: Counter[str] = Counter()
    value_counts: Counter[str] = Counter()
    evidence_value_counts: Counter[str] = Counter()
    cue_total = 0
    valued_total = 0
    for case in cases:
        evidence_ids = list(case.get("cue_evidence_ids") or [])
        value_ids = list(case.get("cue_value_ids") or [])
        if len(evidence_ids) != len(value_ids):
            raise ValueError(f"Cue/value length mismatch for {case.get('base_id')}")
        cue_total += len(evidence_ids)
        for evidence_id, value_id in zip(evidence_ids, value_ids, strict=True):
            evidence = str(evidence_id)
            evidence_counts[evidence] += 1
            value = str(value_id or "")
            if value:
                valued_total += 1
                value_counts[value] += 1
                evidence_value_counts[f"{evidence}::{value}"] += 1
    return {
        "cue_occurrences": cue_total,
        "value_bearing_cue_occurrences": valued_total,
        "unique_evidence_ids": len(evidence_counts),
        "unique_value_ids": len(value_counts),
        "unique_evidence_value_pairs": len(evidence_value_counts),
        "evidence_counts": dict(sorted(evidence_counts.items())),
        "evidence_value_counts": dict(sorted(evidence_value_counts.items())),
    }


def provided_patient_ids(paths: list[Path]) -> set[str]:
    result: set[str] = set()
    for path in paths:
        for row in read_jsonl(path):
            if row.get("patient_id_source") == "provided":
                result.add(str(row.get("source_patient_id") or ""))
    result.discard("")
    return result


def write_summary(path: Path, protocol: dict[str, Any]) -> None:
    scan = protocol["scan"]
    stats = protocol["cue_statistics"]
    lines = [
        "# DDXPlus Probe Training Population",
        "",
        "Train-only development population. Frozen E5 validation/test files were not "
        "read as training rows.",
        "",
        f"- cases: **{protocol['cases']}**",
        f"- diagnoses fixed by E5 protocol: **{protocol['diagnosis_count']}**",
        f"- sampled diagnoses represented in train: **{protocol['represented_diagnosis_count']}**",
        f"- CoT-P0 activation rows: **{protocol['activation_rows']}**",
        f"- cue occurrences: **{stats['cue_occurrences']}**",
        f"- unique evidence IDs: **{stats['unique_evidence_ids']}**",
        f"- value-bearing cue occurrences: **{stats['value_bearing_cue_occurrences']}**",
        f"- unique evidence/value pairs: **{stats['unique_evidence_value_pairs']}**",
        "- provided patient-ID overlap with supplied evaluation cases: "
        f"**{protocol['provided_patient_id_overlap']}**",
        "",
        "## Eligibility Audit",
        "",
        f"- rows scanned: **{scan.get('rows_scanned', 0)}**",
        f"- eligible rows before E5 diagnosis filter: **{scan.get('eligible_rows', 0)}**",
        f"- fewer than three clean cues: **{scan.get('fewer_than_three_cues', 0)}**",
        f"- gold diagnosis named in prompt: **{scan.get('gold_named_in_prompt', 0)}**",
        "",
        "The finding and value vocabularies must be derived from this train population only.",
        "Validation may select layer, regularization, and thresholds; official test "
        "remains locked.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True, type=Path)
    parser.add_argument("--evidences", required=True, type=Path)
    parser.add_argument("--e5-protocol", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--reference-cases", action="append", default=[], type=Path)
    parser.add_argument("--examples-per-diagnosis", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.examples_per_diagnosis <= 0:
        raise ValueError("--examples-per-diagnosis must be positive")
    frozen = json.loads(args.e5_protocol.read_text(encoding="utf-8"))
    diagnoses = [str(value) for value in frozen.get("common_diagnoses") or []]
    if not diagnoses:
        raise ValueError("E5 protocol has no common_diagnoses")
    evidence_meta = read_json(args.evidences)
    if not isinstance(evidence_meta, dict):
        raise ValueError("--evidences must contain a JSON object")

    sampled, scan = sample_split(
        args.train_csv,
        split="train",
        evidence_meta=evidence_meta,
        seed=args.seed,
        quota=args.examples_per_diagnosis,
    )
    diagnosis_set = set(diagnoses)
    cases = [row for row in sampled if str(row["diagnosis_id"]) in diagnosis_set]
    represented = sorted({str(row["diagnosis_id"]) for row in cases})
    missing = sorted(diagnosis_set - set(represented))
    if missing:
        raise ValueError(f"Official train has no eligible rows for E5 diagnoses: {missing}")

    activation_rows = [make_activation_row(case, condition="cot") for case in cases]
    reference_ids = provided_patient_ids(args.reference_cases)
    train_ids = {
        str(row.get("source_patient_id"))
        for row in cases
        if row.get("patient_id_source") == "provided"
    }
    overlap = train_ids & reference_ids
    if overlap:
        raise ValueError(
            f"Provided patient identifiers overlap train and evaluation: {len(overlap)}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "cases_train.jsonl", cases)
    write_jsonl(args.out_dir / "activation_rows_train.jsonl", activation_rows)
    protocol = {
        "schema_version": 1,
        "purpose": "DDXPlus finding-presence and native-value probe training only",
        "seed": args.seed,
        "examples_per_diagnosis_cap": args.examples_per_diagnosis,
        "primary_hidden_state": "CoT-P0/HS16,HS24,HS32/last_token",
        "e5_protocol_path": str(args.e5_protocol),
        "e5_protocol_sha256": sha256_file(args.e5_protocol),
        "train_csv_path": str(args.train_csv),
        "train_csv_sha256": sha256_file(args.train_csv),
        "evidences_sha256": sha256_file(args.evidences),
        "diagnosis_count": len(diagnoses),
        "diagnoses": diagnoses,
        "represented_diagnosis_count": len(represented),
        "cases": len(cases),
        "case_id_sha256": sha256_values(str(row["base_id"]) for row in cases),
        "activation_rows": len(activation_rows),
        "provided_patient_id_overlap": len(overlap),
        "scan": scan,
        "cue_statistics": cue_statistics(cases),
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", protocol)
    print(
        f"[train] cases={len(cases)} diagnoses={len(represented)} "
        f"activation_rows={len(activation_rows)}",
        flush=True,
    )
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
