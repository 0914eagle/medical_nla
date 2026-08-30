from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from scripts.audit_ddxplus_vanilla_zero_sample import finalize, prepare
from src.jsonl import read_jsonl, write_jsonl


def test_prepare_samples_one_case_per_diagnosis_and_preserves_null_value(
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    readouts = tmp_path / "readouts.jsonl"
    out = tmp_path / "audit"
    rows = []
    outputs = []
    for index in range(22):
        for duplicate in range(2):
            row_id = f"case_{index}_{duplicate}"
            rows.append(
                {
                    "id": row_id,
                    "variant": "original",
                    "diagnosis_id": f"dx_{index}",
                    "cue_evidence_ids": ["E1"],
                    "cue_value_ids": [None],
                    "cue_targets": ["a fever"],
                }
            )
            outputs.append({"id": row_id, "nla_output": "generic prose"})
    write_jsonl(manifest, rows)
    write_jsonl(readouts, outputs)

    prepare(
        argparse.Namespace(
            manifest=manifest, readouts=readouts, out_dir=out, cases=20, seed=17
        )
    )

    private = list(read_jsonl(out / "private_index.jsonl"))
    assert len(private) == 20
    assert len({row["diagnosis_id"] for row in private}) == 20
    assert private[0]["expected"][0]["value_id"] is None


def test_finalize_requires_verbatim_quote_for_possible_mapper_miss(
    tmp_path: Path,
) -> None:
    out = tmp_path / "audit"
    private = out / "private_index.jsonl"
    judgements = out / "judgements.jsonl"
    write_jsonl(
        private,
        [
            {
                "id": "audit",
                "source_id": "case",
                "diagnosis_id": "dx",
                "readout": "The patient has an elevated temperature.",
                "expected": [
                    {
                        "evidence_id": "E_FEVER",
                        "value_id": None,
                        "reference_phrase": "a fever",
                    }
                ],
                "readout_characters": 40,
            }
        ],
    )
    response = {
        "category": "expected_finding_paraphrase_missed",
        "matched_expected_evidence_ids": ["E_FEVER"],
        "supporting_quotes": ["invented quote"],
        "reason": "explicit paraphrase",
    }
    write_jsonl(
        judgements,
        [{"id": "audit", "response": json.dumps(response), "judge_model": "m"}],
    )

    with pytest.raises(ValueError, match="non-verbatim"):
        finalize(
            argparse.Namespace(private_index=private, judgements=judgements, out_dir=out)
        )


def test_finalize_writes_aggregate_failure_modes(tmp_path: Path) -> None:
    out = tmp_path / "audit"
    private = out / "private_index.jsonl"
    judgements = out / "judgements.jsonl"
    write_jsonl(
        private,
        [
            {
                "id": "audit",
                "source_id": "case",
                "diagnosis_id": "dx",
                "readout": "General medical discussion only.",
                "expected": [{"evidence_id": "E1", "value_id": None}],
                "readout_characters": 32,
            }
        ],
    )
    response = {
        "category": "generic_clinical_only",
        "matched_expected_evidence_ids": [],
        "supporting_quotes": [],
        "reason": "not patient-specific",
    }
    write_jsonl(
        judgements,
        [{"id": "audit", "response": json.dumps(response), "judge_model": "m"}],
    )

    finalize(
        argparse.Namespace(private_index=private, judgements=judgements, out_dir=out)
    )

    report = json.loads((out / "results.json").read_text())
    assert report["categories"] == {"generic_clinical_only": 1}
    assert report["mapper_miss_cases"] == 0
    assert "General medical discussion" not in (out / "summary.md").read_text()
