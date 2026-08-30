from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.audit_sft_family_raw_outputs import (
    finalize_ddxplus,
    finalize_direct,
    prepare_ddxplus,
    prepare_direct,
)
from src.jsonl import read_jsonl, write_jsonl


def make_semantic_root(
    root: Path, method: str, outputs: dict[str, str]
) -> None:
    index = []
    audits = []
    judgements = []
    for number, (base_id, text) in enumerate(sorted(outputs.items())):
        request_id = f"{method}_{number}"
        relative = f"case_{number}.json"
        index.append(
            {
                "id": request_id,
                "base_id": base_id,
                "method": method,
                "source_relative_path": relative,
                "method_output": text,
            }
        )
        audits.append(
            {
                "id": request_id,
                "base_id": base_id,
                "method": method,
                "accepted_claims": [{"observation": "fever", "rationale": None}],
            }
        )
        judgements.append(
            {"id": request_id, "response": "{}", "judge_model": "extractor"}
        )
        eval_path = root / "evaluations" / method / relative
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(json.dumps({"len_ob_gt": 1, "len_ob_pred": 1}))
    write_jsonl(root / "private_index.jsonl", index)
    write_jsonl(root / "private_extraction_audit.jsonl", audits)
    write_jsonl(root / "extraction_judgements.jsonl", judgements)


def test_prepare_direct_uses_complete_intersection_and_keeps_raw_semantics(
    tmp_path: Path,
) -> None:
    cohort = tmp_path / "cohort.jsonl"
    sources = tmp_path / "sources.jsonl"
    readouts = tmp_path / "readouts.jsonl"
    semantic = tmp_path / "semantic"
    out = tmp_path / "restricted" / "direct" / "out"
    write_jsonl(
        cohort,
        [
            {"base_id": "a", "cue_targets": ["fever"]},
            {"base_id": "b", "cue_targets": ["cough"]},
        ],
    )
    write_jsonl(
        sources,
        [
            {"base_id": "a", "response": "Source fever."},
            {"base_id": "b", "response": "Source cough."},
        ],
    )
    write_jsonl(
        readouts,
        [
            {"base_id": "a", "nla_output": "Readout fever."},
            {"base_id": "b", "nla_output": "Readout cough."},
        ],
    )
    make_semantic_root(
        semantic / "source", "cot", {"a": "Source fever.", "b": "Source cough."}
    )
    make_semantic_root(
        semantic / "readout", "model", {"a": "Readout fever.", "b": "Readout cough."}
    )
    prepare_direct(
        argparse.Namespace(
            cohort=cohort,
            source_answers=[sources],
            method=[
                {
                    "label": "source",
                    "readout": None,
                    "semantic_root": semantic / "source",
                    "semantic_method": "cot",
                    "source_filter": None,
                },
                {
                    "label": "model",
                    "readout": readouts,
                    "semantic_root": semantic / "readout",
                    "semantic_method": "model",
                    "source_filter": None,
                },
            ],
            out_dir=out,
            cases=2,
            seed=17,
        )
    )
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["sampling"] == "complete common population"
    assert protocol["intersection_cases"] == 2
    bundle = list(read_jsonl(out / "private_bundle.jsonl"))
    assert len(bundle) == 2
    assert all(len(row["methods"]) == 2 for row in bundle)
    assert bundle[0]["methods"][0]["official_evaluation"]["len_ob_gt"] == 1


def ddx_rows(method: str) -> list[dict[str, object]]:
    rows = []
    for index in range(3):
        base_id = f"case_{index}"
        common = {
            "base_id": base_id,
            "diagnosis_id": f"dx_{index}",
            "cue_targets": ["fever", "cough", "pain"],
        }
        rows.extend(
            [
                {
                    **common,
                    "id": f"{base_id}__original",
                    "variant": "original",
                    "nla_output": f"{method} original fever cough pain",
                },
                {
                    **common,
                    "id": f"{base_id}__cue_deleted",
                    "variant": "cue_deleted",
                    "cue_targets": ["cough", "pain"],
                    "cf_original_cue": "fever",
                    "cf_original_evidence_id": "E_FEVER",
                    "nla_output": f"{method} deleted cough pain",
                },
                {
                    **common,
                    "id": f"{base_id}__value_edited",
                    "variant": "value_edited",
                    "cue_targets": ["no fever", "cough", "pain"],
                    "cf_original_cue": "fever",
                    "cf_replacement_cue": "no fever",
                    "cf_original_evidence_id": f"E_{index}",
                    "nla_output": f"{method} edited no fever cough pain",
                },
            ]
        )
    return rows


def test_prepare_ddxplus_separates_deletion_and_value_cohorts(tmp_path: Path) -> None:
    paths = []
    for method in ("control", "counterfactual"):
        path = tmp_path / f"{method}.jsonl"
        write_jsonl(path, ddx_rows(method))
        paths.append((method, path))
    out = tmp_path / "out"
    prepare_ddxplus(
        argparse.Namespace(readout=paths, out_dir=out, cases=2, seed=17)
    )
    protocol = json.loads((out / "protocol.json").read_text())
    assert protocol["candidate_counts"] == {"deletion": 3, "value_edit": 3}
    assert protocol["selected_counts"] == {"deletion": 2, "value_edit": 2}
    requests = list(read_jsonl(out / "requests.jsonl"))
    assert len(requests) == 4
    assert sum("<audit_cohort>deletion" in row["prompt"] for row in requests) == 2


def test_finalizers_emit_only_aggregate_text(tmp_path: Path) -> None:
    direct_bundle = tmp_path / "direct_bundle.jsonl"
    direct_judgements = tmp_path / "direct_judgements.jsonl"
    direct_out = tmp_path / "restricted" / "direct" / "direct_out"
    write_jsonl(
        direct_bundle,
        [
            {
                "id": "d",
                "base_id": "private_case",
                "methods": [
                    {
                        "opaque_id": "M01",
                        "method": "model",
                        "method_output": "The patient has fever.",
                    }
                ],
            }
        ],
    )
    direct_response = {
        "items": [
            {
                "opaque_id": "M01",
                **{field: field == "physician_observation_supported" for field in (
                    "physician_observation_supported",
                    "disease_template_only",
                    "boilerplate_or_format_only",
                    "unsupported_patient_claim",
                    "extractor_missed_explicit_aligned_claim",
                )},
                "supporting_quotes": {
                    "physician_observation_supported": ["patient has fever"]
                },
                "reason": "match",
            }
        ]
    }
    write_jsonl(
        direct_judgements,
        [{"id": "d", "response": json.dumps(direct_response), "judge_model": "local"}],
    )
    finalize_direct(
        argparse.Namespace(
            private_bundle=direct_bundle,
            judgements=direct_judgements,
            out_dir=direct_out,
        )
    )
    summary = (direct_out / "summary.md").read_text()
    assert "patient has fever" not in summary
    assert "private_case" not in summary

    ddx_bundle = tmp_path / "ddx_bundle.jsonl"
    ddx_judgements = tmp_path / "ddx_judgements.jsonl"
    ddx_out = tmp_path / "ddx_out"
    write_jsonl(
        ddx_bundle,
        [
            {
                "id": "x",
                "cohort": "deletion",
                "base_id": "case",
                "methods": [
                    {
                        "opaque_id": "M01",
                        "method": "model",
                        "outputs": {"original": "fever cough", "cue_deleted": "cough"},
                    }
                ],
            }
        ],
    )
    ddx_response = {
        "items": [
            {
                "opaque_id": "M01",
                "original_target_mentioned": True,
                "deleted_target_phantom": False,
                "untouched_finding_retained": True,
                "unsupported_patient_claim": False,
                "supporting_quotes": {
                    "original_target_mentioned": ["fever"],
                    "untouched_finding_retained": ["cough"],
                },
                "reason": "expected",
            }
        ]
    }
    write_jsonl(
        ddx_judgements,
        [{"id": "x", "response": json.dumps(ddx_response), "judge_model": "judge"}],
    )
    finalize_ddxplus(
        argparse.Namespace(
            private_bundle=ddx_bundle,
            judgements=ddx_judgements,
            out_dir=ddx_out,
        )
    )
    assert "fever cough" not in (ddx_out / "summary.md").read_text()
