import json
from argparse import Namespace

import pytest
import torch

from scripts.run_ddxplus_structured_reader import (
    LOCKED_CONFIRMATION,
    aligned_cues,
    build_lexicon,
    evaluate,
    evaluate_readouts,
    freeze_protocol,
    render_claims,
    validate_locked_access,
)


def case(identifier, cues):
    return {
        "id": f"{identifier}__original",
        "base_id": identifier,
        "variant": "original",
        "diagnosis_id": "dx",
        "cue_evidence_ids": [item[0] for item in cues],
        "cue_value_ids": [item[1] for item in cues],
        "cue_targets": [item[2] for item in cues],
    }


def protocol(rows):
    labels = ["cough", "fever"]
    values = {"fever": ["high", "low"]}
    return {
        "finding_labels": labels,
        "finding_threshold": 0.5,
        "values_by_evidence": values,
        "lexicon": build_lexicon(rows, labels, values),
    }


def test_train_only_lexicon_is_modal_and_tie_deterministic():
    rows = [
        case("a", [("cough", "", "cough"), ("fever", "high", "high fever")]),
        case("b", [("cough", "", "a cough"), ("fever", "high", "high fever")]),
        case("c", [("cough", "", "cough"), ("fever", "low", "low fever")]),
    ]
    lexicon = build_lexicon(rows, ["cough", "fever"], {"fever": ["high", "low"]})
    assert lexicon["findings"]["cough"]["text"] == "cough"
    assert lexicon["values"]["fever\0high"]["text"] == "high fever"
    assert lexicon["values"]["fever\0low"]["text"] == "low fever"


def test_rendering_uses_probe_state_and_not_case_prompt():
    rows = [
        case("a", [("cough", "", "a cough"), ("fever", "high", "high fever")]),
        case("b", [("cough", "", "a cough"), ("fever", "low", "low fever")]),
    ]
    frozen = protocol(rows)
    claims, observed = render_claims(
        {**rows[0], "prompt": "SECRET PROMPT"},
        torch.tensor([0.8, 0.9]),
        torch.tensor([0.0, 3.0]),
        frozen,
        {"fever": (0, 2)},
    )
    assert [claim["evidence_id"] for claim in claims] == ["fever", "cough"]
    assert claims[0]["value_id"] == "low"
    assert "low fever" in observed
    assert "SECRET" not in observed


def test_structured_metrics_follow_deletion_retention_and_value_edit(tmp_path):
    original = case("a", [("cough", "", "a cough"), ("fever", "high", "high fever")])
    deleted = {
        **case("a", [("cough", "", "a cough")]),
        "id": "a__deleted",
        "variant": "cue_deleted",
        "cf_original_evidence_id": "fever",
    }
    edited = {
        **case("a", [("cough", "", "a cough"), ("fever", "low", "low fever")]),
        "id": "a__edited",
        "variant": "value_edited",
        "cf_original_evidence_id": "fever",
        "cf_original_value_id": "high",
        "cf_replacement_value_id": "low",
    }
    donor = case("b", [("cough", "", "a cough")])
    rows = [original, deleted, edited, donor]
    outputs = [
        {
            "selected_claims": [
                {"evidence_id": "cough", "value_id": None},
                {"evidence_id": "fever", "value_id": "high"},
            ]
        },
        {"selected_claims": [{"evidence_id": "cough", "value_id": None}]},
        {
            "selected_claims": [
                {"evidence_id": "cough", "value_id": None},
                {"evidence_id": "fever", "value_id": "low"},
            ]
        },
        {"selected_claims": [{"evidence_id": "cough", "value_id": None}]},
    ]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"own_base_id": "a", "donor_base_id": "b"}) + "\n")
    result = evaluate_readouts(rows, outputs, protocol([original, edited, donor]), pairs)
    assert result["finding"]["micro_f1"] == 1.0
    assert result["deletion"]["removal_success_given_original_hit"] == 1.0
    assert result["retained"]["preservation_given_original_hit"] == 1.0
    assert result["value_edit"]["clean_switch_given_original_old"] == 1.0


def test_aligned_cues_rejects_mismatched_fields():
    with pytest.raises(ValueError, match="not aligned"):
        aligned_cues(
            {
                "base_id": "bad",
                "cue_targets": ["x"],
                "cue_evidence_ids": [],
                "cue_value_ids": [],
            }
        )


def test_locked_confirmation_constant_is_explicit():
    assert LOCKED_CONFIRMATION == "I_ACCEPT_DDXPLUS_STRUCTURED_READER_LOCKED_TEST"


def test_locked_access_requires_matching_validation_receipt(tmp_path):
    receipt = tmp_path / "results.json"
    receipt.write_text(
        json.dumps({"population": "validation", "protocol_sha256": "abc"}),
        encoding="utf-8",
    )
    validate_locked_access("locked_test", LOCKED_CONFIRMATION, receipt, "abc")
    with pytest.raises(ValueError, match="differs"):
        validate_locked_access("locked_test", LOCKED_CONFIRMATION, receipt, "xyz")
    with pytest.raises(ValueError, match="confirmation"):
        validate_locked_access("locked_test", "wrong", receipt, "abc")


def test_freeze_and_validation_evaluate_end_to_end(tmp_path):
    train_rows = [
        case("a", [("cough", "", "a cough"), ("fever", "high", "high fever")]),
        case("b", [("cough", "", "a cough"), ("fever", "low", "low fever")]),
    ]
    train_cases = tmp_path / "train.jsonl"
    train_cases.write_text(
        "".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8"
    )
    artifact_path = tmp_path / "finding_value_hs24.pt"
    torch.save(
        {
            "layer": 24,
            "feature_mean": torch.zeros((1, 3)),
            "feature_std": torch.ones((1, 3)),
            "finding_labels": ["cough", "fever"],
            "finding_threshold": 0.5,
            "finding_state_dict": {
                "weight": torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0]]),
                "bias": torch.tensor([-5.0, -5.0]),
            },
            "values_by_evidence": {"fever": ["high", "low"]},
            "value_slices": {"fever": (0, 2)},
            "value_state_dict": {
                "weight": torch.tensor([[0.0, 0.0, 10.0], [0.0, 0.0, -10.0]]),
                "bias": torch.zeros(2),
            },
        },
        artifact_path,
    )
    protocol_path = tmp_path / "protocol.json"
    freeze_protocol(
        Namespace(
            artifact=artifact_path,
            train_cases=train_cases,
            output=protocol_path,
            expected_layer=24,
        )
    )

    validation_rows = [
        train_rows[0],
        case("b", [("cough", "", "a cough")]),
    ]
    rows = []
    for row, vector in zip(
        validation_rows,
        (torch.tensor([1.0, 1.0, 1.0]), torch.tensor([1.0, 0.0, 0.0])),
        strict=True,
    ):
        activation = tmp_path / f"{row['base_id']}.pt"
        torch.save(vector, activation)
        rows.append(
            {
                **row,
                "position_family": "P0",
                "activation_path": str(activation),
            }
        )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    hard_pairs = tmp_path / "pairs.jsonl"
    hard_pairs.write_text(
        json.dumps({"own_base_id": "a", "donor_base_id": "b"}) + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"
    evaluate(
        Namespace(
            population="validation",
            confirmation=None,
            validation_results=None,
            protocol=protocol_path,
            artifact=artifact_path,
            manifest=manifest,
            hard_pairs=hard_pairs,
            out_dir=out_dir,
            path_maps=[],
            device="cpu",
        )
    )
    assert len((out_dir / "readouts.jsonl").read_text().splitlines()) == 2
    report = json.loads((out_dir / "results.json").read_text())
    assert report["population"] == "validation"
    assert report["metrics"]["finding"]["micro_f1"] == 1.0
