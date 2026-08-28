from pathlib import Path

import pytest

from scripts.make_ddxplus_counterfactual_sft_dataset import (
    load_split,
    normalize_manifest_row,
    validate_families,
)
from scripts.prepare_ddxplus_counterfactual_train import validate_family
from scripts.prepare_ddxplus_e5 import counterfactual_cases
from src.jsonl import write_jsonl


def evidence_meta() -> dict:
    return {
        "E_PAIN": {
            "question_en": "Where is the pain located?",
            "is_antecedent": False,
            "value_meaning": {"CHEST": "chest", "BACK": "back"},
        },
        "E_FEVER": {"question_en": "Do you have fever?", "is_antecedent": False},
        "E_COUGH": {"question_en": "Do you have cough?", "is_antecedent": False},
    }


def original_case() -> dict:
    return {
        "id": "ddxplus_train_condition_0000001",
        "base_id": "ddxplus_train_condition_0000001",
        "variant": "original",
        "official_split": "train",
        "diagnosis_id": "condition",
        "cue_targets": [
            "the pain is located in the chest",
            "Fever is present.",
            "Cough is present.",
        ],
        "cue_count": 3,
        "cue_types": ["value", "binary", "binary"],
        "cue_polarities": ["positive", "positive", "positive"],
        "cue_evidence_ids": ["E_PAIN", "E_FEVER", "E_COUGH"],
        "cue_evidence_entries": ["E_PAIN_@_CHEST", "E_FEVER", "E_COUGH"],
        "cue_value_ids": ["CHEST", None, None],
        "cue_value_labels": ["chest", None, None],
        "cue_merged_value_counts": [1, 1, 1],
        "age": 42,
        "sex": "M",
    }


def as_manifest(row: dict, activation: Path) -> dict:
    return {
        **row,
        "id": f"{row['id']}__cot_p0",
        "position_family": "P0",
        "layer": 32,
        "activation_path": str(activation),
    }


def test_train_family_has_deletion_and_native_value_edit() -> None:
    case = original_case()
    derived = counterfactual_cases(case, evidence_meta(), seed=17)
    validate_family(case, derived)
    assert {row["variant"] for row in derived} == {"cue_deleted", "value_edited"}


def test_sft_targets_follow_each_counterfactual(tmp_path: Path) -> None:
    case = original_case()
    derived = counterfactual_cases(case, evidence_meta(), seed=17)
    rows = []
    for index, row in enumerate([case, *derived]):
        activation = tmp_path / f"{index}.pt"
        activation.write_bytes(b"activation")
        rows.append(as_manifest(row, activation))

    counts = validate_families(rows, split="train")
    assert counts == {
        "original": 1,
        "cue_deleted": 1,
        "value_edited": 1,
        "families": 1,
    }
    normalized = [
        normalize_manifest_row(
            row, split="train", max_cues=64, require_activation_file=True
        )
        for row in rows
    ]
    by_variant = {row["variant"]: row for row in normalized}
    removed = derived[0]["cf_original_cue"]
    replacement = next(
        row["cf_replacement_cue"] for row in derived if row["variant"] == "value_edited"
    )
    assert removed in by_variant["original"]["target_text"]
    assert removed not in by_variant["cue_deleted"]["target_text"]
    assert removed not in by_variant["value_edited"]["target_text"]
    assert replacement in by_variant["value_edited"]["target_text"]
    assert "condition" not in by_variant["original"]["target_text"].casefold()


def test_loader_rejects_incomplete_family(tmp_path: Path) -> None:
    case = original_case()
    activation = tmp_path / "original.pt"
    activation.write_bytes(b"activation")
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [as_manifest(case, activation)])
    with pytest.raises(ValueError, match="incomplete"):
        load_split(
            [manifest], split="train", max_cues=64, require_activation_file=True
        )


def test_target_builder_refuses_cue_truncation(tmp_path: Path) -> None:
    case = original_case()
    activation = tmp_path / "original.pt"
    activation.write_bytes(b"activation")
    with pytest.raises(ValueError, match="exceed"):
        normalize_manifest_row(
            as_manifest(case, activation),
            split="train",
            max_cues=2,
            require_activation_file=True,
        )
