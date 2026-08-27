import csv
import json
from pathlib import Path

from scripts.prepare_ddxplus_e5 import (
    counterfactual_cases,
    gold_named,
    make_activation_row,
    pair_hard_shuffles,
    sample_split,
)


def test_gold_leakage_uses_phrase_boundaries_for_short_aliases():
    assert not gold_named(
        {
            "prompt": "The pain began after exercise.",
            "diagnosis_name": "Atrial fibrillation",
            "diagnosis_aliases": ["AF"],
        }
    )
    assert gold_named(
        {
            "prompt": "The ECG shows AF with rapid ventricular response.",
            "diagnosis_name": "Atrial fibrillation",
            "diagnosis_aliases": ["AF"],
        }
    )


def evidence_meta():
    return {
        "E_PAIN": {
            "question_en": "Where is the pain located?",
            "is_antecedent": False,
            "value_meaning": {"CHEST": "chest", "BACK": "back"},
        },
        "E_FEVER": {"question_en": "Do you have a fever?", "is_antecedent": False},
        "E_COUGH": {"question_en": "Do you have a cough?", "is_antecedent": False},
    }


def write_patients(path: Path, n_per_label: int = 6) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["PATHOLOGY", "EVIDENCES", "AGE", "SEX", "id"])
        writer.writeheader()
        for label in ("Condition A", "Condition B"):
            for index in range(n_per_label):
                value = "CHEST" if index % 2 == 0 else "BACK"
                writer.writerow(
                    {
                        "PATHOLOGY": label,
                        "EVIDENCES": repr([f"E_PAIN_@_{value}", "E_FEVER", "E_COUGH"]),
                        "AGE": 40 + index,
                        "SEX": "F" if index % 2 else "M",
                        "id": f"{label}-{index}",
                    }
                )


def test_sampling_is_balanced_and_split_namespaced(tmp_path):
    path = tmp_path / "validate.csv"
    write_patients(path)
    cases, summary = sample_split(
        path,
        split="validation",
        evidence_meta=evidence_meta(),
        seed=17,
        quota=3,
        max_diagnoses=2,
    )
    assert len(cases) == 6
    assert summary["diagnoses_selected"] == 2
    assert all(row["base_id"].startswith("ddxplus_validation_") for row in cases)
    assert all(row["cue_count"] == 3 for row in cases)


def test_counterfactual_uses_same_evidence_native_value(tmp_path):
    path = tmp_path / "test.csv"
    write_patients(path, n_per_label=3)
    cases, _ = sample_split(
        path,
        split="test",
        evidence_meta=evidence_meta(),
        seed=17,
        quota=2,
        max_diagnoses=2,
    )
    rows = counterfactual_cases(cases[0], evidence_meta(), seed=17)
    deleted = next(row for row in rows if row["variant"] == "cue_deleted")
    edited = next(row for row in rows if row["variant"] == "value_edited")
    assert len(deleted["cue_targets"]) == len(cases[0]["cue_targets"]) - 1
    for field in (
        "cue_types",
        "cue_polarities",
        "cue_evidence_ids",
        "cue_evidence_entries",
        "cue_value_ids",
        "cue_value_labels",
        "cue_merged_value_counts",
    ):
        assert len(deleted[field]) == len(deleted["cue_targets"])
    assert edited["cf_replacement_evidence_id"] == edited["cf_original_evidence_id"]
    assert edited["cf_replacement_value_id"] != edited["cf_original_value_id"]
    assert edited["cf_replacement_cue"] in edited["prompt"]
    assert edited["cf_original_cue"] not in edited["prompt"]


def test_primary_activation_row_is_cot_p0_and_direct_is_separate(tmp_path):
    path = tmp_path / "test.csv"
    write_patients(path, n_per_label=3)
    cases, _ = sample_split(
        path,
        split="test",
        evidence_meta=evidence_meta(),
        seed=17,
        quota=2,
        max_diagnoses=2,
    )
    case = cases[0]

    primary = make_activation_row(case)
    control = make_activation_row(case, condition="direct")

    assert primary["id"].endswith("__cot_p0")
    assert primary["condition"] == "cot"
    assert primary["prompt"] == case["prompt_cot"]
    assert primary["prompt"] != case["prompt"]
    assert primary["position_family"] == "P0"
    assert primary["position_label"] == "cot_P0_prompt_boundary"
    assert primary["position_mode"] == "last_token"

    assert control["id"].endswith("__direct_p0")
    assert control["condition"] == "direct"
    assert control["prompt"] == case["prompt"]
    assert control["position_family"] == "P0"
    assert control["id"] != primary["id"]


def test_hard_shuffle_is_same_diagnosis_and_a_derangement(tmp_path):
    path = tmp_path / "test.csv"
    write_patients(path)
    cases, _ = sample_split(
        path,
        split="test",
        evidence_meta=evidence_meta(),
        seed=17,
        quota=5,
        max_diagnoses=2,
    )
    by_id = {row["base_id"]: row for row in cases}
    pairs = pair_hard_shuffles(cases)
    assert len(pairs) == len(cases)
    assert {row["own_base_id"] for row in pairs} == set(by_id)
    assert {row["donor_base_id"] for row in pairs} == set(by_id)
    for pair in pairs:
        assert pair["own_base_id"] != pair["donor_base_id"]
        assert (
            by_id[pair["own_base_id"]]["diagnosis_id"]
            == by_id[pair["donor_base_id"]]["diagnosis_id"]
        )
