import json
from pathlib import Path

import torch

from scripts.materialize_ddxplus_oof_finding_teacher import (
    build_teacher,
    same_fold_diagnosis_donor,
)
from scripts.score_ddxplus_selected_changed_cues import fold_of
from src.jsonl import write_jsonl


def identifier_for_fold(fold: int, offset: int) -> str:
    found = []
    candidate = 0
    while len(found) <= offset:
        value = f"teacher_case_{fold}_{candidate}"
        if fold_of(value) == fold:
            found.append(value)
        candidate += 1
    return found[offset]


def make_population(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    originals = []
    deletions = []
    for fold in (0, 1):
        for offset in range(4):
            identifier = identifier_for_fold(fold, offset)
            changed = "B" if offset % 2 == 0 else "A"
            original_path = tmp_path / f"{identifier}_original.pt"
            deleted_path = tmp_path / f"{identifier}_deleted.pt"
            original_vector = torch.tensor(
                [changed == "A", changed == "B", 1.0, offset / 10],
                dtype=torch.float32,
            )
            deleted_vector = original_vector.clone()
            deleted_vector[:2] = 0
            torch.save(original_vector, original_path)
            torch.save(deleted_vector, deleted_path)
            original = {
                "id": f"{identifier}__cot_p0",
                "base_id": identifier,
                "variant": "original",
                "official_split": "train",
                "diagnosis_id": "shared_diagnosis",
                "position_family": "P0",
                "layer": 32,
                "activation_path": str(original_path),
                "cue_targets": [f"{changed} is present", "C is present"],
                "cue_evidence_ids": [changed, "C"],
                "cue_value_ids": [None, None],
            }
            deletion = {
                **original,
                "id": f"{identifier}__cue_deleted__cot_p0",
                "variant": "cue_deleted",
                "activation_path": str(deleted_path),
                "cue_targets": ["C is present"],
                "cue_evidence_ids": ["C"],
                "cue_value_ids": [None],
                "cf_original_evidence_id": changed,
                "cf_original_cue": f"{changed} is present",
            }
            originals.append(original)
            deletions.append(deletion)
    original_manifest = tmp_path / "original.jsonl"
    deletion_manifest = tmp_path / "deletion.jsonl"
    write_jsonl(original_manifest, originals)
    write_jsonl(deletion_manifest, deletions)
    return original_manifest, deletion_manifest, sorted(
        originals, key=lambda row: row["base_id"]
    )


def test_same_fold_diagnosis_donor_is_deterministic(tmp_path: Path) -> None:
    _, _, rows = make_population(tmp_path)
    donor = same_fold_diagnosis_donor(0, rows)
    assert donor is not None
    assert donor != 0
    assert fold_of(rows[donor]["base_id"]) == fold_of(rows[0]["base_id"])
    assert rows[donor]["diagnosis_id"] == rows[0]["diagnosis_id"]


def test_full_label_teacher_is_cross_fitted_and_target_free(tmp_path: Path) -> None:
    original_manifest, deletion_manifest, originals = make_population(tmp_path)
    artifact = tmp_path / "finding_value_hs32.pt"
    torch.save(
        {
            "layer": 32,
            # Deliberately noncanonical to verify selected target ordering.
            "finding_labels": ["C", "B", "A"],
            "finding_threshold": 0.4,
            "finding_selected": {
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "positive_weighting": False,
                "best_epoch": 2,
            },
        },
        artifact,
    )
    output_jsonl = tmp_path / "teacher.jsonl"
    report = build_teacher(
        original_manifest=original_manifest,
        counterfactual_manifest=deletion_manifest,
        probe_artifact=artifact,
        output_jsonl=output_jsonl,
        output_json=tmp_path / "report.json",
        summary_md=tmp_path / "summary.md",
        path_maps=[],
        batch_size=4,
        seed=17,
        device=torch.device("cpu"),
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == 2 * len(originals) == 16
    assert {row["variant"] for row in rows} == {"original", "cue_deleted"}
    assert all(len(row["finding_probabilities"]) == 3 for row in rows)
    assert all(
        row["selected_evidence_ids"] == sorted(row["selected_evidence_ids"])
        for row in rows
    )
    assert all("target_text" not in row for row in rows)
    assert all("activation_path" not in row and "prompt" not in row for row in rows)
    assert report["n_teacher_rows"] == 16
    assert report["complete_pair_coverage"] == 1.0
    assert report["natural_language_target_written"] is False
    assert report["student_dataset_written"] is False
    assert report["student_gate_frozen"] is False
    assert report["validation_read"] is False
    assert report["locked_test_read"] is False
    assert report["same_fold_same_diagnosis_teacher_set_control"]["pairs"] == 8
    summary = (tmp_path / "summary.md").read_text()
    assert "No student target" in summary
    assert "canonical evidence_id ascending" in summary
