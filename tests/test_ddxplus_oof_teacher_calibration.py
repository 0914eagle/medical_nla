import json
from pathlib import Path

import torch

from scripts.audit_ddxplus_oof_teacher_calibration import run_audit, sha256_file
from scripts.score_ddxplus_selected_changed_cues import fold_of
from src.jsonl import write_jsonl


def identifier_for_fold(fold: int) -> str:
    candidate = 0
    while True:
        value = f"audit_case_{fold}_{candidate}"
        if fold_of(value) == fold:
            return value
        candidate += 1


def test_calibration_audit_reports_added_absent_labels(tmp_path: Path) -> None:
    labels = ["A", "B", "C"]
    originals = []
    deletions = []
    teacher_rows = []
    for fold in (0, 1):
        identifier = identifier_for_fold(fold)
        changed = "A" if fold == 0 else "B"
        other = "B" if changed == "A" else "A"
        original_path = tmp_path / f"{identifier}_original.pt"
        deleted_path = tmp_path / f"{identifier}_deleted.pt"
        original_vector = torch.tensor(
            [changed == "A", changed == "B", 1.0], dtype=torch.float32
        )
        deleted_vector = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        torch.save(original_vector, original_path)
        torch.save(deleted_vector, deleted_path)
        original = {
            "id": f"{identifier}__cot_p0",
            "base_id": identifier,
            "variant": "original",
            "diagnosis_id": "shared",
            "position_family": "P0",
            "activation_path": str(original_path),
            "cue_evidence_ids": [changed, "C"],
            "cue_value_ids": [None, None],
        }
        deletion = {
            **original,
            "id": f"{identifier}__cue_deleted__cot_p0",
            "variant": "cue_deleted",
            "activation_path": str(deleted_path),
            "cue_evidence_ids": ["C"],
            "cue_value_ids": [None],
            "cf_original_evidence_id": changed,
            "cf_original_cue": changed,
        }
        originals.append(original)
        deletions.append(deletion)
        original_probabilities = [
            0.9 if label in {changed, "C"} else 0.1 for label in labels
        ]
        deleted_probabilities = [
            0.9 if label in {other, "C"} else 0.1 for label in labels
        ]
        teacher_rows.extend(
            [
                {
                    "id": original["id"],
                    "base_id": identifier,
                    "variant": "original",
                    "finding_probabilities": original_probabilities,
                    "selected_evidence_ids": sorted([changed, "C"]),
                },
                {
                    "id": deletion["id"],
                    "base_id": identifier,
                    "variant": "cue_deleted",
                    "finding_probabilities": deleted_probabilities,
                    "selected_evidence_ids": sorted([other, "C"]),
                },
            ]
        )

    original_manifest = tmp_path / "original.jsonl"
    deletion_manifest = tmp_path / "deletion.jsonl"
    teacher_jsonl = tmp_path / "teacher.jsonl"
    write_jsonl(original_manifest, originals)
    write_jsonl(deletion_manifest, deletions)
    write_jsonl(teacher_jsonl, teacher_rows)
    teacher_report = tmp_path / "teacher_report.json"
    teacher_report.write_text(
        json.dumps(
            {
                "finding_labels": labels,
                "finding_threshold": 0.5,
                "teacher_scores_sha256": sha256_file(teacher_jsonl),
                "validation_read": False,
                "locked_test_read": False,
            }
        )
    )
    artifact = tmp_path / "finding_value_hs32.pt"
    torch.save(
        {
            "layer": 32,
            "finding_labels": labels,
            "finding_threshold": 0.5,
            "feature_mean": torch.zeros((1, 3)),
            "feature_std": torch.ones((1, 3)),
            "finding_state_dict": {
                "weight": torch.eye(3) * 5,
                "bias": torch.full((3,), -2.5),
            },
        },
        artifact,
    )

    report = run_audit(
        teacher_jsonl=teacher_jsonl,
        teacher_report=teacher_report,
        original_manifest=original_manifest,
        counterfactual_manifest=deletion_manifest,
        probe_artifact=artifact,
        output_json=tmp_path / "report.json",
        label_prevalence_jsonl=tmp_path / "prevalence.jsonl",
        summary_md=tmp_path / "summary.md",
        path_maps=[],
        device=torch.device("cpu"),
    )

    transition = report["oof_transition"]
    assert transition["added_count"]["mean"] == 1.0
    assert transition["removed_count"]["mean"] == 1.0
    assert transition["added_labels"] == 2
    assert transition["added_labels_absent_from_deleted_input"] == 2
    assert transition["added_absent_rate"] == 1.0
    assert report["full_data_frozen_probe"]["original"]["micro_f1"] == 1.0
    assert report["threshold_selected_or_changed"] is False
    assert report["student_target_written"] is False
    assert report["validation_read"] is False
    assert report["locked_test_read"] is False
    assert len((tmp_path / "prevalence.jsonl").read_text().splitlines()) == 3
    summary = (tmp_path / "summary.md").read_text()
    assert "OOF Versus Full-Data Frozen Probe" in summary
    assert "does not authorize target building" in summary
