import json
from pathlib import Path

import torch

from scripts.score_ddxplus_selected_changed_cues import (
    build_scores,
    donor_indices,
    fold_of,
)
from scripts.score_ddxplus_validation_changed_cues import build_validation_scores
from src.jsonl import write_jsonl


def identifier_for_fold(fold: int, offset: int) -> str:
    found = []
    candidate = 0
    while len(found) <= offset:
        value = f"case_{fold}_{candidate}"
        if fold_of(value) == fold:
            found.append(value)
        candidate += 1
    return found[offset]


def original_row(identifier: str, changed: str, path: Path) -> dict:
    return {
        "id": f"{identifier}__cot_p0",
        "base_id": identifier,
        "variant": "original",
        "official_split": "train",
        "diagnosis_id": "shared_diagnosis",
        "position_family": "P0",
        "layer": 32,
        "activation_path": str(path),
        "cue_targets": [f"{changed} is present", "C is present"],
        "cue_evidence_ids": [changed, "C"],
        "cue_value_ids": [None, None],
    }


def deletion_row(original: dict, changed: str, path: Path) -> dict:
    return {
        **original,
        "id": f"{original['base_id']}__cue_deleted__cot_p0",
        "variant": "cue_deleted",
        "activation_path": str(path),
        "cue_targets": ["C is present"],
        "cue_evidence_ids": ["C"],
        "cue_value_ids": [None],
        "cf_original_evidence_id": changed,
        "cf_original_cue": f"{changed} is present",
    }


def make_population(tmp_path: Path) -> tuple[Path, Path, list[dict]]:
    originals = []
    deletions = []
    for fold in (0, 1):
        for offset in range(4):
            identifier = identifier_for_fold(fold, offset)
            changed = "A" if offset % 2 == 0 else "B"
            original_path = tmp_path / f"{identifier}_original.pt"
            deleted_path = tmp_path / f"{identifier}_deleted.pt"
            vector = torch.tensor(
                [changed == "A", changed == "B", 1.0, float(offset) / 10],
                dtype=torch.float32,
            )
            deleted_vector = vector.clone()
            deleted_vector[0:2] = 0
            torch.save(vector, original_path)
            torch.save(deleted_vector, deleted_path)
            original = original_row(identifier, changed, original_path)
            originals.append(original)
            deletions.append(deletion_row(original, changed, deleted_path))
    original_manifest = tmp_path / "original.jsonl"
    deletion_manifest = tmp_path / "deletion.jsonl"
    write_jsonl(original_manifest, originals)
    write_jsonl(deletion_manifest, deletions)
    return original_manifest, deletion_manifest, originals


def test_donors_are_same_fold_same_diagnosis_and_cue_absent(tmp_path: Path) -> None:
    _, _, rows = make_population(tmp_path)
    own = rows[0]
    changed = own["cue_evidence_ids"][0]
    selected = donor_indices(0, rows, changed_evidence=changed, maximum=5)
    assert selected
    for index in selected:
        donor = rows[index]
        assert fold_of(donor["base_id"]) == fold_of(own["base_id"])
        assert donor["diagnosis_id"] == own["diagnosis_id"]
        assert changed not in donor["cue_evidence_ids"]


def test_build_scores_is_read_only_and_cross_fitted(tmp_path: Path) -> None:
    original_manifest, deletion_manifest, originals = make_population(tmp_path)
    artifact = tmp_path / "finding_value_hs32.pt"
    torch.save(
        {
            "layer": 32,
            "finding_labels": ["A", "B", "C"],
            "finding_selected": {
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "positive_weighting": False,
                "best_epoch": 2,
            },
        },
        artifact,
    )
    output_jsonl = tmp_path / "scores.jsonl"
    output_json = tmp_path / "report.json"
    summary = tmp_path / "summary.md"
    report = build_scores(
        original_manifest=original_manifest,
        counterfactual_manifest=deletion_manifest,
        probe_artifact=artifact,
        output_jsonl=output_jsonl,
        output_json=output_json,
        summary_md=summary,
        path_maps=[],
        max_donors=5,
        min_fold_positive_count=1,
        batch_size=4,
        seed=17,
        device=torch.device("cpu"),
    )

    rows = [json.loads(line) for line in output_jsonl.read_text().splitlines()]
    assert len(rows) == len(originals) == 8
    assert report["threshold_applied"] is False
    assert report["target_written"] is False
    assert report["locked_test_read"] is False
    assert report["counts"]["score_eligible"] == 8
    assert report["counts"]["donor_available"] == 8
    assert {row["fold"] for row in rows} == {0, 1}
    assert all(row["selected_changed_cue_supported"] is None for row in rows)
    assert all(row["donor_count"] >= 1 for row in rows)
    assert all(row["p_original"] is not None for row in rows)
    assert all(row["p_deleted"] is not None for row in rows)
    assert "No threshold or target is written" in summary.read_text()


def test_validation_null_audit_is_read_only_and_paired(tmp_path: Path) -> None:
    original_manifest, deletion_manifest, originals = make_population(tmp_path)
    combined = tmp_path / "validation.jsonl"
    rows = [json.loads(line) for line in original_manifest.read_text().splitlines()]
    rows.extend(json.loads(line) for line in deletion_manifest.read_text().splitlines())
    write_jsonl(combined, rows)
    artifact = tmp_path / "finding_value_hs32.pt"
    torch.save(
        {
            "layer": 32,
            "feature_mean": torch.zeros((1, 4)),
            "feature_std": torch.ones((1, 4)),
            "finding_labels": ["A", "B", "C"],
            "finding_state_dict": {
                "weight": torch.eye(3, 4),
                "bias": torch.zeros(3),
            },
        },
        artifact,
    )
    output = tmp_path / "validation_scores.jsonl"
    report = build_validation_scores(
        manifest=combined,
        probe_artifact=artifact,
        output_jsonl=output,
        output_json=tmp_path / "validation_report.json",
        summary_md=tmp_path / "validation_summary.md",
        path_maps=[],
        max_donors=5,
        device=torch.device("cpu"),
    )
    scored = [json.loads(line) for line in output.read_text().splitlines()]
    positives = [row for row in scored if row["control_type"] == "selected_changed_cue"]
    nulls = [row for row in scored if row["control_type"] == "cue_absent_null"]
    assert len(positives) == len(originals) == 8
    assert len(nulls) == len(originals) == 8
    assert all(row["selected_changed_cue_supported"] is None for row in scored)
    assert all(row["score_eligible"] for row in scored)
    assert report["threshold_applied"] is False
    assert report["target_written"] is False
    assert report["locked_test_read"] is False
