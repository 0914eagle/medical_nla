import json
from pathlib import Path

import pytest

from scripts.make_ddxplus_d9a_supported_pairs import (
    build_pairs,
    load_approved_protocol,
    select_retained_cue,
)
from scripts.select_ddxplus_d9a_support_thresholds import evaluate_grid
from src.jsonl import write_jsonl


def validation_rows() -> list[dict]:
    rows = []
    for index, (presence, delta, donor) in enumerate(
        [(0.9, 0.4, 0.5), (0.8, 0.2, 0.4), (0.6, 0.1, 0.2)]
    ):
        rows.append(
            {
                "control_type": "selected_changed_cue",
                "base_id": f"positive_{index}",
                "score_eligible": True,
                "p_original": presence,
                "deletion_delta": delta,
                "donor_margin": donor,
            }
        )
    for index, (presence, delta, donor) in enumerate(
        [(0.3, 0.0, 0.1), (0.4, -0.1, 0.0), (0.7, 0.05, 0.1)]
    ):
        rows.append(
            {
                "control_type": "cue_absent_null",
                "base_id": f"null_{index}",
                "score_eligible": True,
                "p_original": presence,
                "deletion_delta": delta,
                "donor_margin": donor,
            }
        )
    return rows


def test_cut_selection_uses_false_support_cap_then_coverage() -> None:
    candidates, selected = evaluate_grid(
        validation_rows(),
        presence_thresholds=[0.5, 0.75],
        deletion_thresholds=[0.1, 0.2],
        donor_thresholds=[0.2, 0.4],
        max_false_support_rate=0.05,
    )
    assert len(candidates) == 8
    assert selected["false_support_rate"] == 0.0
    assert selected["positive_coverage"] == 1.0
    assert selected["presence_threshold"] == 0.5
    assert selected["deletion_delta_threshold"] == 0.1
    assert selected["donor_margin_threshold"] == 0.2


def approved_protocol(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "human_approved": True,
                "approved_by": "fixture-reviewer",
                "approved_at": "2026-08-29T00:00:00+09:00",
                "max_false_support_rate": 0.05,
                "min_fold_positive_count": 5,
                "max_donors": 5,
                "selected": {
                    "presence_threshold": 0.5,
                    "deletion_delta_threshold": 0.1,
                    "donor_margin_threshold": 0.2,
                    "false_support_rate": 0.0,
                    "meets_false_support_cap": True,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def original(identifier: str, activation: Path) -> dict:
    return {
        "id": f"{identifier}__cot_p0",
        "base_id": identifier,
        "variant": "original",
        "official_split": "train",
        "diagnosis_id": "dx",
        "position_family": "P0",
        "layer": 32,
        "activation_path": str(activation),
        "cue_targets": ["finding A", "finding B"],
        "cue_evidence_ids": ["A", "B"],
    }


def deletion(row: dict, activation: Path) -> dict:
    return {
        **row,
        "id": f"{row['base_id']}__cue_deleted__cot_p0",
        "variant": "cue_deleted",
        "activation_path": str(activation),
        "cue_targets": ["finding B"],
        "cue_evidence_ids": ["B"],
        "cf_original_evidence_id": "A",
        "cf_original_cue": "finding A",
    }


def test_target_builder_requires_approval_and_excludes_below_cut(tmp_path: Path) -> None:
    approval = tmp_path / "approved.json"
    validation_scores = tmp_path / "validation_scores.jsonl"
    validation_scores.write_text("{}\n", encoding="utf-8")
    approved_protocol(approval)
    payload = json.loads(approval.read_text())
    from scripts.make_ddxplus_d9a_supported_pairs import sha256_file

    payload["validation_scores_sha256"] = sha256_file(validation_scores)
    approval.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    original_manifest = tmp_path / "original.jsonl"
    deletion_manifest = tmp_path / "deletion.jsonl"
    train_scores = tmp_path / "scores.jsonl"
    originals = []
    deletions = []
    scores = []
    for index in range(2):
        identifier = f"case_{index}"
        own = original(identifier, tmp_path / f"{identifier}_original.pt")
        Path(own["activation_path"]).touch()
        originals.append(own)
        removed = deletion(own, tmp_path / f"{identifier}_deleted.pt")
        Path(removed["activation_path"]).touch()
        deletions.append(removed)
        scores.append(
            {
                "base_id": identifier,
                "changed_evidence_id": "A",
                "score_eligible": True,
                "fold_training_positive_count": 5,
                "donor_count": 1,
                "p_original": 0.9,
                "p_deleted": 0.5,
                "deletion_delta": 0.4,
                "donor_margin": 0.5 if index == 0 else 0.1,
            }
        )
    write_jsonl(original_manifest, originals)
    write_jsonl(deletion_manifest, deletions)
    write_jsonl(train_scores, scores)

    report = build_pairs(
        train_scores=train_scores,
        validation_scores=validation_scores,
        original_manifest=original_manifest,
        counterfactual_manifest=deletion_manifest,
        approved_protocol=approval,
        output_jsonl=tmp_path / "pairs.jsonl",
        protocol_json=tmp_path / "protocol.json",
        summary_md=tmp_path / "summary.md",
    )
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert report["counts"]["supported_pairs"] == 1
    assert report["counts"]["eligible_below_cut_excluded"] == 1
    assert len(rows) == 1
    assert rows[0]["selected_changed_cue_supported"] is True
    assert rows[0]["all_other_cues_support_status"] == "untested"
    assert rows[0]["abstention_target"] is False
    assert rows[0]["value_edit_included"] is False
    assert "finding A" in rows[0]["target_text"]
    assert "finding B" not in rows[0]["target_text"]
    assert rows[0]["retained_evidence_id"] == "B"
    assert rows[0]["retained_cue_text"] == "finding B"
    assert "finding B" in rows[0]["retained_target_text"]
    assert "finding A" not in rows[0]["retained_target_text"]


def test_retained_cue_uses_frozen_hash_and_requires_exact_common_text() -> None:
    own = {
        "base_id": "case",
        "cue_evidence_ids": ["A", "B", "C"],
        "cue_targets": ["changed", "stable B", "stable C"],
    }
    removed = {
        "base_id": "case",
        "cf_original_evidence_id": "A",
        "cue_evidence_ids": ["B", "C"],
        "cue_targets": ["stable B", "changed rendering"],
    }
    evidence_id, cue, digest = select_retained_cue(
        identifier="case", original=own, deleted=removed
    )
    assert evidence_id == "B"
    assert cue == "stable B"
    assert len(digest) == 64


def test_target_builder_excludes_donor_unavailable(tmp_path: Path) -> None:
    approval = tmp_path / "approved.json"
    validation_scores = tmp_path / "validation_scores.jsonl"
    validation_scores.write_text("{}\n", encoding="utf-8")
    approved_protocol(approval)
    payload = json.loads(approval.read_text())
    from scripts.make_ddxplus_d9a_supported_pairs import sha256_file

    payload["validation_scores_sha256"] = sha256_file(validation_scores)
    approval.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    original_path = tmp_path / "original.pt"
    deleted_path = tmp_path / "deleted.pt"
    original_path.touch()
    deleted_path.touch()
    own = original("case_0", original_path)
    removed = deletion(own, deleted_path)
    original_manifest = tmp_path / "original.jsonl"
    deletion_manifest = tmp_path / "deletion.jsonl"
    train_scores = tmp_path / "scores.jsonl"
    write_jsonl(original_manifest, [own])
    write_jsonl(deletion_manifest, [removed])
    write_jsonl(
        train_scores,
        [
            {
                "base_id": "case_0",
                "changed_evidence_id": "A",
                "score_eligible": True,
                "fold_training_positive_count": 5,
                "donor_count": 0,
                "p_original": 0.9,
                "p_deleted": 0.5,
                "deletion_delta": 0.4,
                "donor_margin": None,
            }
        ],
    )
    with pytest.raises(ValueError, match="retained no D9a pairs"):
        build_pairs(
            train_scores=train_scores,
            validation_scores=validation_scores,
            original_manifest=original_manifest,
            counterfactual_manifest=deletion_manifest,
            approved_protocol=approval,
            output_jsonl=tmp_path / "pairs.jsonl",
            protocol_json=tmp_path / "protocol.json",
            summary_md=tmp_path / "summary.md",
        )


def test_unapproved_protocol_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "unapproved.json"
    path.write_text(json.dumps({"human_approved": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="not human-approved"):
        load_approved_protocol(path)
