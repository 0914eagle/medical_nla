import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_ddxplus_d10_specificity import summarize
from scripts.make_ddxplus_d10_validation_pairs import build_validation_pairs
from scripts.make_ddxplus_d9a_supported_pairs import sha256_file
from scripts.train_ddxplus_d10_1x2 import (
    epoch_rows,
    one_by_two_objective,
    paired_variants,
)
from src.jsonl import write_jsonl


def pair(identifier: str) -> dict:
    return {
        "id": identifier,
        "base_id": identifier,
        "target_text": "<explanation>\n- cue\n</explanation>",
        "original_activation_path": f"/{identifier}_original.pt",
        "deleted_activation_path": f"/{identifier}_deleted.pt",
    }


def test_one_by_two_objective_rewards_deleted_nll_above_original() -> None:
    good_loss, good_gap = one_by_two_objective(
        torch.tensor([1.0]), torch.tensor([2.0]), temperature=1.0
    )
    bad_loss, bad_gap = one_by_two_objective(
        torch.tensor([2.0]), torch.tensor([1.0]), temperature=1.0
    )
    assert good_gap.item() > 0
    assert bad_gap.item() < 0
    assert good_loss.item() < bad_loss.item()


def test_pair_variants_change_only_activation_state() -> None:
    row = pair("x")
    original, deleted = paired_variants(row)
    assert original["target_text"] == deleted["target_text"]
    assert original["activation_path"] == "/x_original.pt"
    assert deleted["activation_path"] == "/x_deleted.pt"


def test_pair_order_is_deterministic_per_seed_and_epoch() -> None:
    rows = [pair(str(index)) for index in range(10)]
    first = [row["base_id"] for row in epoch_rows(rows, seed=17, epoch=1)]
    repeated = [row["base_id"] for row in epoch_rows(rows, seed=17, epoch=1)]
    other = [row["base_id"] for row in epoch_rows(rows, seed=29, epoch=1)]
    assert first == repeated
    assert first != other


def test_specificity_summary_subtracts_retained_gap() -> None:
    scores = []
    for identifier, diagnosis, changed, retained in (
        ("a", "dx1", (1.0, 2.0), (1.0, 1.1)),
        ("b", "dx2", (2.0, 2.5), (1.5, 1.4)),
    ):
        for condition, value in (
            ("changed_original", changed[0]),
            ("changed_deleted", changed[1]),
            ("retained_original", retained[0]),
            ("retained_deleted", retained[1]),
        ):
            scores.append(
                {
                    "base_id": identifier,
                    "diagnosis_id": diagnosis,
                    "condition": condition,
                    "content_nll": value,
                }
            )
    report = summarize(scores, expected_rows=2, seed=17)
    assert report["metrics"]["changed_gap"]["mean"] == pytest.approx(0.75)
    assert report["metrics"]["retained_gap"]["mean"] == pytest.approx(0.0)
    assert report["metrics"]["specificity"]["mean"] == pytest.approx(0.75)


def test_validation_builder_uses_positive_rows_and_frozen_cut(tmp_path: Path) -> None:
    validation_scores = tmp_path / "scores.jsonl"
    write_jsonl(
        validation_scores,
        [
            {
                "control_type": "selected_changed_cue",
                "base_id": "case",
                "changed_evidence_id": "A",
                "score_eligible": True,
                "donor_count": 1,
                "p_original": 0.9,
                "deletion_delta": 0.2,
                "donor_margin": 0.3,
            },
            {
                "control_type": "cue_absent_null",
                "base_id": "other",
                "score_eligible": True,
                "donor_count": 1,
                "p_original": 0.9,
                "deletion_delta": 0.2,
                "donor_margin": 0.3,
            },
        ],
    )
    approval = tmp_path / "approved.json"
    approval.write_text(
        json.dumps(
            {
                "human_approved": True,
                "approved_by": "fixture",
                "approved_at": "2026-08-29T00:00:00+09:00",
                "max_false_support_rate": 0.05,
                "min_fold_positive_count": 5,
                "max_donors": 5,
                "validation_scores_sha256": sha256_file(validation_scores),
                "selected": {
                    "presence_threshold": 0.9,
                    "deletion_delta_threshold": 0.0,
                    "donor_margin_threshold": 0.0,
                    "false_support_rate": 0.0,
                    "meets_false_support_cap": True,
                },
            }
        ),
        encoding="utf-8",
    )
    original_pt = tmp_path / "original.pt"
    deleted_pt = tmp_path / "deleted.pt"
    original_pt.touch()
    deleted_pt.touch()
    original = {
        "id": "case__original",
        "base_id": "case",
        "variant": "original",
        "diagnosis_id": "dx",
        "position_family": "P0",
        "activation_path": str(original_pt),
        "cue_evidence_ids": ["A", "B"],
        "cue_targets": ["changed", "retained"],
    }
    deleted = {
        **original,
        "id": "case__deleted",
        "variant": "cue_deleted",
        "activation_path": str(deleted_pt),
        "cue_evidence_ids": ["B"],
        "cue_targets": ["retained"],
        "cf_original_evidence_id": "A",
        "cf_original_cue": "changed",
    }
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [original, deleted])
    report = build_validation_pairs(
        validation_scores=validation_scores,
        validation_manifest=manifest,
        approved_protocol=approval,
        output_jsonl=tmp_path / "pairs.jsonl",
        report_json=tmp_path / "report.json",
        summary_md=tmp_path / "summary.md",
    )
    rows = [json.loads(line) for line in (tmp_path / "pairs.jsonl").read_text().splitlines()]
    assert report["counts"]["supported_pairs"] == 1
    assert len(rows) == 1
    assert rows[0]["retained_cue_text"] == "retained"
