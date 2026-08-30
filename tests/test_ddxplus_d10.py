import json
from pathlib import Path

import pytest
import torch

from scripts.evaluate_ddxplus_d10_specificity import summarize
from scripts.make_ddxplus_d10_validation_pairs import build_validation_pairs
from scripts.make_ddxplus_d9a_supported_pairs import sha256_file
from scripts.train_ddxplus_d10_1x2 import (
    advance_cursor,
    epoch_rows,
    normalize_checkpoint_steps,
    one_by_two_objective,
    paired_variants,
    prepare_metrics_for_resume,
    retained_variants,
    specificity_anchored_objective,
)
from scripts.audit_ddxplus_d20_control_spread import readout_claim_mean, recommendation
from scripts.summarize_ddxplus_d20_arms import build_report as build_d20_report
from scripts.summarize_ddxplus_d10_budget_trajectory import (
    FROZEN_STEPS,
    build_report,
)
from src.jsonl import write_jsonl


def pair(identifier: str) -> dict:
    return {
        "id": identifier,
        "base_id": identifier,
        "target_text": "<explanation>\n- cue\n</explanation>",
        "retained_target_text": "<explanation>\n- retained\n</explanation>",
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


def test_retained_variants_replace_target_but_preserve_activation_pair() -> None:
    original, deleted = retained_variants(pair("x"))
    assert original["target_text"] == "<explanation>\n- retained\n</explanation>"
    assert deleted["target_text"] == original["target_text"]
    assert original["activation_path"] == "/x_original.pt"
    assert deleted["activation_path"] == "/x_deleted.pt"


def test_specificity_anchor_rejects_high_high_retained_nll_shortcut() -> None:
    low, _ = specificity_anchored_objective(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([1.0]),
        torch.tensor([1.0]),
        anchor_weight=1.0,
        ranking_weight=1.0,
        temperature=1.0,
        margin=0.0,
    )
    high, _ = specificity_anchored_objective(
        torch.tensor([1.0]),
        torch.tensor([2.0]),
        torch.tensor([5.0]),
        torch.tensor([5.0]),
        anchor_weight=1.0,
        ranking_weight=1.0,
        temperature=1.0,
        margin=0.0,
    )
    assert high.item() > low.item()


def test_pair_order_is_deterministic_per_seed_and_epoch() -> None:
    rows = [pair(str(index)) for index in range(10)]
    first = [row["base_id"] for row in epoch_rows(rows, seed=17, epoch=1)]
    repeated = [row["base_id"] for row in epoch_rows(rows, seed=17, epoch=1)]
    other = [row["base_id"] for row in epoch_rows(rows, seed=29, epoch=1)]
    assert first == repeated
    assert first != other


def test_frozen_checkpoints_require_final_step() -> None:
    assert normalize_checkpoint_steps(
        [1552, 20, 776, 20], max_steps=1552
    ) == [20, 776, 1552]
    with pytest.raises(ValueError, match="final optimizer step"):
        normalize_checkpoint_steps([20, 776], max_steps=1552)


def test_training_cursor_advances_across_epoch_boundary() -> None:
    assert advance_cursor(epoch=1, row_index=1, n_rows=3) == (1, 2)
    assert advance_cursor(epoch=1, row_index=2, n_rows=3) == (2, 0)


def test_resume_metrics_drop_uncheckpointed_tail(tmp_path: Path) -> None:
    path = tmp_path / "metrics.jsonl"
    write_jsonl(path, [{"step": step} for step in range(1, 6)])
    prepare_metrics_for_resume(path, optimizer_step=3)
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [
        1,
        2,
        3,
    ]


def test_budget_trajectory_requires_all_frozen_steps() -> None:
    comparison = {
        "results": {
            str(seed): {
                "deltas": {
                    metric: {
                        "ranking_minus_control": 0.01,
                        "diagnosis_cluster_bootstrap_95_ci": [0.001, 0.02],
                    }
                    for metric in ("changed_gap", "retained_gap", "specificity")
                }
            }
            for seed in (17, 29, 43)
        },
        "gate": {"teacher_forced_gate_passed": False},
    }
    report = build_report({step: comparison for step in FROZEN_STEPS})
    assert len(report["trajectory"]) == len(FROZEN_STEPS) * 3
    assert report["trajectory"][-1]["step"] == 1552


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


def write_specificity_scores(
    path: Path,
    *,
    changed_original: float,
    changed_deleted: float,
    retained_original: float,
    retained_deleted: float,
) -> None:
    write_jsonl(
        path,
        [
            {
                "base_id": "case",
                "diagnosis_id": "dx",
                "condition": condition,
                "content_nll": value,
            }
            for condition, value in (
                ("changed_original", changed_original),
                ("changed_deleted", changed_deleted),
                ("retained_original", retained_original),
                ("retained_deleted", retained_deleted),
            )
        ],
    )


def test_d20_control_audit_rejects_spread_rule_and_proposes_same_seed_gates(
    tmp_path: Path,
) -> None:
    paths = {}
    for seed, retained_gap, changed_nll, retained_nll in (
        (17, 0.00, 1.00, 2.00),
        (29, 0.02, 1.01, 2.02),
        (43, 0.04, 1.10, 2.03),
    ):
        path = tmp_path / f"seed{seed}.jsonl"
        write_specificity_scores(
            path,
            changed_original=changed_nll,
            changed_deleted=changed_nll + 0.1,
            retained_original=retained_nll,
            retained_deleted=retained_nll + retained_gap,
        )
        paths[seed] = path
    report = recommendation(paths)
    rejected = report["rejected_across_seed_allowances"]
    assert rejected["retained_gap_delta_max"] == pytest.approx(0.08)
    assert report["rejected_across_seed_allowances"][
        "changed_original_nll_delta_max"
    ] == pytest.approx(0.2)
    assert report["rejected_across_seed_allowances"][
        "retained_original_nll_delta_max"
    ] == pytest.approx(0.06)
    assert report["proposed_same_seed_gates"] == {
        "retained_gap_delta_max": 0.01,
        "changed_original_nll_relative_increase_max": 0.10,
        "retained_original_nll_relative_increase_max": 0.10,
        "mean_claim_relative_drop_max": 0.10,
    }


def test_d20_claim_count_excludes_xml_scaffold(tmp_path: Path) -> None:
    path = tmp_path / "readouts.jsonl"
    write_jsonl(
        path,
        [
            {
                "nla_output": (
                    "<explanation>\n<readout>\n<observed>\n"
                    "- first finding\n- second finding\n"
                    "</observed>\n</readout>\n</explanation>"
                )
            }
        ],
    )
    n, mean_claims = readout_claim_mean(path)
    assert n == 1
    assert mean_claims == 2.0


def test_d20_gate_requires_all_seed_specificity_and_noninferiority(
    tmp_path: Path,
) -> None:
    paths = {}
    for seed in (17, 29, 43):
        control = tmp_path / f"control{seed}.jsonl"
        anchored = tmp_path / f"anchored{seed}.jsonl"
        write_specificity_scores(
            control,
            changed_original=1.0,
            changed_deleted=1.0,
            retained_original=1.0,
            retained_deleted=1.0,
        )
        write_specificity_scores(
            anchored,
            changed_original=1.01,
            changed_deleted=1.11,
            retained_original=1.01,
            retained_deleted=1.01,
        )
        paths[seed] = {"control": control, "anchored": anchored}
    protocol = {
        "human_approved": True,
        "control_score_sha256": {
            str(seed): sha256_file(paths[seed]["control"]) for seed in (17, 29, 43)
        },
        "gates": {
            "retained_gap_delta_max": 0.01,
            "changed_original_nll_relative_increase_max": 0.10,
            "retained_original_nll_relative_increase_max": 0.10,
            "mean_claim_relative_drop_max": 0.10,
        },
        "control_baselines": {
            str(seed): {
                "retained_gap": 0.0,
                "changed_original_nll": 1.0,
                "changed_original_nll_max": 1.1,
                "retained_original_nll": 1.0,
                "retained_original_nll_max": 1.1,
                "mean_claims": 1.0,
                "mean_claims_min": 0.9,
            }
            for seed in (17, 29, 43)
        },
    }
    report = build_d20_report(paths, protocol=protocol, report_only=False)
    assert report["gate"]["changed_delta_min_0p05_each_seed"]
    assert report["gate"]["specificity_positive_each_seed"]
    assert report["gate"]["retained_gap_noninferior_each_seed"]
    assert report["gate"]["teacher_forced_gate_passed"]


def test_d20_report_only_checkpoint_does_not_require_final_control_hash(
    tmp_path: Path,
) -> None:
    paths = {}
    for seed in (17, 29, 43):
        control = tmp_path / f"intermediate_control{seed}.jsonl"
        anchored = tmp_path / f"intermediate_anchored{seed}.jsonl"
        write_specificity_scores(
            control,
            changed_original=2.0,
            changed_deleted=2.0,
            retained_original=2.0,
            retained_deleted=2.0,
        )
        write_specificity_scores(
            anchored,
            changed_original=2.0,
            changed_deleted=2.1,
            retained_original=2.0,
            retained_deleted=2.0,
        )
        paths[seed] = {"control": control, "anchored": anchored}
    protocol = {
        "human_approved": True,
        "control_score_sha256": {str(seed): "final-only" for seed in (17, 29, 43)},
        "control_baselines": {
            str(seed): {
                "retained_gap": 0.0,
                "changed_original_nll": 1.0,
                "changed_original_nll_max": 1.1,
                "retained_original_nll": 1.0,
                "retained_original_nll_max": 1.1,
            }
            for seed in (17, 29, 43)
        },
        "gates": {
            "retained_gap_delta_max": 0.01,
            "changed_original_nll_relative_increase_max": 0.10,
            "retained_original_nll_relative_increase_max": 0.10,
            "mean_claim_relative_drop_max": 0.10,
        },
    }
    report = build_d20_report(paths, protocol=protocol, report_only=True)
    assert report["promotion_decision_authorized"] is False


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
