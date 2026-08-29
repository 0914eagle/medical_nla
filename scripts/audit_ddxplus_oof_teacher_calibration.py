"""Audit OOF-teacher calibration before any student target is built.

The audit joins the official-train original/deletion manifests to an existing
full-label OOF teacher. It measures input-cue agreement only as a calibration
diagnostic, not as ground truth for an activation. It also applies the frozen
full-data probe to the same train activations to isolate cross-fit calibration
shift. No threshold is selected and no student artifact is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.score_ddxplus_selected_changed_cues import (
    load_deletion_rows,
    load_original_rows,
    load_vector,
    parse_path_maps,
    quantiles,
)
from scripts.train_ddxplus_finding_value_probes import base_id, predict_linear
from src.jsonl import read_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def set_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def selected_from_probabilities(
    probabilities: torch.Tensor, labels: list[str], threshold: float
) -> list[set[str]]:
    return [
        {
            label
            for label, probability in zip(labels, row.tolist(), strict=True)
            if float(probability) >= threshold
        }
        for row in probabilities
    ]


def cue_set(row: dict[str, Any], label_set: set[str]) -> set[str]:
    return set(map(str, row.get("cue_evidence_ids") or [])) & label_set


def binary_metrics(
    predictions: list[set[str]],
    targets: list[set[str]],
    probabilities: torch.Tensor,
    labels: list[str],
) -> dict[str, Any]:
    tp = sum(len(prediction & target) for prediction, target in zip(predictions, targets, strict=True))
    fp = sum(len(prediction - target) for prediction, target in zip(predictions, targets, strict=True))
    fn = sum(len(target - prediction) for prediction, target in zip(predictions, targets, strict=True))
    precision = ratio(tp, tp + fp)
    recall = ratio(tp, tp + fn)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    label_index = {label: index for index, label in enumerate(labels)}
    gold = torch.zeros_like(probabilities)
    for row_index, target in enumerate(targets):
        for label in target:
            gold[row_index, label_index[label]] = 1.0
    clipped = probabilities.clamp(1e-7, 1 - 1e-7)
    bce = -(gold * clipped.log() + (1 - gold) * (1 - clipped).log()).mean()
    return {
        "rows": len(predictions),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "micro_f1": f1,
        "brier": float(((probabilities - gold) ** 2).mean()),
        "binary_cross_entropy": float(bce),
        "mean_selected": float(np.mean([len(value) for value in predictions])),
        "mean_input_cues_in_ontology": float(np.mean([len(value) for value in targets])),
    }


def load_teacher(
    path: Path, labels: list[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    original_ids = set()
    deleted_ids = set()
    for row in read_jsonl(path):
        identifier = str(row.get("base_id") or "")
        variant = str(row.get("variant") or "")
        if not identifier or variant not in {"original", "cue_deleted"}:
            raise ValueError(f"Invalid teacher row: {identifier!r}/{variant!r}")
        if variant in by_base[identifier]:
            raise ValueError(f"Duplicate teacher row: {identifier}/{variant}")
        probabilities = list(row.get("finding_probabilities") or [])
        if len(probabilities) != len(labels):
            raise ValueError(
                f"Teacher vector mismatch for {identifier}/{variant}: "
                f"{len(probabilities)} vs {len(labels)}"
            )
        selected = list(map(str, row.get("selected_evidence_ids") or []))
        if selected != sorted(selected) or len(selected) != len(set(selected)):
            raise ValueError(f"Noncanonical teacher set: {identifier}/{variant}")
        by_base[identifier][variant] = row
        if variant == "original":
            original_ids.add(identifier)
        else:
            deleted_ids.add(identifier)
    if original_ids != deleted_ids:
        raise ValueError("Teacher original/deleted population mismatch")
    return by_base


def intervention_metrics(
    originals: list[dict[str, Any]],
    deletions: dict[str, dict[str, Any]],
    original_sets: list[set[str]],
    deleted_sets: list[set[str]],
    label_set: set[str],
) -> dict[str, Any]:
    eligible = original_hit = phantom = removed = 0
    untouched_total = untouched_hit = untouched_preserved = 0
    for index, original in enumerate(originals):
        deletion = deletions[base_id(original)]
        changed = str(deletion.get("cf_original_evidence_id") or "")
        if changed in label_set:
            eligible += 1
            before = changed in original_sets[index]
            after = changed in deleted_sets[index]
            original_hit += before
            phantom += after
            removed += before and not after
        common = cue_set(original, label_set) & cue_set(deletion, label_set)
        untouched_total += len(common)
        for evidence in common:
            before = evidence in original_sets[index]
            untouched_hit += before
            untouched_preserved += before and evidence in deleted_sets[index]
    return {
        "changed_eligible": eligible,
        "changed_original_hit": ratio(original_hit, eligible),
        "changed_deleted_phantom": ratio(phantom, eligible),
        "changed_removal_given_original_hit": ratio(removed, original_hit),
        "untouched_eligible": untouched_total,
        "untouched_original_hit": ratio(untouched_hit, untouched_total),
        "untouched_preservation_given_original_hit": ratio(
            untouched_preserved, untouched_hit
        ),
    }


def transition_metrics(
    originals: list[dict[str, Any]],
    deletions: dict[str, dict[str, Any]],
    original_sets: list[set[str]],
    deleted_sets: list[set[str]],
    deleted_probabilities: torch.Tensor,
    labels: list[str],
    threshold: float,
) -> dict[str, Any]:
    label_index = {label: index for index, label in enumerate(labels)}
    retained_counts = []
    removed_counts = []
    added_counts = []
    net_counts = []
    jaccards = []
    added_margins = []
    added_total = added_in_deleted_input = 0
    for index, original in enumerate(originals):
        identifier = base_id(original)
        original_set = original_sets[index]
        deleted_set = deleted_sets[index]
        added = deleted_set - original_set
        retained_counts.append(float(len(original_set & deleted_set)))
        removed_counts.append(float(len(original_set - deleted_set)))
        added_counts.append(float(len(added)))
        net_counts.append(float(len(deleted_set) - len(original_set)))
        jaccards.append(set_jaccard(original_set, deleted_set))
        deleted_inputs = cue_set(deletions[identifier], set(labels))
        for label in added:
            added_total += 1
            added_in_deleted_input += label in deleted_inputs
            added_margins.append(
                float(deleted_probabilities[index, label_index[label]]) - threshold
            )
    return {
        "retained_count": quantiles(retained_counts),
        "removed_count": quantiles(removed_counts),
        "added_count": quantiles(added_counts),
        "net_selected_count": quantiles(net_counts),
        "set_jaccard": quantiles(jaccards),
        "added_probability_margin_above_threshold": quantiles(added_margins),
        "added_labels": added_total,
        "added_labels_present_in_deleted_input": added_in_deleted_input,
        "added_labels_absent_from_deleted_input": added_total - added_in_deleted_input,
        "added_absent_rate": ratio(
            added_total - added_in_deleted_input, added_total
        ),
    }


def prevalence_rows(
    labels: list[str],
    original_sets: list[set[str]],
    deleted_sets: list[set[str]],
) -> list[dict[str, Any]]:
    n = len(original_sets)
    result = []
    for label in labels:
        original_n = sum(label in value for value in original_sets)
        deleted_n = sum(label in value for value in deleted_sets)
        result.append(
            {
                "evidence_id": label,
                "original_selected": original_n,
                "deleted_selected": deleted_n,
                "original_rate": original_n / n,
                "deleted_rate": deleted_n / n,
                "deleted_minus_original": (deleted_n - original_n) / n,
            }
        )
    return result


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def qline(value: dict[str, float] | None) -> str:
    if value is None:
        return "N/A"
    return (
        f"mean {value['mean']:.4f}; q05 {value['q05']:.4f}; "
        f"median {value['median']:.4f}; q95 {value['q95']:.4f}"
    )


def calibration_gate(
    oof: dict[str, Any],
    full: dict[str, Any],
    original_jaccard: dict[str, float],
    fold_audits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Apply the preregistered one-shot K=5 calibration criteria."""

    criteria = {
        "original_precision_ge_0p90": oof["original"]["precision"] >= 0.90,
        "original_recall_ge_0p98": oof["original"]["recall"] >= 0.98,
        "original_mean_selected_relative_gap_le_0p10": (
            abs(oof["original"]["mean_selected"] - full["original"]["mean_selected"])
            / full["original"]["mean_selected"]
            <= 0.10
        ),
        "oof_full_original_jaccard_mean_ge_0p90": original_jaccard["mean"]
        >= 0.90,
        "deleted_mean_selected_relative_gap_le_0p10": (
            abs(
                oof["cue_deleted"]["mean_selected"]
                - full["cue_deleted"]["mean_selected"]
            )
            / full["cue_deleted"]["mean_selected"]
            <= 0.10
        ),
        "deleted_phantom_absolute_gap_le_0p05": (
            abs(
                oof["intervention"]["changed_deleted_phantom"]
                - full["intervention"]["changed_deleted_phantom"]
            )
            <= 0.05
        ),
        "every_fold_original_precision_ge_0p85": all(
            item["original"]["precision"] >= 0.85 for item in fold_audits
        ),
    }
    return {
        "scope": "one-shot K=5 calibration gate approved before K=5 execution",
        "criteria": criteria,
        "passed": all(criteria.values()),
        "failure_policy": "no K/threshold sweep; stop hard-set target building",
    }


def write_summary(path: Path, report: dict[str, Any]) -> None:
    oof = report["oof_teacher"]
    frozen = report["full_data_frozen_probe"]
    transition = report["oof_transition"]
    top_increases = sorted(
        report["label_prevalence"],
        key=lambda row: (-row["deleted_minus_original"], row["evidence_id"]),
    )[:15]
    lines = [
        "# DDXPlus OOF Teacher Calibration Audit",
        "",
        "Official-train-only, read-only audit. Input-cue agreement is a calibration "
        "diagnostic, not ground truth for activation content.",
        "",
        f"- base cases: **{report['base_cases']}**",
        f"- finding labels: **{report['finding_labels']}**",
        f"- OOF folds: **{report['num_folds']}**",
        f"- inherited threshold: **{report['threshold']:.4f}**",
        "- threshold selected or changed: **no**",
        "- student target written: **no**",
        "- validation / locked test read: **no / no**",
        "",
        "## OOF Versus Full-Data Frozen Probe",
        "",
        "| reader | arm | mean selected | cue precision | cue recall | cue F1 | Brier | BCE |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for reader, values in (("OOF teacher", oof), ("full-data frozen", frozen)):
        for arm in ("original", "cue_deleted"):
            item = values[arm]
            lines.append(
                f"| {reader} | {arm} | {item['mean_selected']:.4f} | "
                f"{fmt(item['precision'])} | {fmt(item['recall'])} | "
                f"{fmt(item['micro_f1'])} | {item['brier']:.4f} | "
                f"{item['binary_cross_entropy']:.4f} |"
            )
    lines.extend(
        [
            "",
            f"- OOF/full original-set Jaccard: "
            f"{qline(report['oof_full_original_set_jaccard'])}",
            "",
            "## OOF Original-to-Deletion Transitions",
            "",
            f"- retained labels: {qline(transition['retained_count'])}",
            f"- removed labels: {qline(transition['removed_count'])}",
            f"- newly added labels: {qline(transition['added_count'])}",
            f"- net selected-count change: {qline(transition['net_selected_count'])}",
            f"- set Jaccard: {qline(transition['set_jaccard'])}",
            "- newly-added probability margin above threshold: "
            f"{qline(transition['added_probability_margin_above_threshold'])}",
            f"- newly-added labels absent from deleted input: "
            f"**{transition['added_labels_absent_from_deleted_input']}/"
            f"{transition['added_labels']} "
            f"({fmt(transition['added_absent_rate'])})**",
            "",
            "## Intervention Comparison",
            "",
            "| reader | changed hit | deleted phantom | removal | untouched preservation |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for reader, values in (
        ("OOF teacher", oof["intervention"]),
        ("full-data frozen", frozen["intervention"]),
    ):
        lines.append(
            f"| {reader} | {fmt(values['changed_original_hit'])} | "
            f"{fmt(values['changed_deleted_phantom'])} | "
            f"{fmt(values['changed_removal_given_original_hit'])} | "
            f"{fmt(values['untouched_preservation_given_original_hit'])} |"
        )
    lines.extend(
        [
            "",
            "## OOF Fold Diagnostics",
            "",
            "| fold | arm | rows | mean selected | precision | recall | F1 | Brier | BCE |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in report["fold_audits"]:
        for arm in ("original", "cue_deleted"):
            item = fold[arm]
            lines.append(
                f"| {fold['fold']} | {arm} | {item['rows']} | "
                f"{item['mean_selected']:.4f} | {fmt(item['precision'])} | "
                f"{fmt(item['recall'])} | {fmt(item['micro_f1'])} | "
                f"{item['brier']:.4f} | {item['binary_cross_entropy']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Largest Label Prevalence Increases After Deletion",
            "",
            "| evidence ID | original | deleted | delta |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in top_increases:
        lines.append(
            f"| {row['evidence_id']} | {row['original_rate']:.4f} | "
            f"{row['deleted_rate']:.4f} | {row['deleted_minus_original']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Preregistered K=5 Calibration Gate",
            "",
            "| criterion | pass |",
            "|---|---:|",
        ]
    )
    for criterion, passed in report["calibration_gate"]["criteria"].items():
        lines.append(f"| `{criterion}` | {'PASS' if passed else 'FAIL'} |")
    lines.extend(
        [
            "",
            f"- overall calibration gate: "
            f"**{'PASS' if report['calibration_gate']['passed'] else 'FAIL'}**",
            "- failure policy: no additional K or threshold sweep",
            "",
            "This report does not authorize target building. A large OOF/full-data gap or "
            "broad absent-label additions indicates cross-fit calibration shift or "
            "counterfactual OOD behavior that must be resolved before P2-P4 approval.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_audit(
    *,
    teacher_jsonl: Path,
    teacher_report: Path,
    original_manifest: Path,
    counterfactual_manifest: Path,
    probe_artifact: Path,
    output_json: Path,
    label_prevalence_jsonl: Path,
    summary_md: Path,
    path_maps: list[tuple[str, str]],
    device: torch.device,
) -> dict[str, Any]:
    materialization = json.loads(teacher_report.read_text(encoding="utf-8"))
    labels = [str(value) for value in materialization["finding_labels"]]
    threshold = float(materialization["finding_threshold"])
    num_folds = int(materialization.get("num_folds") or 2)
    if materialization.get("validation_read") or materialization.get("locked_test_read"):
        raise ValueError("Teacher materialization is not train-only")
    if sha256_file(teacher_jsonl) != materialization["teacher_scores_sha256"]:
        raise ValueError("Teacher JSONL hash does not match materialization report")

    originals = load_original_rows(original_manifest)
    deletions = load_deletion_rows(counterfactual_manifest)
    ordered_ids = [base_id(row) for row in originals]
    if set(ordered_ids) != set(deletions):
        raise ValueError("Manifest original/deletion population mismatch")
    teacher = load_teacher(teacher_jsonl, labels)
    if set(teacher) != set(ordered_ids):
        raise ValueError("Teacher and manifest population mismatch")

    oof_original_probabilities = torch.tensor(
        [teacher[key]["original"]["finding_probabilities"] for key in ordered_ids],
        dtype=torch.float32,
    )
    oof_deleted_probabilities = torch.tensor(
        [teacher[key]["cue_deleted"]["finding_probabilities"] for key in ordered_ids],
        dtype=torch.float32,
    )
    oof_original_sets = [
        set(map(str, teacher[key]["original"]["selected_evidence_ids"]))
        for key in ordered_ids
    ]
    oof_deleted_sets = [
        set(map(str, teacher[key]["cue_deleted"]["selected_evidence_ids"]))
        for key in ordered_ids
    ]
    label_set = set(labels)
    original_targets = [cue_set(row, label_set) for row in originals]
    deleted_targets = [cue_set(deletions[key], label_set) for key in ordered_ids]

    artifact = torch.load(probe_artifact, map_location="cpu", weights_only=False)
    artifact_labels = [str(value) for value in artifact["finding_labels"]]
    if artifact_labels != labels or float(artifact["finding_threshold"]) != threshold:
        raise ValueError("Frozen artifact labels/threshold differ from teacher protocol")
    original_features = torch.stack([load_vector(row, path_maps) for row in originals])
    deleted_features = torch.stack(
        [load_vector(deletions[key], path_maps) for key in ordered_ids]
    )
    mean = artifact["feature_mean"].float()
    std = artifact["feature_std"].float()
    full_original_probabilities = predict_linear(
        artifact["finding_state_dict"], (original_features - mean) / std, device
    ).sigmoid()
    full_deleted_probabilities = predict_linear(
        artifact["finding_state_dict"], (deleted_features - mean) / std, device
    ).sigmoid()
    full_original_sets = selected_from_probabilities(
        full_original_probabilities, labels, threshold
    )
    full_deleted_sets = selected_from_probabilities(
        full_deleted_probabilities, labels, threshold
    )

    oof = {
        "original": binary_metrics(
            oof_original_sets, original_targets, oof_original_probabilities, labels
        ),
        "cue_deleted": binary_metrics(
            oof_deleted_sets, deleted_targets, oof_deleted_probabilities, labels
        ),
        "intervention": intervention_metrics(
            originals,
            deletions,
            oof_original_sets,
            oof_deleted_sets,
            label_set,
        ),
    }
    full = {
        "original": binary_metrics(
            full_original_sets, original_targets, full_original_probabilities, labels
        ),
        "cue_deleted": binary_metrics(
            full_deleted_sets, deleted_targets, full_deleted_probabilities, labels
        ),
        "intervention": intervention_metrics(
            originals,
            deletions,
            full_original_sets,
            full_deleted_sets,
            label_set,
        ),
    }
    transitions = transition_metrics(
        originals,
        deletions,
        oof_original_sets,
        oof_deleted_sets,
        oof_deleted_probabilities,
        labels,
        threshold,
    )
    prevalence = prevalence_rows(labels, oof_original_sets, oof_deleted_sets)
    write_jsonl(label_prevalence_jsonl, prevalence)

    folds = [int(teacher[key]["original"]["fold"]) for key in ordered_ids]
    if set(folds) != set(range(num_folds)):
        raise ValueError(
            f"Teacher fold IDs do not match num_folds={num_folds}: {sorted(set(folds))}"
        )
    fold_audits = []
    for fold in range(num_folds):
        indices = [index for index, value in enumerate(folds) if value == fold]
        fold_audits.append(
            {
                "fold": fold,
                "original": binary_metrics(
                    [oof_original_sets[index] for index in indices],
                    [original_targets[index] for index in indices],
                    oof_original_probabilities[indices],
                    labels,
                ),
                "cue_deleted": binary_metrics(
                    [oof_deleted_sets[index] for index in indices],
                    [deleted_targets[index] for index in indices],
                    oof_deleted_probabilities[indices],
                    labels,
                ),
            }
        )

    original_jaccard = quantiles(
        [
            set_jaccard(oof_original_sets[index], full_original_sets[index])
            for index in range(len(originals))
        ]
    )
    assert original_jaccard is not None
    gate = calibration_gate(oof, full, original_jaccard, fold_audits)
    report = {
        "schema_version": 1,
        "method": "ddxplus_oof_teacher_calibration_audit",
        "base_cases": len(originals),
        "finding_labels": len(labels),
        "num_folds": num_folds,
        "threshold": threshold,
        "oof_teacher": oof,
        "full_data_frozen_probe": full,
        "oof_transition": transitions,
        "oof_full_original_set_jaccard": original_jaccard,
        "fold_audits": fold_audits,
        "calibration_gate": gate,
        "label_prevalence": prevalence,
        "inputs": {
            "teacher_jsonl": str(teacher_jsonl),
            "teacher_jsonl_sha256": sha256_file(teacher_jsonl),
            "teacher_report": str(teacher_report),
            "teacher_report_sha256": sha256_file(teacher_report),
            "original_manifest": str(original_manifest),
            "original_manifest_sha256": sha256_file(original_manifest),
            "counterfactual_manifest": str(counterfactual_manifest),
            "counterfactual_manifest_sha256": sha256_file(counterfactual_manifest),
            "probe_artifact": str(probe_artifact),
            "probe_artifact_sha256": sha256_file(probe_artifact),
        },
        "input_cue_agreement_is_activation_ground_truth": False,
        "threshold_selected_or_changed": False,
        "student_target_written": False,
        "student_gate_frozen": False,
        "validation_read": False,
        "locked_test_read": False,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(summary_md, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-jsonl", required=True, type=Path)
    parser.add_argument("--teacher-report", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--counterfactual-manifest", required=True, type=Path)
    parser.add_argument("--probe-artifact", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--label-prevalence-jsonl", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    report = run_audit(
        teacher_jsonl=args.teacher_jsonl,
        teacher_report=args.teacher_report,
        original_manifest=args.original_manifest,
        counterfactual_manifest=args.counterfactual_manifest,
        probe_artifact=args.probe_artifact,
        output_json=args.output_json,
        label_prevalence_jsonl=args.label_prevalence_jsonl,
        summary_md=args.summary_md,
        path_maps=parse_path_maps(args.path_map),
        device=torch.device(args.device),
    )
    print(
        f"[audit] bases={report['base_cases']} labels={report['finding_labels']}",
        flush=True,
    )
    print(f"[summary] {args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
