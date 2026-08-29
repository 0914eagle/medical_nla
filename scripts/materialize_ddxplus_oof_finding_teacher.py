"""Materialize the official-train HS32 full-label OOF finding teacher.

This stage is deliberately read-only with respect to student supervision. It
cross-fits the validation-selected finding-probe configuration on deterministic
official-train folds and stores one 91-label probability vector for every
original and cue-deleted activation. It does not render natural-language
targets, freeze student gates, read DDXPlus validation, or open locked test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import zlib
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
    predict_probabilities,
    quantiles,
    train_fixed_probe,
)
from scripts.train_ddxplus_finding_value_probes import (
    base_id,
    finding_targets,
)
from src.jsonl import write_jsonl


def teacher_fold_of(identifier: str, num_folds: int) -> int:
    if num_folds < 2:
        raise ValueError("num_folds must be at least 2")
    return zlib.crc32(identifier.encode("utf-8")) % num_folds


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


def set_f1(left: set[str], right: set[str]) -> float:
    denominator = len(left) + len(right)
    return 2 * len(left & right) / denominator if denominator else 1.0


def same_fold_diagnosis_donor(
    own_index: int, rows: list[dict[str, Any]], *, num_folds: int = 2
) -> int | None:
    """Choose one deterministic same-fold, same-diagnosis hard control."""

    own = rows[own_index]
    own_fold = teacher_fold_of(base_id(own), num_folds)
    diagnosis = str(own.get("diagnosis_id") or own.get("diagnosis_name") or "")
    own_count = len(own.get("cue_evidence_ids") or [])
    candidates = []
    for index, row in enumerate(rows):
        if (
            index == own_index
            or teacher_fold_of(base_id(row), num_folds) != own_fold
        ):
            continue
        other_diagnosis = str(
            row.get("diagnosis_id") or row.get("diagnosis_name") or ""
        )
        if other_diagnosis != diagnosis:
            continue
        cue_difference = abs(len(row.get("cue_evidence_ids") or []) - own_count)
        candidates.append((cue_difference, base_id(row), index))
    candidates.sort()
    return candidates[0][2] if candidates else None


def selected_ids(
    probabilities: torch.Tensor, labels: list[str], threshold: float
) -> list[str]:
    """Return a set rendered in canonical evidence-ID order."""

    return sorted(
        label
        for label, probability in zip(labels, probabilities.tolist(), strict=True)
        if float(probability) >= threshold
    )


def build_teacher(
    *,
    original_manifest: Path,
    counterfactual_manifest: Path,
    probe_artifact: Path,
    output_jsonl: Path,
    output_json: Path,
    summary_md: Path,
    path_maps: list[tuple[str, str]],
    num_folds: int = 2,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    artifact = torch.load(probe_artifact, map_location="cpu", weights_only=False)
    if int(artifact.get("layer") or -1) != 32:
        raise ValueError("--probe-artifact must be an HS32 finding/value artifact")
    labels = [str(value) for value in artifact["finding_labels"]]
    if len(labels) != len(set(labels)):
        raise ValueError("Probe artifact contains duplicate finding labels")
    selected = artifact["finding_selected"]
    threshold = float(artifact["finding_threshold"])
    epochs = int(selected.get("best_epoch") or 0)
    if epochs <= 0:
        raise ValueError("Probe artifact has no positive finding best_epoch")

    originals = load_original_rows(original_manifest)
    deletions = load_deletion_rows(counterfactual_manifest)
    original_ids = {base_id(row) for row in originals}
    missing = original_ids - set(deletions)
    extra = set(deletions) - original_ids
    if missing or extra:
        raise ValueError(
            f"Original/deletion population mismatch: missing={len(missing)} "
            f"extra={len(extra)}"
        )

    original_features = torch.stack([load_vector(row, path_maps) for row in originals])
    deletion_features = torch.stack(
        [load_vector(deletions[base_id(row)], path_maps) for row in originals]
    )
    if original_features.shape != deletion_features.shape:
        raise ValueError(
            f"Original/deletion shape mismatch: {original_features.shape} vs "
            f"{deletion_features.shape}"
        )
    if num_folds < 2 or num_folds > len(originals):
        raise ValueError("--num-folds must be between 2 and the base-case count")
    targets = finding_targets(originals, labels)
    folds = torch.tensor(
        [teacher_fold_of(base_id(row), num_folds) for row in originals],
        dtype=torch.long,
    )
    original_probabilities = torch.empty((len(originals), len(labels)))
    deletion_probabilities = torch.empty_like(original_probabilities)
    fold_audits = []

    for heldout_fold in range(num_folds):
        train_indices = (folds != heldout_fold).nonzero(as_tuple=False).flatten()
        heldout_indices = (folds == heldout_fold).nonzero(as_tuple=False).flatten()
        if not len(train_indices) or not len(heldout_indices):
            raise ValueError(f"Empty train or held-out fold: {heldout_fold}")
        train_x = original_features[train_indices]
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_y = targets[train_indices]
        state = train_fixed_probe(
            (train_x - mean) / std,
            train_y,
            learning_rate=float(selected["learning_rate"]),
            weight_decay=float(selected["weight_decay"]),
            weighted=bool(selected["positive_weighting"]),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + heldout_fold,
            device=device,
        )
        original_probabilities[heldout_indices] = predict_probabilities(
            state, (original_features[heldout_indices] - mean) / std, device
        )
        deletion_probabilities[heldout_indices] = predict_probabilities(
            state, (deletion_features[heldout_indices] - mean) / std, device
        )
        positive_counts = train_y.sum(dim=0).to(torch.int64)
        fold_audits.append(
            {
                "heldout_fold": heldout_fold,
                "training_folds": [
                    value for value in range(num_folds) if value != heldout_fold
                ],
                "train_rows": int(len(train_indices)),
                "heldout_rows": int(len(heldout_indices)),
                "labels_selected_in_heldout_originals": int(
                    (original_probabilities[heldout_indices] >= threshold)
                    .any(dim=0)
                    .sum()
                ),
                "labels_with_zero_train_positives": int((positive_counts == 0).sum()),
                "labels_below_five_train_positives": int((positive_counts < 5).sum()),
                "minimum_train_positive_count": int(positive_counts.min()),
                "maximum_train_positive_count": int(positive_counts.max()),
            }
        )

    original_sets = [
        set(selected_ids(row, labels, threshold)) for row in original_probabilities
    ]
    deletion_sets = [
        set(selected_ids(row, labels, threshold)) for row in deletion_probabilities
    ]
    rows_out = []
    for index, original in enumerate(originals):
        identifier = base_id(original)
        deletion = deletions[identifier]
        for variant, source, probabilities, selected_set in (
            ("original", original, original_probabilities[index], original_sets[index]),
            ("cue_deleted", deletion, deletion_probabilities[index], deletion_sets[index]),
        ):
            rows_out.append(
                {
                    "id": str(source.get("id") or f"{identifier}__{variant}"),
                    "base_id": identifier,
                    "variant": variant,
                    "official_split": "train",
                    "diagnosis_id": original.get("diagnosis_id"),
                    "fold": teacher_fold_of(identifier, num_folds),
                    "layer": 32,
                    "position_family": "P0",
                    "finding_probabilities": [float(value) for value in probabilities],
                    "selected_evidence_ids": sorted(selected_set),
                    "selected_count": len(selected_set),
                    "changed_evidence_id": (
                        str(deletion.get("cf_original_evidence_id") or "")
                        if variant == "cue_deleted"
                        else None
                    ),
                }
            )
    write_jsonl(output_jsonl, rows_out)

    pair_jaccards = [
        set_jaccard(original_sets[index], deletion_sets[index])
        for index in range(len(originals))
    ]
    changed_in_ontology = changed_original_hit = changed_deleted_phantom = 0
    changed_removed = 0
    retained_common = retained_original_hit = retained_preserved = 0
    label_set = set(labels)
    for index, original in enumerate(originals):
        deletion = deletions[base_id(original)]
        changed = str(deletion.get("cf_original_evidence_id") or "")
        if changed in label_set:
            changed_in_ontology += 1
            before = changed in original_sets[index]
            after = changed in deletion_sets[index]
            changed_original_hit += before
            changed_deleted_phantom += after
            changed_removed += before and not after
        common = (
            set(map(str, original.get("cue_evidence_ids") or []))
            & set(map(str, deletion.get("cue_evidence_ids") or []))
            & label_set
        )
        retained_common += len(common)
        for evidence in common:
            before = evidence in original_sets[index]
            retained_original_hit += before
            retained_preserved += before and evidence in deletion_sets[index]

    shuffle_f1 = []
    shuffle_jaccard = []
    donor_rows = 0
    for index in range(len(originals)):
        donor = same_fold_diagnosis_donor(
            index, originals, num_folds=num_folds
        )
        if donor is None:
            continue
        donor_rows += 1
        shuffle_f1.append(set_f1(original_sets[index], original_sets[donor]))
        shuffle_jaccard.append(
            set_jaccard(original_sets[index], original_sets[donor])
        )

    diagnosis_rows: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(originals):
        diagnosis = str(row.get("diagnosis_id") or row.get("diagnosis_name") or "")
        diagnosis_rows[diagnosis].append(index)
    diagnosis_audits = []
    for diagnosis in sorted(diagnosis_rows):
        indices = diagnosis_rows[diagnosis]
        union = set().union(*(original_sets[index] for index in indices))
        diagnosis_audits.append(
            {
                "diagnosis_id": diagnosis,
                "base_cases": len(indices),
                "mean_selected_original": float(
                    np.mean([len(original_sets[index]) for index in indices])
                ),
                "labels_selected_anywhere": len(union),
                "label_coverage": len(union) / len(labels),
            }
        )

    original_counts = [float(len(value)) for value in original_sets]
    deletion_counts = [float(len(value)) for value in deletion_sets]
    empty_original = sum(not value for value in original_sets)
    empty_deleted = sum(not value for value in deletion_sets)
    shuffled_mean_f1 = float(np.mean(shuffle_f1)) if shuffle_f1 else None
    report = {
        "schema_version": 1,
        "method": "ddxplus_hs32_full_label_oof_finding_teacher",
        "scope": "official DDXPlus train original plus one cue-deleted arm",
        "n_base_cases": len(originals),
        "n_teacher_rows": len(rows_out),
        "complete_pair_coverage": 1.0,
        "finding_labels": labels,
        "n_finding_labels": len(labels),
        "finding_threshold": threshold,
        "selected_order": "canonical evidence_id ascending",
        "probability_vector_order": "finding_labels array order",
        "fold_method": f"crc32(base_id) % {num_folds}",
        "num_folds": num_folds,
        "fixed_probe_hyperparameters": {
            "learning_rate": float(selected["learning_rate"]),
            "weight_decay": float(selected["weight_decay"]),
            "positive_weighting": bool(selected["positive_weighting"]),
            "epochs": epochs,
            "seed_base": seed,
        },
        "fold_audits": fold_audits,
        "selected_count": {
            "original": quantiles(original_counts),
            "deleted": quantiles(deletion_counts),
            "empty_original": empty_original,
            "empty_deleted": empty_deleted,
        },
        "original_deleted_teacher_set_jaccard": quantiles(pair_jaccards),
        "changed_cue": {
            "eligible": changed_in_ontology,
            "original_hit": ratio(changed_original_hit, changed_in_ontology),
            "deleted_phantom": ratio(changed_deleted_phantom, changed_in_ontology),
            "removal_success_given_original_hit": ratio(
                changed_removed, changed_original_hit
            ),
            "original_hit_n": changed_original_hit,
        },
        "untouched_finding": {
            "common_input_cues": retained_common,
            "original_hit": ratio(retained_original_hit, retained_common),
            "preservation_given_original_hit": ratio(
                retained_preserved, retained_original_hit
            ),
            "original_hit_n": retained_original_hit,
        },
        "same_fold_same_diagnosis_teacher_set_control": {
            "pairs": donor_rows,
            "shuffled_set_f1_mean": shuffled_mean_f1,
            "matched_minus_shuffled_f1": (
                1.0 - shuffled_mean_f1 if shuffled_mean_f1 is not None else None
            ),
            "shuffled_jaccard": quantiles(shuffle_jaccard),
            "interpretation": (
                "target-set specificity ceiling only; matched teacher self-F1 is 1 by definition"
            ),
        },
        "diagnosis_audits": diagnosis_audits,
        "inputs": {
            "original_manifest": str(original_manifest),
            "original_manifest_sha256": sha256_file(original_manifest),
            "counterfactual_manifest": str(counterfactual_manifest),
            "counterfactual_manifest_sha256": sha256_file(counterfactual_manifest),
            "probe_artifact": str(probe_artifact),
            "probe_artifact_sha256": sha256_file(probe_artifact),
        },
        "teacher_scores_sha256": sha256_file(output_jsonl),
        "natural_language_target_written": False,
        "student_dataset_written": False,
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


def fmt(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.4f}"


def qline(values: dict[str, float] | None) -> str:
    if values is None:
        return "N/A"
    return (
        f"mean {values['mean']:.4f}; q05 {values['q05']:.4f}; "
        f"median {values['median']:.4f}; q95 {values['q95']:.4f}"
    )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    changed = report["changed_cue"]
    retained = report["untouched_finding"]
    control = report["same_fold_same_diagnosis_teacher_set_control"]
    lines = [
        "# DDXPlus HS32 Full-Label OOF Finding Teacher",
        "",
        "Official-train-only read-only materialization. The probe is used only as an "
        "out-of-fold teacher. No student target, adapter input, or validation gate is written.",
        "",
        f"- base cases: **{report['n_base_cases']}**",
        f"- original + deleted teacher rows: **{report['n_teacher_rows']}**",
        f"- complete pair coverage: **{report['complete_pair_coverage']:.4f}**",
        f"- finding labels: **{report['n_finding_labels']}**",
        f"- OOF folds: **{report['num_folds']}**",
        f"- inherited finding threshold: **{report['finding_threshold']:.4f}**",
        "- selected-set order: `canonical evidence_id ascending`",
        "- natural-language target written: **no**",
        "- validation / locked test read: **no / no**",
        "",
        "## Fold Coverage",
        "",
        "| held-out fold | train | held out | selected labels | zero-positive | below-5 | min / max positives |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["fold_audits"]:
        lines.append(
            f"| {fold['heldout_fold']} | {fold['train_rows']} | {fold['heldout_rows']} | "
            f"{fold['labels_selected_in_heldout_originals']} | "
            f"{fold['labels_with_zero_train_positives']} | "
            f"{fold['labels_below_five_train_positives']} | "
            f"{fold['minimum_train_positive_count']} / "
            f"{fold['maximum_train_positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Teacher-Set Distribution",
            "",
            f"- original selected count: {qline(report['selected_count']['original'])}",
            f"- deleted selected count: {qline(report['selected_count']['deleted'])}",
            f"- empty original / deleted sets: "
            f"**{report['selected_count']['empty_original']} / "
            f"{report['selected_count']['empty_deleted']}**",
            "- original/deleted set Jaccard: "
            f"{qline(report['original_deleted_teacher_set_jaccard'])}",
            "",
            "## Intervention Diagnostics",
            "",
            "| metric | value | denominator |",
            "|---|---:|---:|",
            f"| changed cue original hit | {fmt(changed['original_hit'])} | "
            f"{changed['eligible']} |",
            f"| changed cue deleted phantom | {fmt(changed['deleted_phantom'])} | "
            f"{changed['eligible']} |",
            "| removal success given original hit | "
            f"{fmt(changed['removal_success_given_original_hit'])} | "
            f"{changed['original_hit_n']} |",
            f"| untouched cue original hit | {fmt(retained['original_hit'])} | "
            f"{retained['common_input_cues']} |",
            "| untouched finding preservation | "
            f"{fmt(retained['preservation_given_original_hit'])} | "
            f"{retained['original_hit_n']} |",
            "",
            "## Same-Fold Same-Diagnosis Control",
            "",
            f"- paired bases: **{control['pairs']}**",
            f"- shuffled teacher-set F1: **{fmt(control['shuffled_set_f1_mean'])}**",
            f"- matched-minus-shuffled teacher-set F1: "
            f"**{fmt(control['matched_minus_shuffled_f1'])}**",
            f"- shuffled set Jaccard: {qline(control['shuffled_jaccard'])}",
            "",
            "This control measures target-set heterogeneity within diagnosis. Matched self-F1 "
            "is 1 by construction, so this is a specificity ceiling, not student performance.",
            "",
            "## Diagnosis Coverage",
            "",
            "| diagnosis | bases | mean selected | labels selected | label coverage |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in report["diagnosis_audits"]:
        lines.append(
            f"| {item['diagnosis_id']} | {item['base_cases']} | "
            f"{item['mean_selected_original']:.4f} | "
            f"{item['labels_selected_anywhere']} | {item['label_coverage']:.4f} |"
        )
    lines.extend(
        [
            "",
            "The deletion phantom rate is residual decodability under this intervention. "
            "It is not, by itself, proof of a representation failure because remaining cues "
            "may support inference of the deleted finding.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--counterfactual-manifest", required=True, type=Path)
    parser.add_argument("--probe-artifact", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--num-folds", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    report = build_teacher(
        original_manifest=args.original_manifest,
        counterfactual_manifest=args.counterfactual_manifest,
        probe_artifact=args.probe_artifact,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
        summary_md=args.summary_md,
        path_maps=parse_path_maps(args.path_map),
        num_folds=args.num_folds,
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        f"[teacher] bases={report['n_base_cases']} rows={report['n_teacher_rows']} "
        f"labels={report['n_finding_labels']}",
        flush=True,
    )
    print(f"[scores] {args.output_jsonl}", flush=True)
    print(f"[summary] {args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
