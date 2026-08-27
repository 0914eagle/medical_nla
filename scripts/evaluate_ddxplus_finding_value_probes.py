"""Evaluate a frozen DDXPlus finding/value probe on a locked manifest.

The checkpoint, layer, finding threshold, and train-derived ontologies are read
from one validation-selected artifact.  No fitting or threshold selection is
performed here.  Original, same-diagnosis hard-shuffle, cue-deletion, and
native-value-edit results are reported together.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_ddxplus_finding_value_probes import (
    base_id,
    eligible_finding_coverage,
    finding_metrics,
    finding_targets,
    load_features,
    predict_linear,
    row_value_map,
    value_metrics,
    value_targets,
)
from src.jsonl import read_jsonl


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No rows in {path}")
    ids: set[str] = set()
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in ids:
            raise ValueError(f"Missing or duplicate row id: {identifier!r}")
        ids.add(identifier)
        if str(row.get("position_family") or "P0") != "P0":
            raise ValueError(f"Non-P0 row: {identifier}")
        activation_path = Path(str(row.get("activation_path") or ""))
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
    return rows


def selected_rows(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("variant") or "original") == variant]


def value_coverage(
    rows: list[dict[str, Any]], values_by_evidence: dict[str, list[str]]
) -> dict[str, float | int]:
    total = 0
    eligible = 0
    eligible_set = {evidence: set(values) for evidence, values in values_by_evidence.items()}
    for row in rows:
        for evidence, value in row_value_map(row).items():
            total += 1
            eligible += value in eligible_set.get(evidence, set())
    return {
        "eligible_targets": eligible,
        "all_single_value_targets": total,
        "target_coverage": eligible / total if total else 0.0,
    }


def normalize_slices(raw: dict[str, Any]) -> dict[str, tuple[int, int]]:
    return {
        str(evidence): (int(bounds[0]), int(bounds[1]))
        for evidence, bounds in raw.items()
    }


def hard_shuffle_indices(
    originals: list[dict[str, Any]], pairs_path: Path
) -> tuple[list[int], list[dict[str, Any]]]:
    row_by_id = {base_id(row): row for row in originals}
    index_by_id = {base_id(row): index for index, row in enumerate(originals)}
    indices = []
    donors = []
    for pair in read_jsonl(pairs_path):
        if not pair.get("primary_pair_eligible", True):
            continue
        own = str(pair.get("own_base_id") or "")
        donor = str(pair.get("donor_base_id") or "")
        if own in index_by_id and donor in row_by_id:
            indices.append(index_by_id[own])
            donors.append(row_by_id[donor])
    if not indices:
        raise ValueError("No hard-shuffle pairs joined the locked population")
    return indices, donors


def finding_f1(logits: torch.Tensor, targets: torch.Tensor, threshold: float) -> float:
    predicted = logits.sigmoid() >= threshold
    gold = targets.bool()
    tp = int((predicted & gold).sum())
    fp = int((predicted & ~gold).sum())
    fn = int((~predicted & gold).sum())
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def value_accuracy(
    logits: torch.Tensor,
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    slices: dict[str, tuple[int, int]],
) -> float:
    return float(value_metrics(logits, targets, slices)["accuracy"])


def finding_confusion_by_case(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = logits.sigmoid() >= threshold
    gold = targets.bool()
    return (
        (predicted & gold).sum(dim=1).numpy(),
        (predicted & ~gold).sum(dim=1).numpy(),
        (~predicted & gold).sum(dim=1).numpy(),
    )


def value_correct_by_case(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    values_by_evidence: dict[str, list[str]],
    slices: dict[str, tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    correct = np.zeros(len(rows), dtype=np.int64)
    total = np.zeros(len(rows), dtype=np.int64)
    value_indices = {
        evidence: {value: index for index, value in enumerate(values)}
        for evidence, values in values_by_evidence.items()
    }
    for row_index, row in enumerate(rows):
        for evidence, value in row_value_map(row).items():
            if evidence not in slices or value not in value_indices[evidence]:
                continue
            start, end = slices[evidence]
            prediction = int(logits[row_index, start:end].argmax())
            correct[row_index] += prediction == value_indices[evidence][value]
            total[row_index] += 1
    return correct, total


def bootstrap_gaps(
    finding_logits: torch.Tensor,
    own_finding: torch.Tensor,
    donor_finding: torch.Tensor,
    value_logits: torch.Tensor,
    own_rows: list[dict[str, Any]],
    donor_rows: list[dict[str, Any]],
    values_by_evidence: dict[str, list[str]],
    slices: dict[str, tuple[int, int]],
    threshold: float,
    *,
    replicates: int,
    seed: int,
) -> dict[str, list[float] | float]:
    rng = np.random.default_rng(seed)
    finding_gaps = []
    value_gaps = []
    n = len(own_rows)
    own_tp, own_fp, own_fn = finding_confusion_by_case(
        finding_logits, own_finding, threshold
    )
    donor_tp, donor_fp, donor_fn = finding_confusion_by_case(
        finding_logits, donor_finding, threshold
    )
    own_value_correct, own_value_total = value_correct_by_case(
        value_logits, own_rows, values_by_evidence, slices
    )
    donor_value_correct, donor_value_total = value_correct_by_case(
        value_logits, donor_rows, values_by_evidence, slices
    )
    for _ in range(replicates):
        sample = rng.integers(0, n, size=n)
        own_denominator = 2 * own_tp[sample].sum() + own_fp[sample].sum() + own_fn[sample].sum()
        donor_denominator = (
            2 * donor_tp[sample].sum()
            + donor_fp[sample].sum()
            + donor_fn[sample].sum()
        )
        own_f1 = 2 * own_tp[sample].sum() / own_denominator if own_denominator else 0.0
        donor_f1 = (
            2 * donor_tp[sample].sum() / donor_denominator
            if donor_denominator
            else 0.0
        )
        finding_gaps.append(float(own_f1 - donor_f1))
        own_total = own_value_total[sample].sum()
        donor_total = donor_value_total[sample].sum()
        own_accuracy = (
            own_value_correct[sample].sum() / own_total if own_total else 0.0
        )
        donor_accuracy = (
            donor_value_correct[sample].sum() / donor_total if donor_total else 0.0
        )
        value_gaps.append(float(own_accuracy - donor_accuracy))

    def interval(values: list[float]) -> list[float]:
        return [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))]

    return {
        "replicates": replicates,
        "finding_gap_mean": float(np.mean(finding_gaps)),
        "finding_gap_ci95": interval(finding_gaps),
        "value_gap_mean": float(np.mean(value_gaps)),
        "value_gap_ci95": interval(value_gaps),
    }


def paired_counterfactual_metrics(
    rows: list[dict[str, Any]],
    finding_logits: torch.Tensor,
    value_logits: torch.Tensor,
    finding_labels: list[str],
    values_by_evidence: dict[str, list[str]],
    slices: dict[str, tuple[int, int]],
    threshold: float,
) -> dict[str, Any]:
    finding_index = {label: index for index, label in enumerate(finding_labels)}
    value_index = {
        evidence: {value: index for index, value in enumerate(values)}
        for evidence, values in values_by_evidence.items()
    }
    original_index = {
        base_id(row): index
        for index, row in enumerate(rows)
        if str(row.get("variant") or "original") == "original"
    }
    deletion_drops = []
    deletion_absent = []
    deletion_original_hit = []
    deletion_removal_success = []
    edit_new_hits = []
    edit_old_persistence = []
    edit_clean_switch = []
    edit_margin_changes = []

    finding_probabilities = finding_logits.sigmoid()
    for index, row in enumerate(rows):
        variant = str(row.get("variant") or "original")
        original = original_index.get(base_id(row))
        if original is None or variant == "original":
            continue
        evidence = str(row.get("cf_original_evidence_id") or "")
        if variant == "cue_deleted" and evidence in finding_index:
            column = finding_index[evidence]
            before = float(finding_probabilities[original, column])
            after = float(finding_probabilities[index, column])
            original_hit = before >= threshold
            absent_after = after < threshold
            deletion_drops.append(before - after)
            deletion_absent.append(absent_after)
            deletion_original_hit.append(original_hit)
            if original_hit:
                deletion_removal_success.append(absent_after)
        elif variant == "value_edited" and evidence in slices:
            old_value = str(row.get("cf_original_value_id") or "")
            new_value = str(row.get("cf_replacement_value_id") or "")
            mapping = value_index[evidence]
            if old_value not in mapping or new_value not in mapping:
                continue
            start, end = slices[evidence]
            before_scores = value_logits[original, start:end]
            after_scores = value_logits[index, start:end]
            old_index = mapping[old_value]
            new_index = mapping[new_value]
            before_prediction = int(before_scores.argmax())
            after_prediction = int(after_scores.argmax())
            edit_new_hits.append(after_prediction == new_index)
            edit_old_persistence.append(after_prediction == old_index)
            if before_prediction == old_index:
                edit_clean_switch.append(after_prediction == new_index)
            before_margin = float(before_scores[new_index] - before_scores[old_index])
            after_margin = float(after_scores[new_index] - after_scores[old_index])
            edit_margin_changes.append(after_margin - before_margin)

    def mean(values: list[Any]) -> float | None:
        return float(np.mean(values)) if values else None

    return {
        "deletion": {
            "eligible_pairs": len(deletion_drops),
            "mean_target_probability_drop": mean(deletion_drops),
            "absent_after_deletion": mean(deletion_absent),
            "original_target_hit": mean(deletion_original_hit),
            "removal_success_given_original_hit": mean(deletion_removal_success),
            "conditional_denominator": len(deletion_removal_success),
        },
        "value_edit": {
            "eligible_pairs": len(edit_new_hits),
            "replacement_hit": mean(edit_new_hits),
            "old_value_persistence": mean(edit_old_persistence),
            "clean_switch_given_original_old": mean(edit_clean_switch),
            "conditional_denominator": len(edit_clean_switch),
            "mean_new_minus_old_margin_change": mean(edit_margin_changes),
        },
    }


def format_value(value: float | None, *, signed: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def write_summary(path: Path, result: dict[str, Any]) -> None:
    finding = result["finding"]
    value = result["value"]
    deletion = result["counterfactual"]["deletion"]
    edit = result["counterfactual"]["value_edit"]
    bootstrap = result["bootstrap"]
    lines = [
        "# DDXPlus Locked-Test Finding And Value Probe",
        "",
        f"- frozen layer: **HS{result['layer']}**",
        f"- original test cases: **{result['n_original']}**",
        f"- finding labels: **{finding['labels']}**",
        f"- value evidence tasks/classes: **{value['evidence_tasks']}/{value['value_classes']}**",
        "",
        "## Original And Hard-Shuffled",
        "",
        "| target | own | same-diagnosis shuffled | gap | bootstrap 95% CI | coverage |",
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| finding micro F1 | {finding['own']['micro_f1']:.4f} | "
            f"{finding['shuffled']['micro_f1']:.4f} | {finding['gap']:+.4f} | "
            f"[{bootstrap['finding_gap_ci95'][0]:+.4f}, "
            f"{bootstrap['finding_gap_ci95'][1]:+.4f}] | "
            f"{finding['coverage']['positive_occurrence_coverage']:.4f} |"
        ),
        (
            f"| conditional native-value accuracy | {value['own']['accuracy']:.4f} | "
            f"{value['shuffled']['accuracy']:.4f} | {value['gap']:+.4f} | "
            f"[{bootstrap['value_gap_ci95'][0]:+.4f}, "
            f"{bootstrap['value_gap_ci95'][1]:+.4f}] | "
            f"{value['coverage']['target_coverage']:.4f} |"
        ),
        "",
        "## Counterfactual Response",
        "",
        "| intervention | eligible | primary response | side effect/control |",
        "|---|---:|---:|---:|",
        (
            f"| cue deletion | {deletion['eligible_pairs']} | probability drop "
            f"{format_value(deletion['mean_target_probability_drop'], signed=True)} | "
            f"removal success {format_value(deletion['removal_success_given_original_hit'])} "
            f"(n={deletion['conditional_denominator']}) |"
        ),
        (
            f"| native value edit | {edit['eligible_pairs']} | replacement hit "
            f"{format_value(edit['replacement_hit'])} | old persistence "
            f"{format_value(edit['old_value_persistence'])}; clean switch "
            f"{format_value(edit['clean_switch_given_original_old'])} "
            f"(n={edit['conditional_denominator']}) |"
        ),
        "",
        "No layer, threshold, ontology, or checkpoint was selected on this test population.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--hard-pairs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    layer = int(artifact["layer"])
    if f"layer{layer}" not in str(args.manifest):
        raise ValueError(
            f"Frozen artifact is HS{layer}, but manifest path is {args.manifest}"
        )
    rows = load_rows(args.manifest)
    originals = selected_rows(rows, "original")
    if not originals:
        raise ValueError("Locked manifest has no original rows")
    features = load_features(rows)
    mean = artifact["feature_mean"]
    std = artifact["feature_std"]
    features = (features - mean) / std
    device = torch.device(args.device)
    finding_logits = predict_linear(artifact["finding_state_dict"], features, device)
    value_logits = predict_linear(artifact["value_state_dict"], features, device)
    original_indices = [
        index
        for index, row in enumerate(rows)
        if str(row.get("variant") or "original") == "original"
    ]
    original_finding_logits = finding_logits[original_indices]
    original_value_logits = value_logits[original_indices]
    finding_labels = [str(value) for value in artifact["finding_labels"]]
    values_by_evidence = {
        str(evidence): [str(value) for value in values]
        for evidence, values in artifact["values_by_evidence"].items()
    }
    slices = normalize_slices(artifact["value_slices"])
    threshold = float(artifact["finding_threshold"])
    own_finding = finding_targets(originals, finding_labels)
    own_values = value_targets(originals, values_by_evidence)
    own_finding_metrics = finding_metrics(
        original_finding_logits, own_finding, threshold
    )
    own_value_metrics = value_metrics(original_value_logits, own_values, slices)

    pair_indices, donors = hard_shuffle_indices(originals, args.hard_pairs)
    pair_tensor = torch.tensor(pair_indices, dtype=torch.long)
    pair_own_rows = [originals[index] for index in pair_indices]
    pair_finding_logits = original_finding_logits[pair_tensor]
    pair_value_logits = original_value_logits[pair_tensor]
    pair_own_finding = finding_targets(pair_own_rows, finding_labels)
    donor_finding = finding_targets(donors, finding_labels)
    donor_values = value_targets(donors, values_by_evidence)
    shuffled_finding_metrics = finding_metrics(
        pair_finding_logits, donor_finding, threshold
    )
    shuffled_value_metrics = value_metrics(pair_value_logits, donor_values, slices)
    pair_own_finding_metrics = finding_metrics(
        pair_finding_logits, pair_own_finding, threshold
    )
    pair_own_value_metrics = value_metrics(
        pair_value_logits, value_targets(pair_own_rows, values_by_evidence), slices
    )
    bootstrap = bootstrap_gaps(
        pair_finding_logits,
        pair_own_finding,
        donor_finding,
        pair_value_logits,
        pair_own_rows,
        donors,
        values_by_evidence,
        slices,
        threshold,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    result = {
        "schema_version": 1,
        "artifact": str(args.artifact),
        "manifest": str(args.manifest),
        "layer": layer,
        "finding_threshold": threshold,
        "n_rows": len(rows),
        "n_original": len(originals),
        "hard_shuffle_pairs": len(pair_indices),
        "selection_performed_on_test": False,
        "finding": {
            "labels": len(finding_labels),
            "coverage": eligible_finding_coverage(originals, set(finding_labels)),
            "own_all_originals": own_finding_metrics,
            "own": pair_own_finding_metrics,
            "shuffled": shuffled_finding_metrics,
            "gap": pair_own_finding_metrics["micro_f1"]
            - shuffled_finding_metrics["micro_f1"],
        },
        "value": {
            "evidence_tasks": len(values_by_evidence),
            "value_classes": sum(len(values) for values in values_by_evidence.values()),
            "coverage": value_coverage(originals, values_by_evidence),
            "own_all_originals": own_value_metrics,
            "own": pair_own_value_metrics,
            "shuffled": shuffled_value_metrics,
            "gap": pair_own_value_metrics["accuracy"]
            - shuffled_value_metrics["accuracy"],
        },
        "counterfactual": paired_counterfactual_metrics(
            rows,
            finding_logits,
            value_logits,
            finding_labels,
            values_by_evidence,
            slices,
            threshold,
        ),
        "bootstrap": bootstrap,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", result)
    print(f"[done] {args.out_dir}", flush=True)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
