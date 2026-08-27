"""Train validation-selected DDXPlus finding and native-value probes.

The finding head is one multi-label linear map from a CoT-P0 hidden state to
evidence IDs.  The value head is one linear map whose outputs are grouped by
evidence ID and trained with conditional cross entropy.  Ontologies are built
from official training rows only.  Validation selects layer, regularization,
and a single finding threshold; no test manifest is accepted by this script.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from src.jsonl import read_jsonl


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    for row in read_jsonl(path):
        if str(row.get("variant") or "original") != "original":
            continue
        if str(row.get("position_family") or "P0") != "P0":
            raise ValueError(f"Non-P0 row in {path}: {row.get('id')}")
        identifier = base_id(row)
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate base_id in {path}: {identifier!r}")
        activation_path = Path(str(row.get("activation_path") or ""))
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        evidence_ids = list(row.get("cue_evidence_ids") or [])
        value_ids = list(row.get("cue_value_ids") or [])
        if len(evidence_ids) != len(value_ids):
            raise ValueError(f"Cue/value mismatch for {identifier}")
        seen.add(identifier)
        rows.append(row)
    if not rows:
        raise ValueError(f"No original P0 rows in {path}")
    return rows


def load_features(rows: list[dict[str, Any]]) -> torch.Tensor:
    vectors = [
        torch.load(row["activation_path"], map_location="cpu", weights_only=True)
        .flatten()
        .float()
        for row in rows
    ]
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent activation shapes: {sorted(shapes)}")
    return torch.stack(vectors)


def finding_vocabulary(
    rows: list[dict[str, Any]], *, min_count: int
) -> tuple[list[str], Counter[str]]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts.update({str(value) for value in row.get("cue_evidence_ids") or []})
    labels = sorted(label for label, count in counts.items() if count >= min_count)
    if not labels:
        raise ValueError(f"No finding meets min train count {min_count}")
    return labels, counts


def finding_targets(rows: list[dict[str, Any]], labels: list[str]) -> torch.Tensor:
    index = {label: offset for offset, label in enumerate(labels)}
    targets = torch.zeros((len(rows), len(labels)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for label in set(str(value) for value in row.get("cue_evidence_ids") or []):
            if label in index:
                targets[row_index, index[label]] = 1.0
    return targets


def eligible_finding_coverage(rows: list[dict[str, Any]], labels: set[str]) -> dict[str, float]:
    all_positive = 0
    known_positive = 0
    cases_with_known = 0
    for row in rows:
        values = {str(value) for value in row.get("cue_evidence_ids") or []}
        all_positive += len(values)
        known_positive += len(values & labels)
        cases_with_known += bool(values & labels)
    return {
        "positive_occurrence_coverage": known_positive / all_positive if all_positive else 0.0,
        "case_coverage": cases_with_known / len(rows),
        "known_positive_occurrences": known_positive,
        "all_positive_occurrences": all_positive,
    }


def value_ontology(
    rows: list[dict[str, Any]], *, min_value_count: int
) -> tuple[dict[str, list[str]], Counter[tuple[str, str]]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        for evidence_id, value_id in zip(
            row.get("cue_evidence_ids") or [],
            row.get("cue_value_ids") or [],
            strict=True,
        ):
            evidence = str(evidence_id)
            value = str(value_id or "")
            if value and "," not in value:
                counts[(evidence, value)] += 1
    values_by_evidence: dict[str, list[str]] = {}
    grouped: dict[str, list[str]] = defaultdict(list)
    for (evidence, value), count in counts.items():
        if count >= min_value_count:
            grouped[evidence].append(value)
    for evidence, values in grouped.items():
        unique = sorted(set(values))
        if len(unique) >= 2:
            values_by_evidence[evidence] = unique
    if not values_by_evidence:
        raise ValueError(
            f"No evidence has at least two values with count >= {min_value_count}"
        )
    return dict(sorted(values_by_evidence.items())), counts


def row_value_map(row: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    conflicts: set[str] = set()
    for evidence_id, value_id in zip(
        row.get("cue_evidence_ids") or [], row.get("cue_value_ids") or [], strict=True
    ):
        evidence = str(evidence_id)
        value = str(value_id or "")
        if not value or "," in value:
            continue
        if evidence in result and result[evidence] != value:
            conflicts.add(evidence)
        result[evidence] = value
    for evidence in conflicts:
        result.pop(evidence, None)
    return result


def value_targets(
    rows: list[dict[str, Any]], values_by_evidence: dict[str, list[str]]
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    result: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    maps = [row_value_map(row) for row in rows]
    for evidence, values in values_by_evidence.items():
        value_index = {value: index for index, value in enumerate(values)}
        row_indices = []
        labels = []
        for row_index, mapping in enumerate(maps):
            value = mapping.get(evidence)
            if value in value_index:
                row_indices.append(row_index)
                labels.append(value_index[value])
        result[evidence] = (
            torch.tensor(row_indices, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )
    return result


def binary_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int8, copy=False)
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(labels.size, dtype=np.float64)
    start = 0
    while start < labels.size:
        end = start + 1
        while end < labels.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[labels == 1].sum())
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def finding_metrics(
    logits: torch.Tensor, targets: torch.Tensor, threshold: float
) -> dict[str, float]:
    probabilities = logits.sigmoid().cpu()
    gold = targets.cpu().bool()
    predicted = probabilities >= threshold
    tp = int((predicted & gold).sum())
    fp = int((predicted & ~gold).sum())
    fn = int((~predicted & gold).sum())
    micro_f1 = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    per_label_f1 = []
    per_label_auc = []
    for column in range(gold.shape[1]):
        y = gold[:, column]
        p = predicted[:, column]
        ctp = int((p & y).sum())
        cfp = int((p & ~y).sum())
        cfn = int((~p & y).sum())
        per_label_f1.append(2 * ctp / (2 * ctp + cfp + cfn) if ctp + cfp + cfn else 0.0)
        auc = binary_auroc(y.numpy(), probabilities[:, column].numpy())
        if not math.isnan(auc):
            per_label_auc.append(auc)
    micro_auc = binary_auroc(gold.numpy().reshape(-1), probabilities.numpy().reshape(-1))
    return {
        "bce": float(F.binary_cross_entropy_with_logits(logits.cpu(), targets.cpu()).item()),
        "micro_f1": micro_f1,
        "macro_f1": float(np.mean(per_label_f1)),
        "micro_auroc": micro_auc,
        "macro_auroc": float(np.mean(per_label_auc)) if per_label_auc else math.nan,
        "predicted_positive_rate": float(predicted.float().mean().item()),
    }


def finding_pos_weight(targets: torch.Tensor) -> torch.Tensor:
    positives = targets.sum(dim=0).clamp_min(1)
    negatives = targets.shape[0] - positives
    return (negatives / positives).clamp(1.0, 20.0)


def train_finding_candidate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    weighted: bool,
    epochs: int,
    patience: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    torch.manual_seed(seed)
    model = torch.nn.Linear(train_x.shape[1], train_y.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    weight = finding_pos_weight(train_y).to(device) if weighted else None
    generator = torch.Generator().manual_seed(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = model(train_x[indices].to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, train_y[indices].to(device), pos_weight=weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        if epoch % 2:
            continue
        model.eval()
        with torch.inference_mode():
            val_logits = model(val_x.to(device)).cpu()
        val_loss = float(F.binary_cross_entropy_with_logits(val_logits, val_y).item())
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Finding probe produced no checkpoint")
    return best_state, best_epoch, best_loss


def value_slices(values_by_evidence: dict[str, list[str]]) -> dict[str, tuple[int, int]]:
    result = {}
    start = 0
    for evidence, values in values_by_evidence.items():
        result[evidence] = (start, start + len(values))
        start += len(values)
    return result


def conditional_value_loss(
    logits: torch.Tensor,
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    slices: dict[str, tuple[int, int]],
    device: torch.device,
) -> torch.Tensor:
    losses = []
    weights = []
    for evidence, (rows, labels) in targets.items():
        if not len(rows):
            continue
        start, end = slices[evidence]
        losses.append(F.cross_entropy(logits[rows.to(device), start:end], labels.to(device)))
        weights.append(len(rows))
    if not losses:
        raise ValueError("No eligible conditional value targets")
    total = sum(loss * weight for loss, weight in zip(losses, weights, strict=True))
    return total / sum(weights)


def value_metrics(
    logits: torch.Tensor,
    targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    slices: dict[str, tuple[int, int]],
) -> dict[str, float]:
    correct = 0
    total = 0
    recalls = []
    reciprocal_ranks = []
    for evidence, (rows, labels) in targets.items():
        if not len(rows):
            continue
        start, end = slices[evidence]
        scores = logits[rows, start:end]
        predictions = scores.argmax(dim=1)
        correct += int((predictions == labels).sum())
        total += len(rows)
        order = scores.argsort(dim=1, descending=True)
        ranks = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
        reciprocal_ranks.extend((1.0 / ranks.float()).tolist())
        for value in sorted(set(labels.tolist())):
            mask = labels == value
            recalls.append(float((predictions[mask] == labels[mask]).float().mean()))
    return {
        "targets": total,
        "accuracy": correct / total if total else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "mrr": float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0,
    }


def train_value_candidate(
    train_x: torch.Tensor,
    train_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    val_x: torch.Tensor,
    val_targets: dict[str, tuple[torch.Tensor, torch.Tensor]],
    slices: dict[str, tuple[int, int]],
    *,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], int, float]:
    torch.manual_seed(seed)
    output_dim = max(end for _, end in slices.values())
    model = torch.nn.Linear(train_x.shape[1], output_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_device = train_x.to(device)
    val_device = val_x.to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = math.inf
    best_epoch = 0
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        loss = conditional_value_loss(model(train_device), train_targets, slices, device)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.inference_mode():
            val_loss = float(
                conditional_value_loss(model(val_device), val_targets, slices, device).item()
            )
        if val_loss < best_loss - 1e-7:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(
                {key: value.detach().cpu() for key, value in model.state_dict().items()}
            )
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None:
        raise RuntimeError("Value probe produced no checkpoint")
    return best_state, best_epoch, best_loss


def predict_linear(
    state: dict[str, torch.Tensor], features: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model = torch.nn.Linear(features.shape[1], state["weight"].shape[0]).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        return model(features.to(device)).cpu()


def donor_targets(
    val_rows: list[dict[str, Any]],
    pairs_path: Path,
    finding_labels: list[str],
    values_by_evidence: dict[str, list[str]],
) -> tuple[list[int], torch.Tensor, dict[str, tuple[torch.Tensor, torch.Tensor]]]:
    row_by_id = {base_id(row): row for row in val_rows}
    index_by_id = {base_id(row): index for index, row in enumerate(val_rows)}
    selected_indices = []
    donor_rows = []
    for pair in read_jsonl(pairs_path):
        if not pair.get("primary_pair_eligible", True):
            continue
        own = str(pair.get("own_base_id") or "")
        donor = str(pair.get("donor_base_id") or "")
        if own in index_by_id and donor in row_by_id:
            selected_indices.append(index_by_id[own])
            donor_rows.append(row_by_id[donor])
    if not selected_indices:
        raise ValueError("No eligible validation hard-shuffle pairs joined")
    return (
        selected_indices,
        finding_targets(donor_rows, finding_labels),
        value_targets(donor_rows, values_by_evidence),
    )


def standardize(
    train_x: torch.Tensor, val_x: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = train_x.mean(dim=0, keepdim=True)
    std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (train_x - mean) / std, (val_x - mean) / std, mean, std


def fit_layer(
    *,
    layer: int,
    train_path: Path,
    val_path: Path,
    hard_pairs: Path,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    train_rows = load_manifest(train_path)
    val_rows = load_manifest(val_path)
    train_ids = {base_id(row) for row in train_rows}
    val_ids = {base_id(row) for row in val_rows}
    if train_ids & val_ids:
        raise ValueError(f"Train/validation overlap at HS{layer}: {len(train_ids & val_ids)}")

    finding_labels, finding_counts = finding_vocabulary(
        train_rows, min_count=args.min_finding_train_count
    )
    values_by_evidence, value_counts = value_ontology(
        train_rows, min_value_count=args.min_value_train_count
    )
    train_x, val_x = load_features(train_rows), load_features(val_rows)
    train_x, val_x, mean, std = standardize(train_x, val_x)
    train_finding = finding_targets(train_rows, finding_labels)
    val_finding = finding_targets(val_rows, finding_labels)
    train_values = value_targets(train_rows, values_by_evidence)
    val_values = value_targets(val_rows, values_by_evidence)
    slices = value_slices(values_by_evidence)

    finding_candidates = []
    selected_finding = None
    selected_finding_state = None
    for lr in args.learning_rates:
        for wd in args.weight_decays:
            for weighted in args.finding_weighted_options:
                state, epoch, val_loss = train_finding_candidate(
                    train_x,
                    train_finding,
                    val_x,
                    val_finding,
                    learning_rate=lr,
                    weight_decay=wd,
                    weighted=weighted,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    seed=args.seed,
                    device=device,
                )
                candidate = {
                    "learning_rate": lr,
                    "weight_decay": wd,
                    "positive_weighting": weighted,
                    "best_epoch": epoch,
                    "validation_bce": val_loss,
                }
                finding_candidates.append(candidate)
                if selected_finding is None or val_loss < selected_finding["validation_bce"]:
                    selected_finding = candidate
                    selected_finding_state = state
    assert selected_finding is not None and selected_finding_state is not None
    finding_logits = predict_linear(selected_finding_state, val_x, device)
    threshold_rows = []
    for threshold in args.thresholds:
        metrics = finding_metrics(finding_logits, val_finding, threshold)
        threshold_rows.append({"threshold": threshold, **metrics})
    selected_threshold = max(
        threshold_rows,
        key=lambda row: (row["micro_f1"], row["macro_f1"], -abs(row["threshold"] - 0.5)),
    )

    value_candidates = []
    selected_value = None
    selected_value_state = None
    for lr in args.learning_rates:
        for wd in args.weight_decays:
            state, epoch, val_loss = train_value_candidate(
                train_x,
                train_values,
                val_x,
                val_values,
                slices,
                learning_rate=lr,
                weight_decay=wd,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed,
                device=device,
            )
            candidate = {
                "learning_rate": lr,
                "weight_decay": wd,
                "best_epoch": epoch,
                "validation_nll": val_loss,
            }
            value_candidates.append(candidate)
            if selected_value is None or val_loss < selected_value["validation_nll"]:
                selected_value = candidate
                selected_value_state = state
    assert selected_value is not None and selected_value_state is not None
    value_logits = predict_linear(selected_value_state, val_x, device)
    own_value_metrics = value_metrics(value_logits, val_values, slices)

    donor_indices, donor_finding, donor_values = donor_targets(
        val_rows, hard_pairs, finding_labels, values_by_evidence
    )
    donor_logits = finding_logits[donor_indices]
    donor_finding_metrics = finding_metrics(
        donor_logits, donor_finding, float(selected_threshold["threshold"])
    )
    donor_value_metrics = value_metrics(value_logits[donor_indices], donor_values, slices)

    artifact_path = args.out_dir / f"finding_value_hs{layer}.pt"
    torch.save(
        {
            "layer": layer,
            "feature_mean": mean,
            "feature_std": std,
            "finding_labels": finding_labels,
            "finding_state_dict": selected_finding_state,
            "finding_selected": selected_finding,
            "finding_threshold": selected_threshold["threshold"],
            "values_by_evidence": values_by_evidence,
            "value_slices": slices,
            "value_state_dict": selected_value_state,
            "value_selected": selected_value,
        },
        artifact_path,
    )
    finding_result = {
        "labels": len(finding_labels),
        "min_train_count": args.min_finding_train_count,
        "train_label_count_min": min(finding_counts[label] for label in finding_labels),
        "coverage": eligible_finding_coverage(val_rows, set(finding_labels)),
        "selected": selected_finding,
        "selected_threshold": selected_threshold,
        "hard_shuffled": donor_finding_metrics,
        "own_minus_shuffled_micro_f1": selected_threshold["micro_f1"]
        - donor_finding_metrics["micro_f1"],
        "candidates": finding_candidates,
    }
    value_result = {
        "evidence_tasks": len(values_by_evidence),
        "value_classes": sum(len(values) for values in values_by_evidence.values()),
        "min_value_train_count": args.min_value_train_count,
        "eligible_train_targets": sum(len(rows) for rows, _ in train_values.values()),
        "eligible_validation_targets": sum(len(rows) for rows, _ in val_values.values()),
        "selected": selected_value,
        "validation": own_value_metrics,
        "hard_shuffled": donor_value_metrics,
        "own_minus_shuffled_accuracy": own_value_metrics["accuracy"]
        - donor_value_metrics["accuracy"],
        "candidates": value_candidates,
        "minimum_retained_pair_count": min(
            value_counts[(evidence, value)]
            for evidence, values in values_by_evidence.items()
            for value in values
        ),
    }
    return {
        "layer": layer,
        "n_train": len(train_rows),
        "n_validation": len(val_rows),
        "hard_shuffle_pairs": len(donor_indices),
        "finding": finding_result,
        "value": value_result,
        "artifact": str(artifact_path),
    }


def write_summary(path: Path, results: list[dict[str, Any]], selected_layer: int) -> None:
    lines = [
        "# DDXPlus CoT-P0 Finding And Value Probe Validation",
        "",
        "Official train fits the probes; official validation selects hyperparameters, "
        "threshold, and layer. Locked test was not read.",
        "",
        "## Finding Presence",
        "",
        "| HS | train | val | labels | coverage | micro F1 | macro F1 | micro AUROC "
        "| shuffled F1 | own-shuffled |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        finding = result["finding"]
        own = finding["selected_threshold"]
        shuffled = finding["hard_shuffled"]
        lines.append(
            f"| {result['layer']} | {result['n_train']} | {result['n_validation']} | "
            f"{finding['labels']} | {finding['coverage']['positive_occurrence_coverage']:.4f} | "
            f"{own['micro_f1']:.4f} | {own['macro_f1']:.4f} | {own['micro_auroc']:.4f} | "
            f"{shuffled['micro_f1']:.4f} | {finding['own_minus_shuffled_micro_f1']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Native Value (Conditioned On Evidence ID)",
            "",
            "| HS | evidence tasks | value classes | val targets | accuracy | macro recall "
            "| MRR | shuffled accuracy | own-shuffled |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in results:
        value = result["value"]
        own = value["validation"]
        shuffled = value["hard_shuffled"]
        lines.append(
            f"| {result['layer']} | {value['evidence_tasks']} | {value['value_classes']} | "
            f"{value['eligible_validation_targets']} | {own['accuracy']:.4f} | "
            f"{own['macro_recall']:.4f} | {own['mrr']:.4f} | "
            f"{shuffled['accuracy']:.4f} | {value['own_minus_shuffled_accuracy']:+.4f} |"
        )
    lines.extend(
        [
            "",
            f"- validation-selected layer: **HS{selected_layer}**",
            "- Finding scores are conditional on train-supported evidence IDs; coverage is "
            "reported explicitly.",
            "- Value accuracy is conditional on the evidence ID and only includes "
            "train-supported multi-value evidence.",
            "- Hard-shuffled targets come from a different case with the same diagnosis.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_bool_options(values: Iterable[str]) -> list[bool]:
    result = []
    for value in values:
        normalized = value.strip().lower()
        if normalized not in {"true", "false"}:
            raise argparse.ArgumentTypeError("expected true or false")
        result.append(normalized == "true")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", required=True, type=Path)
    parser.add_argument("--validation-root", required=True, type=Path)
    parser.add_argument("--validation-hard-pairs", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument("--min-finding-train-count", type=int, default=20)
    parser.add_argument("--min-value-train-count", type=int, default=10)
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[1e-3, 3e-3])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[0.0, 1e-3])
    parser.add_argument(
        "--finding-weighted-options", nargs="+", default=["true", "false"]
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4, 0.5]
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    args.finding_weighted_options = parse_bool_options(args.finding_weighted_options)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)

    results = []
    for layer in args.layers:
        train_path = args.train_root / f"layer{layer}" / "last_token" / "manifest.jsonl"
        val_path = args.validation_root / f"layer{layer}" / "last_token" / "manifest.jsonl"
        print(f"[probe] HS{layer}", flush=True)
        result = fit_layer(
            layer=layer,
            train_path=train_path,
            val_path=val_path,
            hard_pairs=args.validation_hard_pairs,
            args=args,
            device=device,
        )
        results.append(result)
        print(
            f"[probe] HS{layer} finding_f1="
            f"{result['finding']['selected_threshold']['micro_f1']:.4f} "
            f"value_acc={result['value']['validation']['accuracy']:.4f}",
            flush=True,
        )
    selected = max(
        results,
        key=lambda result: (
            result["finding"]["own_minus_shuffled_micro_f1"],
            result["value"]["own_minus_shuffled_accuracy"],
            -result["layer"],
        ),
    )
    output = {
        "schema_version": 1,
        "selection_rule": "max finding own-minus-hard-shuffled micro F1, then value gap",
        "selected_layer": selected["layer"],
        "locked_test_read": False,
        "results": results,
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", results, selected["layer"])
    print(f"[done] selected=HS{selected['layer']} out={args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
