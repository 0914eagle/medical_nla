"""Cross-fit DDXPlus selected changed-cue support scores.

The current counterfactual training population has exactly one cue-deletion
arm per base case.  Consequently this script scores only that selected changed
cue; it must not be used to claim that every input cue has been audited.

The finding ontology and fixed training hyperparameters come from a
validation-selected HS32 probe artifact.  Two new linear heads are trained on
opposite crc32 folds.  Every original, deletion, and donor probability is then
out-of-fold.  No threshold is applied and no SFT target is written here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import zlib
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from scripts.train_ddxplus_finding_value_probes import (
    base_id,
    finding_pos_weight,
    finding_targets,
)
from src.jsonl import read_jsonl, write_jsonl


def fold_of(identifier: str) -> int:
    return zlib.crc32(identifier.encode("utf-8")) % 2


def parse_path_maps(values: Iterable[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--path-map must be OLD=NEW")
        old, new = value.split("=", 1)
        if not old:
            raise argparse.ArgumentTypeError("--path-map OLD prefix cannot be empty")
        result.append((old, new))
    return result


def activation_path(row: dict[str, Any], path_maps: list[tuple[str, str]]) -> Path:
    value = str(row.get("activation_path") or "")
    for old, new in path_maps:
        if value.startswith(old):
            value = new + value[len(old) :]
            break
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_vector(row: dict[str, Any], path_maps: list[tuple[str, str]]) -> torch.Tensor:
    return torch.load(
        activation_path(row, path_maps), map_location="cpu", weights_only=True
    ).flatten().float()


def load_original_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for row in read_jsonl(path):
        if str(row.get("variant") or "original") != "original":
            continue
        identifier = base_id(row)
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate original base_id: {identifier!r}")
        if str(row.get("position_family") or "P0") != "P0":
            raise ValueError(f"Non-P0 original row: {identifier}")
        seen.add(identifier)
        rows.append(row)
    if not rows:
        raise ValueError(f"No original rows in {path}")
    return sorted(rows, key=base_id)


def load_deletion_rows(path: Path) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        if str(row.get("variant") or "") != "cue_deleted":
            continue
        identifier = base_id(row)
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate deletion base_id: {identifier!r}")
        if str(row.get("position_family") or "P0") != "P0":
            raise ValueError(f"Non-P0 deletion row: {identifier}")
        changed = str(row.get("cf_original_evidence_id") or "")
        if not changed:
            raise ValueError(f"Deletion row lacks cf_original_evidence_id: {identifier}")
        if changed in {str(value) for value in row.get("cue_evidence_ids") or []}:
            raise ValueError(f"Deleted evidence remains in deletion row: {identifier}/{changed}")
        result[identifier] = row
    if not result:
        raise ValueError(f"No cue_deleted rows in {path}")
    return result


def train_fixed_probe(
    features: torch.Tensor,
    targets: torch.Tensor,
    *,
    learning_rate: float,
    weight_decay: float,
    weighted: bool,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Fit without looking at the held-out fold for model selection."""

    torch.manual_seed(seed)
    model = torch.nn.Linear(features.shape[1], targets.shape[1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positive_weight = finding_pos_weight(targets).to(device) if weighted else None
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            logits = model(features[indices].to(device))
            loss = F.binary_cross_entropy_with_logits(
                logits, targets[indices].to(device), pos_weight=positive_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return {
        key: value.detach().cpu() for key, value in model.state_dict().items()
    }


def predict_probabilities(
    state: dict[str, torch.Tensor], features: torch.Tensor, device: torch.device
) -> torch.Tensor:
    model = torch.nn.Linear(features.shape[1], state["weight"].shape[0]).to(device)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        return model(features.to(device)).sigmoid().cpu()


def donor_indices(
    own_index: int,
    rows: list[dict[str, Any]],
    *,
    changed_evidence: str,
    maximum: int,
) -> list[int]:
    """Same-fold, same-diagnosis donors in which the changed cue is absent."""

    own = rows[own_index]
    own_fold = fold_of(base_id(own))
    diagnosis = str(own.get("diagnosis_id") or own.get("diagnosis_name") or "")
    own_count = len(own.get("cue_evidence_ids") or [])
    candidates = []
    for index, row in enumerate(rows):
        if index == own_index or fold_of(base_id(row)) != own_fold:
            continue
        other_diagnosis = str(row.get("diagnosis_id") or row.get("diagnosis_name") or "")
        if other_diagnosis != diagnosis:
            continue
        if changed_evidence in {str(value) for value in row.get("cue_evidence_ids") or []}:
            continue
        cue_difference = abs(len(row.get("cue_evidence_ids") or []) - own_count)
        candidates.append((cue_difference, base_id(row), index))
    candidates.sort()
    return [index for _, _, index in candidates[:maximum]]


def quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "q05": float(np.quantile(array, 0.05)),
        "q25": float(np.quantile(array, 0.25)),
        "median": float(np.quantile(array, 0.5)),
        "q75": float(np.quantile(array, 0.75)),
        "q95": float(np.quantile(array, 0.95)),
    }


def build_scores(
    *,
    original_manifest: Path,
    counterfactual_manifest: Path,
    probe_artifact: Path,
    output_jsonl: Path,
    output_json: Path,
    summary_md: Path,
    path_maps: list[tuple[str, str]],
    max_donors: int,
    min_fold_positive_count: int,
    batch_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    artifact = torch.load(probe_artifact, map_location="cpu", weights_only=False)
    if int(artifact.get("layer") or -1) != 32:
        raise ValueError("--probe-artifact must be an HS32 finding/value artifact")
    labels = [str(value) for value in artifact["finding_labels"]]
    label_index = {label: index for index, label in enumerate(labels)}
    selected = artifact["finding_selected"]
    epochs = int(selected.get("best_epoch") or 0)
    if epochs <= 0:
        raise ValueError("Probe artifact has no positive finding best_epoch")

    originals = load_original_rows(original_manifest)
    deletions = load_deletion_rows(counterfactual_manifest)
    original_ids = {base_id(row) for row in originals}
    missing_deletions = original_ids - set(deletions)
    extra_deletions = set(deletions) - original_ids
    if missing_deletions or extra_deletions:
        raise ValueError(
            "Original/deletion population mismatch: "
            f"missing={len(missing_deletions)} extra={len(extra_deletions)}"
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
    targets = finding_targets(originals, labels)
    folds = torch.tensor([fold_of(base_id(row)) for row in originals], dtype=torch.long)
    original_probabilities = torch.empty((len(originals), len(labels)))
    deletion_probabilities = torch.empty_like(original_probabilities)
    fold_audits = []

    for heldout_fold in (0, 1):
        train_indices = (folds != heldout_fold).nonzero(as_tuple=False).flatten()
        heldout_indices = (folds == heldout_fold).nonzero(as_tuple=False).flatten()
        if not len(train_indices) or not len(heldout_indices):
            raise ValueError(f"Empty train or held-out fold: {heldout_fold}")
        train_x = original_features[train_indices]
        mean = train_x.mean(dim=0, keepdim=True)
        std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
        train_x = (train_x - mean) / std
        train_y = targets[train_indices]
        state = train_fixed_probe(
            train_x,
            train_y,
            learning_rate=float(selected["learning_rate"]),
            weight_decay=float(selected["weight_decay"]),
            weighted=bool(selected["positive_weighting"]),
            epochs=epochs,
            batch_size=batch_size,
            seed=seed + heldout_fold,
            device=device,
        )
        heldout_original = (original_features[heldout_indices] - mean) / std
        heldout_deleted = (deletion_features[heldout_indices] - mean) / std
        original_probabilities[heldout_indices] = predict_probabilities(
            state, heldout_original, device
        )
        deletion_probabilities[heldout_indices] = predict_probabilities(
            state, heldout_deleted, device
        )
        positive_counts = train_y.sum(dim=0).to(torch.int64)
        fold_audits.append(
            {
                "heldout_fold": heldout_fold,
                "train_rows": len(train_indices),
                "heldout_rows": len(heldout_indices),
                "labels_with_zero_train_positives": int((positive_counts == 0).sum()),
                "labels_below_min_fold_positive_count": int(
                    (positive_counts < min_fold_positive_count).sum()
                ),
                "minimum_train_positive_count": int(positive_counts.min()),
            }
        )

    rows_out = []
    deletion_deltas = []
    donor_margins = []
    counts = Counter()
    for index, original in enumerate(originals):
        identifier = base_id(original)
        deletion = deletions[identifier]
        changed = str(deletion["cf_original_evidence_id"])
        label_offset = label_index.get(changed)
        training_fold = 1 - fold_of(identifier)
        training_positive_count = (
            int(targets[folds == training_fold, label_offset].sum())
            if label_offset is not None
            else 0
        )
        eligible = (
            label_offset is not None
            and training_positive_count >= min_fold_positive_count
        )
        donors = (
            donor_indices(index, originals, changed_evidence=changed, maximum=max_donors)
            if eligible
            else []
        )
        p_original = (
            float(original_probabilities[index, label_offset]) if eligible else None
        )
        p_deleted = (
            float(deletion_probabilities[index, label_offset]) if eligible else None
        )
        deletion_delta = (
            p_original - p_deleted
            if p_original is not None and p_deleted is not None
            else None
        )
        donor_values = (
            [float(original_probabilities[item, label_offset]) for item in donors]
            if label_offset is not None
            else []
        )
        donor_mean = float(np.mean(donor_values)) if donor_values else None
        donor_margin = (
            p_original - donor_mean
            if p_original is not None and donor_mean is not None
            else None
        )
        if label_offset is None:
            counts["changed_cue_outside_ontology"] += 1
        elif training_positive_count < min_fold_positive_count:
            counts["changed_cue_low_fold_support"] += 1
        else:
            counts["score_eligible"] += 1
        if not donors:
            counts["donor_unavailable"] += 1
        else:
            counts["donor_available"] += 1
        if deletion_delta is not None:
            deletion_deltas.append(deletion_delta)
        if donor_margin is not None:
            donor_margins.append(donor_margin)
        rows_out.append(
            {
                "base_id": identifier,
                "diagnosis_id": original.get("diagnosis_id"),
                "fold": fold_of(identifier),
                "changed_evidence_id": changed,
                "changed_cue_text": deletion.get("cf_original_cue"),
                "input_cue_count": len(original.get("cue_evidence_ids") or []),
                "fold_training_positive_count": training_positive_count,
                "score_eligible": eligible,
                "p_original": p_original,
                "p_deleted": p_deleted,
                "deletion_delta": deletion_delta,
                "donor_ids": [base_id(originals[item]) for item in donors],
                "donor_count": len(donors),
                "donor_mean_probability": donor_mean,
                "donor_margin": donor_margin,
                "selected_changed_cue_supported": None,
            }
        )

    write_jsonl(output_jsonl, rows_out)
    report = {
        "schema_version": 1,
        "scope": "D9a selected changed cue only; no all-cue support claim",
        "n_base_cases": len(originals),
        "finding_labels": len(labels),
        "fold_method": "crc32(base_id) % 2",
        "max_donors": max_donors,
        "min_fold_positive_count": min_fold_positive_count,
        "fixed_probe_hyperparameters": {
            "learning_rate": float(selected["learning_rate"]),
            "weight_decay": float(selected["weight_decay"]),
            "positive_weighting": bool(selected["positive_weighting"]),
            "epochs": epochs,
        },
        "counts": dict(sorted(counts.items())),
        "fold_audits": fold_audits,
        "deletion_delta": quantiles(deletion_deltas),
        "donor_margin": quantiles(donor_margins),
        "threshold_applied": False,
        "target_written": False,
        "locked_test_read": False,
        "inputs": {
            "original_manifest": str(original_manifest),
            "counterfactual_manifest": str(counterfactual_manifest),
            "probe_artifact": str(probe_artifact),
        },
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(summary_md, report)
    return report


def display_quantiles(value: dict[str, float] | None) -> str:
    if value is None:
        return "N/A"
    return (
        f"mean {value['mean']:+.4f}; q05 {value['q05']:+.4f}; "
        f"median {value['median']:+.4f}; q95 {value['q95']:+.4f}"
    )


def write_summary(path: Path, report: dict[str, Any]) -> None:
    counts = report["counts"]
    lines = [
        "# DDXPlus D9a Selected Changed-Cue Support Audit",
        "",
        "Official-train-only, cross-fitted read-only audit. Each base case contributes "
        "only its one pre-existing selected cue deletion. No threshold or target is written.",
        "",
        f"- base cases: **{report['n_base_cases']}**",
        f"- finding labels: **{report['finding_labels']}**",
        f"- score eligible: **{counts.get('score_eligible', 0)}**",
        f"- donor available: **{counts.get('donor_available', 0)}**",
        f"- changed cue outside ontology: **{counts.get('changed_cue_outside_ontology', 0)}**",
        f"- changed cue below fold support: **{counts.get('changed_cue_low_fold_support', 0)}**",
        f"- donor unavailable: **{counts.get('donor_unavailable', 0)}**",
        "- threshold applied: **no**",
        "- SFT target written: **no**",
        "- locked test read: **no**",
        "",
        "## Fold Audit",
        "",
        "| held-out fold | train | held out | zero-positive labels | below-min labels | min positives |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for fold in report["fold_audits"]:
        lines.append(
            f"| {fold['heldout_fold']} | {fold['train_rows']} | {fold['heldout_rows']} | "
            f"{fold['labels_with_zero_train_positives']} | "
            f"{fold['labels_below_min_fold_positive_count']} | "
            f"{fold['minimum_train_positive_count']} |"
        )
    lines.extend(
        [
            "",
            "## Score Distributions",
            "",
            f"- deletion delta: {display_quantiles(report['deletion_delta'])}",
            f"- cue-absent same-diagnosis donor margin: "
            f"{display_quantiles(report['donor_margin'])}",
            "",
            "These distributions do not define support by themselves. DDXPlus validation must "
            "freeze the false-support-controlled cuts before any target builder or smoke run.",
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
    parser.add_argument("--max-donors", type=int, default=5)
    parser.add_argument("--min-fold-positive-count", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_donors <= 0 or args.min_fold_positive_count <= 0:
        raise ValueError("--max-donors and --min-fold-positive-count must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    report = build_scores(
        original_manifest=args.original_manifest,
        counterfactual_manifest=args.counterfactual_manifest,
        probe_artifact=args.probe_artifact,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
        summary_md=args.summary_md,
        path_maps=parse_path_maps(args.path_map),
        max_donors=args.max_donors,
        min_fold_positive_count=args.min_fold_positive_count,
        batch_size=args.batch_size,
        seed=args.seed,
        device=torch.device(args.device),
    )
    print(
        f"[support] bases={report['n_base_cases']} "
        f"eligible={report['counts'].get('score_eligible', 0)} "
        f"donors={report['counts'].get('donor_available', 0)}",
        flush=True,
    )
    print(f"[scores] {args.output_jsonl}", flush=True)
    print(f"[summary] {args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
