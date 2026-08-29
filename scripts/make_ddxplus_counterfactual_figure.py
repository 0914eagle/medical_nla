"""Render paired DDXPlus deletion/value-edit response from a frozen probe."""

from __future__ import annotations

import argparse
import hashlib
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

from scripts.train_ddxplus_finding_value_probes import load_features, predict_linear
from src.jsonl import read_jsonl


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path_map(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("expected OLD=NEW")
    return source, destination


def mapped_rows(path: Path, mappings: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    ids = [str(row.get("id") or "") for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Manifest has missing or duplicate row IDs")
    for row in rows:
        activation = str(row.get("activation_path") or "")
        for source, destination in mappings:
            if activation.startswith(source):
                activation = destination + activation[len(source) :]
                break
        if not Path(activation).is_file():
            raise FileNotFoundError(activation)
        row["activation_path"] = activation
    return rows


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def compute(
    artifact_path: Path, rows: list[dict[str, Any]], device: torch.device
) -> tuple[dict[str, Any], dict[str, list[float]]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    layer = int(artifact["layer"])
    if layer != 24:
        raise ValueError(f"Canonical counterfactual figure requires HS24, got HS{layer}")
    features = load_features(rows)
    features = (features - artifact["feature_mean"]) / artifact["feature_std"]
    finding_logits = predict_linear(artifact["finding_state_dict"], features, device)
    value_logits = predict_linear(artifact["value_state_dict"], features, device)
    finding_probs = finding_logits.sigmoid()
    labels = [str(value) for value in artifact["finding_labels"]]
    label_index = {label: index for index, label in enumerate(labels)}
    threshold = float(artifact["finding_threshold"])
    slices = {
        str(key): (int(value[0]), int(value[1]))
        for key, value in artifact["value_slices"].items()
    }
    values = {
        str(key): [str(item) for item in items]
        for key, items in artifact["values_by_evidence"].items()
    }
    value_index = {
        evidence: {value: index for index, value in enumerate(items)}
        for evidence, items in values.items()
    }
    originals = {
        base_id(row): index
        for index, row in enumerate(rows)
        if str(row.get("variant") or "original") == "original"
    }
    deletion_before: list[float] = []
    deletion_after: list[float] = []
    deletion_drop: list[float] = []
    deletion_original_hit = deletion_phantom = deletion_removed = 0
    retained_original_hit = retained_preserved = retained_total = 0
    replacement = old_persistence = clean_switch = clean_n = edit_total = 0
    margin_changes: list[float] = []
    for index, row in enumerate(rows):
        variant = str(row.get("variant") or "original")
        if variant == "original":
            continue
        original_index = originals.get(base_id(row))
        if original_index is None:
            raise ValueError(f"No original arm for {base_id(row)}")
        evidence = str(row.get("cf_original_evidence_id") or "")
        if variant == "cue_deleted" and evidence in label_index:
            column = label_index[evidence]
            before = float(finding_probs[original_index, column])
            after = float(finding_probs[index, column])
            hit = before >= threshold
            phantom = after >= threshold
            deletion_before.append(before)
            deletion_after.append(after)
            deletion_drop.append(before - after)
            deletion_original_hit += hit
            deletion_phantom += phantom
            deletion_removed += hit and not phantom
            original_cues = {
                str(value) for value in rows[original_index].get("cue_evidence_ids") or []
            }
            derived_cues = {str(value) for value in row.get("cue_evidence_ids") or []}
            for retained in original_cues & derived_cues & label_index.keys():
                retained_total += 1
                retained_column = label_index[retained]
                retained_hit = float(finding_probs[original_index, retained_column]) >= threshold
                retained_original_hit += retained_hit
                retained_preserved += (
                    retained_hit and float(finding_probs[index, retained_column]) >= threshold
                )
        elif variant == "value_edited" and evidence in slices:
            old = str(row.get("cf_original_value_id") or "")
            new = str(row.get("cf_replacement_value_id") or "")
            if old not in value_index[evidence] or new not in value_index[evidence]:
                continue
            start, end = slices[evidence]
            before = value_logits[original_index, start:end]
            after = value_logits[index, start:end]
            old_index = value_index[evidence][old]
            new_index = value_index[evidence][new]
            before_prediction = int(before.argmax())
            after_prediction = int(after.argmax())
            edit_total += 1
            replacement += after_prediction == new_index
            old_persistence += after_prediction == old_index
            if before_prediction == old_index:
                clean_n += 1
                clean_switch += after_prediction == new_index
            after_margin = after[new_index] - after[old_index]
            before_margin = before[new_index] - before[old_index]
            margin_changes.append(float(after_margin - before_margin))
    if not deletion_drop or not margin_changes:
        raise ValueError("Manifest does not contain eligible deletion and value-edit pairs")
    metrics = {
        "layer": layer,
        "finding_threshold": threshold,
        "deletion": {
            "eligible": len(deletion_drop),
            "mean_probability_before": float(np.mean(deletion_before)),
            "mean_probability_after": float(np.mean(deletion_after)),
            "mean_probability_drop": float(np.mean(deletion_drop)),
            "original_hit": ratio(deletion_original_hit, len(deletion_drop)),
            "phantom": ratio(deletion_phantom, len(deletion_drop)),
            "removal_given_original_hit": ratio(deletion_removed, deletion_original_hit),
        },
        "retained": {
            "eligible": retained_total,
            "preservation_given_original_hit": ratio(retained_preserved, retained_original_hit),
            "conditional_denominator": retained_original_hit,
        },
        "value_edit": {
            "eligible": edit_total,
            "replacement_hit": ratio(replacement, edit_total),
            "old_persistence": ratio(old_persistence, edit_total),
            "clean_switch": ratio(clean_switch, clean_n),
            "clean_switch_denominator": clean_n,
            "mean_new_minus_old_margin_change": float(np.mean(margin_changes)),
        },
    }
    distributions = {
        "deletion_before": deletion_before,
        "deletion_after": deletion_after,
        "deletion_drop": deletion_drop,
        "value_margin_change": margin_changes,
    }
    return metrics, distributions


def render(output: Path, metrics: dict[str, Any], values: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.15))
    before, after = values["deletion_before"], values["deletion_after"]
    violin = axes[0].violinplot([before, after], positions=[0, 1], showmeans=True, widths=0.72)
    for body in violin["bodies"]:
        body.set_facecolor("0.82")
        body.set_edgecolor("0.35")
        body.set_alpha(1)
    axes[0].set_xticks([0, 1], ["original", "deleted"])
    axes[0].set_ylabel("target finding probability", fontsize=8)
    axes[0].set_title("(a) Deleted-cue response", fontsize=8.5)
    axes[0].text(
        0.5,
        0.04,
        f"mean drop {metrics['deletion']['mean_probability_drop']:+.3f}",
        transform=axes[0].transAxes,
        ha="center",
        fontsize=6.8,
    )

    labels = ["phantom", "removal", "retention"]
    rates = [
        metrics["deletion"]["phantom"],
        metrics["deletion"]["removal_given_original_hit"],
        metrics["retained"]["preservation_given_original_hit"],
    ]
    bars = axes[1].bar(range(3), rates, color=["0.65", "0.35", "0.82"], width=0.65)
    axes[1].set_xticks(range(3), labels, rotation=20, ha="right")
    axes[1].set_ylim(0, 1.08)
    axes[1].set_ylabel("rate", fontsize=8)
    axes[1].set_title("(b) Deletion specificity", fontsize=8.5)
    axes[1].bar_label(bars, labels=[f"{value:.3f}" for value in rates], fontsize=6.5)

    edit_labels = ["replacement", "old persists", "clean switch"]
    edit_rates = [
        metrics["value_edit"]["replacement_hit"],
        metrics["value_edit"]["old_persistence"],
        metrics["value_edit"]["clean_switch"],
    ]
    bars = axes[2].bar(range(3), edit_rates, color=["0.35", "0.68", "0.82"], width=0.65)
    axes[2].set_xticks(range(3), edit_labels, rotation=20, ha="right")
    axes[2].set_ylim(0, max(0.72, max(edit_rates) + 0.15))
    axes[2].set_ylabel("rate", fontsize=8)
    axes[2].set_title("(c) Native-value edits", fontsize=8.5)
    axes[2].bar_label(bars, labels=[f"{value:.3f}" for value in edit_rates], fontsize=6.5)
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.24, wspace=0.34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def finite(value: Any) -> bool:
    return value is not None and math.isfinite(float(value))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--values-json", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], type=path_map)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    rows = mapped_rows(args.manifest, args.path_map)
    metrics, distributions = compute(args.artifact, rows, torch.device(args.device))
    if not all(
        finite(value)
        for value in (
            metrics["deletion"]["phantom"],
            metrics["deletion"]["removal_given_original_hit"],
            metrics["retained"]["preservation_given_original_hit"],
            metrics["value_edit"]["replacement_hit"],
            metrics["value_edit"]["old_persistence"],
            metrics["value_edit"]["clean_switch"],
        )
    ):
        raise ValueError("One or more canonical figure metrics have zero denominator")
    payload = {
        "schema_version": 1,
        "population": "locked_test",
        "metrics": metrics,
        "sources": {
            "artifact": {"path": str(args.artifact), "sha256": sha256_file(args.artifact)},
            "manifest": {"path": str(args.manifest), "sha256": sha256_file(args.manifest)},
        },
        "distribution_summary": {
            name: {
                "n": len(values),
                "mean": float(np.mean(values)),
                "q05": float(np.quantile(values, 0.05)),
                "median": float(np.quantile(values, 0.5)),
                "q95": float(np.quantile(values, 0.95)),
            }
            for name, values in distributions.items()
        },
    }
    args.values_json.parent.mkdir(parents=True, exist_ok=True)
    args.values_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    render(args.output, metrics, distributions)
    print(f"[figure] {args.output}")
    print(f"[values] {args.values_json}")


if __name__ == "__main__":
    main()
