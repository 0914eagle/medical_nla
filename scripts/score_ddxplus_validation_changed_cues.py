"""Score D9a changed cues and matched cue-absent nulls on validation.

This is a read-only validation procedure. It applies the frozen HS32 finding
probe to original and deletion activations. Each eligible selected changed cue
gets one deterministic same-fold, same-diagnosis cue-absent null control. The
script does not select thresholds, write targets, or accept a test manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.score_ddxplus_selected_changed_cues import (
    activation_path,
    donor_indices,
    fold_of,
    load_deletion_rows,
    load_original_rows,
    parse_path_maps,
    predict_probabilities,
    quantiles,
)
from src.jsonl import write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_features(
    rows: list[dict[str, Any]], path_maps: list[tuple[str, str]]
) -> torch.Tensor:
    vectors = [
        torch.load(
            activation_path(row, path_maps), map_location="cpu", weights_only=True
        )
        .flatten()
        .float()
        for row in rows
    ]
    shapes = {tuple(vector.shape) for vector in vectors}
    if len(shapes) != 1:
        raise ValueError(f"Inconsistent activation shapes: {sorted(shapes)}")
    return torch.stack(vectors)


def score_triplet(
    *,
    row_index: int,
    changed: str,
    label_offset: int,
    originals: list[dict[str, Any]],
    probabilities: torch.Tensor,
    deletion_probabilities: torch.Tensor,
    max_donors: int,
) -> dict[str, Any]:
    donors = donor_indices(
        row_index, originals, changed_evidence=changed, maximum=max_donors
    )
    p_original = float(probabilities[row_index, label_offset])
    p_deleted = float(deletion_probabilities[row_index, label_offset])
    donor_values = [float(probabilities[index, label_offset]) for index in donors]
    donor_mean = float(np.mean(donor_values)) if donor_values else None
    return {
        "p_original": p_original,
        "p_deleted": p_deleted,
        "deletion_delta": p_original - p_deleted,
        "donor_ids": [str(originals[index].get("base_id")) for index in donors],
        "donor_count": len(donors),
        "donor_mean_probability": donor_mean,
        "donor_margin": p_original - donor_mean if donor_mean is not None else None,
    }


def build_validation_scores(
    *,
    manifest: Path,
    probe_artifact: Path,
    output_jsonl: Path,
    output_json: Path,
    summary_md: Path,
    path_maps: list[tuple[str, str]],
    max_donors: int,
    device: torch.device,
) -> dict[str, Any]:
    artifact = torch.load(probe_artifact, map_location="cpu", weights_only=False)
    if int(artifact.get("layer") or -1) != 32:
        raise ValueError("--probe-artifact must be the frozen HS32 artifact")
    labels = [str(value) for value in artifact["finding_labels"]]
    label_index = {label: index for index, label in enumerate(labels)}

    originals = load_original_rows(manifest)
    deletions = load_deletion_rows(manifest)
    original_ids = {str(row["base_id"]) for row in originals}
    if set(deletions) != original_ids:
        raise ValueError(
            "Validation original/deletion mismatch: "
            f"missing={len(original_ids - set(deletions))} "
            f"extra={len(set(deletions) - original_ids)}"
        )
    original_features = load_features(originals, path_maps)
    deletion_features = load_features(
        [deletions[str(row["base_id"])] for row in originals], path_maps
    )
    mean = artifact["feature_mean"].float()
    std = artifact["feature_std"].float().clamp_min(1e-6)
    probabilities = predict_probabilities(
        artifact["finding_state_dict"], (original_features - mean) / std, device
    )
    deletion_probabilities = predict_probabilities(
        artifact["finding_state_dict"], (deletion_features - mean) / std, device
    )

    output_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    positive_deltas: list[float] = []
    positive_margins: list[float] = []
    positive_presence: list[float] = []
    null_deltas: list[float] = []
    null_margins: list[float] = []
    null_presence: list[float] = []
    for own_index, original in enumerate(originals):
        identifier = str(original["base_id"])
        deletion = deletions[identifier]
        changed = str(deletion["cf_original_evidence_id"])
        label_offset = label_index.get(changed)
        if label_offset is None:
            counts["changed_cue_outside_ontology"] += 1
            continue
        positive = score_triplet(
            row_index=own_index,
            changed=changed,
            label_offset=label_offset,
            originals=originals,
            probabilities=probabilities,
            deletion_probabilities=deletion_probabilities,
            max_donors=max_donors,
        )
        positive_row = {
            "control_type": "selected_changed_cue",
            "base_id": identifier,
            "diagnosis_id": original.get("diagnosis_id"),
            "fold": fold_of(identifier),
            "changed_evidence_id": changed,
            "changed_cue_text": deletion.get("cf_original_cue"),
            "score_eligible": positive["donor_count"] > 0,
            **positive,
            "selected_changed_cue_supported": None,
        }
        output_rows.append(positive_row)
        counts["positive_rows"] += 1
        if positive_row["score_eligible"]:
            counts["positive_eligible"] += 1
            positive_presence.append(float(positive_row["p_original"]))
            positive_deltas.append(float(positive_row["deletion_delta"]))
            positive_margins.append(float(positive_row["donor_margin"]))

        # One deterministic 1:1 null. The candidate cue is absent from this
        # case; its own deletion removes an unrelated cue. Passing all three
        # gates here is therefore a false support event.
        null_candidates = donor_indices(
            own_index, originals, changed_evidence=changed, maximum=max_donors
        )
        if not null_candidates:
            counts["null_unavailable"] += 1
            continue
        null_index = null_candidates[0]
        null_triplet = score_triplet(
            row_index=null_index,
            changed=changed,
            label_offset=label_offset,
            originals=originals,
            probabilities=probabilities,
            deletion_probabilities=deletion_probabilities,
            max_donors=max_donors,
        )
        null_identifier = str(originals[null_index]["base_id"])
        null_row = {
            "control_type": "cue_absent_null",
            "base_id": null_identifier,
            "source_positive_base_id": identifier,
            "diagnosis_id": originals[null_index].get("diagnosis_id"),
            "fold": fold_of(null_identifier),
            "changed_evidence_id": changed,
            "changed_cue_text": deletion.get("cf_original_cue"),
            "score_eligible": null_triplet["donor_count"] > 0,
            **null_triplet,
            "selected_changed_cue_supported": None,
        }
        output_rows.append(null_row)
        counts["null_rows"] += 1
        if null_row["score_eligible"]:
            counts["null_eligible"] += 1
            null_presence.append(float(null_row["p_original"]))
            null_deltas.append(float(null_row["deletion_delta"]))
            null_margins.append(float(null_row["donor_margin"]))

    report = {
        "schema_version": 1,
        "scope": "DDXPlus validation selected changed cue plus 1:1 cue-absent null",
        "n_validation_cases": len(originals),
        "finding_labels": len(labels),
        "max_donors": max_donors,
        "counts": dict(sorted(counts.items())),
        "positive_presence": quantiles(positive_presence),
        "positive_deletion_delta": quantiles(positive_deltas),
        "positive_donor_margin": quantiles(positive_margins),
        "null_presence": quantiles(null_presence),
        "null_deletion_delta": quantiles(null_deltas),
        "null_donor_margin": quantiles(null_margins),
        "threshold_applied": False,
        "target_written": False,
        "locked_test_read": False,
        "inputs": {
            "manifest": str(manifest),
            "manifest_sha256": sha256_file(manifest),
            "probe_artifact": str(probe_artifact),
            "probe_artifact_sha256": sha256_file(probe_artifact),
        },
    }
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, output_rows)
    output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "\n".join(
            [
                "# DDXPlus D9a Validation Null Audit",
                "",
                "Frozen HS32 probe; no threshold selection and no target writing.",
                "",
                f"- validation cases: **{len(originals)}**",
                f"- positive eligible: **{counts['positive_eligible']}/{counts['positive_rows']}**",
                f"- null eligible: **{counts['null_eligible']}/{counts['null_rows']}**",
                "- null definition: candidate cue absent; paired deletion changes another cue",
                "- threshold applied: **no**",
                "- locked test read: **no**",
                "",
                "The score distributions must be inspected before an explicit candidate grid is supplied.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--probe-artifact", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument("--max-donors", type=int, default=5)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()
    if args.max_donors != 5:
        raise ValueError("D11 freezes --max-donors=5")
    report = build_validation_scores(
        manifest=args.manifest,
        probe_artifact=args.probe_artifact,
        output_jsonl=args.output_jsonl,
        output_json=args.output_json,
        summary_md=args.summary_md,
        path_maps=parse_path_maps(args.path_map),
        max_donors=args.max_donors,
        device=torch.device(args.device),
    )
    print(
        f"[validation] cases={report['n_validation_cases']} "
        f"positive_eligible={report['counts'].get('positive_eligible', 0)} "
        f"null_eligible={report['counts'].get('null_eligible', 0)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
