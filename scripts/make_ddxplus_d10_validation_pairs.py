"""Build the frozen D10 validation pairs from approved D9a support cuts.

Only validation selected-changed-cue rows are consumed. Cue-absent null rows
remain threshold-selection controls and never become evaluation targets.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.make_ddxplus_d9a_supported_pairs import (
    clean,
    load_approved_protocol,
    one_cue_target,
    select_retained_cue,
    sha256_file,
)
from scripts.score_ddxplus_selected_changed_cues import (
    load_deletion_rows,
    load_original_rows,
)
from scripts.select_ddxplus_d9a_support_thresholds import passes
from src.jsonl import read_jsonl, write_jsonl


def build_validation_pairs(
    *,
    validation_scores: Path,
    validation_manifest: Path,
    approved_protocol: Path,
    output_jsonl: Path,
    report_json: Path,
    summary_md: Path,
) -> dict[str, Any]:
    approval = load_approved_protocol(approved_protocol)
    observed_hash = sha256_file(validation_scores)
    if observed_hash != str(approval.get("validation_scores_sha256") or ""):
        raise ValueError("Approved protocol does not match validation score SHA256")
    cuts = approval["selected"]
    thresholds = (
        float(cuts["presence_threshold"]),
        float(cuts["deletion_delta_threshold"]),
        float(cuts["donor_margin_threshold"]),
    )

    positive_rows = [
        row
        for row in read_jsonl(validation_scores)
        if row.get("control_type") == "selected_changed_cue"
    ]
    score_by_id = {str(row.get("base_id") or ""): row for row in positive_rows}
    if "" in score_by_id or len(score_by_id) != len(positive_rows):
        raise ValueError("Validation positive scores have missing or duplicate base_id")
    originals = {
        str(row["base_id"]): row for row in load_original_rows(validation_manifest)
    }
    deletions = load_deletion_rows(validation_manifest)
    if set(originals) != set(deletions):
        raise ValueError("Validation original/deletion population mismatch")
    if not set(score_by_id).issubset(originals):
        raise ValueError("Validation scores contain IDs outside the manifest")

    output_rows = []
    counts: Counter[str] = Counter()
    diagnosis_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for identifier in sorted(originals):
        diagnosis = clean(originals[identifier].get("diagnosis_id")) or "<missing>"
        score = score_by_id.get(identifier)
        if score is None:
            counts["outside_ontology_excluded"] += 1
            diagnosis_counts[diagnosis]["outside_ontology_excluded"] += 1
            continue
        if not score.get("score_eligible") or int(score.get("donor_count") or 0) <= 0:
            counts["donor_unavailable_excluded"] += 1
            diagnosis_counts[diagnosis]["donor_unavailable_excluded"] += 1
            continue
        if not passes(score, *thresholds):
            counts["below_cut_excluded"] += 1
            diagnosis_counts[diagnosis]["below_cut_excluded"] += 1
            continue

        original = originals[identifier]
        deleted = deletions[identifier]
        changed = str(deleted.get("cf_original_evidence_id") or "")
        if changed != str(score.get("changed_evidence_id") or ""):
            raise ValueError(f"Changed-cue mismatch for {identifier}")
        claim = clean(deleted.get("cf_original_cue"))
        if not claim:
            raise ValueError(f"Missing changed-cue text for {identifier}")
        original_activation = Path(str(original.get("activation_path") or ""))
        deleted_activation = Path(str(deleted.get("activation_path") or ""))
        if not original_activation.is_file() or not deleted_activation.is_file():
            raise FileNotFoundError(
                original_activation
                if not original_activation.is_file()
                else deleted_activation
            )
        retained_id, retained_cue, retained_hash = select_retained_cue(
            identifier=identifier, original=original, deleted=deleted
        )
        output_rows.append(
            {
                "id": f"{identifier}__d10_validation_pair",
                "base_id": identifier,
                "source_dataset": "ddxplus",
                "diagnosis_id": original.get("diagnosis_id"),
                "position_family": "P0",
                "layer": 32,
                "changed_evidence_id": changed,
                "selected_changed_cue_text": claim,
                "retained_evidence_id": retained_id,
                "retained_cue_text": retained_cue,
                "retained_cue_sha256": retained_hash,
                "retained_cue_selection_rule": (
                    "minimum sha256(base_id + NUL + exact retained cue text)"
                ),
                "original_activation_path": str(original_activation),
                "deleted_activation_path": str(deleted_activation),
                "target_text": one_cue_target(original, claim, "changed_target"),
                "retained_target_text": one_cue_target(
                    original, retained_cue, "retained_target"
                ),
                "support_scores": {
                    "p_original": score["p_original"],
                    "deletion_delta": score["deletion_delta"],
                    "donor_margin": score["donor_margin"],
                },
            }
        )
        counts["supported_pairs"] += 1
        diagnosis_counts[diagnosis]["supported_pairs"] += 1

    if not output_rows:
        raise ValueError("Approved support cuts retained no validation pairs")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, output_rows)
    report = {
        "schema_version": 1,
        "scope": "D10 validation selected changed cue only",
        "counts": dict(sorted(counts.items())),
        "cuts": {
            "presence_threshold": thresholds[0],
            "deletion_delta_threshold": thresholds[1],
            "donor_margin_threshold": thresholds[2],
        },
        "retained_cue_selection_rule": (
            "minimum sha256(base_id + NUL + exact retained cue text)"
        ),
        "validation_scores_sha256": observed_hash,
        "approved_protocol_sha256": sha256_file(approved_protocol),
        "locked_test_read": False,
        "disposition_by_diagnosis": {
            diagnosis: dict(sorted(values.items()))
            for diagnosis, values in sorted(diagnosis_counts.items())
        },
    }
    report_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# DDXPlus D10 Frozen Validation Pairs",
        "",
        f"- supported pairs: **{counts['supported_pairs']}**",
        f"- donor unavailable excluded: **{counts['donor_unavailable_excluded']}**",
        f"- below cut excluded: **{counts['below_cut_excluded']}**",
        f"- outside ontology excluded: **{counts['outside_ontology_excluded']}**",
        "- locked test read: **no**",
        "",
        "| diagnosis | retained | donor unavailable | below cut | outside ontology |",
        "|---|---:|---:|---:|---:|",
    ]
    for diagnosis, values in sorted(diagnosis_counts.items()):
        lines.append(
            f"| {diagnosis} | {values['supported_pairs']} | "
            f"{values['donor_unavailable_excluded']} | "
            f"{values['below_cut_excluded']} | "
            f"{values['outside_ontology_excluded']} |"
        )
    summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
    parser.add_argument("--approved-protocol", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--report-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    report = build_validation_pairs(
        validation_scores=args.validation_scores,
        validation_manifest=args.validation_manifest,
        approved_protocol=args.approved_protocol,
        output_jsonl=args.output_jsonl,
        report_json=args.report_json,
        summary_md=args.summary_md,
    )
    print(f"[validation pairs] {report['counts']['supported_pairs']}", flush=True)


if __name__ == "__main__":
    main()
