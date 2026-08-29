"""Build D9a selected-cue pairs from a human-approved support protocol.

Unsupported and untested cues are never written as negative or abstention
targets. Each retained case contributes exactly one selected changed-cue claim,
its original activation, and its paired cue-deleted activation. This dataset is
only a mechanism-smoke input; it is not a multi-claim Medical-NLA corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.make_medical_nla_v3_cue_first_targets import cue_first_target_text
from scripts.score_ddxplus_selected_changed_cues import (
    load_deletion_rows,
    load_original_rows,
)
from scripts.select_ddxplus_d9a_support_thresholds import passes
from src.jsonl import read_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_approved_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if protocol.get("human_approved") is not True:
        raise ValueError("Support protocol is not human-approved")
    if not str(protocol.get("approved_by") or "").strip():
        raise ValueError("Approved protocol must record approved_by")
    if not str(protocol.get("approved_at") or "").strip():
        raise ValueError("Approved protocol must record approved_at")
    if float(protocol.get("max_false_support_rate", -1)) != 0.05:
        raise ValueError("D11 freezes max_false_support_rate=0.05")
    if int(protocol.get("min_fold_positive_count", -1)) != 5:
        raise ValueError("D11 freezes min_fold_positive_count=5")
    if int(protocol.get("max_donors", -1)) != 5:
        raise ValueError("D11 freezes max_donors=5")
    selected = protocol.get("selected") or {}
    if not selected.get("meets_false_support_cap"):
        raise ValueError("Approved cut does not satisfy the frozen false-support cap")
    if float(selected.get("false_support_rate", 1.0)) > 0.05:
        raise ValueError("Approved cut exceeds false-support rate 0.05")
    return protocol


def build_pairs(
    *,
    train_scores: Path,
    validation_scores: Path,
    original_manifest: Path,
    counterfactual_manifest: Path,
    approved_protocol: Path,
    output_jsonl: Path,
    protocol_json: Path,
    summary_md: Path,
) -> dict[str, Any]:
    approval = load_approved_protocol(approved_protocol)
    expected_validation_hash = str(approval.get("validation_scores_sha256") or "")
    observed_validation_hash = sha256_file(validation_scores)
    if not expected_validation_hash or observed_validation_hash != expected_validation_hash:
        raise ValueError("Approved protocol does not match --validation-scores SHA256")
    cuts = approval["selected"]
    presence = float(cuts["presence_threshold"])
    deletion = float(cuts["deletion_delta_threshold"])
    donor = float(cuts["donor_margin_threshold"])

    score_rows = list(read_jsonl(train_scores))
    score_by_id = {str(row.get("base_id") or ""): row for row in score_rows}
    if "" in score_by_id or len(score_by_id) != len(score_rows):
        raise ValueError("Train score table has missing or duplicate base_id")
    originals = {str(row["base_id"]): row for row in load_original_rows(original_manifest)}
    deletions = load_deletion_rows(counterfactual_manifest)
    if set(score_by_id) != set(originals) or set(originals) != set(deletions):
        raise ValueError(
            "Score/original/deletion population mismatch: "
            f"scores={len(score_by_id)} original={len(originals)} deletion={len(deletions)}"
        )

    rows_out = []
    counts: Counter[str] = Counter()
    for identifier in sorted(originals):
        score = score_by_id[identifier]
        if not score.get("score_eligible"):
            counts["ineligible_excluded"] += 1
            continue
        if int(score.get("fold_training_positive_count") or 0) < 5:
            raise ValueError(f"Eligible row violates min fold positives: {identifier}")
        if int(score.get("donor_count") or 0) <= 0:
            raise ValueError(f"Eligible row has no donor: {identifier}")
        if not passes(score, presence, deletion, donor):
            counts["eligible_below_cut_excluded"] += 1
            continue
        original = originals[identifier]
        deleted = deletions[identifier]
        changed = str(deleted.get("cf_original_evidence_id") or "")
        if changed != str(score.get("changed_evidence_id") or ""):
            raise ValueError(f"Changed-cue mismatch for {identifier}")
        claim = " ".join(str(deleted.get("cf_original_cue") or "").split())
        if not claim:
            raise ValueError(f"Missing selected changed-cue text for {identifier}")
        original_activation = Path(str(original.get("activation_path") or ""))
        deleted_activation = Path(str(deleted.get("activation_path") or ""))
        if not original_activation.is_file() or not deleted_activation.is_file():
            raise FileNotFoundError(
                original_activation if not original_activation.is_file() else deleted_activation
            )
        target_row = {
            **original,
            "id": f"{identifier}__d9a_selected_changed_cue",
            "cue_targets": [claim],
        }
        rows_out.append(
            {
                "id": target_row["id"],
                "base_id": identifier,
                "source_dataset": "ddxplus",
                "diagnosis_id": original.get("diagnosis_id"),
                "position_family": "P0",
                "layer": 32,
                "changed_evidence_id": changed,
                "selected_changed_cue_text": claim,
                "original_activation_path": str(original_activation),
                "deleted_activation_path": str(deleted_activation),
                "target_text": cue_first_target_text(
                    target_row,
                    max_cues=1,
                    seed=17,
                    include_assessment=False,
                    cue_order="source",
                ),
                "support_scores": {
                    "p_original": score["p_original"],
                    "deletion_delta": score["deletion_delta"],
                    "donor_margin": score["donor_margin"],
                },
                "selected_changed_cue_supported": True,
                "all_other_cues_support_status": "untested",
                "abstention_target": False,
                "value_edit_included": False,
            }
        )
        counts["supported_pairs"] += 1

    if not rows_out:
        raise ValueError("Approved support cuts retained no D9a pairs")
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_jsonl, rows_out)
    report = {
        "schema_version": 1,
        "scope": "D9a selected changed cue only",
        "counts": dict(sorted(counts.items())),
        "cuts": {
            "presence_threshold": presence,
            "deletion_delta_threshold": deletion,
            "donor_margin_threshold": donor,
        },
        "approved_protocol": str(approved_protocol),
        "approved_protocol_sha256": sha256_file(approved_protocol),
        "train_scores": str(train_scores),
        "train_scores_sha256": sha256_file(train_scores),
        "validation_scores": str(validation_scores),
        "validation_scores_sha256": observed_validation_hash,
        "unsupported_policy": "exclude; never abstention",
        "target_claims_per_case": 1,
        "value_edit_included": False,
        "locked_test_read": False,
    }
    protocol_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary_md.write_text(
        "\n".join(
            [
                "# DDXPlus D9a Approved Selected-Cue Pairs",
                "",
                "Mechanism-smoke dataset; not a multi-claim Medical-NLA corpus.",
                "",
                f"- supported pairs: **{counts['supported_pairs']}**",
                f"- ineligible excluded: **{counts['ineligible_excluded']}**",
                f"- eligible below cut excluded: **{counts['eligible_below_cut_excluded']}**",
                "- target claims per retained case: **1**",
                "- unsupported policy: **exclude**",
                "- abstention targets: **none**",
                "- value-edit arms: **none**",
                "- locked test read: **no**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-scores", required=True, type=Path)
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--original-manifest", required=True, type=Path)
    parser.add_argument("--counterfactual-manifest", required=True, type=Path)
    parser.add_argument("--approved-protocol", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--protocol-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    report = build_pairs(
        train_scores=args.train_scores,
        validation_scores=args.validation_scores,
        original_manifest=args.original_manifest,
        counterfactual_manifest=args.counterfactual_manifest,
        approved_protocol=args.approved_protocol,
        output_jsonl=args.output_jsonl,
        protocol_json=args.protocol_json,
        summary_md=args.summary_md,
    )
    print(f"[pairs] supported={report['counts']['supported_pairs']}", flush=True)


if __name__ == "__main__":
    main()
