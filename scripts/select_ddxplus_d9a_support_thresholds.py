"""Evaluate an explicitly supplied D9a support-cut grid on validation.

The candidate grid has no defaults: it may be supplied only after inspection
of the read-only train and validation score distributions. The selected row is
written as an unapproved recommendation. It cannot be consumed by the D9a
target builder until a human explicitly sets ``human_approved`` to true.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def passes(row: dict[str, Any], presence: float, deletion: float, donor: float) -> bool:
    return bool(
        row.get("score_eligible")
        and row.get("p_original") is not None
        and row.get("deletion_delta") is not None
        and row.get("donor_margin") is not None
        and float(row["p_original"]) >= presence
        and float(row["deletion_delta"]) >= deletion
        and float(row["donor_margin"]) >= donor
    )


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def evaluate_grid(
    rows: list[dict[str, Any]],
    *,
    presence_thresholds: list[float],
    deletion_thresholds: list[float],
    donor_thresholds: list[float],
    max_false_support_rate: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    positives = [row for row in rows if row.get("control_type") == "selected_changed_cue"]
    nulls = [row for row in rows if row.get("control_type") == "cue_absent_null"]
    eligible_positives = [row for row in positives if row.get("score_eligible")]
    eligible_nulls = [row for row in nulls if row.get("score_eligible")]
    if not eligible_positives or not eligible_nulls:
        raise ValueError("Need nonempty eligible positive and null validation rows")

    candidates = []
    for presence, deletion, donor in product(
        sorted(set(presence_thresholds)),
        sorted(set(deletion_thresholds)),
        sorted(set(donor_thresholds)),
    ):
        positive_hits = sum(passes(row, presence, deletion, donor) for row in eligible_positives)
        false_hits = sum(passes(row, presence, deletion, donor) for row in eligible_nulls)
        false_rate = false_hits / len(eligible_nulls)
        candidates.append(
            {
                "presence_threshold": presence,
                "deletion_delta_threshold": deletion,
                "donor_margin_threshold": donor,
                "positive_supported": positive_hits,
                "positive_eligible": len(eligible_positives),
                "positive_coverage": positive_hits / len(eligible_positives),
                "null_false_supported": false_hits,
                "null_eligible": len(eligible_nulls),
                "false_support_rate": false_rate,
                "meets_false_support_cap": false_rate <= max_false_support_rate,
            }
        )
    feasible = [row for row in candidates if row["meets_false_support_cap"]]
    if not feasible:
        raise ValueError(
            "No candidate satisfies the frozen false-support cap; do not loosen it "
            "without a new human-approved decision entry"
        )
    # Maximize coverage. Remaining ties choose lower empirical false support,
    # then the stricter aggregate threshold, then lexicographically stricter
    # individual thresholds. No downstream smoke metric enters this ordering.
    selected = max(
        feasible,
        key=lambda row: (
            row["positive_coverage"],
            -row["false_support_rate"],
            row["presence_threshold"]
            + row["deletion_delta_threshold"]
            + row["donor_margin_threshold"],
            row["presence_threshold"],
            row["deletion_delta_threshold"],
            row["donor_margin_threshold"],
        ),
    )
    population = {
        "positive_all": len(positives),
        "positive_eligible": len(eligible_positives),
        "positive_eligibility_rate": safe_rate(len(eligible_positives), len(positives)),
        "null_all": len(nulls),
        "null_eligible": len(eligible_nulls),
        "null_eligibility_rate": safe_rate(len(eligible_nulls), len(nulls)),
    }
    return candidates, {**selected, "population": population}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-scores", required=True, type=Path)
    parser.add_argument("--presence-thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--deletion-thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--donor-thresholds", required=True, nargs="+", type=float)
    parser.add_argument("--max-false-support-rate", type=float, default=0.05)
    parser.add_argument("--min-fold-positive-count", type=int, default=5)
    parser.add_argument("--candidates-jsonl", required=True, type=Path)
    parser.add_argument("--recommendation-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    if args.min_fold_positive_count != 5:
        raise ValueError("D11 freezes --min-fold-positive-count=5")
    if args.max_false_support_rate != 0.05:
        raise ValueError("D11 freezes --max-false-support-rate=0.05")
    for name, values in (
        ("presence", args.presence_thresholds),
        ("deletion", args.deletion_thresholds),
        ("donor", args.donor_thresholds),
    ):
        if not values or any(not -1.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid explicit {name} threshold grid")

    rows = list(read_jsonl(args.validation_scores))
    candidates, selected = evaluate_grid(
        rows,
        presence_thresholds=args.presence_thresholds,
        deletion_thresholds=args.deletion_thresholds,
        donor_thresholds=args.donor_thresholds,
        max_false_support_rate=args.max_false_support_rate,
    )
    args.candidates_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.candidates_jsonl, candidates)
    recommendation = {
        "schema_version": 1,
        "decision": "D9a/D11 selected changed-cue support cuts",
        "selection_rule": "false-support <= 0.05, then maximum positive coverage",
        "validation_scores": str(args.validation_scores),
        "validation_scores_sha256": sha256_file(args.validation_scores),
        "max_false_support_rate": args.max_false_support_rate,
        "min_fold_positive_count": args.min_fold_positive_count,
        "max_donors": 5,
        "selected": selected,
        "candidate_grid": {
            "presence_thresholds": sorted(set(args.presence_thresholds)),
            "deletion_thresholds": sorted(set(args.deletion_thresholds)),
            "donor_thresholds": sorted(set(args.donor_thresholds)),
        },
        "human_approved": False,
        "target_written": False,
        "locked_test_read": False,
    }
    args.recommendation_json.write_text(
        json.dumps(recommendation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_md.write_text(
        "\n".join(
            [
                "# DDXPlus D9a Support-Cut Recommendation",
                "",
                "Validation-only recommendation. It is not an approved training protocol.",
                "",
                f"- candidate combinations: **{len(candidates)}**",
                f"- positive eligible: **{selected['population']['positive_eligible']}**",
                f"- null eligible: **{selected['population']['null_eligible']}**",
                f"- presence threshold: **{selected['presence_threshold']:.4f}**",
                f"- deletion-delta threshold: **{selected['deletion_delta_threshold']:.4f}**",
                f"- donor-margin threshold: **{selected['donor_margin_threshold']:.4f}**",
                f"- positive coverage: **{selected['positive_coverage']:.4f}**",
                f"- null false-support rate: **{selected['false_support_rate']:.4f}**",
                "- human approved: **no**",
                "- target written: **no**",
                "- locked test read: **no**",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(
        f"[cuts] candidates={len(candidates)} coverage={selected['positive_coverage']:.4f} "
        f"false_support={selected['false_support_rate']:.4f} approved=false",
        flush=True,
    )


if __name__ == "__main__":
    main()
