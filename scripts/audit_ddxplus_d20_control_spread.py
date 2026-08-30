"""Audit D20 controls and emit the approved same-seed gate proposal.

This script is read-only. It never approves a protocol or reads locked test data.
Across-seed spread remains a diagnostic. It is not used as a non-inferiority
allowance because the observed seed spread makes that rule non-informative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.summarize_ddxplus_d10_arms import metrics_by_id
from scripts.score_medical_nla_v2_readouts import extract_tag
from src.cue_readout_scoring import observed_items
from src.jsonl import read_jsonl
from src.nla_text import extract_explanation

SEEDS = (17, 29, 43)


def parse_seed_path(value: str) -> tuple[int, Path]:
    seed_text, separator, path_text = value.partition("=")
    if not separator or int(seed_text) not in SEEDS or not path_text:
        raise argparse.ArgumentTypeError("Expected 17|29|43=PATH")
    return int(seed_text), Path(path_text)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def seed_means(path: Path) -> dict[str, float]:
    rows = metrics_by_id(path)
    if not rows:
        raise ValueError(f"No control rows in {path}")
    return {
        "n": len(rows),
        "retained_gap": mean(row["retained_gap"] for row in rows.values()),
        "changed_original_nll": mean(
            row["changed_original_nll"] for row in rows.values()
        ),
        "retained_original_nll": mean(
            row["retained_original_nll"] for row in rows.values()
        ),
    }


def readout_claim_mean(path: Path) -> tuple[int, float]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No control readouts in {path}")
    counts = []
    for row in rows:
        text = str(row.get("nla_output") or row.get("response") or "")
        explanation, _parsed = extract_explanation(text)
        observed, parsed_observed = extract_tag(explanation, "observed")
        counts.append(len(observed_items(observed if parsed_observed else explanation)))
    return len(rows), mean(counts)


def spread(values: list[float]) -> float:
    return max(values) - min(values)


def recommendation(
    score_paths: dict[int, Path], readout_paths: dict[int, Path] | None = None
) -> dict[str, Any]:
    if set(score_paths) != set(SEEDS):
        raise ValueError("Exactly three D10 control score paths are required")
    score_metrics = {seed: seed_means(score_paths[seed]) for seed in SEEDS}
    n_values = {item["n"] for item in score_metrics.values()}
    if len(n_values) != 1:
        raise ValueError("Control score populations differ across seeds")

    retained_spread = spread(
        [score_metrics[seed]["retained_gap"] for seed in SEEDS]
    )
    changed_nll_spread = spread(
        [score_metrics[seed]["changed_original_nll"] for seed in SEEDS]
    )
    retained_nll_spread = spread(
        [score_metrics[seed]["retained_original_nll"] for seed in SEEDS]
    )
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "unapproved_recommendation",
        "human_approved": False,
        "locked_test_read": False,
        "control_scores": {
            str(seed): {
                "path": str(score_paths[seed]),
                "sha256": sha256_file(score_paths[seed]),
                **score_metrics[seed],
            }
            for seed in SEEDS
        },
        "rejected_spread_rule": "max(2 * three-seed control range, absolute floor)",
        "rejected_absolute_floors": {
            "retained_gap_delta_max": 0.01,
            "changed_original_nll_delta_max": 0.05,
            "retained_original_nll_delta_max": 0.05,
            "mean_claim_relative_drop_max": 0.10,
        },
        "control_spreads": {
            "retained_gap": retained_spread,
            "changed_original_nll": changed_nll_spread,
            "retained_original_nll": retained_nll_spread,
        },
        "rejected_across_seed_allowances": {
            "retained_gap_delta_max": max(2 * retained_spread, 0.01),
            "changed_original_nll_delta_max": max(2 * changed_nll_spread, 0.05),
            "retained_original_nll_delta_max": max(2 * retained_nll_spread, 0.05),
            "mean_claim_relative_drop_max": None,
        },
        "proposed_same_seed_gates": {
            "retained_gap_delta_max": 0.01,
            "changed_original_nll_relative_increase_max": 0.10,
            "retained_original_nll_relative_increase_max": 0.10,
            "mean_claim_relative_drop_max": 0.10,
        },
    }

    if readout_paths is not None:
        if set(readout_paths) != set(SEEDS):
            raise ValueError("Control readouts must be supplied for all three seeds")
        readouts = {}
        means = []
        row_counts = set()
        for seed in SEEDS:
            n_rows, mean_claims = readout_claim_mean(readout_paths[seed])
            row_counts.add(n_rows)
            means.append(mean_claims)
            readouts[str(seed)] = {
                "path": str(readout_paths[seed]),
                "sha256": sha256_file(readout_paths[seed]),
                "n": n_rows,
                "mean_claims": mean_claims,
            }
        if len(row_counts) != 1:
            raise ValueError("Control readout populations differ across seeds")
        mean_level = mean(means)
        if mean_level <= 0:
            raise ValueError("Control readouts emitted no claims")
        relative_spread = spread(means) / mean_level
        result["control_readouts"] = readouts
        result["control_spreads"]["mean_claims_relative"] = relative_spread
        result["rejected_across_seed_allowances"]["mean_claim_relative_drop_max"] = max(
            2 * relative_spread, 0.10
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-score", action="append", required=True, type=parse_seed_path)
    parser.add_argument("--control-readout", action="append", type=parse_seed_path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    score_paths = dict(args.control_score)
    readout_paths = dict(args.control_readout) if args.control_readout else None
    report = recommendation(score_paths, readout_paths)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# D20 Control-Spread Audit",
        "",
        "Read-only recommendation; it does not authorize D20 training.",
        "",
        "| metric | seed 17 | seed 29 | seed 43 | range | rejected 2x-range allowance |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    rows = (
        ("retained gap", "retained_gap", "retained_gap_delta_max"),
        ("changed original NLL", "changed_original_nll", "changed_original_nll_delta_max"),
        ("retained original NLL", "retained_original_nll", "retained_original_nll_delta_max"),
    )
    for label, metric, gate in rows:
        values = [report["control_scores"][str(seed)][metric] for seed in SEEDS]
        lines.append(
            f"| {label} | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
            f"{report['control_spreads'][metric]:.6f} | "
            f"{report['rejected_across_seed_allowances'][gate]:.6f} |"
        )
    if "control_readouts" in report:
        values = [
            report["control_readouts"][str(seed)]["mean_claims"] for seed in SEEDS
        ]
        rejected_claim_drop = report["rejected_across_seed_allowances"][
            "mean_claim_relative_drop_max"
        ]
        lines.append(
            f"| mean claims | {values[0]:.6f} | {values[1]:.6f} | {values[2]:.6f} | "
            f"{report['control_spreads']['mean_claims_relative']:.6f} relative | "
            f"{rejected_claim_drop:.6f} relative |"
        )
    else:
        lines.extend(
            [
                "",
                "- generation claim-count allowance: **pending three control readouts**",
            ]
        )
    lines.extend(
        [
            "",
            "The 2x-range allowances are reported but rejected: the observed seed "
            "spread would permit severe degradation, including a 75% claim-count drop.",
            "",
            "Proposed same-seed paired gates: retained-gap delta <= .01; changed and "
            "retained original NLL <= 1.10x the same-seed control; mean claims >= "
            "0.90x the same-seed control. Human approval is still required.",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
