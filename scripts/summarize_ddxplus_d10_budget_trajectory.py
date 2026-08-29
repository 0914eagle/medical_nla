"""Summarize the preregistered D10 budget-calibration checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


FROZEN_STEPS = (20, 194, 388, 776, 1164, 1552)
SEEDS = (17, 29, 43)


def parse_comparison(value: str) -> tuple[int, Path]:
    step, separator, path = value.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("--comparison must be STEP=PATH")
    return int(step), Path(path)


def build_report(comparisons: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if tuple(sorted(comparisons)) != FROZEN_STEPS:
        raise ValueError(f"Expected frozen steps {FROZEN_STEPS}")
    trajectory = []
    for step in FROZEN_STEPS:
        comparison = comparisons[step]
        for seed in SEEDS:
            item = comparison["results"][str(seed)]
            trajectory.append(
                {
                    "step": step,
                    "epochs": step / 776,
                    "seed": seed,
                    "changed_gap_delta": item["deltas"]["changed_gap"][
                        "ranking_minus_control"
                    ],
                    "retained_gap_delta": item["deltas"]["retained_gap"][
                        "ranking_minus_control"
                    ],
                    "specificity_delta": item["deltas"]["specificity"][
                        "ranking_minus_control"
                    ],
                    "changed_gap_cluster_ci": item["deltas"]["changed_gap"][
                        "diagnosis_cluster_bootstrap_95_ci"
                    ],
                    "specificity_cluster_ci": item["deltas"]["specificity"][
                        "diagnosis_cluster_bootstrap_95_ci"
                    ],
                }
            )
    final = comparisons[FROZEN_STEPS[-1]]
    return {
        "schema_version": 1,
        "status": "post_hoc_exploratory_budget_calibration",
        "frozen_steps": list(FROZEN_STEPS),
        "steps_per_epoch": 776,
        "trajectory": trajectory,
        "final_gate": final["gate"],
        "locked_test_read": False,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# DDXPlus D10 Budget-Calibration Trajectory",
        "",
        "Human-approved post-hoc exploratory calibration. Intermediate checkpoints are",
        "report-only; promotion is decided at step 1,552 with the frozen D5 gate.",
        "Locked DDXPlus test was not read.",
        "",
        "| step | epochs | seed | changed-gap delta | cluster 95% CI | retained-gap delta | specificity delta | specificity cluster 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["trajectory"]:
        changed_ci = row["changed_gap_cluster_ci"]
        specificity_ci = row["specificity_cluster_ci"]
        lines.append(
            f"| {row['step']} | {row['epochs']:.2f} | {row['seed']} | "
            f"{row['changed_gap_delta']:+.4f} | "
            f"[{changed_ci[0]:+.4f}, {changed_ci[1]:+.4f}] | "
            f"{row['retained_gap_delta']:+.4f} | "
            f"{row['specificity_delta']:+.4f} | "
            f"[{specificity_ci[0]:+.4f}, {specificity_ci[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Across-Seed Means",
            "",
            "| step | changed-gap delta | retained-gap delta | specificity delta |",
            "|---:|---:|---:|---:|",
        ]
    )
    for step in report["frozen_steps"]:
        rows = [row for row in report["trajectory"] if row["step"] == step]
        lines.append(
            f"| {step} | {mean(row['changed_gap_delta'] for row in rows):+.4f} | "
            f"{mean(row['retained_gap_delta'] for row in rows):+.4f} | "
            f"{mean(row['specificity_delta'] for row in rows):+.4f} |"
        )
    gate = report["final_gate"]
    lines.extend(
        [
            "",
            "## Final Frozen Gate",
            "",
            f"- three-seed changed sign: **{gate['three_seed_changed_sign_consistent']}**",
            f"- changed delta >= .05 in every seed: **{gate['changed_delta_min_0p05_each_seed']}**",
            f"- changed-gap cluster CI > 0 in every seed: **{gate['changed_cluster_ci_above_zero_each_seed']}**",
            f"- three-seed specificity sign: **{gate['three_seed_specificity_sign_consistent']}**",
            f"- specificity cluster CI > 0 in every seed: **{gate['specificity_cluster_ci_above_zero_each_seed']}**",
            f"- teacher-forced gate: **{'PASS' if gate['teacher_forced_gate_passed'] else 'FAIL'}**",
            "",
            "No automatic extension beyond step 1,552 is authorized.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison", action="append", required=True, type=parse_comparison
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    paths = dict(args.comparison)
    if len(paths) != len(args.comparison):
        raise ValueError("Duplicate checkpoint comparison")
    comparisons = {
        step: json.loads(path.read_text(encoding="utf-8"))
        for step, path in paths.items()
    }
    report = build_report(comparisons)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.summary_md.write_text(render_markdown(report), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
