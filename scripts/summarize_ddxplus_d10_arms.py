"""Compare D10 ranking and original-only teacher-forced audits by seed."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from scripts.evaluate_ddxplus_d10_specificity import (
    bootstrap_ci,
    clean,
    cluster_bootstrap_ci,
)
from src.jsonl import read_jsonl


def parse_arm(value: str) -> tuple[int, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--arm must be SEED=control|ranking=PATH")
    seed = int(parts[0])
    label = parts[1]
    if seed not in {17, 29, 43} or label not in {"control", "ranking"}:
        raise argparse.ArgumentTypeError("Seeds are 17/29/43; arms are control/ranking")
    return seed, label, Path(parts[2])


def metrics_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    conditions: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        conditions[str(row["condition"])][str(row["base_id"])] = row
    expected = {
        "changed_original",
        "changed_deleted",
        "retained_original",
        "retained_deleted",
    }
    if set(conditions) != expected:
        raise ValueError(f"Incomplete conditions in {path}: {sorted(conditions)}")
    identifiers = set.intersection(*(set(values) for values in conditions.values()))
    if any(set(values) != identifiers for values in conditions.values()):
        raise ValueError(f"Condition populations differ in {path}")
    result = {}
    for identifier in sorted(identifiers):
        changed_gap = (
            conditions["changed_deleted"][identifier]["content_nll"]
            - conditions["changed_original"][identifier]["content_nll"]
        )
        retained_gap = (
            conditions["retained_deleted"][identifier]["content_nll"]
            - conditions["retained_original"][identifier]["content_nll"]
        )
        result[identifier] = {
            "diagnosis_id": conditions["changed_original"][identifier].get("diagnosis_id"),
            "changed_gap": changed_gap,
            "retained_gap": retained_gap,
            "specificity": changed_gap - retained_gap,
            "changed_original_nll": conditions["changed_original"][identifier]["content_nll"],
            "changed_deleted_nll": conditions["changed_deleted"][identifier]["content_nll"],
        }
    return result


def compare_seed(
    control: dict[str, dict[str, Any]],
    ranking: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if set(control) != set(ranking):
        raise ValueError(f"Seed {seed} control/ranking populations differ")
    identifiers = sorted(control)
    result: dict[str, Any] = {"n": len(identifiers), "deltas": {}}
    for offset, metric in enumerate(("changed_gap", "retained_gap", "specificity")):
        deltas = [ranking[item][metric] - control[item][metric] for item in identifiers]
        clustered: dict[str, list[float]] = defaultdict(list)
        for identifier, value in zip(identifiers, deltas, strict=True):
            diagnosis = clean(control[identifier].get("diagnosis_id")) or "<missing>"
            clustered[diagnosis].append(value)
        result["deltas"][metric] = {
            "ranking_minus_control": mean(deltas),
            "row_bootstrap_95_ci": bootstrap_ci(deltas, seed=seed + offset),
            "diagnosis_cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                clustered, seed=seed + offset
            ),
            "ranking_win_rate": sum(value > 0 for value in deltas) / len(deltas),
        }
    original_nll_delta = [
        ranking[item]["changed_original_nll"] - control[item]["changed_original_nll"]
        for item in identifiers
    ]
    result["changed_original_nll_delta"] = mean(original_nll_delta)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    paths: dict[int, dict[str, Path]] = defaultdict(dict)
    for seed, label, path in args.arm:
        if label in paths[seed]:
            raise ValueError(f"Duplicate {seed}/{label}")
        paths[seed][label] = path
    incomplete = any(
        set(items) != {"control", "ranking"} for items in paths.values()
    )
    if set(paths) != {17, 29, 43} or incomplete:
        raise ValueError("D10 comparison requires control/ranking for seeds 17, 29, and 43")

    results = {}
    for seed in (17, 29, 43):
        results[str(seed)] = compare_seed(
            metrics_by_id(paths[seed]["control"]),
            metrics_by_id(paths[seed]["ranking"]),
            seed=seed,
        )
    changed = [
        results[str(seed)]["deltas"]["changed_gap"]["ranking_minus_control"]
        for seed in (17, 29, 43)
    ]
    specificity = [
        results[str(seed)]["deltas"]["specificity"]["ranking_minus_control"]
        for seed in (17, 29, 43)
    ]
    gate: dict[str, Any] = {
        "three_seed_changed_sign_consistent": all(value > 0 for value in changed),
        "three_seed_specificity_sign_consistent": all(value > 0 for value in specificity),
        "changed_delta_min_0p05_each_seed": all(value >= 0.05 for value in changed),
        "changed_cluster_ci_above_zero_each_seed": all(
            results[str(seed)]["deltas"]["changed_gap"][
                "diagnosis_cluster_bootstrap_95_ci"
            ][0]
            > 0
            for seed in (17, 29, 43)
        ),
        "specificity_cluster_ci_above_zero_each_seed": all(
            results[str(seed)]["deltas"]["specificity"][
                "diagnosis_cluster_bootstrap_95_ci"
            ][0]
            > 0
            for seed in (17, 29, 43)
        ),
        "generation_original_hit_maintained": None,
        "generation_deleted_phantom_nonincrease": None,
    }
    teacher_forced_keys = (
        "three_seed_changed_sign_consistent",
        "changed_delta_min_0p05_each_seed",
        "changed_cluster_ci_above_zero_each_seed",
        "three_seed_specificity_sign_consistent",
        "specificity_cluster_ci_above_zero_each_seed",
    )
    gate["teacher_forced_gate_passed"] = all(gate[key] for key in teacher_forced_keys)
    gate["full_d5_gate_passed"] = False
    if gate["teacher_forced_gate_passed"]:
        gate["generation_audit_status"] = "pending"
        gate["full_gate_reason"] = "generation hit/phantom audit not run"
    else:
        gate["generation_audit_status"] = "not required for promotion decision"
        gate["full_gate_reason"] = (
            "mandatory teacher-forced changed-gap or specificity conditions failed"
        )
    report = {
        "schema_version": 1,
        "results": results,
        "gate": gate,
        "locked_test_read": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# DDXPlus D10 Paired Arm Comparison",
        "",
        "Validation-only. Positive deltas favor ranking over the original-only control.",
        "",
        "| seed | n | changed-gap delta | cluster 95% CI | retained-gap delta | "
        "specificity delta | specificity cluster 95% CI |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (17, 29, 43):
        item = results[str(seed)]
        changed_item = item["deltas"]["changed_gap"]
        retained_item = item["deltas"]["retained_gap"]
        specificity_item = item["deltas"]["specificity"]
        changed_ci = changed_item["diagnosis_cluster_bootstrap_95_ci"]
        specificity_ci = specificity_item["diagnosis_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {seed} | {item['n']} | {changed_item['ranking_minus_control']:+.4f} | "
            f"[{changed_ci[0]:+.4f}, {changed_ci[1]:+.4f}] | "
            f"{retained_item['ranking_minus_control']:+.4f} | "
            f"{specificity_item['ranking_minus_control']:+.4f} | "
            f"[{specificity_ci[0]:+.4f}, {specificity_ci[1]:+.4f}] |"
        )
    lines.extend(
        [
            "",
            "## Frozen Gate Status",
            "",
            f"- three-seed changed sign: **{gate['three_seed_changed_sign_consistent']}**",
            f"- changed delta >= .05 in every seed: **{gate['changed_delta_min_0p05_each_seed']}**",
            "- changed-gap cluster CI > 0 in every seed: "
            f"**{gate['changed_cluster_ci_above_zero_each_seed']}**",
            f"- three-seed specificity sign: **{gate['three_seed_specificity_sign_consistent']}**",
            "- specificity cluster CI > 0 in every seed: "
            f"**{gate['specificity_cluster_ci_above_zero_each_seed']}**",
            "- teacher-forced gate: "
            f"**{'PASS' if gate['teacher_forced_gate_passed'] else 'FAIL'}**",
            "- original-hit and deleted-phantom generation checks: "
            f"**{gate['generation_audit_status']}**",
            "- full D5 promotion: **FAIL**"
            if not gate["teacher_forced_gate_passed"]
            else "- full D5 promotion: **pending generation checks**",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
