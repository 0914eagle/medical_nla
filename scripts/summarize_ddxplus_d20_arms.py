"""Compare D20 specificity-anchored adapters with frozen D10 controls."""

from __future__ import annotations

import argparse
import hashlib
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
from scripts.summarize_ddxplus_d10_arms import metrics_by_id

SEEDS = (17, 29, 43)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_arm(value: str) -> tuple[int, str, Path]:
    parts = value.split("=", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--arm must be SEED=control|anchored=PATH")
    seed = int(parts[0])
    label = parts[1]
    if seed not in SEEDS or label not in {"control", "anchored"}:
        raise argparse.ArgumentTypeError("Seeds are 17/29/43; arms are control/anchored")
    return seed, label, Path(parts[2])


def metric_delta(
    control: dict[str, dict[str, Any]],
    anchored: dict[str, dict[str, Any]],
    metric: str,
    *,
    seed: int,
    offset: int,
) -> dict[str, Any]:
    identifiers = sorted(control)
    deltas = [anchored[item][metric] - control[item][metric] for item in identifiers]
    clusters: dict[str, list[float]] = defaultdict(list)
    for identifier, value in zip(identifiers, deltas, strict=True):
        diagnosis = clean(control[identifier].get("diagnosis_id")) or "<missing>"
        clusters[diagnosis].append(value)
    return {
        "anchored_minus_control": mean(deltas),
        "row_bootstrap_95_ci": bootstrap_ci(deltas, seed=seed + offset),
        "diagnosis_cluster_bootstrap_95_ci": cluster_bootstrap_ci(
            clusters, seed=seed + offset
        ),
        "anchored_win_rate": sum(value > 0 for value in deltas) / len(deltas),
    }


def compare_seed(
    control: dict[str, dict[str, Any]],
    anchored: dict[str, dict[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if set(control) != set(anchored):
        raise ValueError(f"Seed {seed} control/anchored populations differ")
    metrics = (
        "changed_gap",
        "retained_gap",
        "specificity",
        "changed_original_nll",
        "retained_original_nll",
    )
    return {
        "n": len(control),
        "deltas": {
            metric: metric_delta(
                control, anchored, metric, seed=seed, offset=offset
            )
            for offset, metric in enumerate(metrics)
        },
    }


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if not protocol.get("human_approved"):
        raise ValueError("D20 gate protocol is not human-approved")
    gates = protocol.get("gates") or {}
    required = {
        "retained_gap_delta_max",
        "changed_original_nll_delta_max",
        "retained_original_nll_delta_max",
        "mean_claim_relative_drop_max",
    }
    if set(gates) < required or any(gates[name] is None for name in required):
        raise ValueError("D20 gate protocol lacks frozen numeric allowances")
    hashes = protocol.get("control_score_sha256") or {}
    if set(hashes) != {str(seed) for seed in SEEDS}:
        raise ValueError("D20 gate protocol lacks all three control score hashes")
    return protocol


def build_report(
    paths: dict[int, dict[str, Path]],
    *,
    protocol: dict[str, Any],
    report_only: bool,
) -> dict[str, Any]:
    incomplete = any(set(items) != {"control", "anchored"} for items in paths.values())
    if set(paths) != set(SEEDS) or incomplete:
        raise ValueError("D20 comparison requires control/anchored for all three seeds")
    for seed in SEEDS:
        observed = sha256_file(paths[seed]["control"])
        expected = protocol["control_score_sha256"][str(seed)]
        if observed != expected:
            raise ValueError(f"Seed {seed} control score hash does not match protocol")
    results = {
        str(seed): compare_seed(
            metrics_by_id(paths[seed]["control"]),
            metrics_by_id(paths[seed]["anchored"]),
            seed=seed,
        )
        for seed in SEEDS
    }
    gates = protocol["gates"]
    changed = [
        results[str(seed)]["deltas"]["changed_gap"]["anchored_minus_control"]
        for seed in SEEDS
    ]
    specificity = [
        results[str(seed)]["deltas"]["specificity"]["anchored_minus_control"]
        for seed in SEEDS
    ]
    retained = [
        results[str(seed)]["deltas"]["retained_gap"]["anchored_minus_control"]
        for seed in SEEDS
    ]
    changed_original = [
        results[str(seed)]["deltas"]["changed_original_nll"]["anchored_minus_control"]
        for seed in SEEDS
    ]
    retained_original = [
        results[str(seed)]["deltas"]["retained_original_nll"]["anchored_minus_control"]
        for seed in SEEDS
    ]
    gate = {
        "changed_delta_min_0p05_each_seed": all(value >= 0.05 for value in changed),
        "changed_cluster_ci_above_zero_each_seed": all(
            results[str(seed)]["deltas"]["changed_gap"][
                "diagnosis_cluster_bootstrap_95_ci"
            ][0]
            > 0
            for seed in SEEDS
        ),
        "specificity_positive_each_seed": all(value > 0 for value in specificity),
        "specificity_cluster_ci_above_zero_each_seed": all(
            results[str(seed)]["deltas"]["specificity"][
                "diagnosis_cluster_bootstrap_95_ci"
            ][0]
            > 0
            for seed in SEEDS
        ),
        "retained_gap_noninferior_each_seed": all(
            value <= gates["retained_gap_delta_max"] for value in retained
        ),
        "changed_original_nll_noninferior_each_seed": all(
            value <= gates["changed_original_nll_delta_max"]
            for value in changed_original
        ),
        "retained_original_nll_noninferior_each_seed": all(
            value <= gates["retained_original_nll_delta_max"]
            for value in retained_original
        ),
    }
    gate["teacher_forced_gate_passed"] = all(gate.values())
    return {
        "schema_version": 1,
        "results": results,
        "gate": gate,
        "report_only": report_only,
        "promotion_decision_authorized": not report_only,
        "protocol": protocol,
        "locked_test_read": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", action="append", required=True, type=parse_arm)
    parser.add_argument("--gate-protocol", required=True, type=Path)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    paths: dict[int, dict[str, Path]] = defaultdict(dict)
    for seed, label, path in args.arm:
        if label in paths[seed]:
            raise ValueError(f"Duplicate {seed}/{label}")
        paths[seed][label] = path
    report = build_report(
        paths, protocol=load_protocol(args.gate_protocol), report_only=args.report_only
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# DDXPlus D20 Specificity-Anchored Comparison",
        "",
        "Validation-only. Positive gap/specificity deltas favor D20; positive NLL "
        "deltas are degradation.",
        f"Promotion decision authorized: **{'no' if args.report_only else 'yes'}**.",
        "",
        "| seed | changed gap | retained gap | specificity | changed orig NLL | retained orig NLL |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        deltas = report["results"][str(seed)]["deltas"]
        lines.append(
            f"| {seed} | {deltas['changed_gap']['anchored_minus_control']:+.4f} | "
            f"{deltas['retained_gap']['anchored_minus_control']:+.4f} | "
            f"{deltas['specificity']['anchored_minus_control']:+.4f} | "
            f"{deltas['changed_original_nll']['anchored_minus_control']:+.4f} | "
            f"{deltas['retained_original_nll']['anchored_minus_control']:+.4f} |"
        )
    lines.extend(["", "## Teacher-Forced Gate", ""])
    gates = report["protocol"]["gates"]
    lines.extend(
        [
            f"- retained-gap delta upper bound: **{gates['retained_gap_delta_max']:.6f}**",
            "- changed-original NLL delta upper bound: "
            f"**{gates['changed_original_nll_delta_max']:.6f}**",
            "- retained-original NLL delta upper bound: "
            f"**{gates['retained_original_nll_delta_max']:.6f}**",
            "- mean-claim relative-drop upper bound (generation): "
            f"**{gates['mean_claim_relative_drop_max']:.6f}**",
            "",
        ]
    )
    for name, passed in report["gate"].items():
        lines.append(f"- {name}: **{passed}**")
    if args.report_only:
        lines.append("- final decision: **report-only checkpoint; no selection**")
    else:
        lines.append(
            "- final teacher-forced decision: "
            f"**{'PASS' if report['gate']['teacher_forced_gate_passed'] else 'FAIL'}**"
        )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
