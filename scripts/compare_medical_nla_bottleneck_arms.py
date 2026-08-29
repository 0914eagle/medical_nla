"""Compare D16 proposed versus seed-matched controls on paired Direct scores."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl
from src.nla_bottleneck import sha256_file


def parse_arms(values: list[str]) -> dict[int, tuple[Path, Path]]:
    result = {}
    for value in values:
        parts = value.split("=", 2)
        if len(parts) != 3:
            raise ValueError("--seed-scores must be SEED=CONTROL_JSONL=PROPOSED_JSONL")
        seed = int(parts[0])
        result[seed] = (Path(parts[1]), Path(parts[2]))
    if set(result) != {17, 29, 43}:
        raise ValueError("Seeds must be exactly 17,29,43")
    return result


def symmetric_by_id(path: Path) -> dict[str, dict[str, Any]]:
    rows = list(read_jsonl(path))
    by_condition = {
        condition: {
            str(row["base_id"]): row
            for row in rows
            if row.get("condition") == condition
        }
        for condition in ("matched", "target_shuffled", "activation_shuffled")
    }
    ids = set.intersection(*(set(values) for values in by_condition.values()))
    output = {}
    for identifier in sorted(ids):
        donor = str(by_condition["target_shuffled"][identifier]["donor_base_id"])
        if donor not in by_condition["matched"]:
            continue
        value = 0.5 * (
            float(by_condition["target_shuffled"][identifier]["content_nll"])
            + float(by_condition["activation_shuffled"][identifier]["content_nll"])
            - float(by_condition["matched"][identifier]["content_nll"])
            - float(by_condition["matched"][donor]["content_nll"])
        )
        output[identifier] = {
            "value": value,
            "category": str(
                by_condition["target_shuffled"][identifier].get("disease_category")
                or "<missing>"
            ).casefold(),
        }
    return output


def cluster_ci(values: dict[str, list[float]], *, seed: int, draws: int = 5000) -> list[float]:
    clusters = sorted(key for key, rows in values.items() if rows)
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(clusters) for _ in clusters]
        estimates.append(mean(value for key in sampled for value in values[key]))
    estimates.sort()
    return [
        estimates[int(0.025 * (draws - 1))],
        estimates[int(0.975 * (draws - 1))],
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-scores", action="append", required=True)
    parser.add_argument("--floor-protocol", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    floor_protocol = json.loads(args.floor_protocol.read_text(encoding="utf-8"))
    if not floor_protocol.get("frozen_before_proposed"):
        raise ValueError("Effect floor is not immutable")
    floor = float(floor_protocol["effect_floor"])
    arms = parse_arms(args.seed_scores)
    results = {}
    for seed in (17, 29, 43):
        control_path, proposed_path = arms[seed]
        frozen_control = floor_protocol["control_gaps"][str(seed)]
        if sha256_file(control_path) != frozen_control.get("private_scores_sha256"):
            raise ValueError(f"Control private-score SHA changed after floor freeze: seed {seed}")
        control = symmetric_by_id(control_path)
        proposed = symmetric_by_id(proposed_path)
        ids = sorted(set(control) & set(proposed))
        if not ids or set(control) != set(proposed):
            raise ValueError(f"Control/proposed paired population mismatch for seed {seed}")
        deltas = [proposed[key]["value"] - control[key]["value"] for key in ids]
        by_category: dict[str, list[float]] = defaultdict(list)
        for key, value in zip(ids, deltas, strict=True):
            by_category[control[key]["category"]].append(value)
        delta = mean(deltas)
        ci = cluster_ci(by_category, seed=seed)
        control_gap = mean(control[key]["value"] for key in ids)
        proposed_gap = mean(proposed[key]["value"] for key in ids)
        frozen_gap = float(frozen_control["gap"])
        if abs(control_gap - frozen_gap) > 1e-9:
            raise ValueError(
                f"Control score file does not reproduce frozen seed-{seed} gap: "
                f"{control_gap} vs {frozen_gap}"
            )
        results[str(seed)] = {
            "n": len(ids),
            "clusters": len(by_category),
            "control_gap": control_gap,
            "proposed_gap": proposed_gap,
            "proposed_minus_control": delta,
            "category_cluster_bootstrap_95_ci": ci,
            "meets_floor": delta >= floor,
            "cluster_ci_above_zero": ci[0] > 0,
            "positive": delta > 0,
            "control_private_scores": str(control_path),
            "control_private_scores_sha256": sha256_file(control_path),
            "proposed_private_scores": str(proposed_path),
            "proposed_private_scores_sha256": sha256_file(proposed_path),
        }
    primary_pass = all(
        row["meets_floor"] and row["cluster_ci_above_zero"] and row["positive"]
        for row in results.values()
    )
    report = {
        "decision": "D16",
        "effect_floor": floor,
        "floor_protocol": str(args.floor_protocol),
        "floor_protocol_sha256": sha256_file(args.floor_protocol),
        "seeds": results,
        "three_seed_positive": all(row["positive"] for row in results.values()),
        "primary_gate_pass": primary_pass,
        "gate_c_obscomp_threshold": 0.2130,
        "gate_c_status": "pending_separate_semantic_evaluation",
        "locked_test_read": False,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# D16 Seed-Matched Bottleneck Arm Comparison",
        "",
        f"- frozen effect floor: **{floor:.6f}**",
        "",
        "| seed | n | control gap | proposed gap | delta | cluster 95% CI | >= floor |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in (17, 29, 43):
        row = results[str(seed)]
        ci = row["category_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {seed} | {row['n']} | {row['control_gap']:+.6f} | "
            f"{row['proposed_gap']:+.6f} | {row['proposed_minus_control']:+.6f} | "
            f"[{ci[0]:+.6f}, {ci[1]:+.6f}] | "
            f"{'yes' if row['meets_floor'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"- primary three-seed gate: **{'PASS' if primary_pass else 'FAIL'}**",
            "- Gate C Obscomp > .2130: **pending separate semantic evaluation**",
            "- locked test read: **no**",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
