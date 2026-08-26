"""Create private gold-oracle predictions in the official DiReCT schema.

The input manifest and generated JSON files contain restricted clinical text.
Keep both under the private DiReCT data root and never commit them.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def official_oracle_prediction(
    row: dict[str, Any],
    official_gold: dict[str, Any] | None = None,
    official_chain_leaf_to_root: list[str] | None = None,
) -> tuple[dict[str, Any], int]:
    prediction: dict[str, Any] = {}
    duplicate_observations = 0
    if official_gold is None:
        for deduction in row["gold_deductions"]:
            observation = deduction["observation"]
            if observation in prediction:
                duplicate_observations += 1
            prediction[observation] = [
                deduction.get("rationale"),
                deduction.get("annotated_source_section"),
                deduction.get("diagnosis"),
            ]
    else:
        prediction.update(official_gold)

    # cal_a_json() returns its diagnosis chain leaf-to-root. The official
    # baseline prediction starts with the disease category and proceeds
    # root-to-leaf, which is the shape expected by statistics.py.
    gold_chain_leaf_to_root = list(
        official_chain_leaf_to_root
        if official_chain_leaf_to_root is not None
        else row["annotation_chain"]
    )
    prediction["chain"] = [
        row["disease_category"],
        *reversed(gold_chain_leaf_to_root),
    ]
    return prediction, duplicate_observations


def load_official_parser(official_repo: Path) -> tuple[Any, Any]:
    repo = str(official_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    data_analysis = importlib.import_module("utils.data_analysis")
    return data_analysis.cal_a_json, data_analysis.deduction_assemble


def relative_source_path(row: dict[str, Any], samples_root: Path) -> Path:
    source_path = Path(row["source_path"]).resolve()
    try:
        return source_path.relative_to(samples_root.resolve())
    except ValueError as exc:
        raise ValueError(
            f"Source path for row {row['id']} is outside samples root"
        ) from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--samples-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    rows = sorted(read_jsonl(args.manifest), key=lambda row: row["id"])
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[: args.limit]

    args.output_root.mkdir(parents=True, exist_ok=True)
    cal_a_json, deduction_assemble = load_official_parser(args.official_repo)
    duplicate_observations = 0
    manifest_official_observation_count_mismatches = 0
    manifest_official_chain_mismatches = 0
    observation_counts: list[int] = []
    category_counts: Counter[str] = Counter()
    for row in rows:
        record_node, _, official_chain = cal_a_json(row["source_path"])
        official_gold = deduction_assemble(record_node)
        prediction, duplicate_count = official_oracle_prediction(
            row,
            official_gold=official_gold,
            official_chain_leaf_to_root=official_chain,
        )
        relative_path = relative_source_path(row, args.samples_root)
        output_path = args.output_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(prediction, ensure_ascii=False),
            encoding="utf-8",
        )
        duplicate_observations += duplicate_count
        observation_counts.append(len(prediction) - 1)
        category_counts[row["disease_category"]] += 1
        manifest_unique_observations = {
            deduction["observation"] for deduction in row["gold_deductions"]
        }
        if len(manifest_unique_observations) != len(official_gold):
            manifest_official_observation_count_mismatches += 1
        if list(row["annotation_chain"]) != list(official_chain):
            manifest_official_chain_mismatches += 1

    mean_observations = (
        sum(observation_counts) / len(observation_counts) if observation_counts else 0
    )
    summary = [
        "# DiReCT Official-Schema Oracle Prediction Summary",
        "",
        "Aggregate-only summary. Prediction JSON files contain restricted text.",
        "",
        f"- rows: **{len(rows)}**",
        f"- output files: **{sum(1 for _ in args.output_root.rglob('*.json'))}**",
        f"- duplicate observation keys overwritten: **{duplicate_observations}**",
        (
            "- manifest/official observation-count mismatches: "
            f"**{manifest_official_observation_count_mismatches}**"
        ),
        f"- manifest/official chain mismatches: **{manifest_official_chain_mismatches}**",
        (
            "- observations per prediction: "
            f"**min {min(observation_counts, default=0)}, "
            f"mean {mean_observations:.2f}, "
            f"max {max(observation_counts, default=0)}**"
        ),
        f"- disease categories represented: **{len(category_counts)}**",
        "",
        "The oracle is an evaluator smoke-test fixture, not a model result.",
        "",
    ]
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(summary), encoding="utf-8")
    print(f"[oracle] rows={len(rows)} output_root={args.output_root}")
    print(f"[oracle] summary={args.summary_md}")


if __name__ == "__main__":
    main()
