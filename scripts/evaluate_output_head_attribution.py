"""Evaluate source output-head likelihood on note-caused answer movement.

The causal label comes from paired no-note/wrong-note answers, but every score
is computed from the wrong-note run alone. This places the source model's final
candidate distribution between generated-text baselines and hidden-state
probe/readout channels in Table 2b.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_hint_effect import answer_names, group_by_case
from scripts.compare_channels_on_attribution import moved, stratified_auroc
from scripts.predict_error_from_readouts import auroc
from src.answer_matching import is_correct
from src.ddxplus_aliases import aliases_for
from src.jsonl import read_jsonl, write_jsonl


def candidate_probabilities(
    row: dict[str, Any], rank_field: str
) -> list[tuple[dict[str, Any], float]]:
    candidates = list(row.get("top_candidates") or [])
    expected = int(row.get("num_candidates") or 0)
    if expected and len(candidates) != expected:
        raise ValueError(
            f"{row.get('id')}: top_candidates has {len(candidates)} rows, expected "
            f"{expected}. Re-run the scorer with --top-k-output {expected}."
        )
    if not candidates:
        raise ValueError(f"{row.get('id')}: no candidate scores found")
    values = [float(candidate[rank_field]) for candidate in candidates]
    offset = max(values)
    weights = [math.exp(value - offset) for value in values]
    total = sum(weights)
    return [(candidate, weight / total) for candidate, weight in zip(candidates, weights)]


def matching_probability(
    distribution: list[tuple[dict[str, Any], float]], diagnosis_name: str
) -> float:
    aliases = aliases_for(diagnosis_name)
    return sum(
        probability
        for candidate, probability in distribution
        if is_correct(candidate.get("diagnosis_name"), diagnosis_name, aliases)
    )


def output_head_features(
    case: dict[str, dict[str, Any]], row: dict[str, Any], rank_field: str
) -> dict[str, float]:
    distribution = candidate_probabilities(row, rank_field)
    ranked = sorted(distribution, key=lambda item: item[1], reverse=True)
    top_candidate, top1_prob = ranked[0]
    top2_prob = ranked[1][1] if len(ranked) > 1 else 0.0
    suggestion = str(case["wrong"].get("hint_diagnosis_name") or "")
    answer = str(case["wrong"].get("answer") or "")
    top_name = str(top_candidate.get("diagnosis_name") or "")
    entropy = -sum(probability * math.log(max(probability, 1e-30)) for _, probability in ranked)
    entropy_norm = entropy / math.log(len(ranked)) if len(ranked) > 1 else 0.0

    features = {
        "output-head entropy": entropy_norm,
        "output-head low top1 probability": 1.0 - top1_prob,
        "output-head small top1-top2 margin": 1.0 - (top1_prob - top2_prob),
        "output-head probability of suggestion": matching_probability(distribution, suggestion),
        "output-head top1 is suggestion": float(
            is_correct(top_name, suggestion, aliases_for(suggestion))
        ),
        "output-head top1 disagrees with generated answer": 1.0
        - float(is_correct(answer, top_name, aliases_for(top_name))),
    }

    # Diagnostic-only quantities: useful for mechanism analysis but unavailable
    # in deployment because the gold diagnosis is unknown.
    gold = str(case["none"].get("diagnosis_name") or "")
    p_gold = matching_probability(distribution, gold)
    p_suggestion = features["output-head probability of suggestion"]
    features["[analysis only] low gold probability"] = 1.0 - p_gold
    features["[analysis only] suggestion minus gold probability"] = p_suggestion - p_gold
    return features


def summarize_group(
    rows: list[dict[str, Any]], feature_names: list[str]
) -> dict[str, dict[str, float | int]]:
    labels = [bool(row["moved"]) for row in rows]
    diagnoses = [str(row["diagnosis_name"]) for row in rows]
    out: dict[str, dict[str, float | int]] = {}
    for feature in feature_names:
        values = [float(row[feature]) for row in rows]
        within, pairs = stratified_auroc(values, labels, diagnoses)
        out[feature] = {
            "pooled_auroc": auroc(values, labels),
            "within_diagnosis_auroc": within,
            "within_diagnosis_pairs": pairs,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--cases", default=None)
    parser.add_argument("--logprobs", required=True)
    parser.add_argument(
        "--rank-field",
        default="first_token_logprob",
        choices=[
            "logprob_mean",
            "logprob_sum",
            "first_token_logprob",
            "calibrated_logprob_mean",
            "calibrated_logprob_sum",
            "calibrated_first_token_logprob",
        ],
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    cases = group_by_case(args.answers, args.cases)
    logprobs = {
        str(row.get("base_id") or row.get("id")): row
        for row in read_jsonl(args.logprobs)
        if str(row.get("hint_variant") or "wrong") == "wrong"
    }

    evaluated: list[dict[str, Any]] = []
    missing: list[str] = []
    for base_id, case in cases.items():
        row = logprobs.get(base_id)
        if row is None:
            missing.append(base_id)
            continue
        features = output_head_features(case, row, args.rank_field)
        evaluated.append(
            {
                "base_id": base_id,
                "diagnosis_name": str(case["none"].get("diagnosis_name") or ""),
                "moved": moved(case),
                "silent": not answer_names(
                    case["wrong"], case["wrong"].get("hint_diagnosis_name")
                ),
                "rank_field": args.rank_field,
                **features,
            }
        )
    if not evaluated:
        raise ValueError("No source likelihood rows joined to answer cases.")

    feature_names = [
        key
        for key in evaluated[0]
        if key
        not in {"base_id", "diagnosis_name", "moved", "silent", "rank_field"}
    ]
    silent = [row for row in evaluated if row["silent"]]
    all_scores = summarize_group(evaluated, feature_names)
    silent_scores = summarize_group(silent, feature_names)
    write_jsonl(Path(args.output_jsonl), evaluated)

    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write("# Source Output-Head Note-Influence Detection\n\n")
        handle.write(f"- rank_field: `{args.rank_field}`\n")
        handle.write(f"- joined: {len(evaluated)}\n")
        handle.write(f"- missing: {len(missing)}\n")
        handle.write(f"- moved: {sum(bool(row['moved']) for row in evaluated)}\n")
        handle.write(f"- silent: {len(silent)}\n")
        handle.write(f"- silent_moved: {sum(bool(row['moved']) for row in silent)}\n\n")
        handle.write(
            "Main values are within-diagnosis AUROCs. Rows marked analysis-only "
            "use the gold label and are not deployable detector features.\n\n"
        )
        handle.write("| feature | all pooled | all within-dx | silent pooled | silent within-dx |\n")
        handle.write("|---|---:|---:|---:|---:|\n")
        for feature in feature_names:
            handle.write(
                f"| {feature} | {all_scores[feature]['pooled_auroc']:.4f} | "
                f"{all_scores[feature]['within_diagnosis_auroc']:.4f} | "
                f"{silent_scores[feature]['pooled_auroc']:.4f} | "
                f"{silent_scores[feature]['within_diagnosis_auroc']:.4f} |\n"
            )

    print(
        f"[done] joined {len(evaluated):,}/{len(cases):,} cases; "
        f"summary -> {summary_path}"
    )


if __name__ == "__main__":
    main()
