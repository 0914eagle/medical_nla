"""Run the preregistered D22 A1-A5 geometry audit.

The audit is validation-only. It consumes reconstruction vectors written by
``audit_medical_nla_ar_roundtrip.py`` and emits aggregate metrics. DiReCT text
and row-level geometry remain under the restricted output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl
from src.reconstruction_scoring import load_activation


POSITIVE_CONTROLS = ("ddxplus::structured_reader", "direct::source_cot")


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_path_map(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected OLD=NEW")
    old, new = value.split("=", 1)
    if not old or not new:
        raise argparse.ArgumentTypeError("Expected non-empty OLD=NEW")
    return old, new


def mapped_path(value: Any, mappings: list[tuple[str, str]]) -> Path:
    text = str(value or "")
    for old, new in mappings:
        if text.startswith(old):
            text = new + text[len(old) :]
            break
    return Path(text)


def row_base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def row_stratum(row: dict[str, Any], dataset: str) -> str:
    if dataset == "ddxplus":
        return clean(row.get("diagnosis_id"))
    return clean(
        row.get("disease_category")
        or row.get("diagnosis_id")
        or row.get("canonical_pdd")
    )


def unit(vector: torch.Tensor) -> torch.Tensor:
    vector = vector.float().flatten()
    norm = float(vector.norm())
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot normalize a zero or non-finite vector")
    return vector / norm


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    if left.numel() != right.numel():
        raise ValueError("Vector-width mismatch")
    value = float(F.cosine_similarity(left.float(), right.float(), dim=0))
    if not math.isfinite(value):
        raise ValueError("Non-finite cosine")
    return value


def remove_mean_direction(vector: torch.Tensor, mean_direction: torch.Tensor) -> torch.Tensor:
    direction = unit(mean_direction)
    source = vector.float().flatten()
    return source - torch.dot(source, direction) * direction


def average_rank(scores: list[float], own_index: int) -> float:
    own = scores[own_index]
    greater = sum(value > own for value in scores)
    equal = sum(value == own for value in scores)
    return 1.0 + greater + 0.5 * (equal - 1)


def row_bootstrap_ci(
    values: list[float], *, seed: int = 17, replicates: int = 10_000
) -> list[float]:
    if not values:
        raise ValueError("Cannot bootstrap no values")
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        estimates.append(
            statistics.fmean(values[rng.randrange(len(values))] for _ in values)
        )
    estimates.sort()
    return [
        estimates[int(0.025 * replicates)],
        estimates[int(0.975 * replicates) - 1],
    ]


def cluster_bootstrap_ci(
    values: list[float],
    clusters: list[str],
    *,
    seed: int = 17,
    replicates: int = 10_000,
) -> list[float]:
    if len(values) != len(clusters) or not values:
        raise ValueError("Cluster bootstrap inputs are empty or unaligned")
    grouped: dict[str, list[float]] = defaultdict(list)
    for value, cluster in zip(values, clusters, strict=True):
        grouped[cluster].append(value)
    keys = sorted(grouped)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sample = []
        for _ in keys:
            sample.extend(grouped[keys[rng.randrange(len(keys))]])
        estimates.append(statistics.fmean(sample))
    estimates.sort()
    return [
        estimates[int(0.025 * replicates)],
        estimates[int(0.975 * replicates) - 1],
    ]


def summarize_values(values: list[float], clusters: list[str]) -> dict[str, Any]:
    return {
        "n": len(values),
        "clusters": len(set(clusters)),
        "mean": statistics.fmean(values),
        "row_bootstrap_95_ci": row_bootstrap_ci(values),
        "diagnosis_cluster_bootstrap_95_ci": cluster_bootstrap_ci(values, clusters),
        "positive_rate": sum(value > 0 for value in values) / len(values),
    }


def load_original_pool(
    path: Path,
    *,
    dataset: str,
    mappings: list[tuple[str, str]],
) -> dict[str, dict[str, Any]]:
    result = {}
    for row in read_jsonl(path):
        if str(row.get("variant") or "original") != "original":
            continue
        identifier = row_base_id(row)
        stratum = row_stratum(row, dataset)
        activation_path = mapped_path(row.get("activation_path"), mappings)
        if not identifier or not stratum or identifier in result:
            raise ValueError(f"Invalid {dataset} original row: {identifier!r}")
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        result[identifier] = {
            "base_id": identifier,
            "stratum": stratum,
            "activation_path": str(activation_path),
        }
    if not result:
        raise ValueError(f"No original rows in {path}")
    return result


def train_mean(
    path: Path,
    *,
    dataset: str,
    mappings: list[tuple[str, str]],
) -> tuple[torch.Tensor, int]:
    total = None
    count = 0
    seen = set()
    for row in read_jsonl(path):
        if str(row.get("variant") or "original") != "original":
            continue
        identifier = row_base_id(row)
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        path_value = mapped_path(row.get("activation_path"), mappings)
        vector = load_activation(path_value).double()
        if total is None:
            total = torch.zeros_like(vector)
        if vector.shape != total.shape:
            raise ValueError("Train activation width mismatch")
        total += vector
        count += 1
    if total is None or count == 0:
        raise ValueError(f"No train activations in {path}")
    return (total / count).float(), count


def deterministic_different_donor(
    base_id: str, stratum: str, pool: dict[str, dict[str, Any]]
) -> str:
    candidates = [
        identifier
        for identifier, row in pool.items()
        if row["stratum"] != stratum
    ]
    if not candidates:
        raise ValueError(f"No different-diagnosis donor for {base_id}")
    return min(
        candidates,
        key=lambda value: hashlib.sha256(
            f"d22-different\0{base_id}\0{value}".encode()
        ).hexdigest(),
    )


def load_reconstruction(row: dict[str, Any], mappings: list[tuple[str, str]]) -> torch.Tensor:
    path = mapped_path(row.get("reconstruction_path"), mappings)
    if not path.is_file():
        raise FileNotFoundError(path)
    expected = clean(row.get("reconstruction_sha256"))
    if expected and sha256_file(path) != expected:
        raise ValueError(f"Reconstruction hash mismatch: {path}")
    return load_activation(path)


def harmonic_chance_mrr(count: int) -> float:
    return sum(1.0 / rank for rank in range(1, count + 1)) / count


def arm_key(row: dict[str, Any]) -> str:
    return f"{row['dataset']}::{row['arm']}"


def evaluate(args: argparse.Namespace) -> None:
    scores = list(read_jsonl(args.scores))
    if not scores:
        raise ValueError("No D22 score rows")
    if any(not row.get("reconstruction_path") for row in scores):
        raise ValueError("D22 scores do not contain persisted reconstruction vectors")

    pools = {
        "ddxplus": load_original_pool(
            args.ddx_validation_manifest,
            dataset="ddxplus",
            mappings=args.path_map,
        ),
        "direct": load_original_pool(
            args.direct_validation_manifest,
            dataset="direct",
            mappings=args.path_map,
        ),
    }
    means = {}
    mean_counts = {}
    for dataset, path in (
        ("ddxplus", args.ddx_train_manifest),
        ("direct", args.direct_train_manifest),
    ):
        means[dataset], mean_counts[dataset] = train_mean(
            path, dataset=dataset, mappings=args.path_map
        )

    activation_cache: dict[str, torch.Tensor] = {}

    def activation(path: str) -> torch.Tensor:
        if path not in activation_cache:
            activation_cache[path] = load_activation(path)
        return activation_cache[path]

    candidate_vectors = {
        dataset: {
            identifier: activation(str(row["activation_path"]))
            for identifier, row in pool.items()
        }
        for dataset, pool in pools.items()
    }

    private_rows = []
    for row in scores:
        dataset = str(row["dataset"])
        identifier = str(row["base_id"])
        if dataset not in pools:
            raise ValueError(f"Unknown score dataset: {dataset}")
        pool_identifier = identifier
        if pool_identifier not in pools[dataset]:
            score_activation = str(mapped_path(row.get("activation_path"), args.path_map))
            matches = [
                candidate_id
                for candidate_id, candidate in pools[dataset].items()
                if str(candidate["activation_path"]) == score_activation
            ]
            if len(matches) != 1:
                raise ValueError(
                    "Score row cannot be joined to validation pool by base ID or "
                    f"activation path: {dataset}/{identifier} matches={len(matches)}"
                )
            pool_identifier = matches[0]
        pool_row = pools[dataset][pool_identifier]
        stratum = str(pool_row["stratum"])
        own = activation(str(pool_row["activation_path"]))
        same = activation(str(mapped_path(row["donor_activation_path"], args.path_map)))
        different_id = deterministic_different_donor(
            pool_identifier, stratum, pools[dataset]
        )
        different = candidate_vectors[dataset][different_id]
        reconstruction = load_reconstruction(row, args.path_map)
        mean = means[dataset]

        own_centered = remove_mean_direction(own, mean)
        same_centered = remove_mean_direction(same, mean)
        recon_centered = remove_mean_direction(reconstruction, mean)
        a2_gap = cosine(recon_centered, own_centered) - cosine(
            recon_centered, same_centered
        )
        model_error = float((unit(own) - unit(reconstruction)).pow(2).sum())
        mean_error = float((unit(own) - unit(mean)).pow(2).sum())

        candidates = sorted(
            (
                candidate_id,
                candidate,
            )
            for candidate_id, candidate in candidate_vectors[dataset].items()
            if pools[dataset][candidate_id]["stratum"] == stratum
        )
        candidate_ids = [candidate_id for candidate_id, _ in candidates]
        retrieval_scores = [cosine(reconstruction, vector) for _, vector in candidates]
        own_index = candidate_ids.index(pool_identifier)
        rank = average_rank(retrieval_scores, own_index)
        candidate_count = len(candidates)
        if candidate_count < 2:
            rank = float("nan")

        private_rows.append(
            {
                "id": row["id"],
                "dataset": dataset,
                "arm": row["arm"],
                "base_id": identifier,
                "validation_pool_base_id": pool_identifier,
                "stratum": stratum,
                "a1_same_diagnosis_activation_cosine": cosine(own, same),
                "a1_different_diagnosis_activation_cosine": cosine(own, different),
                "a2_centered_matched_cosine": cosine(recon_centered, own_centered),
                "a2_centered_shuffled_cosine": cosine(recon_centered, same_centered),
                "a2_centered_gap": a2_gap,
                "a3_model_error": model_error,
                "a3_train_mean_error": mean_error,
                "a4_same_diagnosis_reconstruction_cosine": cosine(reconstruction, same),
                "a4_different_diagnosis_reconstruction_cosine": cosine(
                    reconstruction, different
                ),
                "a4_same_minus_different_gap": cosine(reconstruction, same)
                - cosine(reconstruction, different),
                "a5_candidate_count": candidate_count,
                "a5_rank": rank,
                "a5_reciprocal_rank": (1.0 / rank if math.isfinite(rank) else None),
                "a5_top1": (rank == 1.0 if math.isfinite(rank) else None),
                "a5_top1_minus_chance": (
                    float(rank == 1.0) - 1.0 / candidate_count
                    if math.isfinite(rank)
                    else None
                ),
                "a5_mrr_minus_chance": (
                    1.0 / rank - harmonic_chance_mrr(candidate_count)
                    if math.isfinite(rank)
                    else None
                ),
                "different_donor_base_id": different_id,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in private_rows:
        grouped[arm_key(row)].append(row)
    arms = {}
    for key, rows in sorted(grouped.items()):
        clusters = [str(row["stratum"]) for row in rows]
        a1_same = [float(row["a1_same_diagnosis_activation_cosine"]) for row in rows]
        a1_diff = [float(row["a1_different_diagnosis_activation_cosine"]) for row in rows]
        a2 = [float(row["a2_centered_gap"]) for row in rows]
        a4 = [float(row["a4_same_minus_different_gap"]) for row in rows]
        retrieval = [
            row
            for row in rows
            if row["a5_rank"] is not None
            and math.isfinite(float(row["a5_rank"]))
        ]
        retrieval_clusters = [str(row["stratum"]) for row in retrieval]
        top1_gap = [float(row["a5_top1_minus_chance"]) for row in retrieval]
        mrr_gap = [float(row["a5_mrr_minus_chance"]) for row in retrieval]
        model_error = sum(float(row["a3_model_error"]) for row in rows)
        mean_error = sum(float(row["a3_train_mean_error"]) for row in rows)
        arms[key] = {
            "n": len(rows),
            "diagnosis_clusters": len(set(clusters)),
            "a1": {
                "same_diagnosis_mean_cosine": statistics.fmean(a1_same),
                "different_diagnosis_mean_cosine": statistics.fmean(a1_diff),
                "same_minus_different": summarize_values(
                    [left - right for left, right in zip(a1_same, a1_diff, strict=True)],
                    clusters,
                ),
            },
            "a2": summarize_values(a2, clusters),
            "a3": {
                "model_error_sum": model_error,
                "train_mean_predictor_error_sum": mean_error,
                "direction_normalized_fve": 1.0 - model_error / mean_error,
            },
            "a4": summarize_values(a4, clusters),
            "a5": {
                "n": len(retrieval),
                "mean_candidate_count": statistics.fmean(
                    float(row["a5_candidate_count"]) for row in retrieval
                ),
                "top1": statistics.fmean(float(row["a5_top1"]) for row in retrieval),
                "mrr": statistics.fmean(float(row["a5_reciprocal_rank"]) for row in retrieval),
                "median_rank": statistics.median(float(row["a5_rank"]) for row in retrieval),
                "top1_minus_chance": summarize_values(top1_gap, retrieval_clusters),
                "mrr_minus_chance": summarize_values(mrr_gap, retrieval_clusters),
            },
        }

    controls = {}
    for key in POSITIVE_CONTROLS:
        if key not in arms:
            raise ValueError(f"Missing positive-control arm: {key}")
        item = arms[key]
        a2_pass = item["a2"]["diagnosis_cluster_bootstrap_95_ci"][0] > 0
        a5_pass = (
            item["a5"]["top1_minus_chance"]["diagnosis_cluster_bootstrap_95_ci"][0]
            > 0
            or item["a5"]["mrr_minus_chance"]["diagnosis_cluster_bootstrap_95_ci"][0]
            > 0
        )
        a3_pass = item["a3"]["direction_normalized_fve"] > 0
        controls[key] = {"a2_pass": a2_pass, "a3_pass": a3_pass, "a5_pass": a5_pass}

    limited = all(item["a2_pass"] or item["a5_pass"] for item in controls.values())
    reward = all(
        item["a2_pass"] and item["a5_pass"] and item["a3_pass"]
        for item in controls.values()
    )
    discarded = all(
        not item["a2_pass"] and not item["a5_pass"] for item in controls.values()
    )
    decision = (
        "av_reward_eligible"
        if reward
        else "limited_diagnostic_only"
        if limited
        else "discard_public_ar"
        if discarded
        else "inconclusive_positive_controls"
    )
    result = {
        "schema_version": 1,
        "validation_only": True,
        "locked_test_read": False,
        "train_mean_counts": mean_counts,
        "arms": arms,
        "positive_control_gates": controls,
        "decision": decision,
        "public_ar_limited_diagnostic_accepted": limited,
        "public_ar_reward_accepted": reward,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private_geometry_rows.jsonl", private_rows)
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# D22 Public-AR Geometry Audit",
        "",
        "Validation-only A1-A5 audit; locked test was not read.",
        "",
        f"- decision: **{decision}**",
        f"- limited diagnostic accepted: **{limited}**",
        f"- AV reward accepted: **{reward}**",
        f"- train means: DDXPlus {mean_counts['ddxplus']}, DiReCT {mean_counts['direct']}",
        "",
        "| dataset/arm | n | A1 same-diff | A2 centered gap [cluster CI] | "
        "A3 FVE | A4 same-diff donor | A5 top1 | A5 MRR | A5 median rank | "
        "A5 mean candidates | A5 top1-minus-chance [cluster CI] | "
        "A5 MRR-minus-chance [cluster CI] |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for key, item in arms.items():
        a2_ci = item["a2"]["diagnosis_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {key} | {item['n']} | {item['a1']['same_minus_different']['mean']:+.4f} | "
            f"{item['a2']['mean']:+.4f} [{a2_ci[0]:+.4f}, {a2_ci[1]:+.4f}] | "
            f"{item['a3']['direction_normalized_fve']:+.4f} | "
            f"{item['a4']['mean']:+.4f} | {item['a5']['top1']:.4f} | "
            f"{item['a5']['mrr']:.4f} | {item['a5']['median_rank']:.1f} | "
            f"{item['a5']['mean_candidate_count']:.1f} | "
            f"{item['a5']['top1_minus_chance']['mean']:+.4f} "
            f"[{item['a5']['top1_minus_chance']['diagnosis_cluster_bootstrap_95_ci'][0]:+.4f}, "
            f"{item['a5']['top1_minus_chance']['diagnosis_cluster_bootstrap_95_ci'][1]:+.4f}] | "
            f"{item['a5']['mrr_minus_chance']['mean']:+.4f} "
            f"[{item['a5']['mrr_minus_chance']['diagnosis_cluster_bootstrap_95_ci'][0]:+.4f}, "
            f"{item['a5']['mrr_minus_chance']['diagnosis_cluster_bootstrap_95_ci'][1]:+.4f}] |"
        )
    lines.extend(["", "## Positive-Control Gates", ""])
    for key, item in controls.items():
        lines.append(
            f"- `{key}`: A2={item['a2_pass']}, A3={item['a3_pass']}, "
            f"A5={item['a5_pass']}"
        )
    lines.append("")
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True, type=Path)
    parser.add_argument("--ddx-validation-manifest", required=True, type=Path)
    parser.add_argument("--direct-validation-manifest", required=True, type=Path)
    parser.add_argument("--ddx-train-manifest", required=True, type=Path)
    parser.add_argument("--direct-train-manifest", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], type=parse_path_map)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    evaluate(args)


if __name__ == "__main__":
    main()
