"""Teacher-forced D10 changed-claim and retained-cue specificity audit."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_medical_nla_contrastive import losses_for_rows
from src.config import load_config
from src.jsonl import read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter
from src.nla import adapter_av_prompt, load_nla_sidecar


CONDITIONS = (
    "changed_original",
    "changed_deleted",
    "retained_original",
    "retained_deleted",
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def scoring_variant(row: dict[str, Any], condition: str) -> dict[str, Any]:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown D10 condition: {condition}")
    target_field = "retained_target_text" if condition.startswith("retained") else "target_text"
    activation_field = (
        "deleted_activation_path"
        if condition.endswith("deleted")
        else "original_activation_path"
    )
    if not row.get(target_field) or not row.get(activation_field):
        raise ValueError(f"D10 row lacks {target_field}/{activation_field}")
    variant = dict(row)
    variant["target_text"] = row[target_field]
    variant["activation_path"] = row[activation_field]
    variant["d10_condition"] = condition
    return variant


def bootstrap_ci(values: list[float], *, seed: int, draws: int = 5000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = sorted(mean(rng.choice(values) for _ in values) for _ in range(draws))
    return [
        estimates[int(0.025 * (draws - 1))],
        estimates[int(0.975 * (draws - 1))],
    ]


def cluster_bootstrap_ci(
    values_by_cluster: dict[str, list[float]], *, seed: int, draws: int = 5000
) -> list[float]:
    clusters = sorted(key for key, values in values_by_cluster.items() if values)
    if not clusters:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        sampled = [rng.choice(clusters) for _ in clusters]
        values = [value for key in sampled for value in values_by_cluster[key]]
        estimates.append(mean(values))
    estimates.sort()
    return [
        estimates[int(0.025 * (draws - 1))],
        estimates[int(0.975 * (draws - 1))],
    ]


@torch.inference_mode()
def score_rows(
    *,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    output = []
    for condition in CONDITIONS:
        variants = [scoring_variant(row, condition) for row in rows]
        for start in range(0, len(variants), batch_size):
            batch = variants[start : start + batch_size]
            _sums, counts, content_nll = losses_for_rows(
                rows=batch,
                model=model,
                tokenizer=tokenizer,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt_template,
            )
            for row, count, nll in zip(batch, counts, content_nll, strict=True):
                output.append(
                    {
                        "base_id": str(row["base_id"]),
                        "diagnosis_id": row.get("diagnosis_id"),
                        "condition": condition,
                        "content_nll": float(nll.item()),
                        "supervised_tokens": count,
                    }
                )
            print(
                f"[score] {condition} {min(start + batch_size, len(variants))}/"
                f"{len(variants)}",
                flush=True,
            )
    return output


def summarize(scores: list[dict[str, Any]], *, expected_rows: int, seed: int) -> dict[str, Any]:
    by_condition = {
        condition: {
            str(row["base_id"]): row
            for row in scores
            if row.get("condition") == condition
        }
        for condition in CONDITIONS
    }
    ids = sorted(set.intersection(*(set(rows) for rows in by_condition.values())))
    if len(ids) != expected_rows:
        raise ValueError(f"Complete D10 score rows {len(ids)} != expected {expected_rows}")
    rows_out = []
    values: dict[str, list[float]] = defaultdict(list)
    clusters: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    diagnosis_counts: Counter[str] = Counter()
    for identifier in ids:
        changed_original = by_condition["changed_original"][identifier]
        changed_gap = (
            by_condition["changed_deleted"][identifier]["content_nll"]
            - changed_original["content_nll"]
        )
        retained_gap = (
            by_condition["retained_deleted"][identifier]["content_nll"]
            - by_condition["retained_original"][identifier]["content_nll"]
        )
        specificity = changed_gap - retained_gap
        diagnosis = clean(changed_original.get("diagnosis_id")) or "<missing>"
        diagnosis_counts[diagnosis] += 1
        row = {
            "base_id": identifier,
            "diagnosis_id": changed_original.get("diagnosis_id"),
            "changed_original_content_nll": changed_original["content_nll"],
            "changed_deleted_content_nll": by_condition["changed_deleted"][identifier][
                "content_nll"
            ],
            "retained_original_content_nll": by_condition["retained_original"][identifier][
                "content_nll"
            ],
            "retained_deleted_content_nll": by_condition["retained_deleted"][identifier][
                "content_nll"
            ],
            "changed_gap": changed_gap,
            "retained_gap": retained_gap,
            "specificity": specificity,
        }
        rows_out.append(row)
        for metric in ("changed_gap", "retained_gap", "specificity"):
            values[metric].append(row[metric])
            clusters[metric][diagnosis].append(row[metric])

    metrics = {}
    for offset, metric in enumerate(("changed_gap", "retained_gap", "specificity")):
        metric_values = values[metric]
        metrics[metric] = {
            "mean": mean(metric_values),
            "row_bootstrap_95_ci": bootstrap_ci(metric_values, seed=seed + offset),
            "diagnosis_cluster_bootstrap_95_ci": cluster_bootstrap_ci(
                clusters[metric], seed=seed + offset
            ),
            "positive_rate": sum(value > 0 for value in metric_values)
            / len(metric_values),
        }
    return {
        "n": len(ids),
        "diagnosis_clusters": len(diagnosis_counts),
        "diagnosis_counts": dict(sorted(diagnosis_counts.items())),
        "metrics": metrics,
        "private_rows": rows_out,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--pairs", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume-scores", action="store_true")
    args = parser.parse_args()
    rows = list(read_jsonl(args.pairs))
    if not rows:
        raise ValueError("No D10 validation pairs")

    if args.resume_scores:
        raw_scores = list(read_jsonl(args.output_jsonl))
        expected = 4 * len(rows)
        if len(raw_scores) != expected:
            raise ValueError(f"Existing scores {len(raw_scores)} != expected {expected}")
    else:
        cfg = load_config(args.config)
        nla_cfg = cfg["nla_model"]
        cache_dir = cfg["paths"].get("cache_dir")
        tokenizer = load_tokenizer(
            nla_cfg["model_id"],
            cache_dir=cache_dir,
            trust_remote_code=nla_cfg.get("trust_remote_code", True),
        )
        sidecar = load_nla_sidecar(
            nla_cfg["model_id"],
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            filename=nla_cfg.get("sidecar_filename", "nla_meta.yaml"),
            expected_d_model=nla_cfg.get("expected_d_model"),
            expected_injection_token_id=nla_cfg.get("expected_injection_token_id"),
        )
        actor_prompt_template = adapter_av_prompt(str(args.adapter))
        if actor_prompt_template is None:
            raise FileNotFoundError(f"Adapter has no AV prompt: {args.adapter}")
        model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
        model = maybe_load_peft_adapter(model, str(args.adapter), cache_dir=cache_dir)
        raw_scores = score_rows(
            rows=rows,
            model=model,
            tokenizer=tokenizer,
            embed_layer=model.get_input_embeddings(),
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
            batch_size=args.batch_size,
        )

    result = summarize(raw_scores, expected_rows=len(rows), seed=args.seed)
    private_rows = result.pop("private_rows")
    result.update(
        {
            "pairs": str(args.pairs),
            "adapter": str(args.adapter),
            "seed": args.seed,
            "specificity_definition": "changed_gap - retained_gap",
            "locked_test_read": False,
        }
    )
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume_scores:
        write_jsonl(args.output_jsonl, raw_scores)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# DDXPlus D10 Teacher-Forced Specificity Audit",
        "",
        f"- validation pairs: **{result['n']}**",
        f"- diagnosis clusters: **{result['diagnosis_clusters']}**",
        "- retained cue selection: frozen SHA256 rule",
        "- locked test read: **no**",
        "",
        "| metric | mean | row bootstrap 95% CI | diagnosis-cluster 95% CI | positive rate |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in ("changed_gap", "retained_gap", "specificity"):
        item = result["metrics"][metric]
        row_ci = item["row_bootstrap_95_ci"]
        cluster_ci = item["diagnosis_cluster_bootstrap_95_ci"]
        lines.append(
            f"| {metric} | {item['mean']:+.4f} | "
            f"[{row_ci[0]:+.4f}, {row_ci[1]:+.4f}] | "
            f"[{cluster_ci[0]:+.4f}, {cluster_ci[1]:+.4f}] | "
            f"{item['positive_rate']:.4f} |"
        )
    lines.extend(
        ["", "## Validation Diagnosis Distribution", "", "| diagnosis | n |", "|---|---:|"]
    )
    for diagnosis, count in result["diagnosis_counts"].items():
        lines.append(f"| {diagnosis} | {count} |")
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
