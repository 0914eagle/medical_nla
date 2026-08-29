"""Paired bootstrap, threshold sensitivity, and verbosity audit for CF readouts.

The counterfactual grounding numbers in the tuning strategy are point
estimates at a single lexical threshold (.5), and seed 17/29 disagree far
beyond evaluation noise. Before the cue-level ranking objective runs, this
script answers three questions on the SAME scored readout files, with the
SAME hit instrument as scripts/score_ddxplus_e5_readout_pilot.py:

1. Threshold sensitivity: do the deletion/value metrics survive at
   thresholds .3/.5/.7, or does the seed-17 contrast live at one threshold?
2. Paired uncertainty: base-case cluster bootstrap CIs for every rate, and
   paired method-vs-method deltas (counterfactual vs original-only, and
   seed vs seed) computed per shared base case.
3. Verbosity: claims per readout, content tokens per readout, and the
   unsupported-claim rate. Recall/phantom both rise when a model simply
   says more, so seed gaps must be read next to these columns.

CPU only; no torch import.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_ddxplus_e5_readout_pilot import (
    contrastive_cue_hit,
    cue_hit,
    emitted_text,
    parse_named_path,
)
from scripts.summarize_cue_position_readouts import content_tokens
from src.jsonl import read_jsonl
from src.paired_stats import paired_delta_ci

DELETION_METRICS = (
    "original_target_hit_rate",
    "deleted_target_phantom_rate",
    "deletion_contrast",
    "removal_success_given_original_hit",
)
VALUE_METRICS = (
    "replacement_hit_rate",
    "original_persistence_rate",
    "clean_switch_rate",
)


def claim_texts(row: dict[str, Any]) -> list[str]:
    """Bullet lines of the emitted readout; the whole text if none parse."""
    raw = str(row.get("nla_output") or row.get("raw_nla_output") or "")
    claims = [
        line.strip()[2:].strip()
        for line in raw.splitlines()
        if line.strip().startswith("- ")
    ]
    if claims:
        return [claim for claim in claims if claim]
    whole = emitted_text(row)
    return [whole] if whole else []


def claim_supported(claim: str, cue_union: set[str], threshold: float) -> bool:
    tokens = content_tokens(claim)
    if not tokens:
        return False
    return len(tokens & cue_union) / len(tokens) >= threshold


def base_stats(
    group: dict[str, dict[str, Any]], threshold: float
) -> dict[str, Any]:
    """Reduce one base case's rows to the numbers the bootstrap resamples."""
    original = group.get("original")
    if original is None:
        raise ValueError("base case has no original row")
    original_text = emitted_text(original)

    row_stats = []
    for variant, row in group.items():
        text = emitted_text(row)
        cues = [str(cue) for cue in row.get("cue_targets") or [] if str(cue).strip()]
        recall = (
            sum(cue_hit(cue, text, threshold) for cue in cues) / len(cues)
            if cues
            else None
        )
        claims = claim_texts(row)
        cue_union: set[str] = set()
        for cue in cues:
            cue_union |= content_tokens(cue)
        supported = sum(claim_supported(claim, cue_union, threshold) for claim in claims)
        row_stats.append(
            {
                "variant": variant,
                "recall": recall,
                "claims": len(claims),
                "tokens": len(content_tokens(text)),
                "supported_claims": supported,
            }
        )

    deletion = None
    deleted = group.get("cue_deleted")
    if deleted is not None:
        target = str(deleted.get("cf_original_cue") or "")
        hit = int(cue_hit(target, original_text, threshold))
        phantom = int(cue_hit(target, emitted_text(deleted), threshold))
        deletion = {"hit": hit, "phantom": phantom}

    value = None
    edited = group.get("value_edited")
    if edited is not None:
        old = str(edited.get("cf_original_cue") or "")
        new = str(edited.get("cf_replacement_cue") or "")
        edited_text = emitted_text(edited)
        old_after = int(contrastive_cue_hit(old, new, edited_text, threshold))
        new_after = int(contrastive_cue_hit(new, old, edited_text, threshold))
        value = {
            "new_after": new_after,
            "old_after": old_after,
            "clean_switch": int(new_after and not old_after),
        }

    return {"rows": row_stats, "deletion": deletion, "value": value}


def load_bases(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in read_jsonl(path):
        base_id = str(row.get("base_id") or "")
        variant = str(row.get("variant") or "")
        if not base_id or not variant:
            raise ValueError(f"Missing base_id/variant on row {row.get('id')!r}")
        if variant in by_base[base_id]:
            raise ValueError(f"Duplicate {base_id}/{variant}")
        by_base[base_id][variant] = row
    return dict(by_base)


def safe_rate(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def aggregate(bases: list[dict[str, Any]]) -> dict[str, float | None]:
    recalls = [
        row["recall"]
        for base in bases
        for row in base["rows"]
        if row["recall"] is not None
    ]
    claims = [row["claims"] for base in bases for row in base["rows"]]
    tokens = [row["tokens"] for base in bases for row in base["rows"]]
    total_claims = sum(claims)
    supported = sum(row["supported_claims"] for base in bases for row in base["rows"])

    deletions = [base["deletion"] for base in bases if base["deletion"] is not None]
    hits = sum(item["hit"] for item in deletions)
    phantoms = sum(item["phantom"] for item in deletions)
    removed_after_hit = sum(
        item["hit"] and not item["phantom"] for item in deletions
    )

    values = [base["value"] for base in bases if base["value"] is not None]

    hit_rate = safe_rate(hits, len(deletions))
    phantom_rate = safe_rate(phantoms, len(deletions))
    return {
        "rows": float(sum(len(base["rows"]) for base in bases)),
        "mean_current_finding_recall": safe_rate(sum(recalls), len(recalls)),
        "mean_claims_per_readout": safe_rate(total_claims, len(claims)),
        "mean_content_tokens_per_readout": safe_rate(sum(tokens), len(tokens)),
        "unsupported_claim_rate": (
            1 - supported / total_claims if total_claims else None
        ),
        "deletion_pairs": float(len(deletions)),
        "original_target_hit_rate": hit_rate,
        "deleted_target_phantom_rate": phantom_rate,
        "deletion_contrast": (
            hit_rate - phantom_rate
            if hit_rate is not None and phantom_rate is not None
            else None
        ),
        "removal_success_given_original_hit": safe_rate(removed_after_hit, hits),
        "value_pairs": float(len(values)),
        "replacement_hit_rate": safe_rate(
            sum(item["new_after"] for item in values), len(values)
        ),
        "original_persistence_rate": safe_rate(
            sum(item["old_after"] for item in values), len(values)
        ),
        "clean_switch_rate": safe_rate(
            sum(item["clean_switch"] for item in values), len(values)
        ),
    }


def percentile_ci(values: list[float], alpha: float = 0.05) -> tuple[float, float]:
    ordered = sorted(values)
    n = len(ordered)
    lo = int((alpha / 2) * n)
    hi = int((1 - alpha / 2) * n) - 1
    return ordered[max(0, min(lo, n - 1))], ordered[max(0, min(hi, n - 1))]


def bootstrap_method(
    bases: list[dict[str, Any]], draws: int, seed: int
) -> dict[str, Any]:
    point = aggregate(bases)
    rng = random.Random(seed)
    samples: dict[str, list[float]] = defaultdict(list)
    for _ in range(draws):
        resampled = [bases[rng.randrange(len(bases))] for _ in bases]
        estimate = aggregate(resampled)
        for key, value in estimate.items():
            if value is not None:
                samples[key].append(value)
    intervals = {
        key: percentile_ci(values) for key, values in samples.items() if values
    }
    return {"point": point, "ci95": intervals}


def paired_comparisons(
    stats: dict[str, dict[str, dict[str, Any]]],
    pairs: list[tuple[str, str]],
    threshold: float,
    draws: int,
    seed: int,
) -> list[dict[str, Any]]:
    del threshold  # stats are already reduced at the requested threshold
    results = []
    for left, right in pairs:
        shared = sorted(set(stats[left]) & set(stats[right]))
        entry: dict[str, Any] = {
            "left": left,
            "right": right,
            "shared_base_cases": len(shared),
            "deltas": {},
        }

        def per_base(name: str, extract) -> list[tuple[float, float]]:
            collected = []
            for base_id in shared:
                a = extract(stats[left][base_id])
                b = extract(stats[right][base_id])
                if a is not None and b is not None:
                    collected.append((a, b))
            return collected

        def base_recall(base: dict[str, Any]) -> float | None:
            recalls = [
                row["recall"] for row in base["rows"] if row["recall"] is not None
            ]
            return sum(recalls) / len(recalls) if recalls else None

        extractors = {
            "mean_current_finding_recall": base_recall,
            "deletion_contrast": lambda base: (
                base["deletion"]["hit"] - base["deletion"]["phantom"]
                if base["deletion"]
                else None
            ),
            "deleted_target_phantom_rate": lambda base: (
                float(base["deletion"]["phantom"]) if base["deletion"] else None
            ),
            "mean_claims_per_readout": lambda base: (
                sum(row["claims"] for row in base["rows"]) / len(base["rows"])
                if base["rows"]
                else None
            ),
        }
        for name, extract in extractors.items():
            pairs_for_metric = per_base(name, extract)
            if pairs_for_metric:
                entry["deltas"][name] = {
                    "pairs": len(pairs_for_metric),
                    **paired_delta_ci(pairs_for_metric, draws=draws, seed=seed),
                }
        results.append(entry)
    return results


def format_value(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def write_summary(
    path: Path,
    report: dict[str, Any],
    methods: list[str],
) -> None:
    lines = [
        "# CF readout uncertainty audit",
        "",
        f"Draws: {report['draws']}, seed {report['seed']}. "
        "Same hit instrument as score_ddxplus_e5_readout_pilot.py; "
        "cluster bootstrap resamples base cases.",
        "",
    ]
    for threshold_key, block in report["thresholds"].items():
        lines.append(f"## Threshold {threshold_key}")
        lines.append("")
        header = "| metric | " + " | ".join(methods) + " |"
        lines.append(header)
        lines.append("|---|" + "---|" * len(methods))
        metric_names = (
            ("mean_current_finding_recall",)
            + DELETION_METRICS
            + VALUE_METRICS
            + (
                "mean_claims_per_readout",
                "mean_content_tokens_per_readout",
                "unsupported_claim_rate",
            )
        )
        for metric in metric_names:
            cells = []
            for method in methods:
                result = block["methods"][method]
                point = result["point"].get(metric)
                ci = result["ci95"].get(metric)
                if point is None:
                    cells.append("n/a")
                elif ci is None:
                    cells.append(format_value(point))
                else:
                    cells.append(
                        f"{point:.4f} [{ci[0]:.4f}, {ci[1]:.4f}]"
                    )
            lines.append(f"| {metric} | " + " | ".join(cells) + " |")
        lines.append("")
        if block["comparisons"]:
            lines.append("### Paired deltas (left - right, per shared base case)")
            lines.append("")
            lines.append("| pair | metric | n | delta | 95% CI |")
            lines.append("|---|---|---:|---:|---|")
            for comparison in block["comparisons"]:
                for metric, delta in comparison["deltas"].items():
                    lines.append(
                        f"| {comparison['left']} vs {comparison['right']} "
                        f"| {metric} | {delta['pairs']} "
                        f"| {delta['delta']:.4f} "
                        f"| [{delta['lo']:.4f}, {delta['hi']:.4f}] |"
                    )
            lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--readout", action="append", required=True, type=parse_named_path
    )
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[0.3, 0.5, 0.7]
    )
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--compare",
        action="append",
        default=None,
        help="METHOD_A:METHOD_B paired delta; default is every method pair",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    methods = [name for name, _ in args.readout]
    if len(set(methods)) != len(methods):
        raise SystemExit("duplicate --readout method names")
    grouped = {name: load_bases(path) for name, path in args.readout}

    if args.compare:
        pairs = []
        for spec in args.compare:
            left, _, right = spec.partition(":")
            if left not in grouped or right not in grouped:
                raise SystemExit(f"--compare names unknown method: {spec}")
            pairs.append((left, right))
    else:
        pairs = list(itertools.combinations(methods, 2))

    report: dict[str, Any] = {
        "draws": args.draws,
        "seed": args.seed,
        "thresholds": {},
    }
    for threshold in args.thresholds:
        stats = {
            name: {
                base_id: base_stats(group, threshold)
                for base_id, group in bases.items()
            }
            for name, bases in grouped.items()
        }
        block = {
            "methods": {
                name: bootstrap_method(
                    list(per_base.values()), args.draws, args.seed
                )
                for name, per_base in stats.items()
            },
            "comparisons": paired_comparisons(
                stats, pairs, threshold, args.draws, args.seed
            ),
        }
        report["thresholds"][f"{threshold:g}"] = block

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    write_summary(args.summary_md, report, methods)
    print(f"[done] thresholds={args.thresholds} out={args.summary_md}", flush=True)


if __name__ == "__main__":
    main()
