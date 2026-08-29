"""Reindex existing DiReCT source outputs to the frozen locked pools and score them.

This script never generates a new backbone response. It joins materialized
Direct/CoT outputs to the frozen ``test_seen`` and ``test_pdd_heldout`` files,
recomputes strict/category correctness from the frozen labels, and reports
patient-group cluster bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, token_f1
from src.jsonl import read_jsonl, write_jsonl


LOCKED_CONFIRMATION = "I_ACCEPT_DIRECT_LOCKED_SOURCE_REINDEX"
LOCKED_SPLITS = ("test_seen", "test_pdd_heldout")
EXPECTED_COUNTS = {"test_seen": 72, "test_pdd_heldout": 106}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def named_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected ARM=PATH")
    return label.strip(), Path(raw_path)


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "").strip()


def index_unique(rows: Iterable[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = case_id(row)
        if not identifier or identifier in indexed:
            raise ValueError(f"Missing or duplicate {label} ID: {identifier!r}")
        indexed[identifier] = row
    return indexed


def load_locked_splits(split_dir: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result = {}
    seen: set[str] = set()
    for split in LOCKED_SPLITS:
        path = split_dir / f"{split}.jsonl"
        rows = index_unique(read_jsonl(path), split)
        expected = EXPECTED_COUNTS[split]
        if len(rows) != expected:
            raise ValueError(f"{split} has {len(rows)} rows; expected {expected}")
        overlap = seen & rows.keys()
        if overlap:
            raise ValueError(f"Locked splits overlap on {len(overlap)} IDs")
        seen.update(rows)
        result[split] = rows
    return result


def aliases_for(answer_row: dict[str, Any], frozen_gold: str) -> list[str]:
    aliases = {
        str(value).strip()
        for value in answer_row.get("diagnosis_aliases") or []
        if str(value).strip()
    }
    old_gold = str(answer_row.get("diagnosis_name") or "").strip()
    if old_gold and old_gold != frozen_gold:
        aliases.discard(old_gold)
    return sorted(aliases, key=str.casefold)


def annotate(
    arm: str,
    split: str,
    frozen: dict[str, Any],
    answer_row: dict[str, Any],
) -> dict[str, Any]:
    answer = answer_row.get("answer")
    gold = str(frozen.get("canonical_pdd") or "")
    category = str(frozen.get("disease_category") or "")
    if not gold or not category:
        raise ValueError(f"Frozen row {case_id(frozen)} lacks PDD/category")
    aliases = aliases_for(answer_row, gold)
    return {
        "id": case_id(frozen),
        "patient_group": str(frozen.get("patient_group") or case_id(frozen)),
        "split": split,
        "arm": arm,
        "parsed": bool(answer_row.get("answer_parsed", answer)),
        "strict_correct": is_correct(answer, gold, aliases),
        "category_correct": is_correct(answer, category, []),
        "token_f1": float(token_f1(answer, gold, aliases)),
        "answer_forced": bool(answer_row.get("answer_forced")),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else math.nan


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def cluster_bootstrap(
    rows: list[dict[str, Any]], field: str, *, replicates: int, seed: int
) -> list[float]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[row["patient_group"]].append(row)
    groups = sorted(by_group)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        values = [float(row[field]) for group in sampled for row in by_group[group]]
        estimates.append(mean(values))
    return estimates


def summarize(rows: list[dict[str, Any]], *, replicates: int, seed: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "n": len(rows),
        "patient_groups": len({row["patient_group"] for row in rows}),
        "parse_rate": mean([float(row["parsed"]) for row in rows]),
        "strict_pdd_accuracy": mean([float(row["strict_correct"]) for row in rows]),
        "category_accuracy": mean([float(row["category_correct"]) for row in rows]),
        "mean_token_f1": mean([row["token_f1"] for row in rows]),
        "forced_rate": mean([float(row["answer_forced"]) for row in rows]),
    }
    for field, name in (
        ("strict_correct", "strict_pdd_ci95"),
        ("category_correct", "category_ci95"),
    ):
        estimates = cluster_bootstrap(rows, field, replicates=replicates, seed=seed)
        result[name] = [percentile(estimates, 0.025), percentile(estimates, 0.975)]
    return result


def exact_mcnemar(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, value) for value in range(min(left_only, right_only) + 1))
    return min(1.0, 2 * tail / (2**discordant))


def paired_summary(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    field: str,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    if left_by_id.keys() != right_by_id.keys():
        raise ValueError("Paired arms do not contain the same locked IDs")
    joined = []
    for identifier in sorted(left_by_id):
        lrow, rrow = left_by_id[identifier], right_by_id[identifier]
        joined.append(
            {
                "id": identifier,
                "patient_group": lrow["patient_group"],
                "left": bool(lrow[field]),
                "right": bool(rrow[field]),
            }
        )
    left_only = sum(row["left"] and not row["right"] for row in joined)
    right_only = sum(row["right"] and not row["left"] for row in joined)
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in joined:
        by_group[row["patient_group"]].append(row)
    groups = sorted(by_group)
    rng = random.Random(seed)
    differences = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        rows = [row for group in sampled for row in by_group[group]]
        differences.append(mean([float(row["right"]) - float(row["left"]) for row in rows]))
    return {
        "n": len(joined),
        "both_correct": sum(row["left"] and row["right"] for row in joined),
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "neither_correct": sum(not row["left"] and not row["right"] for row in joined),
        "right_minus_left": mean(
            [float(row["right"]) - float(row["left"]) for row in joined]
        ),
        "cluster_ci95": [percentile(differences, 0.025), percentile(differences, 0.975)],
        "mcnemar_exact_p": exact_mcnemar(left_only, right_only),
    }


def fmt(value: Any) -> str:
    return "N/A" if value is None or math.isnan(float(value)) else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--answers", action="append", required=True, type=named_path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.confirmation != LOCKED_CONFIRMATION:
        raise ValueError(f"Locked source reindex requires --confirmation {LOCKED_CONFIRMATION}")
    if args.bootstrap_replicates < 100:
        raise ValueError("--bootstrap-replicates must be at least 100")

    frozen_splits = load_locked_splits(args.split_dir)
    locked_ids = {identifier for rows in frozen_splits.values() for identifier in rows}
    paths_by_arm: dict[str, list[Path]] = defaultdict(list)
    for arm, path in args.answers:
        paths_by_arm[arm].append(path)
    if set(paths_by_arm) != {"direct", "cot"}:
        raise ValueError("Exactly direct and cot answer arms are required")

    annotated_by_arm: dict[str, list[dict[str, Any]]] = {}
    source_hashes: dict[str, list[dict[str, str]]] = {}
    for arm, paths in sorted(paths_by_arm.items()):
        all_rows = [row for path in paths for row in read_jsonl(path)]
        indexed = index_unique(all_rows, f"{arm} source answer")
        missing = locked_ids - indexed.keys()
        if missing:
            raise ValueError(f"{arm} misses {len(missing)} locked IDs")
        output_rows = []
        for split in LOCKED_SPLITS:
            for identifier, frozen in frozen_splits[split].items():
                output_rows.append(annotate(arm, split, frozen, indexed[identifier]))
        annotated_by_arm[arm] = output_rows
        source_hashes[arm] = [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ]

    report: dict[str, Any] = {
        "schema_version": 1,
        "locked_confirmation": args.confirmation,
        "split_hashes": {
            split: sha256_file(args.split_dir / f"{split}.jsonl") for split in LOCKED_SPLITS
        },
        "source_hashes": source_hashes,
        "bootstrap": {
            "unit": "patient_group",
            "replicates": args.bootstrap_replicates,
            "seed": args.seed,
        },
        "arms": {},
        "paired": {},
    }
    for arm, rows in annotated_by_arm.items():
        report["arms"][arm] = {
            split: summarize(
                [row for row in rows if row["split"] == split],
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            for split in LOCKED_SPLITS
        }
    for split in LOCKED_SPLITS:
        left = [row for row in annotated_by_arm["direct"] if row["split"] == split]
        right = [row for row in annotated_by_arm["cot"] if row["split"] == split]
        report["paired"][split] = {
            name: paired_summary(
                left,
                right,
                field,
                replicates=args.bootstrap_replicates,
                seed=args.seed,
            )
            for field, name in (
                ("strict_correct", "strict_pdd"),
                ("category_correct", "disease_category"),
            )
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(
        args.out_dir / "private_case_scores.jsonl",
        sorted(
            annotated_by_arm["direct"] + annotated_by_arm["cot"],
            key=lambda row: (row["split"], row["arm"], row["id"]),
        ),
    )
    lines = [
        "# DiReCT Frozen Source Diagnostic Behavior",
        "",
        "Existing source outputs reindexed to the frozen downstream split; no generation was run.",
        "",
        "| arm | pool | n | groups | parse | strict PDD | strict CI | category | category CI |",
        "|---|---|---:|---:|---:|---:|---|---:|---|",
    ]
    for arm in ("direct", "cot"):
        for split in LOCKED_SPLITS:
            item = report["arms"][arm][split]
            strict_ci = item["strict_pdd_ci95"]
            category_ci = item["category_ci95"]
            lines.append(
                f"| {arm} | {split} | {item['n']} | {item['patient_groups']} | "
                f"{fmt(item['parse_rate'])} | {fmt(item['strict_pdd_accuracy'])} | "
                f"[{fmt(strict_ci[0])}, {fmt(strict_ci[1])}] | "
                f"{fmt(item['category_accuracy'])} | "
                f"[{fmt(category_ci[0])}, {fmt(category_ci[1])}] |"
            )
    lines.extend(["", "## Paired Direct versus CoT", ""])
    for split in LOCKED_SPLITS:
        for metric, item in report["paired"][split].items():
            ci = item["cluster_ci95"]
            lines.append(
                f"- {split} / {metric}: CoT-Direct {item['right_minus_left']:+.4f} "
                f"(cluster 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]); "
                f"McNemar p={item['mcnemar_exact_p']:.4f}."
            )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
