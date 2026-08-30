"""Combine a completed DiReCT locked baseline batch into paper-ready tables."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("test_seen", "test_pdd_heldout")
METHODS = ("cot", "vanilla")
METRICS = (
    ("acc_diag", "Accdiag"),
    ("comp_pre", "Obspre"),
    ("comp_re", "Obsrec"),
    ("comp_coverage", "Obscomp"),
    ("faith_ob", "Expcom"),
    ("faith_all", "Expall"),
)


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def extraction_stats(path: Path, expected: int) -> dict[str, dict[str, int | float]]:
    rows = read_jsonl(path)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row.get("method"))].append(row)
    if set(by_method) != set(METHODS):
        raise ValueError(f"Extraction methods mismatch in {path}: {sorted(by_method)}")
    result = {}
    for method in METHODS:
        items = by_method[method]
        if len(items) != expected:
            raise ValueError(f"{method} extraction rows={len(items)}, expected={expected}")
        observation_rows = sum(bool(row.get("accepted_claims")) for row in items)
        observations = sum(len(row.get("accepted_claims") or []) for row in items)
        parse_errors = sum(bool(row.get("parse_error")) for row in items)
        result[method] = {
            "rows": len(items),
            "rows_with_observation": observation_rows,
            "extraction_coverage": observation_rows / len(items),
            "observations": observations,
            "parse_errors": parse_errors,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    source = load_json(args.root / "table1a_source" / "results.json")
    probes = load_json(args.root / "table1b_probes" / "results.json")
    if probes.get("population") != "test_seen" or probes.get("locked_test_read") is not True:
        raise ValueError("Table 1B report is not the frozen test-seen evaluation")

    table2: dict[str, dict[str, Any]] = {}
    expected_by_split = {"test_seen": 72, "test_pdd_heldout": 106}
    for split in SPLITS:
        expected = expected_by_split[split]
        stats = extraction_stats(args.root / split / "private_extraction_audit.jsonl", expected)
        table2[split] = {}
        for method in METHODS:
            report = load_json(args.root / split / "reports" / f"{method}.json")
            population = report.get("population") or {}
            if population.get("expected_predictions") != expected:
                raise ValueError(f"{split}/{method} expected population mismatch")
            metrics = report.get("metrics") or {}
            table2[split][method] = {
                **stats[method],
                "zero_scored": int(population.get("zero_scored", 0)),
                **{alias: float(metrics[field]["mean"]) for field, alias in METRICS},
            }

    output = {
        "schema_version": 1,
        "table_1a": source,
        "table_1b": probes,
        "table_2": table2,
        "medical_nla_locked_row": "excluded: no validation-promoted checkpoint",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    lines = ["# DiReCT Locked Paper Tables", "", "## Table 1A - Source Diagnostic Behavior", ""]
    lines.extend([
        "| arm | pool | n | strict PDD | category | token F1 |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for arm in ("direct", "cot"):
        for split in SPLITS:
            item = source["arms"][arm][split]
            lines.append(
                f"| {arm} | {split} | {item['n']} | {item['strict_pdd_accuracy']:.4f} | "
                f"{item['category_accuracy']:.4f} | {item['mean_token_f1']:.4f} |"
            )
    lines.extend(["", "## Table 1B - Frozen HS24 Probes", "", "| target | n | top-1 | shuffled | gap |", "|---|---:|---:|---:|---:|"])
    for item in probes["results"]:
        lines.append(
            f"| {item['label_field']} | {item['n']} | {item['own']['acc1']:.4f} | "
            f"{item['label_shuffle']['acc1']:.4f} | {item['own_minus_shuffle']:+.4f} |"
        )
    for split in SPLITS:
        lines.extend([
            "",
            f"## Table 2 - {split}",
            "",
            "| method | extraction | observations | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall | zero-scored |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            item = table2[split][method]
            lines.append(
                f"| {method} | {item['rows_with_observation']}/{item['rows']} "
                f"({item['extraction_coverage']:.4f}) | {item['observations']} | "
                f"{item['Accdiag']:.4f} | {item['Obspre']:.4f} | {item['Obsrec']:.4f} | "
                f"{item['Obscomp']:.4f} | {item['Expcom']:.4f} | {item['Expall']:.4f} | "
                f"{item['zero_scored']} |"
            )
    lines.extend(["", "Medical-NLA locked row: **excluded; no validation-promoted checkpoint**.", ""])
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[paper tables] {args.summary_md}")


if __name__ == "__main__":
    main()
