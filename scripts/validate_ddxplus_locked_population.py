"""Validate the frozen DDXPlus E5 locked population without scoring outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_counts(values: list[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        name, separator, raw_count = value.partition("=")
        if not separator or not name or name in result:
            raise ValueError(f"Invalid or duplicate count specification: {value!r}")
        result[name] = int(raw_count)
    return result


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("case_id") or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--expected-variant", action="append", default=[])
    parser.add_argument("--expected-layer", type=int)
    parser.add_argument(
        "--require-activation-files",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    if len(rows) != args.expected_rows:
        raise ValueError(f"Read {len(rows)} rows; expected {args.expected_rows}")

    seen: set[str] = set()
    variants: Counter[str] = Counter()
    families: dict[str, Counter[str]] = defaultdict(Counter)
    missing_paths = 0
    for row in rows:
        row_id = str(row.get("id") or "").strip()
        if not row_id or row_id in seen:
            raise ValueError(f"Missing or duplicate row ID: {row_id!r}")
        seen.add(row_id)
        variant = str(row.get("variant") or "original")
        variants[variant] += 1
        identifier = base_id(row)
        if not identifier:
            raise ValueError(f"Row {row_id!r} has no base_id")
        families[identifier][variant] += 1
        if str(row.get("position_family") or "") != "P0":
            raise ValueError(f"Non-P0 row in locked population: {row_id}")
        if str(row.get("condition") or "") != "cot":
            raise ValueError(f"Non-CoT row in locked population: {row_id}")
        if args.expected_layer is not None:
            if int(row.get("layer", -1)) != args.expected_layer:
                raise ValueError(f"Layer mismatch for {row_id}")
        if args.require_activation_files:
            path = Path(str(row.get("activation_path") or ""))
            if not path.is_file():
                missing_paths += 1

    expected_variants = parse_counts(args.expected_variant)
    if expected_variants and variants != Counter(expected_variants):
        raise ValueError(
            f"Variant counts {dict(variants)} != expected {expected_variants}"
        )
    for identifier, counts in families.items():
        if counts["original"] != 1 or counts["cue_deleted"] != 1:
            raise ValueError(
                f"Incomplete original/deletion family {identifier}: {dict(counts)}"
            )
        if counts["value_edited"] not in (0, 1) or sum(counts.values()) not in (2, 3):
            raise ValueError(f"Invalid family {identifier}: {dict(counts)}")
    if missing_paths:
        raise FileNotFoundError(f"Missing activation files: {missing_paths}")

    report = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "rows": len(rows),
        "base_cases": len(families),
        "variants": dict(sorted(variants.items())),
        "position_family": "P0",
        "condition": "cot",
        "layer": args.expected_layer,
        "activation_files_verified": bool(args.require_activation_files),
        "missing_activation_files": 0,
        "population_exact": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[population] rows={len(rows)} bases={len(families)} "
        f"variants={dict(variants)} report={args.report}",
        flush=True,
    )


if __name__ == "__main__":
    main()
