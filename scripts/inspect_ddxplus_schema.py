"""Inspect local DDXPlus patient/evidence files before conversion.

DDXPlus releases are commonly distributed as patient CSV files plus an
evidence metadata JSON file. This script prints the actual column/key schema so
the conversion step can be configured without guessing.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_patient_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in islice(f, limit):
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        data = read_json(path)
        if isinstance(data, list):
            return list(islice(data, limit))
        if isinstance(data, dict):
            for key in ("data", "rows", "patients"):
                if isinstance(data.get(key), list):
                    return list(islice(data[key], limit))
        raise ValueError(f"Could not find patient rows in JSON file {path}")

    with path.open(encoding="utf-8", newline="") as f:
        return list(islice(csv.DictReader(f), limit))


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def evidence_items(value: Any) -> list[str]:
    parsed = parse_maybe_json(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        return [str(item) for item in parsed]
    if isinstance(parsed, str):
        for sep in (";", "|"):
            if sep in parsed:
                return [part.strip() for part in parsed.split(sep) if part.strip()]
        if "," in parsed:
            return [part.strip() for part in parsed.split(",") if part.strip()]
        if parsed.strip():
            return [parsed.strip()]
    return []


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True, help="DDXPlus patients CSV/JSON/JSONL")
    parser.add_argument("--evidences", required=True, help="DDXPlus evidence metadata JSON")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    patient_path = Path(args.patients)
    evidence_path = Path(args.evidences)
    rows = read_patient_rows(patient_path, args.limit)
    evidences = read_json(evidence_path)

    print(f"patients_file: {patient_path}")
    print(f"sample_rows: {len(rows)}")
    if rows:
        print(f"patient_columns: {list(rows[0].keys())}")
        print("\npatient_samples:")
        for row in rows[: args.limit]:
            print(json.dumps(row, ensure_ascii=False)[:2000])

    print(f"\nevidences_file: {evidence_path}")
    if isinstance(evidences, dict):
        print(f"evidence_count: {len(evidences)}")
        keys = list(evidences.keys())
        print(f"first_evidence_ids: {keys[:10]}")
        meta_key_counts = Counter()
        for meta in evidences.values():
            if isinstance(meta, dict):
                meta_key_counts.update(meta.keys())
        print(f"common_evidence_meta_keys: {meta_key_counts.most_common(30)}")
        print("\nevidence_samples:")
        for key in keys[: args.limit]:
            print(key, json.dumps(evidences[key], ensure_ascii=False)[:2000])
    else:
        print(f"evidences_type: {type(evidences).__name__}")
        print(json.dumps(evidences, ensure_ascii=False)[:4000])

    evidence_col = None
    if rows:
        lowered = {key.lower(): key for key in rows[0]}
        for candidate in ("evidences", "evidence", "symptoms"):
            if candidate in lowered:
                evidence_col = lowered[candidate]
                break
    if evidence_col:
        print(f"\ndetected_evidence_column: {evidence_col}")
        for idx, row in enumerate(rows[: args.limit]):
            items = evidence_items(row.get(evidence_col))
            print(f"row_{idx}_evidence_count: {len(items)}")
            print(f"row_{idx}_evidences: {items[:20]}")


if __name__ == "__main__":
    main()
