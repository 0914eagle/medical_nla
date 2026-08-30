"""Merge semantic-judge shards against an immutable request population."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def merge_judgements(
    requests_path: Path,
    judgement_paths: list[Path],
    output_path: Path,
    expected_model: str,
    report_path: Path,
) -> dict[str, Any]:
    requests = list(read_jsonl(requests_path))
    request_ids = [str(row.get("id") or "").strip() for row in requests]
    if not request_ids or any(not row_id for row_id in request_ids):
        raise ValueError("Requests contain a missing ID or are empty")
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("Requests contain duplicate IDs")

    by_id: dict[str, dict[str, Any]] = {}
    models: set[str] = set()
    for path in judgement_paths:
        for row in read_jsonl(path):
            row_id = str(row.get("id") or "").strip()
            if not row_id:
                raise ValueError(f"Missing judgement ID in {path}")
            if row_id in by_id:
                raise ValueError(f"Duplicate judgement ID: {row_id}")
            model = str(row.get("judge_model") or "").strip()
            if not model:
                raise ValueError(f"Missing judge_model for {row_id}")
            if not str(row.get("response") or "").strip():
                raise ValueError(f"Empty response for {row_id}")
            models.add(model)
            by_id[row_id] = row

    missing = sorted(set(request_ids) - set(by_id))
    extra = sorted(set(by_id) - set(request_ids))
    if missing or extra:
        raise ValueError(
            f"Judgement population mismatch: missing={len(missing)} "
            f"extra={len(extra)}"
        )
    if models != {expected_model}:
        raise ValueError(
            f"Judge-model mismatch: observed={sorted(models)} "
            f"expected={[expected_model]}"
        )

    ordered = [by_id[row_id] for row_id in request_ids]
    write_jsonl(output_path, ordered)
    report = {
        "schema_version": 1,
        "requests": str(requests_path),
        "requests_sha256": sha256_file(requests_path),
        "judgement_shards": [str(path) for path in judgement_paths],
        "expected_model": expected_model,
        "rows": len(ordered),
        "exact_request_population": True,
        "duplicate_ids": 0,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--judgement", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-model", required=True)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    report = merge_judgements(
        args.requests,
        args.judgement,
        args.output,
        args.expected_model,
        args.report,
    )
    print(
        f"[merge-judgements] shards={len(args.judgement)} "
        f"rows={report['rows']} model={args.expected_model} "
        f"output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
