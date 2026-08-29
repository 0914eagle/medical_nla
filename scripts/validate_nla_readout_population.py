"""Validate that NLA readouts exactly cover a canonical activation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or "").strip()


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result = {}
    for row in rows:
        identifier = row_id(row)
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate {label} ID: {identifier!r}")
        result[identifier] = row
    return result


def parse_counts(specs: list[str]) -> dict[str, int]:
    counts = {}
    for spec in specs:
        variant, separator, raw_count = spec.partition("=")
        if not separator or not variant or variant in counts:
            raise ValueError(f"Invalid or duplicate --expected-variant {spec!r}")
        counts[variant] = int(raw_count)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--readout", action="append", required=True, type=Path)
    parser.add_argument("--expected-rows", required=True, type=int)
    parser.add_argument("--expected-variant", action="append", default=[])
    parser.add_argument("--expected-max-new-tokens", required=True, type=int)
    parser.add_argument("--expected-do-sample", choices=("true", "false"), default="false")
    parser.add_argument("--expected-actor-prompt-file", type=Path)
    parser.add_argument("--expected-model-revision")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    manifest_rows = list(read_jsonl(args.manifest))
    manifest = index_unique(manifest_rows, "manifest")
    if len(manifest) != args.expected_rows:
        raise ValueError(f"Manifest has {len(manifest)} rows; expected {args.expected_rows}")
    missing_activations = sum(
        not Path(str(row.get("activation_path") or "")).is_file() for row in manifest_rows
    )
    if missing_activations:
        raise FileNotFoundError(f"Manifest has {missing_activations} missing activation files")
    expected_variants = parse_counts(args.expected_variant)
    manifest_variants = Counter(
        str(row.get("variant") or "original") for row in manifest_rows
    )
    if expected_variants and manifest_variants != Counter(expected_variants):
        raise ValueError(
            f"Manifest variants {dict(manifest_variants)} != expected {expected_variants}"
        )

    output_rows = [row for path in args.readout for row in read_jsonl(path)]
    outputs = index_unique(output_rows, "readout")
    missing = manifest.keys() - outputs.keys()
    extra = outputs.keys() - manifest.keys()
    if missing or extra:
        raise ValueError(f"Readout population mismatch: missing={len(missing)} extra={len(extra)}")
    if len(outputs) != args.expected_rows:
        raise ValueError(f"Readouts have {len(outputs)} rows; expected {args.expected_rows}")
    expected_sample = args.expected_do_sample == "true"
    query_hashes = set()
    sidecars = set()
    actor_prompt_files = set()
    for identifier, row in outputs.items():
        source = manifest[identifier]
        if str(row.get("variant") or "original") != str(source.get("variant") or "original"):
            raise ValueError(f"Variant mismatch for {identifier}")
        config = row.get("gen_config") or {}
        if int(config.get("max_new_tokens", -1)) != args.expected_max_new_tokens:
            raise ValueError(f"max_new_tokens mismatch for {identifier}")
        if bool(config.get("do_sample", False)) != expected_sample:
            raise ValueError(f"do_sample mismatch for {identifier}")
        if row.get("adapter_id") not in (None, ""):
            raise ValueError(f"Vanilla readout unexpectedly used adapter for {identifier}")
        query_hashes.add(hashlib.sha256(str(row.get("query") or "").encode()).hexdigest())
        sidecars.add(str(row.get("sidecar_path") or ""))
        actor_prompt_files.add(str(row.get("actor_prompt_template_file") or ""))
    if len(query_hashes) != 1:
        raise ValueError(f"Readouts used {len(query_hashes)} actor prompt queries")
    if len(sidecars) != 1 or "" in sidecars:
        raise ValueError(f"Readouts used inconsistent sidecars: {sorted(sidecars)}")
    if args.expected_actor_prompt_file is not None:
        expected_prompt = str(args.expected_actor_prompt_file)
        if actor_prompt_files != {expected_prompt}:
            raise ValueError(
                f"Actor prompt file mismatch: {sorted(actor_prompt_files)} != {expected_prompt}"
            )
    if args.expected_model_revision is not None:
        sidecar = next(iter(sidecars))
        marker = "/snapshots/"
        revision = sidecar.split(marker, 1)[1].split("/", 1)[0] if marker in sidecar else ""
        if revision != args.expected_model_revision:
            raise ValueError(
                f"Model snapshot mismatch: {revision!r} != {args.expected_model_revision!r}"
            )

    report: dict[str, Any] = {
        "schema_version": 1,
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "readouts": [
            {"path": str(path), "sha256": sha256_file(path)} for path in args.readout
        ],
        "rows": len(outputs),
        "variants": dict(sorted(manifest_variants.items())),
        "missing_activation_files": 0,
        "gen_config": {
            "max_new_tokens": args.expected_max_new_tokens,
            "do_sample": expected_sample,
        },
        "query_sha256": next(iter(query_hashes)),
        "actor_prompt_template_file": next(iter(actor_prompt_files)),
        "sidecar_path": next(iter(sidecars)),
        "population_exact": True,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[validated] rows={len(outputs)} report={args.report}")


if __name__ == "__main__":
    main()
