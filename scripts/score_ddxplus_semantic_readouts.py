"""Prepare and finalize quote-constrained semantic mapping of DDXPlus readouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_ddxplus_structured_reader import evaluate_readouts, write_summary
from src.ddxplus_semantic_mapping import (
    canonical_json,
    make_batch_prompt,
    materialize_items,
    parse_batch_response,
    prepare_items,
    protocol_cache_key,
    sha256_file,
)
from src.jsonl import read_jsonl, write_jsonl


def read_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    alias_path = Path(protocol["alias_table"]["path"])
    ontology_path = Path(protocol["ontology"]["path"])
    prompt_path = Path(protocol["mapper_prompt"]["path"])
    for item, artifact in (
        (alias_path, protocol["alias_table"]),
        (ontology_path, protocol["ontology"]),
        (prompt_path, protocol["mapper_prompt"]),
        (
            Path(protocol["structured_protocol"]["path"]),
            protocol["structured_protocol"],
        ),
        (
            Path(protocol["release_evidences"]["path"]),
            protocol["release_evidences"],
        ),
    ):
        if sha256_file(item) != artifact["sha256"]:
            raise ValueError(f"Frozen mapper artifact hash mismatch: {item}")
    return (
        protocol,
        json.loads(alias_path.read_text(encoding="utf-8")),
        json.loads(ontology_path.read_text(encoding="utf-8")),
        prompt_path.read_text(encoding="utf-8"),
    )


def readout_text(row: dict[str, Any]) -> str:
    return str(row.get("nla_output") or row.get("observed") or row.get("response") or "")


def prepare(args: argparse.Namespace) -> None:
    protocol, alias_table, ontology, template = read_protocol(args.protocol)
    source = list(read_jsonl(args.readouts))
    items = [
        {
            "id": str(row["id"]),
            "base_id": str(row.get("base_id") or row["id"]),
            "variant": str(row.get("variant") or "original"),
            "text": readout_text(row),
        }
        for row in source
    ]
    if len({row["id"] for row in items}) != len(items):
        raise ValueError("Readouts contain duplicate IDs")
    prepared, residual = prepare_items(items, alias_table)
    ordered = [{"claim_id": key, "claim": residual[key]} for key in sorted(residual)]
    requests = []
    batch_size = int(protocol["batch_size"])
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        prompt = make_batch_prompt(batch, ontology, template)
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        request_id = f"semantic_batch_{start // batch_size:06d}_{request_hash}"
        requests.append(
            {
                "id": request_id,
                "prompt": prompt,
                "claim_ids": [item["claim_id"] for item in batch],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "prepared_items.jsonl", prepared)
    write_jsonl(args.out_dir / "semantic_requests.jsonl", requests)
    report = {
        "schema_version": 1,
        "readout_rows": len(items),
        "claims": sum(len(row["claims"]) for row in prepared),
        "unique_residual_claims": len(residual),
        "requests": len(requests),
        "batch_size": batch_size,
        "readouts_sha256": sha256_file(args.readouts),
        "protocol_sha256": sha256_file(args.protocol),
        "locked_test_read": args.population == "locked_test",
    }
    (args.out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[prepare] rows={len(items)} residual={len(residual)} "
        f"requests={len(requests)}"
    )


def semantic_decisions(
    prepared: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    judgements: list[dict[str, Any]],
    *,
    protocol: dict[str, Any],
    ontology: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], str, list[dict[str, Any]]]:
    expected_claims = {
        claim["claim_id"]: claim["text"]
        for row in prepared
        for claim in row["claims"]
        if not claim["lexical_mappings"]
    }
    by_request = {str(row["id"]): row for row in requests}
    by_judgement = {str(row["id"]): row for row in judgements}
    if set(by_request) != set(by_judgement):
        raise ValueError("Semantic judgement population mismatch")
    models = {str(row.get("judge_model") or "") for row in judgements}
    if len(models) != 1 or not next(iter(models)):
        raise ValueError(f"Expected one non-empty mapper model ID, got {models}")
    model = next(iter(models))
    decisions: dict[str, list[dict[str, Any]]] = {}
    audit = []
    for request_id in sorted(by_request):
        request = by_request[request_id]
        subset = {claim_id: expected_claims[claim_id] for claim_id in request["claim_ids"]}
        parsed = parse_batch_response(
            by_judgement[request_id].get("response"),
            expected_claims=subset,
            ontology=ontology,
        )
        for claim_id, mappings in parsed.items():
            cache_key = protocol_cache_key(
                subset[claim_id],
                ontology_sha256=protocol["ontology"]["sha256"],
                alias_sha256=protocol["alias_table"]["sha256"],
                prompt_sha256=protocol["mapper_prompt"]["sha256"],
                model_id=model,
            )
            decisions[claim_id] = mappings
            audit.append(
                {
                    "claim_id": claim_id,
                    "claim": subset[claim_id],
                    "cache_key": cache_key,
                    "mappings": mappings,
                    "model_id": model,
                    "request_id": request_id,
                }
            )
    if set(decisions) != set(expected_claims):
        raise ValueError("Not every residual claim received a semantic decision")
    return decisions, model, audit


def finalize(args: argparse.Namespace) -> None:
    protocol, _alias, ontology, _template = read_protocol(args.protocol)
    prepared = list(read_jsonl(args.prepared_items))
    requests = list(read_jsonl(args.requests))
    judgements = list(read_jsonl(args.judgements))
    if requests:
        decisions, model, audit = semantic_decisions(
            prepared, requests, judgements, protocol=protocol, ontology=ontology
        )
    else:
        decisions, audit = {}, []
        if args.mapper_receipt is None:
            raise ValueError("A lexical-only run still requires the frozen mapper receipt")
        frozen_receipt = json.loads(args.mapper_receipt.read_text(encoding="utf-8"))
        model = str(frozen_receipt.get("primary_model_id") or "")
    if args.mapper_receipt is not None:
        receipt = json.loads(args.mapper_receipt.read_text(encoding="utf-8"))
        if receipt.get("all_gates_passed") is not True:
            raise ValueError("Mapper receipt has not passed G1-G4")
        if receipt.get("protocol_sha256") != sha256_file(args.protocol):
            raise ValueError("Mapper receipt protocol hash mismatch")
        if str(receipt.get("primary_model_id") or "") != model:
            raise ValueError("Locked mapper model differs from validation receipt")
    outputs = materialize_items(prepared, decisions)
    manifest = list(read_jsonl(args.manifest))
    by_id = {str(row["id"]): row for row in manifest}
    if set(by_id) != {str(row["id"]) for row in outputs}:
        raise ValueError("Manifest/readout population mismatch")
    ordered_manifest = [by_id[str(row["id"])] for row in outputs]
    structured_protocol = json.loads(
        Path(protocol["structured_protocol"]["path"]).read_text(encoding="utf-8")
    )
    result = evaluate_readouts(
        ordered_manifest, outputs, structured_protocol, args.hard_pairs
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "mapped_readouts.jsonl", outputs)
    write_jsonl(args.out_dir / "semantic_decisions.jsonl", audit)
    report = {
        "schema_version": 1,
        "population": args.population,
        "mapper_model_id": model,
        "protocol_sha256": sha256_file(args.protocol),
        "readouts_sha256": args.readouts_sha256,
        "manifest_sha256": sha256_file(args.manifest),
        "metrics": result,
        "locked_test_read": args.population == "locked_test",
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", args.population, result)
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--readouts", required=True, type=Path)
    prep.add_argument("--protocol", required=True, type=Path)
    prep.add_argument("--population", choices=["validation", "locked_test"], required=True)
    prep.add_argument("--out-dir", required=True, type=Path)

    final = sub.add_parser("finalize")
    final.add_argument("--prepared-items", required=True, type=Path)
    final.add_argument("--requests", required=True, type=Path)
    final.add_argument("--judgements", required=True, type=Path)
    final.add_argument("--manifest", required=True, type=Path)
    final.add_argument("--protocol", required=True, type=Path)
    final.add_argument("--mapper-receipt", type=Path)
    final.add_argument("--hard-pairs", type=Path)
    final.add_argument("--readouts-sha256", required=True)
    final.add_argument("--population", choices=["validation", "locked_test"], required=True)
    final.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
