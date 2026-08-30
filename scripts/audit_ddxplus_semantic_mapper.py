"""Run validation-only G1-G4 audits for the DDXPlus semantic mapper."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from scripts.score_ddxplus_semantic_readouts import read_protocol, semantic_decisions
from src.ddxplus_semantic_mapping import (
    canonical_json,
    make_batch_prompt,
    materialize_items,
    prepare_items,
    sha256_file,
)
from src.jsonl import read_jsonl, write_jsonl


def selected(row: dict[str, Any]) -> set[str]:
    return {str(item["evidence_id"]) for item in row.get("selected_claims") or []}


def values(row: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["evidence_id"]): str(item["value_id"])
        for item in row.get("selected_claims") or []
        if item.get("value_id") is not None
    }


def micro_f1(predictions: list[set[str]], targets: list[set[str]]) -> float:
    tp = sum(len(pred & target) for pred, target in zip(predictions, targets, strict=True))
    fp = sum(len(pred - target) for pred, target in zip(predictions, targets, strict=True))
    fn = sum(len(target - pred) for pred, target in zip(predictions, targets, strict=True))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def wilson_upper(hits: int, total: int, z: float = 1.959963984540054) -> float | None:
    if total <= 0:
        return None
    proportion = hits / total
    denominator = 1 + z * z / total
    centre = proportion + z * z / (2 * total)
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
    return (centre + radius) / denominator


def stable_key(identifier: str, seed: int = 17) -> str:
    return hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()


def requests_for_residual(
    residual: dict[str, str], *, ontology: dict[str, Any], template: str, batch_size: int
) -> list[dict[str, Any]]:
    ordered = [{"claim_id": key, "claim": residual[key]} for key in sorted(residual)]
    requests = []
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        requests.append(
            {
                "id": f"semantic_batch_{start // batch_size:06d}",
                "prompt": make_batch_prompt(batch, ontology, template),
                "claim_ids": [item["claim_id"] for item in batch],
            }
        )
    return requests


def prepare(args: argparse.Namespace) -> None:
    protocol, alias_table, ontology, template = read_protocol(args.protocol)
    reader_rows = list(read_jsonl(args.reader_readouts))
    reader_originals = {
        str(row.get("base_id") or row["id"]): row
        for row in reader_rows
        if str(row.get("variant") or "original") == "original"
    }
    source_items = []
    for row in reader_rows:
        source_items.append(
            {
                "id": f"g1::{row['id']}",
                "audit_group": "G1",
                "source_id": str(row["id"]),
                "text": str(row.get("observed") or ""),
                "expected_claims": row.get("selected_claims") or [],
            }
        )

    g2_total = g2_no_target = 0
    for pair in read_jsonl(args.hard_pairs):
        if not pair.get("primary_pair_eligible", True):
            continue
        own = str(pair.get("own_base_id") or "")
        donor = str(pair.get("donor_base_id") or "")
        if own not in reader_originals or donor not in reader_originals:
            continue
        own_set = selected(reader_originals[own])
        donor_set = selected(reader_originals[donor])
        candidates = sorted(own_set - donor_set)
        if not candidates:
            g2_no_target += 1
            continue
        target = candidates[0]
        g2_total += 1
        source_items.append(
            {
                "id": f"g2::{own}::{donor}::{target}",
                "audit_group": "G2",
                "source_id": donor,
                "text": str(reader_originals[donor].get("observed") or ""),
                "target_evidence_id": target,
            }
        )

    open_rows = list(read_jsonl(args.open_readouts))
    if args.open_source_dataset:
        open_rows = [
            row
            for row in open_rows
            if str(row.get("source_dataset") or "") == args.open_source_dataset
        ]
    for row in open_rows:
        source_items.append(
            {
                "id": f"g4pool::{row['id']}",
                "audit_group": "G4_POOL",
                "source_id": str(row["id"]),
                "text": str(row.get("nla_output") or row.get("observed") or ""),
            }
        )
    prepared, residual = prepare_items(source_items, alias_table)
    requests = requests_for_residual(
        residual,
        ontology=ontology,
        template=template,
        batch_size=int(protocol["batch_size"]),
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "prepared_items.jsonl", prepared)
    write_jsonl(args.out_dir / "primary_requests.jsonl", requests)
    report = {
        "schema_version": 1,
        "reader_rows": len(reader_rows),
        "g2_eligible": g2_total,
        "g2_no_absent_target": g2_no_target,
        "open_rows": len(open_rows),
        "unique_residual_claims": len(residual),
        "primary_requests": len(requests),
        "estimated_input_characters": sum(len(row["prompt"]) for row in requests),
        "protocol_sha256": sha256_file(args.protocol),
        "locked_test_read": False,
    }
    (args.out_dir / "dry_run_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[prepare] reader={len(reader_rows)} g2={g2_total} open={len(open_rows)} "
        f"residual={len(residual)} requests={len(requests)}"
    )


def apply_primary(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol, _alias, ontology, template = read_protocol(args.protocol)
    prepared = list(read_jsonl(args.prepared_items))
    requests = list(read_jsonl(args.requests))
    judgements = list(read_jsonl(args.judgements))
    semantic, model, decision_audit = semantic_decisions(
        prepared, requests, judgements, protocol=protocol, ontology=ontology
    )
    mapped = materialize_items(prepared, semantic)

    g1 = [row for row in mapped if row["audit_group"] == "G1"]
    predictions = [selected(row) for row in g1]
    targets = [
        {str(item["evidence_id"]) for item in row.get("expected_claims") or []}
        for row in g1
    ]
    value_total = value_correct = 0
    for row in g1:
        predicted_values = values(row)
        for item in row.get("expected_claims") or []:
            if item.get("value_id") is None:
                continue
            value_total += 1
            value_correct += predicted_values.get(str(item["evidence_id"])) == str(
                item["value_id"]
            )
    finding_f1 = micro_f1(predictions, targets)
    value_accuracy = value_correct / value_total if value_total else None

    g2 = [row for row in mapped if row["audit_group"] == "G2"]
    false_maps = sum(row["target_evidence_id"] in selected(row) for row in g2)
    false_rate = false_maps / len(g2) if g2 else None

    cache_rows = sorted(decision_audit, key=lambda row: row["cache_key"])
    cache_path = args.out_dir / "primary_semantic_cache.jsonl"
    write_jsonl(cache_path, cache_rows)
    replay_rows = list(read_jsonl(cache_path))
    if len({row["cache_key"] for row in replay_rows}) != len(replay_rows):
        raise ValueError("Protocol-bound semantic cache contains duplicate keys")
    replay_semantic = {
        str(row["claim_id"]): list(row["mappings"]) for row in replay_rows
    }
    replay_a = canonical_json(materialize_items(prepared, semantic))
    replay_b = canonical_json(materialize_items(prepared, replay_semantic))

    g4_claim_ids = {
        claim["claim_id"]
        for row in prepared
        if row["audit_group"] == "G4_POOL"
        for claim in row["claims"]
        if not claim["lexical_mappings"]
    }
    g4_decisions = [
        row for row in decision_audit if row["claim_id"] in g4_claim_ids
    ]
    value_decisions = [
        row
        for row in g4_decisions
        if any(mapping.get("value_id") for mapping in row["mappings"])
    ]
    value_ids = {row["claim_id"] for row in value_decisions}
    mapped_decisions = [
        row
        for row in g4_decisions
        if row["mappings"] and row["claim_id"] not in value_ids
    ]
    null_decisions = [row for row in g4_decisions if not row["mappings"]]
    for bucket in (value_decisions, mapped_decisions, null_decisions):
        bucket.sort(key=lambda row: stable_key(row["claim_id"]))
    sample = value_decisions[:30]
    for bucket in (mapped_decisions, null_decisions, value_decisions[30:]):
        sample.extend(bucket[: max(0, 100 - len(sample))])
    if len(sample) < 100:
        raise ValueError(f"G4 requires 100 unique Stage-2 decisions; found {len(sample)}")
    sample = sample[:100]
    residual = {row["claim_id"]: row["claim"] for row in sample}
    auditor_requests = requests_for_residual(
        residual,
        ontology=ontology,
        template=template,
        batch_size=int(protocol["batch_size"]),
    )
    cold_sample = sorted(decision_audit, key=lambda row: stable_key(row["claim_id"]))[:20]
    cold_requests = requests_for_residual(
        {row["claim_id"]: row["claim"] for row in cold_sample},
        ontology=ontology,
        template=template,
        batch_size=int(protocol["batch_size"]),
    )
    report = {
        "schema_version": 1,
        "primary_model_id": model,
        "G1": {
            "finding_micro_f1": finding_f1,
            "finding_threshold": protocol["gates"]["G1_finding_micro_f1_min"],
            "native_value_accuracy": value_accuracy,
            "native_value_n": value_total,
            "value_threshold": protocol["gates"]["G1_native_value_accuracy_min"],
            "passed": finding_f1 >= protocol["gates"]["G1_finding_micro_f1_min"]
            and value_accuracy is not None
            and value_accuracy >= protocol["gates"]["G1_native_value_accuracy_min"],
        },
        "G2": {
            "eligible": len(g2),
            "false_maps": false_maps,
            "false_map_rate": false_rate,
            "wilson_95_upper": wilson_upper(false_maps, len(g2)),
            "threshold": protocol["gates"]["G2_false_map_max"],
            "passed": false_rate is not None
            and false_rate <= protocol["gates"]["G2_false_map_max"],
        },
        "G3": {
            "cache_replay_byte_identical": replay_a.encode() == replay_b.encode(),
            "cache_rows": len(cache_rows),
            "cache_sha256": sha256_file(cache_path),
            "cold_duplicate_agreement": None,
            "cold_duplicate_n": len(cold_sample),
            "passed": replay_a.encode() == replay_b.encode(),
        },
        "G4_sample": {
            "n": len(sample),
            "value_primary_n": sum(
                any(mapping.get("value_id") for mapping in row["mappings"])
                for row in sample
            ),
        },
        "protocol_sha256": sha256_file(args.protocol),
        "locked_test_read": False,
    }
    write_jsonl(args.out_dir / "primary_mapped_items.jsonl", mapped)
    write_jsonl(args.out_dir / "primary_semantic_decisions.jsonl", decision_audit)
    write_jsonl(args.out_dir / "g4_primary_sample.jsonl", sample)
    write_jsonl(args.out_dir / "auditor_requests.jsonl", auditor_requests)
    write_jsonl(args.out_dir / "g3_primary_sample.jsonl", cold_sample)
    write_jsonl(args.out_dir / "cold_requests.jsonl", cold_requests)
    (args.out_dir / "primary_gate_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def apply_cold(args: argparse.Namespace) -> None:
    protocol, _alias, ontology, _template = read_protocol(args.protocol)
    report = json.loads(args.primary_report.read_text(encoding="utf-8"))
    sample = list(read_jsonl(args.primary_sample))
    requests = list(read_jsonl(args.cold_requests))
    judgements = list(read_jsonl(args.cold_judgements))
    fake_prepared = [
        {
            "id": row["claim_id"],
            "text": row["claim"],
            "claims": [
                {
                    "claim_id": row["claim_id"],
                    "text": row["claim"],
                    "lexical_mappings": [],
                }
            ],
        }
        for row in sample
    ]
    cold, model, _audit = semantic_decisions(
        fake_prepared, requests, judgements, protocol=protocol, ontology=ontology
    )
    if model != report["primary_model_id"]:
        raise ValueError("Cold duplicate used a different primary model")
    agreements = [
        canonical_json(row["mappings"]) == canonical_json(cold[row["claim_id"]])
        for row in sample
    ]
    report["G3"]["cold_duplicate_agreement"] = (
        sum(agreements) / len(agreements) if agreements else None
    )
    report["G3"]["cold_duplicate_n"] = len(agreements)
    args.primary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[cold] n={len(agreements)} agreement="
        f"{report['G3']['cold_duplicate_agreement']}"
    )
    print(
        f"[primary] model={model} G1={report['G1']['passed']} "
        f"G2={report['G2']['passed']} G3={report['G3']['passed']}"
    )


def finalize(args: argparse.Namespace) -> None:
    protocol, _alias, ontology, _template = read_protocol(args.protocol)
    primary_report = json.loads(args.primary_report.read_text(encoding="utf-8"))
    sample = list(read_jsonl(args.primary_sample))
    requests = list(read_jsonl(args.auditor_requests))
    judgements = list(read_jsonl(args.auditor_judgements))
    fake_prepared = [
        {
            "id": row["claim_id"],
            "text": row["claim"],
            "claims": [
                {
                    "claim_id": row["claim_id"],
                    "text": row["claim"],
                    "lexical_mappings": [],
                }
            ],
        }
        for row in sample
    ]
    auditor, auditor_model, _audit = semantic_decisions(
        fake_prepared, requests, judgements, protocol=protocol, ontology=ontology
    )
    primary_model = str(primary_report["primary_model_id"])
    if not primary_model or not auditor_model or primary_model == auditor_model:
        raise ValueError("G4 requires distinct non-empty primary and auditor model IDs")
    if any("gemma" in model.casefold() for model in (primary_model, auditor_model)):
        raise ValueError("Gemma-family models cannot run G4")
    evidence_disagreement = 0
    value_disagreement = value_n = 0
    for row in sample:
        claim_id = row["claim_id"]
        primary_map = {item["evidence_id"]: item.get("value_id") for item in row["mappings"]}
        auditor_map = {item["evidence_id"]: item.get("value_id") for item in auditor[claim_id]}
        evidence_disagreement += set(primary_map) != set(auditor_map)
        for evidence, value_id in primary_map.items():
            if value_id is None:
                continue
            value_n += 1
            value_disagreement += auditor_map.get(evidence) != value_id
    evidence_rate = evidence_disagreement / len(sample)
    value_rate = value_disagreement / value_n if value_n else None
    gates = protocol["gates"]
    g4_pass = (
        len(sample) == 100
        and evidence_rate <= gates["G4_evidence_disagreement_max"]
        and value_n >= gates["G4_value_denominator_min"]
        and value_rate is not None
        and value_rate <= gates["G4_value_disagreement_max"]
    )
    final_gates = {
        name: primary_report[name] for name in ("G1", "G2", "G3")
    }
    final_gates["G4"] = {
        "n": len(sample),
        "primary_model_id": primary_model,
        "auditor_model_id": auditor_model,
        "evidence_disagreements": evidence_disagreement,
        "evidence_disagreement_rate": evidence_rate,
        "value_n": value_n,
        "value_disagreements": value_disagreement,
        "value_disagreement_rate": value_rate,
        "passed": g4_pass,
    }
    protocol_hash = sha256_file(args.protocol)
    receipt = {
        "schema_version": 1,
        "all_gates_passed": all(final_gates[name]["passed"] for name in ("G1", "G2", "G3", "G4")),
        "locked_test_read": False,
        "protocol_sha256": protocol_hash,
        "primary_model_id": primary_model,
        "auditor_model_id": auditor_model,
        "gates": final_gates,
        "alias_table": protocol["alias_table"],
        "mapper_prompt": protocol["mapper_prompt"],
        "ontology": protocol["ontology"],
        "scorer": {
            "path": str(args.scorer),
            "sha256": sha256_file(args.scorer),
        },
        "inputs": {
            "primary_report_sha256": sha256_file(args.primary_report),
            "primary_sample_sha256": sha256_file(args.primary_sample),
            "auditor_judgements_sha256": sha256_file(args.auditor_judgements),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    g1, g2, g3, g4 = (final_gates[name] for name in ("G1", "G2", "G3", "G4"))
    g4_value = g4["value_disagreement_rate"]
    g4_value_text = "N/A" if g4_value is None else f"{g4_value:.4f}"
    g4_value_threshold = gates["G4_value_disagreement_max"]
    g4_value_denominator = gates["G4_value_denominator_min"]
    summary = [
        "# DDXPlus Semantic Mapper Validation Gates",
        "",
        f"- primary model: `{primary_model}`",
        f"- auditor model: `{auditor_model}`",
        f"- all gates passed: **{receipt['all_gates_passed']}**",
        "- locked test read: **no**",
        "",
        "| gate | metric | result | threshold | pass |",
        "|---|---|---:|---:|---:|",
        "| G1 | reader finding micro F1 | "
        f"{g1['finding_micro_f1']:.4f} | >= {g1['finding_threshold']:.2f} | "
        f"{g1['passed']} |",
        "| G1 | reader native-value accuracy | "
        f"{g1['native_value_accuracy']:.4f} | >= {g1['value_threshold']:.2f} | "
        f"{g1['passed']} |",
        "| G2 | absent-target false map | "
        f"{g2['false_map_rate']:.4f} | <= {g2['threshold']:.2f} | "
        f"{g2['passed']} |",
        "| G3 | cache replay byte-identical | "
        f"{g3['cache_replay_byte_identical']} | true | {g3['passed']} |",
        "| G3 | cold duplicate agreement | "
        f"{g3['cold_duplicate_agreement']:.4f} | report only | - |",
        "| G4 | evidence disagreement | "
        f"{g4['evidence_disagreement_rate']:.4f} | "
        f"<= {gates['G4_evidence_disagreement_max']:.2f} | {g4['passed']} |",
        "| G4 | conditional value disagreement | "
        f"{g4_value_text} | <= {g4_value_threshold:.2f}; "
        f"n >= {g4_value_denominator} | {g4['passed']} |",
        "",
    ]
    (args.output.parent / "summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    print(
        f"[final] G4={g4_pass} all={receipt['all_gates_passed']} "
        f"receipt={args.output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--protocol", required=True, type=Path)
    prep.add_argument("--reader-readouts", required=True, type=Path)
    prep.add_argument("--hard-pairs", required=True, type=Path)
    prep.add_argument("--open-readouts", required=True, type=Path)
    prep.add_argument("--open-source-dataset")
    prep.add_argument("--out-dir", required=True, type=Path)

    primary = sub.add_parser("apply-primary")
    primary.add_argument("--protocol", required=True, type=Path)
    primary.add_argument("--prepared-items", required=True, type=Path)
    primary.add_argument("--requests", required=True, type=Path)
    primary.add_argument("--judgements", required=True, type=Path)
    primary.add_argument("--out-dir", required=True, type=Path)

    cold = sub.add_parser("apply-cold")
    cold.add_argument("--protocol", required=True, type=Path)
    cold.add_argument("--primary-report", required=True, type=Path)
    cold.add_argument("--primary-sample", required=True, type=Path)
    cold.add_argument("--cold-requests", required=True, type=Path)
    cold.add_argument("--cold-judgements", required=True, type=Path)

    final = sub.add_parser("finalize")
    final.add_argument("--protocol", required=True, type=Path)
    final.add_argument("--primary-report", required=True, type=Path)
    final.add_argument("--primary-sample", required=True, type=Path)
    final.add_argument("--auditor-requests", required=True, type=Path)
    final.add_argument("--auditor-judgements", required=True, type=Path)
    final.add_argument(
        "--scorer", default=Path("scripts/score_ddxplus_semantic_readouts.py"), type=Path
    )
    final.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "apply-primary":
        apply_primary(args)
    elif args.command == "apply-cold":
        apply_cold(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
