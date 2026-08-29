"""Freeze, seal, and verify an NLA generation artifact before semantic scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)


def git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def read_json(path: Path) -> dict[str, Any]:
    require_file(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def dependency(path: Path) -> dict[str, str]:
    require_file(path)
    return {"path": str(path.resolve()), "sha256": sha256_file(path)}


def verify_dependency(value: dict[str, Any], label: str) -> None:
    path = Path(str(value.get("path") or ""))
    expected = str(value.get("sha256") or "")
    require_file(path)
    actual = sha256_file(path)
    if not expected or actual != expected:
        raise ValueError(f"{label} hash mismatch: {actual} != {expected}")


def freeze(args: argparse.Namespace) -> None:
    if args.confirmation != "I_FREEZE_DDXPLUS_VANILLA_GENERATION":
        raise ValueError("Missing exact generation-freeze confirmation")
    if args.model_id != "kitft/nla-gemma3-12b-L32-av":
        raise ValueError("The frozen Vanilla AV checkpoint must be L32")
    population = read_json(args.population_report)
    if population.get("population_exact") is not True:
        raise ValueError("Population report is not exact")
    if int(population.get("rows", -1)) != args.expected_rows:
        raise ValueError("Population report row count mismatch")
    model_metadata = read_json(args.model_metadata)
    if model_metadata.get("model_id") != args.model_id:
        raise ValueError("Resolved model metadata does not match frozen model ID")
    if not str(model_metadata.get("snapshot_revision") or "").strip():
        raise ValueError("Resolved model metadata has no snapshot revision")
    payload = {
        "schema_version": 1,
        "status": "generation_frozen_scoring_not_authorized",
        "scope": "DDXPlus E5 locked-test Vanilla AV open-text generation",
        "model_id": args.model_id,
        "model_snapshot_revision": model_metadata["snapshot_revision"],
        "adapter_id": None,
        "activation_layer": 32,
        "position": "CoT-P0",
        "expected_rows": args.expected_rows,
        "expected_variants": {
            "original": 4543,
            "cue_deleted": 4543,
            "value_edited": 942,
        },
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "manifest": dependency(args.manifest),
        "actor_prompt": dependency(args.actor_prompt),
        "config": dependency(args.config),
        "model_metadata": dependency(args.model_metadata),
        "population_report": dependency(args.population_report),
        "git_commit": git_commit(),
        "semantic_scoring_authorized": False,
        "locked_outputs_may_be_generated": True,
        "locked_outputs_may_be_semantically_inspected": False,
    }
    write_json(args.output, payload)
    print(f"[frozen] {args.output}", flush=True)


def verify_protocol(
    protocol: dict[str, Any], *, require_current_git: bool = False
) -> None:
    if protocol.get("status") != "generation_frozen_scoring_not_authorized":
        raise ValueError("Unexpected generation protocol status")
    if protocol.get("semantic_scoring_authorized") is not False:
        raise ValueError("Generation protocol improperly authorizes scoring")
    for key in (
        "manifest",
        "actor_prompt",
        "config",
        "model_metadata",
        "population_report",
    ):
        verify_dependency(protocol[key], key)
    if require_current_git and protocol.get("git_commit") != git_commit():
        raise ValueError(
            "Git commit changed after generation freeze; use the frozen commit to resume"
        )


def verify_protocol_command(args: argparse.Namespace) -> None:
    protocol = read_json(args.protocol)
    verify_protocol(protocol, require_current_git=args.require_current_git)
    print(
        f"[protocol verified] revision={protocol['model_snapshot_revision']} "
        f"rows={protocol['expected_rows']}",
        flush=True,
    )


def seal(args: argparse.Namespace) -> None:
    if args.operator_attestation != "NO_LOCKED_TEXT_INSPECTED":
        raise ValueError("Missing exact no-inspection attestation")
    protocol = read_json(args.protocol)
    verify_protocol(protocol)
    report = read_json(args.population_report)
    if report.get("population_exact") is not True:
        raise ValueError("Readout population was not exact")
    if int(report.get("rows", -1)) != int(protocol["expected_rows"]):
        raise ValueError("Readout row count differs from frozen protocol")
    require_file(args.readout)
    receipt = {
        "schema_version": 1,
        "status": "sealed_generation_waiting_for_semantic_mapper",
        "generation_protocol": dependency(args.protocol),
        "readout": dependency(args.readout),
        "population_report": dependency(args.population_report),
        "rows": int(report["rows"]),
        "semantic_scoring_performed": False,
        "locked_output_text_inspected": False,
        "operator_attestation": args.operator_attestation,
        "reuse_without_regeneration": True,
    }
    write_json(args.output, receipt)
    print(f"[sealed] rows={receipt['rows']} receipt={args.output}", flush=True)


def verify(args: argparse.Namespace) -> None:
    receipt = read_json(args.receipt)
    if receipt.get("status") != "sealed_generation_waiting_for_semantic_mapper":
        raise ValueError("Unexpected seal status")
    if receipt.get("semantic_scoring_performed") is not False:
        raise ValueError("Seal says semantic scoring already occurred")
    for key in ("generation_protocol", "readout", "population_report"):
        verify_dependency(receipt[key], key)
    protocol_path = Path(receipt["generation_protocol"]["path"])
    verify_protocol(read_json(protocol_path))
    print(
        f"[verified] rows={receipt['rows']} readout={receipt['readout']['path']}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze_parser = subparsers.add_parser("freeze")
    freeze_parser.add_argument("--manifest", required=True, type=Path)
    freeze_parser.add_argument("--actor-prompt", required=True, type=Path)
    freeze_parser.add_argument("--config", required=True, type=Path)
    freeze_parser.add_argument("--model-metadata", required=True, type=Path)
    freeze_parser.add_argument("--population-report", required=True, type=Path)
    freeze_parser.add_argument("--model-id", required=True)
    freeze_parser.add_argument("--expected-rows", required=True, type=int)
    freeze_parser.add_argument("--max-new-tokens", required=True, type=int)
    freeze_parser.add_argument("--batch-size", required=True, type=int)
    freeze_parser.add_argument("--confirmation", required=True)
    freeze_parser.add_argument("--output", required=True, type=Path)
    freeze_parser.set_defaults(func=freeze)

    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--protocol", required=True, type=Path)
    seal_parser.add_argument("--readout", required=True, type=Path)
    seal_parser.add_argument("--population-report", required=True, type=Path)
    seal_parser.add_argument("--operator-attestation", required=True)
    seal_parser.add_argument("--output", required=True, type=Path)
    seal_parser.set_defaults(func=seal)

    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", required=True, type=Path)
    verify_parser.set_defaults(func=verify)

    protocol_parser = subparsers.add_parser("verify-protocol")
    protocol_parser.add_argument("--protocol", required=True, type=Path)
    protocol_parser.add_argument(
        "--require-current-git",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    protocol_parser.set_defaults(func=verify_protocol_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
