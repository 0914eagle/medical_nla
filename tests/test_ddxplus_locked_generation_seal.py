import json
import sys
from pathlib import Path

import pytest

from scripts import (
    manage_nla_generation_seal,
    validate_ddxplus_locked_population,
    validate_semantic_mapper_freeze_receipt,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def locked_rows(tmp_path: Path) -> list[dict]:
    rows = []
    for variant in ("original", "cue_deleted", "value_edited"):
        activation = tmp_path / f"{variant}.pt"
        activation.write_bytes(b"activation")
        rows.append(
            {
                "id": f"case-a__{variant}__cot_p0",
                "base_id": "case-a",
                "variant": variant,
                "condition": "cot",
                "position_family": "P0",
                "layer": 32,
                "activation_path": str(activation),
            }
        )
    return rows


def run_population(monkeypatch, source: Path, report: Path, expected_rows: int) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--input",
            str(source),
            "--expected-rows",
            str(expected_rows),
            "--expected-layer",
            "32",
            "--require-activation-files",
            "--report",
            str(report),
        ],
    )
    validate_ddxplus_locked_population.main()


def test_locked_population_requires_original_deletion_family(monkeypatch, tmp_path):
    source = tmp_path / "manifest.jsonl"
    report = tmp_path / "population.json"
    rows = locked_rows(tmp_path)
    write_jsonl(source, rows)
    run_population(monkeypatch, source, report, 3)
    assert json.loads(report.read_text())["population_exact"] is True

    write_jsonl(source, [row for row in rows if row["variant"] != "cue_deleted"])
    with pytest.raises(ValueError, match="Incomplete original/deletion family"):
        run_population(monkeypatch, source, report, 2)


def test_generation_seal_detects_readout_mutation(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    actor_prompt = tmp_path / "actor.txt"
    config = tmp_path / "config.yaml"
    population = tmp_path / "manifest_population.json"
    model_metadata = tmp_path / "model_metadata.json"
    protocol = tmp_path / "generation_protocol.json"
    readout = tmp_path / "readout.jsonl"
    output_population = tmp_path / "output_population.json"
    receipt = tmp_path / "generation_seal.json"
    for path, value in (
        (manifest, "{}\n"),
        (actor_prompt, "read {injection_char}\n"),
        (config, "generation: {}\n"),
        (readout, '{"id":"a"}\n'),
    ):
        path.write_text(value, encoding="utf-8")
    population.write_text(
        json.dumps({"population_exact": True, "rows": 10028}) + "\n",
        encoding="utf-8",
    )
    model_metadata.write_text(
        json.dumps(
            {
                "model_id": "kitft/nla-gemma3-12b-L32-av",
                "snapshot_revision": "abc123",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal",
            "freeze",
            "--manifest",
            str(manifest),
            "--actor-prompt",
            str(actor_prompt),
            "--config",
            str(config),
            "--population-report",
            str(population),
            "--model-metadata",
            str(model_metadata),
            "--model-id",
            "kitft/nla-gemma3-12b-L32-av",
            "--expected-rows",
            "10028",
            "--max-new-tokens",
            "512",
            "--batch-size",
            "4",
            "--confirmation",
            "I_FREEZE_DDXPLUS_VANILLA_GENERATION",
            "--output",
            str(protocol),
        ],
    )
    manage_nla_generation_seal.main()

    output_population.write_text(
        json.dumps({"population_exact": True, "rows": 10028}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "seal",
            "seal",
            "--protocol",
            str(protocol),
            "--readout",
            str(readout),
            "--population-report",
            str(output_population),
            "--operator-attestation",
            "NO_LOCKED_TEXT_INSPECTED",
            "--output",
            str(receipt),
        ],
    )
    manage_nla_generation_seal.main()

    readout.write_text('{"id":"changed"}\n', encoding="utf-8")
    monkeypatch.setattr(
        sys, "argv", ["seal", "verify", "--receipt", str(receipt)]
    )
    with pytest.raises(ValueError, match="readout hash mismatch"):
        manage_nla_generation_seal.main()


def test_mapper_receipt_requires_distinct_non_gemma_models(monkeypatch, tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}\n", encoding="utf-8")
    artifacts = {}
    for name in ("alias_table", "mapper_prompt", "ontology", "scorer"):
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        artifacts[name] = {
            "path": str(path),
            "sha256": validate_semantic_mapper_freeze_receipt.sha256_file(path),
        }
    receipt = {
        "all_gates_passed": True,
        "locked_test_read": False,
        "protocol_sha256": validate_semantic_mapper_freeze_receipt.sha256_file(protocol),
        "primary_model_id": "gemma-primary",
        "auditor_model_id": "independent-auditor",
        "gates": {name: {"passed": True} for name in ("G1", "G2", "G3", "G4")},
        **artifacts,
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--receipt",
            str(receipt_path),
            "--expected-protocol-sha256",
            receipt["protocol_sha256"],
        ],
    )
    with pytest.raises(ValueError, match="Gemma-family"):
        validate_semantic_mapper_freeze_receipt.main()
