import json
import sys

import pytest
import torch

from scripts import validate_nla_readout_population


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def run_validator(monkeypatch, tmp_path, manifest_rows, readout_rows):
    manifest = tmp_path / "manifest.jsonl"
    readout = tmp_path / "readout.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(manifest, manifest_rows)
    write_jsonl(readout, readout_rows)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate",
            "--manifest",
            str(manifest),
            "--readout",
            str(readout),
            "--expected-rows",
            str(len(manifest_rows)),
            "--expected-max-new-tokens",
            "512",
            "--report",
            str(report),
        ],
    )
    validate_nla_readout_population.main()
    return report


def test_population_validator_accepts_exact_union(monkeypatch, tmp_path):
    activation = tmp_path / "a.pt"
    torch.save(torch.ones(2), activation)
    manifest = [{"id": "a", "variant": "original", "activation_path": str(activation)}]
    readout = [
        {
            "id": "a",
            "variant": "original",
            "gen_config": {"max_new_tokens": 512, "do_sample": False},
            "adapter_id": None,
            "query": "fixed prompt",
            "sidecar_path": "nla_meta.yaml",
        }
    ]
    report = run_validator(monkeypatch, tmp_path, manifest, readout)
    assert json.loads(report.read_text())["population_exact"] is True


def test_population_validator_rejects_wrong_decoding(monkeypatch, tmp_path):
    activation = tmp_path / "a.pt"
    torch.save(torch.ones(2), activation)
    manifest = [{"id": "a", "variant": "original", "activation_path": str(activation)}]
    readout = [
        {
            "id": "a",
            "variant": "original",
            "gen_config": {"max_new_tokens": 256, "do_sample": False},
            "adapter_id": None,
            "query": "fixed prompt",
            "sidecar_path": "nla_meta.yaml",
        }
    ]
    with pytest.raises(ValueError, match="max_new_tokens"):
        run_validator(monkeypatch, tmp_path, manifest, readout)
