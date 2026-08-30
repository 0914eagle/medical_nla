from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.audit_medical_nla_ar_roundtrip import cosine, prepare, summarize
from src.jsonl import read_jsonl, write_jsonl


def test_cosine_matches_direction() -> None:
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])) == 1.0
    assert cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 3.0])) == 0.0


def test_summary_requires_both_positive_controls(tmp_path: Path) -> None:
    scored = tmp_path / "scores.jsonl"
    rows = []
    for dataset, arm in (
        ("ddxplus", "structured_reader"),
        ("direct", "source_cot"),
        ("direct", "sft"),
    ):
        for index in range(8):
            rows.append(
                {
                    "dataset": dataset,
                    "arm": arm,
                    "reconstruction_cosine_own": 0.8,
                    "reconstruction_cosine_shuffled": 0.6,
                    "matched_over_shuffled_gap": 0.2,
                    "word_count": 10 + index,
                }
            )
    write_jsonl(scored, rows)
    output_json = tmp_path / "results.json"
    summary_md = tmp_path / "summary.md"
    summarize(
        argparse.Namespace(
            scored=scored, output_json=output_json, summary_md=summary_md
        )
    )
    result = json.loads(output_json.read_text())
    assert result["positive_controls_passed"] is True
    assert "direct::sft" in result["arms"]


def test_prepare_joins_original_rows_and_assigns_donors(tmp_path: Path) -> None:
    activation_paths = {}
    for name in ("direct_a", "direct_b", "ddx_a", "ddx_b", "ddx_deleted"):
        path = tmp_path / f"{name}.pt"
        torch.save(torch.tensor([1.0, 0.0]), path)
        activation_paths[name] = path

    direct_manifest = tmp_path / "direct.jsonl"
    write_jsonl(
        direct_manifest,
        [
            {
                "id": f"{name}__p0",
                "base_id": name,
                "disease_category": "category",
                "activation_path": str(activation_paths[name]),
            }
            for name in ("direct_a", "direct_b")
        ],
    )
    private_bundle = tmp_path / "private.jsonl"
    write_jsonl(
        private_bundle,
        [
            {
                "base_id": name,
                "methods": [
                    {"method": "source_cot", "method_output": f"text {name}"}
                ],
            }
            for name in ("direct_a", "direct_b")
        ],
    )
    ddx_manifest = tmp_path / "ddx.jsonl"
    write_jsonl(
        ddx_manifest,
        [
            {
                "id": "ddx_a",
                "variant": "original",
                "diagnosis_id": "diagnosis",
                "activation_path": str(activation_paths["ddx_a"]),
            },
            {
                "id": "ddx_b",
                "variant": "original",
                "diagnosis_id": "diagnosis",
                "activation_path": str(activation_paths["ddx_b"]),
            },
            {
                "id": "ddx_a_deleted",
                "variant": "cue_deleted",
                "diagnosis_id": "diagnosis",
                "activation_path": str(activation_paths["ddx_deleted"]),
            },
        ],
    )
    reader = tmp_path / "reader.jsonl"
    write_jsonl(
        reader,
        [
            {"id": "ddx_a", "variant": "original", "observed": "finding a"},
            {"id": "ddx_b", "variant": "original", "observed": "finding b"},
            {
                "id": "ddx_a_deleted",
                "variant": "cue_deleted",
                "observed": "deleted",
            },
        ],
    )
    out = tmp_path / "out"
    prepare(
        argparse.Namespace(
            direct_manifest=direct_manifest,
            direct_private_bundle=private_bundle,
            ddx_manifest=ddx_manifest,
            structured_reader=reader,
            path_map=[],
            limit_per_arm=None,
            out_dir=out,
        )
    )
    rows = list(read_jsonl(out / "private_manifest.jsonl"))
    assert len(rows) == 4
    assert {row["donor_base_id"] for row in rows if row["dataset"] == "direct"} == {
        "direct_a",
        "direct_b",
    }
    assert all(row["base_id"] != row["donor_base_id"] for row in rows)
