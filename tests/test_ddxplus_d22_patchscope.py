from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from scripts.run_ddxplus_d22_patchscope import (
    find_subsequence,
    marker_token_span,
    output_contract_valid,
    patched_prefill,
    prepare,
    round_robin_sample,
)
from src.jsonl import read_jsonl, write_jsonl


def test_output_contract_accepts_only_none_or_bullets() -> None:
    assert output_contract_valid("NONE")
    assert output_contract_valid("- Chest pain is substernal.\n- The patient is febrile.")
    assert not output_contract_valid("Yes")
    assert not output_contract_valid("Chest pain is substernal.")


def test_find_subsequence_requires_one_marker() -> None:
    assert find_subsequence([1, 2, 3, 4], [2, 3]) == 2


def test_marker_token_span_uses_rendered_character_offsets() -> None:
    text = "prefix <STATE> suffix"
    offsets = [(0, 6), (7, 8), (8, 13), (13, 14), (15, 21)]
    assert marker_token_span(text, offsets) == (1, 4)


def test_patched_prefill_replaces_only_the_marker_position() -> None:
    layer = torch.nn.Identity()
    hidden = torch.zeros((2, 4, 3))
    vectors = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    with patched_prefill(layer, vectors, marker_position=2):
        output = layer(hidden)
    assert torch.equal(output[:, 2, :], vectors)
    assert torch.count_nonzero(output[:, :2, :]) == 0
    assert torch.count_nonzero(output[:, 3:, :]) == 0


def test_round_robin_sample_spreads_diagnoses_before_second_cases() -> None:
    def case(identifier: str, diagnosis: str):
        return {
            variant: {"diagnosis_id": diagnosis}
            for variant in ("original", "cue_deleted", "value_edited")
        }

    eligible = {
        "a1": case("a1", "a"),
        "a2": case("a2", "a"),
        "b1": case("b1", "b"),
        "b2": case("b2", "b"),
    }
    selected = round_robin_sample(eligible, 2)
    diagnoses = {eligible[item]["original"]["diagnosis_id"] for item in selected}
    assert diagnoses == {"a", "b"}


def test_prepare_freezes_eight_unique_generations_and_twelve_cells_per_case(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.jsonl"
    train = tmp_path / "train.jsonl"
    rows = []
    for case in ("case_a", "case_b"):
        for variant in ("original", "cue_deleted", "value_edited"):
            activation = tmp_path / f"{case}_{variant}.pt"
            torch.save(torch.tensor([1.0, 2.0]), activation)
            rows.append(
                {
                    "id": f"{case}_{variant}",
                    "base_id": case,
                    "variant": variant,
                    "diagnosis_id": "diagnosis",
                    "activation_path": str(activation),
                    "cue_evidence_ids": ["E_1"],
                    "cf_original_evidence_id": "E_1",
                    "cf_original_value_id": "old",
                    "cf_replacement_value_id": "new",
                }
            )
    write_jsonl(validation, rows)
    write_jsonl(train, [rows[0], rows[3]])
    structured = tmp_path / "structured.json"
    structured.write_text(
        json.dumps(
            {
                "layer": 24,
                "values_by_evidence": {"E_1": ["old", "new"]},
            }
        )
    )
    out = tmp_path / "out"
    prepare(
        argparse.Namespace(
            validation_manifest=validation,
            train_manifest=train,
            structured_protocol=structured,
            path_map=[],
            cases=2,
            out_dir=out,
        )
    )
    assert len(list(read_jsonl(out / "generation_manifest.jsonl"))) == 16
    assert len(list(read_jsonl(out / "logical_manifest.jsonl"))) == 24
