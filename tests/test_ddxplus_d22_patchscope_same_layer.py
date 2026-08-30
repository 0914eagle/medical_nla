from __future__ import annotations

import json
from pathlib import Path

from scripts.calibrate_ddxplus_d22_patchscope_same_layer import (
    control_cell_eligible,
    exact_layer_map,
    fixed_donors,
    parse_clinical_cell,
    requested_control_cell,
)
from src.jsonl import write_jsonl


def test_exact_layer_map_requires_all_three_layers(tmp_path: Path) -> None:
    values = []
    for layer in (16, 24, 32):
        path = tmp_path / f"layer{layer}.jsonl"
        path.write_text("{}\n")
        values.append((layer, path))
    assert set(exact_layer_map(values, "test")) == {16, 24, 32}


def test_fixed_donors_reads_frozen_original_pairs(tmp_path: Path) -> None:
    path = tmp_path / "generation.jsonl"
    write_jsonl(
        path,
        [
            {
                "base_id": "a",
                "variant": "original",
                "condition": "same_diagnosis_shuffled",
                "donor_base_id": "b",
            },
            {
                "base_id": "a",
                "variant": "original",
                "condition": "real",
                "donor_base_id": None,
            },
        ],
    )
    assert fixed_donors(path) == {"a": "b"}


def test_protocol_candidate_shape_is_json_serializable() -> None:
    cells = [
        {"family": family, "source_layer": layer, "target_layer": layer}
        for family in ("entity_description", "relation_specific")
        for layer in (16, 24, 32)
    ]
    assert len(json.loads(json.dumps(cells))) == 6


def test_parse_clinical_cell() -> None:
    assert parse_clinical_cell("relation_specific:16") == (
        "relation_specific",
        16,
    )


def test_requested_control_cell_must_be_eligible() -> None:
    summaries = [
        {
            "family": "relation_specific",
            "source_layer": 16,
            "target_layer": 16,
            "keyword_hits": 3,
            "keyword_gain": 0.6,
            "outputs_differing_from_no_patch": 5,
        }
    ]
    assert control_cell_eligible(summaries[0])
    assert requested_control_cell(summaries, ("relation_specific", 16)) == summaries[0]
