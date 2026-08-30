from __future__ import annotations

import json
from pathlib import Path

from scripts.calibrate_ddxplus_d22_patchscope_same_layer import (
    exact_layer_map,
    fixed_donors,
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
