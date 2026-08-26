import json
from pathlib import Path

import pytest

from scripts.reindex_direct_activations import (
    load_assignments,
    merge_rows,
    parse_path_maps,
    remap_path,
    validate_grid,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_path_map_only_replaces_prefix() -> None:
    mappings = parse_path_maps(["/data1/heejae=/data/heejae"])
    assert remap_path("/data1/heejae/a.pt", mappings) == "/data/heejae/a.pt"
    assert remap_path("/data1/heejae2/a.pt", mappings) == "/data1/heejae2/a.pt"


def test_merge_and_validate_complete_grid(tmp_path: Path) -> None:
    split_dir = tmp_path / "splits"
    write_jsonl(split_dir / "train.jsonl", [{"id": "a"}])
    write_jsonl(split_dir / "val_seen.jsonl", [{"id": "b"}])
    write_jsonl(split_dir / "test_seen.jsonl", [{"id": "c"}])
    write_jsonl(split_dir / "test_pdd_heldout.jsonl", [{"id": "d"}])
    assignments, counts = load_assignments(split_dir)
    assert counts == {"train": 1, "val_seen": 1, "test_seen": 1, "test_pdd_heldout": 1}

    root = tmp_path / "activations"
    for layer in (16, 24, 32):
        p0 = []
        p12 = []
        for case_id in assignments:
            for family, bucket, selection in (
                ("P0", p0, "last_token"),
                ("P1", p12, "last_subtoken"),
                ("P2", p12, "last_subtoken"),
            ):
                tensor = tmp_path / "tensors" / f"{case_id}_{layer}_{family}.pt"
                tensor.parent.mkdir(parents=True, exist_ok=True)
                tensor.touch()
                bucket.append(
                    {
                        "id": f"{case_id}_{family}",
                        "base_id": case_id,
                        "layer": layer,
                        "position_family": family,
                        "position_mode": selection,
                        "activation_path": str(tensor),
                    }
                )
        write_jsonl(root / f"layer{layer}" / "last_token" / "manifest.jsonl", p0)
        write_jsonl(root / f"layer{layer}" / "last_subtoken" / "manifest.jsonl", p12)

    rows, stats = merge_rows([root], assignments, [], True)
    assert len(rows) == 4 * 3 * 3
    assert stats["unassigned"] == 0
    validate_grid(rows, assignments, [16, 24, 32])


def test_validate_grid_rejects_missing_position() -> None:
    rows = [
        {"base_id": "a", "layer": 16, "position_family": "P0"},
        {"base_id": "a", "layer": 16, "position_family": "P1"},
    ]
    with pytest.raises(ValueError, match="incomplete"):
        validate_grid(rows, {"a": "train"}, [16])
