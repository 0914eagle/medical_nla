import json
import sys
from pathlib import Path

from scripts.remap_activation_manifest_paths import main, remap


def test_remap_requires_path_boundary():
    mappings = [("/data/heejae", "/data1/heejae")]
    assert remap("/data/heejae/a.pt", mappings) == "/data1/heejae/a.pt"
    assert remap("/data/heejae-other/a.pt", mappings) == "/data/heejae-other/a.pt"


def test_remap_manifest_and_verify_files(tmp_path, monkeypatch):
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    new_root.mkdir()
    tensor = new_root / "a.pt"
    tensor.write_bytes(b"tensor")
    source = tmp_path / "source.jsonl"
    output = tmp_path / "output.jsonl"
    source.write_text(
        json.dumps({"id": "a", "activation_path": str(old_root / "a.pt")}) + "\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "remap_activation_manifest_paths.py",
            "--input",
            str(source),
            "--output",
            str(output),
            "--path-map",
            f"{old_root}={new_root}",
            "--expected-rows",
            "1",
        ],
    )
    main()
    row = json.loads(output.read_text())
    assert row["activation_path"] == str(tensor)
