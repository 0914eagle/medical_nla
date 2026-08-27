import json
import sys
from pathlib import Path

import pytest

from scripts.merge_jsonl_files import main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_merge_jsonl_files_rejects_duplicates(tmp_path, monkeypatch):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(first, [{"id": "a"}])
    write_jsonl(second, [{"id": "a"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_jsonl_files.py",
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
        ],
    )
    with pytest.raises(ValueError, match="Duplicate row ID"):
        main()


def test_merge_jsonl_files_preserves_input_order(tmp_path, monkeypatch):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "output.jsonl"
    write_jsonl(first, [{"id": "a"}, {"id": "b"}])
    write_jsonl(second, [{"id": "c"}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_jsonl_files.py",
            "--input",
            str(first),
            "--input",
            str(second),
            "--output",
            str(output),
            "--expected-rows",
            "3",
        ],
    )
    main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["a", "b", "c"]
