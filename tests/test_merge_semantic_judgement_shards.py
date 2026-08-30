from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.merge_semantic_judgement_shards import merge_judgements
from src.jsonl import read_jsonl, write_jsonl


def judgement(row_id: str, model: str = "gpt-5.6-sol") -> dict[str, str]:
    return {
        "id": row_id,
        "response": json.dumps({"selected": []}),
        "judge_model": model,
        "judge_backend": "codex",
    }


def test_merge_reorders_to_frozen_request_population(tmp_path: Path) -> None:
    requests = tmp_path / "requests.jsonl"
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    output = tmp_path / "merged.jsonl"
    report = tmp_path / "report.json"
    write_jsonl(requests, [{"id": "r2", "prompt": "b"}, {"id": "r1", "prompt": "a"}])
    write_jsonl(shard_a, [judgement("r1")])
    write_jsonl(shard_b, [judgement("r2")])

    result = merge_judgements(
        requests, [shard_a, shard_b], output, "gpt-5.6-sol", report
    )

    assert [row["id"] for row in read_jsonl(output)] == ["r2", "r1"]
    assert result["rows"] == 2
    assert result["exact_request_population"] is True


@pytest.mark.parametrize("failure", ["missing", "duplicate", "model"])
def test_merge_rejects_invalid_population(tmp_path: Path, failure: str) -> None:
    requests = tmp_path / "requests.jsonl"
    shard_a = tmp_path / "a.jsonl"
    shard_b = tmp_path / "b.jsonl"
    write_jsonl(requests, [{"id": "r1", "prompt": "a"}, {"id": "r2", "prompt": "b"}])
    write_jsonl(shard_a, [judgement("r1")])
    if failure == "missing":
        write_jsonl(shard_b, [])
    elif failure == "duplicate":
        write_jsonl(shard_b, [judgement("r1"), judgement("r2")])
    else:
        write_jsonl(shard_b, [judgement("r2", "gpt-5.4")])

    with pytest.raises(ValueError):
        merge_judgements(
            requests,
            [shard_a, shard_b],
            tmp_path / "merged.jsonl",
            "gpt-5.6-sol",
            tmp_path / "report.json",
        )
