from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_semantic_judgement_batches import audit_batches
from src.jsonl import read_jsonl, write_jsonl


def test_audit_emits_only_parser_failures(tmp_path: Path) -> None:
    ontology = tmp_path / "ontology.json"
    protocol = tmp_path / "protocol.json"
    prepared = tmp_path / "prepared.jsonl"
    requests = tmp_path / "requests.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    retry = tmp_path / "retry.jsonl"
    ontology.write_text(
        json.dumps(
            {
                "evidence": [
                    {"evidence_id": "E1", "values": []},
                ]
            }
        ),
        encoding="utf-8",
    )
    protocol.write_text(
        json.dumps({"ontology": {"path": str(ontology)}}), encoding="utf-8"
    )
    write_jsonl(
        prepared,
        [
            {
                "claims": [
                    {"claim_id": "c1", "text": "alpha", "lexical_mappings": []},
                    {"claim_id": "c2", "text": "beta", "lexical_mappings": []},
                ]
            }
        ],
    )
    write_jsonl(
        requests,
        [
            {"id": "r1", "prompt": "p1", "claim_ids": ["c1"]},
            {"id": "r2", "prompt": "p2", "claim_ids": ["c2"]},
        ],
    )
    write_jsonl(
        judgements,
        [
            {
                "id": "r1",
                "response": json.dumps(
                    {"results": [{"claim_id": "c1", "mappings": []}]}
                ),
            },
            {"id": "r2", "response": json.dumps({"results": []})},
        ],
    )

    report = audit_batches(
        prepared,
        requests,
        judgements,
        protocol,
        retry,
        tmp_path / "report.json",
    )

    assert report["valid"] == 1
    assert report["invalid"] == 1
    assert [row["id"] for row in read_jsonl(retry)] == ["r2"]
