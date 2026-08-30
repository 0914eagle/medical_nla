from __future__ import annotations

import json
from pathlib import Path

from scripts.audit_ddxplus_vanilla_semantic_scores import audit
from src.jsonl import write_jsonl


def test_audit_distinguishes_raw_rejection_from_true_null(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared.jsonl"
    judgements = tmp_path / "judgements.jsonl"
    decisions = tmp_path / "decisions.jsonl"
    mapped = tmp_path / "mapped.jsonl"
    results = tmp_path / "results.json"
    write_jsonl(
        prepared,
        [{"claims": [{"claim_id": "c1", "text": "x", "lexical_mappings": []}]}],
    )
    write_jsonl(
        judgements,
        [
            {
                "id": "r1",
                "response": json.dumps(
                    {
                        "results": [
                            {
                                "claim_id": "c1",
                                "mappings": [{"evidence_id": "E1"}],
                            }
                        ]
                    }
                ),
            }
        ],
    )
    write_jsonl(decisions, [{"claim_id": "c1", "mappings": []}])
    write_jsonl(mapped, [{"variant": "original", "selected_claims": []}])
    results.write_text(json.dumps({"population": "locked_test", "metrics": {}}))

    report = audit(prepared, judgements, decisions, mapped, results)

    assert report["raw_mapper_mappings"] == 1
    assert report["accepted_semantic_mappings"] == 0
    assert report["interpretation"] == "raw_mappings_rejected_do_not_report_metrics"
