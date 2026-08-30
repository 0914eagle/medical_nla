"""Find semantic-mapper batch responses that fail the frozen parser."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ddxplus_semantic_mapping import parse_batch_response
from src.jsonl import read_jsonl, write_jsonl


def audit_batches(
    prepared_path: Path,
    requests_path: Path,
    judgements_path: Path,
    protocol_path: Path,
    retry_requests_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    ontology = json.loads(
        Path(protocol["ontology"]["path"]).read_text(encoding="utf-8")
    )
    expected_claims = {
        str(claim["claim_id"]): str(claim["text"])
        for row in read_jsonl(prepared_path)
        for claim in row["claims"]
        if not claim["lexical_mappings"]
    }
    requests = list(read_jsonl(requests_path))
    judgements = {str(row["id"]): row for row in read_jsonl(judgements_path)}
    invalid = []
    for request in requests:
        request_id = str(request["id"])
        judgement = judgements.get(request_id)
        if judgement is None:
            invalid.append({"id": request_id, "error": "missing judgement"})
            continue
        subset = {
            str(claim_id): expected_claims[str(claim_id)]
            for claim_id in request["claim_ids"]
        }
        try:
            parse_batch_response(
                judgement.get("response"),
                expected_claims=subset,
                ontology=ontology,
            )
        except Exception as exc:  # noqa: BLE001 - report every frozen-parser failure
            invalid.append({"id": request_id, "error": str(exc)})

    invalid_ids = {row["id"] for row in invalid}
    retry = [row for row in requests if str(row["id"]) in invalid_ids]
    write_jsonl(retry_requests_path, retry)
    report = {
        "schema_version": 1,
        "requests": len(requests),
        "valid": len(requests) - len(invalid),
        "invalid": len(invalid),
        "invalid_requests": invalid,
        "retry_requests": str(retry_requests_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[audit-judgements] valid={report['valid']} invalid={report['invalid']} "
        f"retry={retry_requests_path}",
        flush=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--requests", required=True, type=Path)
    parser.add_argument("--judgements", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--retry-requests", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    audit_batches(
        args.prepared,
        args.requests,
        args.judgements,
        args.protocol,
        args.retry_requests,
        args.report,
    )


if __name__ == "__main__":
    main()
