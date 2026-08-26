import json
import sys
from pathlib import Path

from scripts.apply_direct_e4_claim_extractions import (
    extract_json_object,
    ontology,
    quoted,
)
from scripts.make_direct_e4_claim_requests import main as make_requests


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_quote_validation_rejects_paraphrases():
    source = "The patient has chest pain at rest. This supports unstable angina."
    assert quoted("chest pain at rest", source) == "chest pain at rest"
    assert quoted("pain while resting", source) is None


def test_json_parser_accepts_fence_but_not_non_object():
    assert extract_json_object('```json\n{"claims": []}\n```') == {"claims": []}


def test_ontology_builds_official_root_to_leaf_chain():
    names, chains = ontology(
        [
            {
                "canonical_pdd": "NSTEMI",
                "disease_category": "Acute Coronary Syndrome",
                "annotation_chain": ["NSTEMI", "NSTE-ACS", "Suspected ACS"],
            }
        ]
    )
    assert names["nstemi"] == "NSTEMI"
    assert chains["nstemi"] == [
        "Acute Coronary Syndrome",
        "Suspected ACS",
        "NSTE-ACS",
        "NSTEMI",
    ]


def test_request_builder_balances_methods_and_hides_method_label(tmp_path, monkeypatch):
    ids = ["case-a", "case-b"]
    cohort = tmp_path / "cohort.jsonl"
    cases = tmp_path / "cases.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    sources = tmp_path / "sources.jsonl"
    vanilla = tmp_path / "vanilla.jsonl"
    tuned = tmp_path / "tuned.jsonl"
    requests = tmp_path / "requests.jsonl"
    private_index = tmp_path / "index.jsonl"
    summary = tmp_path / "summary.md"

    write_jsonl(cohort, [{"base_id": identifier} for identifier in ids])
    write_jsonl(
        cases,
        [
            {
                "base_id": identifier,
                "source_relative_path": f"Category/PDD/{identifier}.json",
            }
            for identifier in ids
        ],
    )
    write_jsonl(
        candidates,
        [
            {
                "canonical_pdd": "NSTEMI",
                "disease_category": "ACS",
                "annotation_chain": ["NSTEMI"],
            }
        ],
    )
    write_jsonl(
        sources,
        [
            {
                "base_id": identifier,
                "response": "The patient has chest pain. The answer is NSTEMI.",
            }
            for identifier in ids
        ],
    )
    for path, prefix in ((vanilla, "Vanilla"), (tuned, "Tuned")):
        write_jsonl(
            path,
            [
                {"base_id": identifier, "nla_output": f"{prefix} patient finding"}
                for identifier in ids
            ],
        )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "make_direct_e4_claim_requests.py",
            "--cohort",
            str(cohort),
            "--case-manifest",
            str(cases),
            "--candidate-manifest",
            str(candidates),
            "--source-answers",
            str(sources),
            "--readout",
            f"vanilla={vanilla}",
            "--readout",
            f"medical_nla_seed17={tuned}",
            "--requests",
            str(requests),
            "--private-index",
            str(private_index),
            "--summary-md",
            str(summary),
            "--expected-cases",
            "2",
        ],
    )
    make_requests()

    request_rows = [json.loads(line) for line in requests.read_text().splitlines()]
    index_rows = [json.loads(line) for line in private_index.read_text().splitlines()]
    assert len(request_rows) == 6
    assert len(index_rows) == 6
    assert {row["method"] for row in index_rows} == {
        "cot",
        "vanilla",
        "medical_nla_seed17",
    }
    request_by_id = {row["id"]: row for row in request_rows}
    for row in index_rows:
        prompt = request_by_id[row["id"]]["prompt"]
        assert set(request_by_id[row["id"]]) == {"id", "prompt"}
        assert row["base_id"] not in prompt
