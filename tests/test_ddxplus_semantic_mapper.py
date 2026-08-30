import json
from pathlib import Path

import pytest

from scripts.audit_ddxplus_semantic_mapper import micro_f1, wilson_upper
from scripts.score_ddxplus_semantic_readouts import semantic_decisions
from src.ddxplus_semantic_mapping import (
    build_alias_and_ontology,
    claims_from_text,
    lexical_mappings,
    materialize_items,
    parse_batch_response,
    prepare_items,
    protocol_cache_key,
)


def structured_protocol() -> dict:
    return {
        "finding_labels": ["E_FEVER", "E_PAIN"],
        "values_by_evidence": {"E_PAIN": ["CHEST", "BACK"]},
        "lexicon": {
            "findings": {
                "E_FEVER": {"text": "a fever"},
                "E_PAIN": {"text": "pain"},
            },
            "values": {
                "E_PAIN\0CHEST": {"text": "pain in the chest"},
                "E_PAIN\0BACK": {"text": "pain in the back"},
            },
        },
    }


def evidence_meta() -> dict:
    return {
        "E_FEVER": {"question_en": "Do you have a fever?"},
        "E_PAIN": {
            "question_en": "Where is the pain located?",
            "value_meaning": {"CHEST": "chest", "BACK": "back"},
        },
    }


def test_claim_splitter_prefers_observed_bullets() -> None:
    text = (
        "<explanation><observed>\n- a fever\n- pain in the chest\n"
        "</observed></explanation>"
    )
    assert claims_from_text(text) == ["a fever", "pain in the chest"]


def test_claim_splitter_handles_prose_and_fixed_abbreviations() -> None:
    assert claims_from_text("Dr. Smith noted fever. Cough followed.") == [
        "Dr. Smith noted fever.",
        "Cough followed.",
    ]


def test_lexical_mapper_handles_multiple_findings_and_explicit_value() -> None:
    aliases, _ontology = build_alias_and_ontology(
        structured_protocol(), evidence_meta()
    )
    mappings = lexical_mappings("a fever and pain in the chest", aliases)
    by_id = {item["evidence_id"]: item for item in mappings}
    assert set(by_id) == {"E_FEVER", "E_PAIN"}
    assert by_id["E_PAIN"]["value_id"] == "CHEST"
    assert by_id["E_PAIN"]["supporting_quote"] == "pain in the chest"


def test_lexical_mapper_defers_nonassertive_claims() -> None:
    aliases, _ontology = build_alias_and_ontology(
        structured_protocol(), evidence_meta()
    )
    assert lexical_mappings("The differential may include a fever", aliases) == []


def test_lexical_mapper_suppresses_nested_generic_alias() -> None:
    aliases = {
        "finding_aliases": {"GENERIC": ["pain"], "SPECIFIC": ["pain in the chest"]},
        "value_aliases": {},
    }
    mappings = lexical_mappings("pain in the chest", aliases)
    assert [item["evidence_id"] for item in mappings] == ["SPECIFIC"]


def test_value_mapper_prefers_full_phrase_over_nested_value_alias() -> None:
    aliases = {
        "finding_aliases": {
            "E_RASH": ["the rash is swollen (rated 4)", "the rash is swollen"]
        },
        "value_aliases": {
            "E_RASH": {
                "1": ["the rash is swollen"],
                "4": ["the rash is swollen (rated 4)"],
            }
        },
    }
    mappings = lexical_mappings("the rash is swollen (rated 4)", aliases)
    assert mappings[0]["value_id"] == "4"


def test_semantic_response_requires_exact_quote_and_known_enums() -> None:
    _aliases, ontology = build_alias_and_ontology(
        structured_protocol(), evidence_meta()
    )
    parsed = parse_batch_response(
        json.dumps(
            {
                "results": [
                    {
                        "claim_id": "x",
                        "mappings": [
                            {
                                "evidence_id": "E_PAIN",
                                "value_id": "NOT_NATIVE",
                                "supporting_quote": "chest discomfort",
                            },
                            {
                                "evidence_id": "E_FEVER",
                                "value_id": None,
                                "supporting_quote": "invented fever quote",
                            },
                        ],
                    }
                ]
            }
        ),
        expected_claims={"x": "The patient reports chest discomfort."},
        ontology=ontology,
    )
    assert parsed["x"] == [
        {
            "evidence_id": "E_PAIN",
            "value_id": None,
            "supporting_quote": "chest discomfort",
            "mapping_stage": "semantic",
        }
    ]


def test_cache_key_is_bound_to_model_and_protocol() -> None:
    kwargs = {
        "ontology_sha256": "o",
        "alias_sha256": "a",
        "prompt_sha256": "p",
    }
    first = protocol_cache_key("a fever", model_id="model-a", **kwargs)
    second = protocol_cache_key("a fever", model_id="model-b", **kwargs)
    assert first != second


def test_semantic_decisions_preserve_batch_population_and_model() -> None:
    _aliases, ontology = build_alias_and_ontology(
        structured_protocol(), evidence_meta()
    )
    prepared = [
        {
            "id": "row",
            "claims": [
                {
                    "claim_id": "claim",
                    "text": "The patient reports chest discomfort.",
                    "lexical_mappings": [],
                }
            ],
        }
    ]
    requests = [{"id": "batch", "claim_ids": ["claim"]}]
    judgements = [
        {
            "id": "batch",
            "judge_model": "independent-model",
            "response": json.dumps(
                {
                    "results": [
                        {
                            "claim_id": "claim",
                            "mappings": [
                                {
                                    "evidence_id": "E_PAIN",
                                    "value_id": None,
                                    "supporting_quote": "chest discomfort",
                                }
                            ],
                        }
                    ]
                }
            ),
        }
    ]
    protocol = {
        "ontology": {"sha256": "ontology"},
        "alias_table": {"sha256": "aliases"},
        "mapper_prompt": {"sha256": "prompt"},
    }
    decisions, model, audit = semantic_decisions(
        prepared,
        requests,
        judgements,
        protocol=protocol,
        ontology=ontology,
    )
    assert model == "independent-model"
    assert decisions["claim"][0]["evidence_id"] == "E_PAIN"
    assert audit[0]["model_id"] == model
    assert audit[0]["cache_key"]


def test_prepare_and_materialize_deduplicate_evidence_per_item() -> None:
    aliases, _ontology = build_alias_and_ontology(
        structured_protocol(), evidence_meta()
    )
    prepared, residual = prepare_items(
        [{"id": "row", "text": "- a fever\n- a fever"}], aliases
    )
    assert residual == {}
    mapped = materialize_items(prepared, {})
    assert [item["evidence_id"] for item in mapped[0]["selected_claims"]] == [
        "E_FEVER"
    ]


def test_audit_helpers_report_micro_f1_and_wilson_bound() -> None:
    assert micro_f1([{"A"}], [{"A", "B"}]) == pytest.approx(2 / 3)
    assert wilson_upper(0, 100) == pytest.approx(0.036993, abs=1e-6)
