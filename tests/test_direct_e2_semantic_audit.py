from scripts.analyze_direct_e2_semantic_audit import (
    parse_object,
    preserve_human_reviews,
    supported_match,
)
from scripts.make_direct_e2_semantic_audit import shuffled_targets
from scripts.run_direct_local_llama_judge import completion_batch
from src.answer_matching import is_correct


def test_targets_are_complete_and_deterministic() -> None:
    row = {"base_id": "case", "canonical_pdd": "HFrEF", "disease_category": "Heart Failure"}
    source = {"answer": "Acute decompensated heart failure"}
    first = shuffled_targets(row, source, arm="default_HS32", seed=17)
    second = shuffled_targets(row, source, arm="default_HS32", seed=17)
    assert first == second
    assert {target["role"] for target in first} == {"source_answer", "gold_pdd", "category"}
    assert {target["label"] for target in first} == {"A", "B", "C"}


def test_parse_object_accepts_fenced_or_prefixed_json() -> None:
    result = parse_object('result:\n```json\n{"A": {"match": true, "evidence": "PE"}}\n```')
    assert result is not None
    assert result["A"]["match"] is True


def test_match_requires_evidence_from_readout() -> None:
    accepted, supported, _ = supported_match(
        {"match": True, "evidence": "PE"}, "The state suggests PE."
    )
    assert accepted and supported
    accepted, supported, _ = supported_match(
        {"match": True, "evidence": "pulmonary embolism"}, "The state suggests PE."
    )
    assert not accepted and not supported


def test_local_judge_batches_dialogs() -> None:
    class FakeGenerator:
        def chat_completion(self, dialogs, **kwargs):
            assert [dialog[0]["content"] for dialog in dialogs] == ["one", "two"]
            assert kwargs["temperature"] == 0.0
            return [
                {"generation": {"content": '{"A": true}'}},
                {"generation": {"content": '{"B": false}'}},
            ]

    assert completion_batch(
        FakeGenerator(),
        ["one", "two"],
        max_gen_len=64,
        temperature=0.0,
        top_p=1.0,
    ) == ['{"A": true}', '{"B": false}']


def test_existing_lexical_scorer_does_not_expand_clinical_abbreviations() -> None:
    assert not is_correct("GERD", "Gastro-oesophageal Reflux Disease", [])
    assert not is_correct("PE", "Pulmonary Embolism", [])


def test_existing_manual_reviews_survive_reaggregation() -> None:
    rows = [
        {
            "id": "a",
            "human_source": None,
            "human_gold": None,
            "human_category": None,
        }
    ]
    existing = [
        {
            "id": "a",
            "human_source": True,
            "human_gold": False,
            "human_category": False,
            "human_reviewer": "reviewer",
        }
    ]
    merged = preserve_human_reviews(rows, existing)
    assert merged[0]["human_source"] is True
    assert merged[0]["human_gold"] is False
    assert merged[0]["human_reviewer"] == "reviewer"
