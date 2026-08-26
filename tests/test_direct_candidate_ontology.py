from scripts.make_direct_candidate_ontology import collect_labels
from scripts.score_source_diagnosis_logprobs import collect_candidates


def test_collect_labels_deduplicates_and_omits_unresolved() -> None:
    rows = [
        {"canonical_pdd": "Beta"},
        {"canonical_pdd": "Alpha"},
        {"canonical_pdd": "Beta"},
        {"canonical_pdd": None},
        {"canonical_pdd": "<unresolved>"},
    ]
    assert collect_labels(rows, "canonical_pdd") == ["Alpha", "Beta"]


def test_source_candidate_collection_accepts_direct_fields() -> None:
    rows = [
        {"canonical_pdd": "Alpha"},
        {"canonical_pdd": "Beta"},
    ]
    assert collect_candidates(
        rows,
        diagnosis_id_field="canonical_pdd",
        diagnosis_name_field="canonical_pdd",
    ) == [
        {"diagnosis_id": "Alpha", "diagnosis_name": "Alpha"},
        {"diagnosis_id": "Beta", "diagnosis_name": "Beta"},
    ]
