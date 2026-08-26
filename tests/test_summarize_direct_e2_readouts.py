from scripts.summarize_direct_e2_readouts import summarize_arm


def test_summary_uses_same_category_different_answer_donor() -> None:
    rows = [
        {
            "id": "a_p0",
            "base_id": "a",
            "position_family": "P0",
            "disease_category": "Heart",
            "canonical_pdd": "Alpha disease",
            "diagnosis_aliases": [],
            "prompt": "The patient has alpha finding today.",
            "nla_output": "Alpha disease with alpha finding.",
            "parsed_explanation_tag": True,
        },
        {
            "id": "b_p0",
            "base_id": "b",
            "position_family": "P0",
            "disease_category": "Heart",
            "canonical_pdd": "Beta disease",
            "diagnosis_aliases": [],
            "prompt": "The patient has beta finding today.",
            "nla_output": "Beta disease with beta finding.",
            "parsed_explanation_tag": True,
        },
    ]
    sources = {
        "a": {"answer": "Alpha disease"},
        "b": {"answer": "Beta disease"},
    }
    result = summarize_arm(rows, sources)
    assert result["n"] == 2
    assert result["source_answer_mention"] == 1.0
    assert result["gold_pdd_mention"] == 1.0
    assert result["derangement_n"] == 2
    assert result["own_source_mention"] == 1.0
    assert result["donor_source_mention"] == 0.0
    assert result["source_mention_gap"] == 1.0


def test_p1_leakage_free_subset_is_reported() -> None:
    rows = [
        {
            "id": "a_p1",
            "base_id": "a",
            "position_family": "P1",
            "diagnosis_alias_in_reasoning": False,
            "disease_category": "Heart",
            "canonical_pdd": "Alpha",
            "prompt": "alpha prompt",
            "nla_output": "Alpha",
            "parsed_explanation_tag": True,
        }
    ]
    result = summarize_arm(rows, {"a": {"answer": "Alpha"}})
    assert result["p1_leakage_free_n"] == 1
    assert result["p1_leakage_free_source_mention"] == 1.0
