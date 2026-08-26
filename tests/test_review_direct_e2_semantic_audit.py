from scripts.review_direct_e2_semantic_audit import (
    parse_labels,
    summary_markdown,
    validate_resume,
)


def test_parse_labels_accepts_spaced_and_compact_values() -> None:
    assert parse_labels("y n y") == (True, False, True)
    assert parse_labels("ynn") == (True, False, False)
    assert parse_labels("q") is None


def test_validate_resume_requires_identical_order() -> None:
    validate_resume(
        [{"id": "a"}, {"id": "b"}], [{"id": "a"}, {"id": "b"}]
    )
    try:
        validate_resume([{"id": "a"}, {"id": "b"}], [{"id": "b"}, {"id": "a"}])
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched order should fail")


def test_summary_counts_completed_reviews_and_agreement() -> None:
    rows = [
        {
            "human_source": True,
            "human_gold": False,
            "human_category": False,
            "verdicts": {
                "source_answer": {"semantic_match": True},
                "gold_pdd": {"semantic_match": False},
                "category": {"semantic_match": True},
            },
        },
        {
            "human_source": None,
            "human_gold": None,
            "human_category": None,
            "verdicts": {},
        },
    ]
    summary = summary_markdown(rows)
    assert "fully reviewed: **1/2**" in summary
    assert "| Source answer | 1 | 1 | 1 | 1.0000 |" in summary
    assert "| Disease category | 1 | 0 | 1 | 0.0000 |" in summary
