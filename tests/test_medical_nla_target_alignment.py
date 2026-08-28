from scripts.audit_medical_nla_target_alignment import (
    condition_rows,
    same_category_pairs,
    summarize,
)


def row(identifier: str, category: str, target: str) -> dict:
    return {
        "id": identifier,
        "base_id": identifier,
        "disease_category": category,
        "target_text": target,
        "cue_targets": [target],
        "activation_path": f"/{identifier}.pt",
    }


def test_pairs_are_same_category_derangements() -> None:
    rows = [
        row("a1", "a", "one"),
        row("a2", "a", "two"),
        row("b1", "b", "three"),
        row("b2", "b", "four"),
        row("singleton", "c", "five"),
    ]
    pairs = same_category_pairs(rows, seed=17)
    assert len(pairs) == 4
    for own, donor in pairs:
        assert own["base_id"] != donor["base_id"]
        assert own["disease_category"] == donor["disease_category"]
        assert own["target_text"] != donor["target_text"]


def test_condition_rows_swap_only_the_named_side() -> None:
    own = row("a1", "a", "one")
    donor = row("a2", "a", "two")
    target = condition_rows([(own, donor)], "target_shuffled")[0]
    activation = condition_rows([(own, donor)], "activation_shuffled")[0]
    assert target["activation_path"] == own["activation_path"]
    assert target["target_text"] == donor["target_text"]
    assert activation["activation_path"] == donor["activation_path"]
    assert activation["target_text"] == own["target_text"]


def test_summary_uses_positive_control_minus_matched_gap() -> None:
    scores = []
    for identifier in ("a", "b"):
        scores.extend(
            [
                {"base_id": identifier, "condition": "matched", "content_nll": 1.0},
                {
                    "base_id": identifier,
                    "donor_base_id": "b" if identifier == "a" else "a",
                    "disease_category": "same",
                    "condition": "target_shuffled",
                    "content_nll": 2.0,
                },
                {
                    "base_id": identifier,
                    "donor_base_id": "b" if identifier == "a" else "a",
                    "disease_category": "same",
                    "condition": "activation_shuffled",
                    "content_nll": 3.0,
                },
            ]
        )
    result = summarize(scores, eligible_rows=2, seed=17)
    assert result["gaps"]["target_shuffled"]["control_minus_matched"] == 1.0
    assert result["gaps"]["activation_shuffled"]["control_minus_matched"] == 2.0
    assert result["gaps"]["target_shuffled"]["matched_win_rate"] == 1.0
    assert result["symmetric_cross"]["cross_minus_matched"] == 1.5
    assert result["symmetric_cross"]["clusters"] == 1
    assert result["symmetric_cross"]["cluster_bootstrap_95_ci"] == [1.5, 1.5]
