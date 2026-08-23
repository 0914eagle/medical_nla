"""The MCR intervention builder's plausibility source and arm construction."""

from collections import Counter

from scripts.make_mcr_hint_cases import confusions, plausible_wrong


def test_confusions_collect_only_wrong_answers():
    answers = [
        {"diagnosis_name": "Anemia", "answer": "Iron deficiency", "source_correct": False},
        {"diagnosis_name": "Anemia", "answer": "Anemia", "source_correct": True},
        {"diagnosis_name": "Anemia", "answer": "Iron deficiency", "source_correct": False},
        {"diagnosis_name": "Anemia", "answer": "Leukemia", "source_correct": False},
    ]
    pool = confusions(answers)
    assert pool["Anemia"].most_common(1) == [("Iron deficiency", 2)]


def test_plausible_wrong_skips_alias_matches_of_the_gold():
    # The most common confusion is just the gold under another name -- passing
    # it as a "wrong" suggestion would build a correct-note arm by accident.
    pool = Counter({"acute anemia": 5, "Leukemia": 2})
    assert plausible_wrong("Anemia", ["acute anemia"], pool) == "Leukemia"
    assert plausible_wrong("Anemia", [], Counter()) is None
    assert plausible_wrong("Anemia", [], None) is None
