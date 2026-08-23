"""The MCR intervention builder's plausibility source and arm construction."""

from collections import Counter

from scripts.make_mcr_hint_cases import (
    confusions,
    cue_words,
    neighbor_gold,
    plausible_wrong,
)


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


def test_neighbor_gold_picks_the_most_cue_similar_other_diagnosis():
    case = {"cue_targets": ["persistent cough", "fever at night", "weight loss"]}
    corpus = [
        (cue_words({"cue_targets": ["persistent cough", "fever at night"]}), "Tuberculosis"),
        (cue_words({"cue_targets": ["knee pain after running"]}), "Meniscus tear"),
        # An alias of the gold must never be offered as its own "wrong".
        (cue_words({"cue_targets": ["persistent cough", "weight loss", "fever at night"]}), "whooping cough"),
    ]
    got, score = neighbor_gold(case, "Whooping cough", [], corpus)
    assert got == "Tuberculosis"
    # The overlap travels with the pick. A neighbour sharing one generic word
    # is not a differential -- it proposed a skin disease for a brain lesion
    # in the real run -- and only the score distinguishes the two.
    assert 0.0 < score <= 1.0
    assert neighbor_gold({"cue_targets": []}, "X", [], corpus) == (None, 0.0)
