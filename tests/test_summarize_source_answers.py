from scripts.summarize_source_answers import diagnosis_only_accuracy, per_diagnosis_table


def rows(spec):
    """(diagnosis, correct) pairs into answer rows."""
    return [
        {"id": f"r{i}", "diagnosis_id": name, "source_correct": ok}
        for i, (name, ok) in enumerate(spec)
    ]


def test_diagnosis_alone_predicts_everything_when_errors_are_concentrated():
    """The failure mode the bound exists to expose: one diagnosis always wrong,
    another always right, so a probe reading only the label scores 1.0 while
    carrying no information about error."""
    spec = [("croup", True)] * 10 + [("ebola", False)] * 10
    result = diagnosis_only_accuracy(rows(spec))
    assert result["diagnosis_only_accuracy"] == 1.0
    assert result["majority_class_accuracy"] == 0.5
    assert result["applicable"] is True


def test_the_bound_falls_to_the_majority_when_errors_are_spread_evenly():
    spec = [("croup", i % 2 == 0) for i in range(10)]
    spec += [("ebola", i % 2 == 0) for i in range(10)]
    result = diagnosis_only_accuracy(rows(spec))
    assert result["diagnosis_only_accuracy"] == 0.5
    assert result["majority_class_accuracy"] == 0.5


def test_the_bound_is_marked_inapplicable_on_one_case_per_diagnosis():
    """With a label per case the bound is 1.0 by construction and means nothing."""
    result = diagnosis_only_accuracy(rows([(f"dx{i}", i % 3 == 0) for i in range(30)]))
    assert result["diagnosis_only_accuracy"] == 1.0
    assert result["applicable"] is False


def test_per_diagnosis_table_is_hardest_first_and_respects_the_floor():
    spec = [("easy", True)] * 6 + [("hard", False)] * 6 + [("rare", True)] * 2
    table = per_diagnosis_table(rows(spec), min_cases=5)
    assert [entry["diagnosis"] for entry in table] == ["hard", "easy"]
    assert table[0]["n_errors"] == 6
    assert table[1]["accuracy"] == 1.0


def test_no_rows_gives_no_bound():
    assert diagnosis_only_accuracy([]) == {}


def test_common_wrong_answers_shows_what_was_said_instead():
    """A diagnosis at 0/n is a model failure or a label the scorer cannot
    match, and only the answers tell them apart."""
    from scripts.summarize_source_answers import common_wrong_answers

    answers = [
        {"diagnosis_id": "larygospasm", "source_correct": False, "answer": "Laryngospasm"},
        {"diagnosis_id": "larygospasm", "source_correct": False, "answer": "Laryngospasm"},
        {"diagnosis_id": "larygospasm", "source_correct": False, "answer": "Croup"},
        {"diagnosis_id": "croup", "source_correct": True, "answer": "Croup"},
    ]
    assert common_wrong_answers(answers, "larygospasm", 4) == [("Laryngospasm", 2), ("Croup", 1)]
    assert common_wrong_answers(answers, "croup", 4) == []


def test_mixed_outcome_rows_drops_diagnoses_with_one_outcome():
    """A diagnosis at 0 of n has zero within-diagnosis signal by construction,
    so leaving it in lets the label stand in for the answer."""
    from scripts.summarize_source_answers import mixed_outcome_rows

    spec = [("mixed", i % 2 == 0) for i in range(10)]
    spec += [("never", False)] * 10 + [("always", True)] * 10
    kept, dropped = mixed_outcome_rows(rows(spec), min_cases=5)
    assert {r["diagnosis_id"] for r in kept} == {"mixed"}
    assert dropped == ["always", "never"]


def test_mixed_outcome_rows_also_drops_diagnoses_below_the_floor():
    from scripts.summarize_source_answers import mixed_outcome_rows

    spec = [("tiny", True), ("tiny", False)]
    kept, dropped = mixed_outcome_rows(rows(spec), min_cases=5)
    assert kept == [] and dropped == ["tiny"]


def test_a_near_constant_diagnosis_is_dropped_too():
    """Requiring merely both outcomes is too weak: a diagnosis at 1 of 100
    hands a label-only predictor 99% just as a constant one does."""
    from scripts.summarize_source_answers import mixed_outcome_rows

    spec = [("lopsided", i == 0) for i in range(100)]
    spec += [("balanced", i % 2 == 0) for i in range(100)]
    kept, dropped = mixed_outcome_rows(rows(spec), min_cases=5)
    assert {r["diagnosis_id"] for r in kept} == {"balanced"}
    assert dropped == ["lopsided"]


def test_the_threshold_is_settable_and_zero_keeps_both_outcomes():
    from scripts.summarize_source_answers import mixed_outcome_rows

    spec = [("lopsided", i == 0) for i in range(100)] + [("never", False)] * 100
    kept, dropped = mixed_outcome_rows(rows(spec), min_cases=5, min_minority_rate=0.0)
    assert {r["diagnosis_id"] for r in kept} == {"lopsided"}
    assert dropped == ["never"]
