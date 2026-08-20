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
