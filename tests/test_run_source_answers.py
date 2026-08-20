from scripts.run_source_answers import differential_rank


def test_differential_rank_reports_where_the_answer_sits():
    # Picking the second-ranked condition is a different failure from picking
    # an unrelated one, which a single accuracy number cannot show.
    differential = [
        {"diagnosis": "Croup", "probability": 0.62},
        {"diagnosis": "Bronchiolitis", "probability": 0.21},
    ]
    assert differential_rank("Croup", differential) == 1
    assert differential_rank("Bronchiolitis", differential) == 2
    assert differential_rank("Pneumonia", differential) is None
    assert differential_rank(None, differential) is None


def test_differential_rank_is_none_without_a_differential():
    assert differential_rank("Croup", []) is None
