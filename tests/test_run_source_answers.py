from scripts.run_source_answers import differential_rank, is_correct, normalize


def test_normalize_folds_case_spacing_and_trailing_period():
    assert normalize("  Acute Otitis  Media. ") == "acute otitis media"


def test_is_correct_matches_containment_in_either_direction():
    # A model may answer more or less specifically than the gold label; both
    # name the same condition.
    assert is_correct("acute otitis media", "Otitis media", [])
    assert is_correct("Otitis media", "Acute otitis media", [])
    assert is_correct("URTI", "Viral URTI", ["URTI"])


def test_is_correct_rejects_a_different_condition_or_no_answer():
    assert not is_correct("Pneumonia", "Otitis media", [])
    assert not is_correct(None, "Otitis media", [])
    assert not is_correct("", "Otitis media", [])


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
