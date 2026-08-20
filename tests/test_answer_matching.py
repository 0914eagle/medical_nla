from src.answer_matching import content_tokens, is_correct, normalize, token_f1


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


def test_containment_misses_an_interposed_word():
    """The failure token_f1 exists to measure, stated as a test."""
    assert not is_correct("phototoxic drug reaction", "phototoxic reaction", [])
    assert token_f1("phototoxic drug reaction", "phototoxic reaction", []) > 0.7


def test_token_f1_penalizes_padding_and_truncation_alike():
    assert round(token_f1("reaction", "phototoxic reaction", []), 2) == 0.67
    assert round(token_f1("phototoxic drug allergic reaction", "phototoxic reaction", []), 2) == 0.67
    assert token_f1("phototoxic reaction", "phototoxic reaction", []) == 1.0
    assert token_f1("Pneumonia", "Otitis media", []) == 0.0


def test_token_f1_takes_the_best_accepted_name():
    assert token_f1("viral URTI", "Something else", ["viral urti"]) == 1.0


def test_content_tokens_keep_words_that_change_the_diagnosis():
    """Dropping 'acute' or 'left' to raise a match rate would score a
    different question, so only true fillers are removed."""
    assert content_tokens("acute renal failure") == {"acute", "renal", "failure"}
    assert content_tokens("inflammation of the left kidney") == {
        "inflammation",
        "left",
        "kidney",
    }


def test_token_f1_treats_a_hyphen_as_a_word_boundary():
    assert token_f1("drug-induced reaction", "drug induced reaction", []) == 1.0


def test_token_f1_without_an_answer_is_zero():
    assert token_f1(None, "Otitis media", []) == 0.0
    assert token_f1("the of", "Otitis media", []) == 0.0
