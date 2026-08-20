from scripts.inspect_source_answers import looks_truncated


def test_a_response_stopping_mid_sentence_is_truncated():
    assert looks_truncated("The findings suggest an infection, so I would")


def test_a_response_ending_on_punctuation_is_not():
    assert not looks_truncated("The answer is Bronchitis.")
    assert not looks_truncated("The answer is **Pneumonia**.\n")


def test_an_empty_response_is_not_reported_as_truncated():
    """Emptiness has its own counter; double-counting it hides which failed."""
    assert not looks_truncated("")
    assert not looks_truncated("   \n")
