from scripts.run_source_answers import (
    answer_row_key,
    differential_rank,
    matches_where,
    parse_where,
    reasoning_before_final_answer,
)


def test_resume_key_distinguishes_stacked_variants():
    assert answer_row_key({"id": "case-1", "variant": "none"}) != answer_row_key(
        {"id": "case-1", "variant": "wrong"}
    )
    assert answer_row_key({"id": "case-1"}) == ("case-1", "")


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


def test_reasoning_stops_at_the_last_answer_boundary():
    response = (
        "The findings favor pneumonia.\n"
        "The answer is unfinished\n\n"
        "The answer is Bronchiolitis."
    )
    assert reasoning_before_final_answer(response) == (
        "The findings favor pneumonia.\nThe answer is unfinished"
    )


def test_prefilled_direct_answer_has_no_reasoning_prefix():
    assert reasoning_before_final_answer("The answer is Pneumonia.") == ""


def test_where_selects_the_arms_a_run_is_actually_for():
    """Case files stack the arms of an experiment. A chain-of-thought pass over
    an arm nothing downstream reads is a third of the run spent on 2048-token
    generations for nobody."""
    rows = [
        {"hint_variant": "none", "gold_in_prompt": False},
        {"hint_variant": "wrong", "gold_in_prompt": True},
        {"hint_variant": "correct", "gold_in_prompt": False},
    ]
    wanted = parse_where(["hint_variant=none,wrong"])
    assert [r["hint_variant"] for r in rows if matches_where(r, wanted)] == ["none", "wrong"]


def test_repeated_where_flags_must_all_hold():
    rows = [
        {"hint_variant": "wrong", "gold_in_prompt": False},
        {"hint_variant": "wrong", "gold_in_prompt": True},
    ]
    wanted = parse_where(["hint_variant=wrong", "gold_in_prompt=false"])
    assert [r["gold_in_prompt"] for r in rows if matches_where(r, wanted)] == [False]


def test_a_boolean_is_matched_as_it_is_written():
    """`gold_in_prompt=false` has to select the JSON boolean without the caller
    knowing how it was serialized."""
    assert matches_where({"gold_in_prompt": False}, parse_where(["gold_in_prompt=false"]))
    assert not matches_where({"gold_in_prompt": True}, parse_where(["gold_in_prompt=false"]))


def test_a_malformed_filter_stops_the_run():
    """It would otherwise be read as a filter nobody wrote and quietly keep
    everything, which is a long run producing the wrong file."""
    import pytest

    with pytest.raises(SystemExit):
        parse_where(["hint_variant"])
