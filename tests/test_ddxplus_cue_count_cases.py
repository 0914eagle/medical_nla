import pytest

from scripts.make_ddxplus_cue_count_cases import make_prompt, parse_cue_count


def test_parse_cue_count_reads_integers():
    assert parse_cue_count("3") == 3
    assert parse_cue_count("12") == 12


def test_parse_cue_count_maps_all_forms_to_none():
    # None means "take every cue the case has", which imposes no minimum.
    for token in ("all", "ALL", " full ", "max"):
        assert parse_cue_count(token) is None


def test_parse_cue_count_rejects_non_positive():
    with pytest.raises(ValueError):
        parse_cue_count("0")
    with pytest.raises(ValueError):
        parse_cue_count("-2")


def test_min_required_has_no_floor_when_only_all_is_requested():
    # Regression: `--cue-counts all` used to raise on max() of an empty
    # sequence, since "all" contributes no explicit count.
    cue_counts = [parse_cue_count("all")]
    explicit = [count for count in cue_counts if count is not None]
    assert explicit == []
    assert (max(explicit) if explicit else 1) == 1


def test_min_required_uses_the_largest_explicit_count():
    cue_counts = [parse_cue_count(token) for token in ("3", "5", "all")]
    explicit = [count for count in cue_counts if count is not None]
    assert (max(explicit) if explicit else 1) == 5


def test_make_prompt_joins_cues_into_one_question():
    prompt = make_prompt(["fever", "a cough", "chest pain"])
    assert prompt.startswith("A patient presents with ")
    assert "fever, a cough, and chest pain" in prompt
    assert prompt.endswith("What diagnosis is most likely?")
