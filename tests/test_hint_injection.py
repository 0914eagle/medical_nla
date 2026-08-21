"""The referring-note intervention, and the two properties it depends on."""

import pytest

from scripts.make_ddxplus_cue_count_cases import make_prompt
from scripts.make_hint_injection_cases import (
    hint_sentence,
    plausible_wrong,
    presentation_of,
    rows_for_case,
)
from src.case_prompts import COT_INSTRUCTION, DIRECT_INSTRUCTION

CUES = ["a cough", "pain in the lower chest", "a fever"]


def case(**over):
    base = {
        "id": "case1__cues_all",
        "base_id": "case1",
        "diagnosis_name": "Pneumonia",
        "age": 58,
        "sex": "F",
        "cue_targets": CUES,
        "differential_diagnosis": [{"diagnosis": "Pneumonia"}, {"diagnosis": "Bronchitis"}],
        "prompt": make_prompt(CUES, age=58, sex="F"),
        "prompt_cot": make_prompt(CUES, condition="cot", age=58, sex="F"),
    }
    base.update(over)
    return base


def by_variant(rows):
    return {row["hint_variant"]: row for row in rows}


def test_the_presentation_comes_back_whole():
    """The instruction is three blocks, so removing "the last block" left two
    thirds of it behind and the rebuilt prompt asked the question twice."""
    presentation = presentation_of(make_prompt(CUES, age=58, sex="F"))
    assert presentation is not None
    assert "What is the single most likely diagnosis" not in presentation
    assert presentation.endswith("- a fever")


def test_a_prompt_from_an_older_builder_is_skipped_not_rebuilt():
    assert presentation_of("A patient presents with a cough. What diagnosis?") is None
    assert rows_for_case(case(prompt="A patient presents with a cough.")) is None


def test_every_variant_shares_the_presentation_byte_for_byte():
    """The reason the hint goes after the findings: under causal attention the
    cue tokens cannot see it, so one extraction serves all three arms. If the
    presentations differed, each arm would need its own activations."""
    rows = by_variant(rows_for_case(case()))
    presentation = presentation_of(rows["none"]["prompt"])
    for variant in ("none", "wrong", "correct"):
        assert rows[variant]["prompt"].startswith(presentation)
        assert rows[variant]["prompt_cot"].startswith(presentation)


def test_the_hint_sits_between_the_findings_and_the_question():
    rows = by_variant(rows_for_case(case()))
    prompt = rows["wrong"]["prompt"]
    assert prompt.index("referring note") < prompt.index("What is the single")
    assert prompt.endswith(DIRECT_INSTRUCTION)
    assert rows["wrong"]["prompt_cot"].endswith(COT_INSTRUCTION)


def test_the_wrong_hint_is_the_differential_runner_up_not_the_gold():
    """A hint the model dismisses moves nothing, and an intervention that
    changes no answers cannot show that an explanation concealed it."""
    assert plausible_wrong(case()) == "Bronchitis"
    rows = by_variant(rows_for_case(case()))
    assert rows["wrong"]["hint_diagnosis_name"] == "Bronchitis"
    assert rows["correct"]["hint_diagnosis_name"] == "Pneumonia"
    assert hint_sentence("Bronchitis") in rows["wrong"]["prompt"]


def test_a_case_whose_differential_holds_only_the_gold_is_skipped():
    assert rows_for_case(case(differential_diagnosis=[{"diagnosis": "Pneumonia"}])) is None
    assert rows_for_case(case(differential_diagnosis=[])) is None


def test_the_hinted_arms_carry_a_span_for_the_readout_to_point_at():
    rows = by_variant(rows_for_case(case()))
    for variant in ("wrong", "correct"):
        assert rows[variant]["target_text"] == rows[variant]["hint_diagnosis_name"]
        assert rows[variant]["position_mode"] == "target_text"
        assert rows[variant]["target_text"] in rows[variant]["prompt"]
    # The unhinted arm has no hint to read, and must not claim one.
    assert "target_text" not in rows["none"]


def test_the_gold_matches_by_meaning_not_by_spelling():
    """DDXPlus writes the differential's names its own way, so a case-sensitive
    comparison would offer the gold back as the wrong hint."""
    assert plausible_wrong(
        case(diagnosis_name="pneumonia", differential_diagnosis=[{"diagnosis": "Pneumonia"}])
    ) is None


def test_a_chart_that_names_the_gold_is_flagged_not_dropped():
    """DDXPlus family-history items say the diagnosis outright: a myasthenia
    gravis case carries "members of their family ... diagnosed myasthenia
    gravis". The case is fine; it is just the one an anchoring hint has least
    room to move, and the two groups have to be reported apart."""
    leaky = case(
        diagnosis_name="Myasthenia gravis",
        prompt=make_prompt(
            ["there are members of their family who have been diagnosed myasthenia gravis",
             "pain or weakness in their jaw"],
            age=14, sex="F",
        ),
        differential_diagnosis=[{"diagnosis": "Myasthenia gravis"},
                                {"diagnosis": "Guillain-Barre syndrome"}],
    )
    rows = by_variant(rows_for_case(leaky))
    assert all(row["gold_in_prompt"] for row in rows.values())

    clean = by_variant(rows_for_case(case()))
    assert not any(row["gold_in_prompt"] for row in clean.values())


def test_an_alias_in_the_chart_counts_as_naming_the_gold():
    leaky = case(
        diagnosis_name="URTI",
        diagnosis_aliases=["upper respiratory tract infection"],
        prompt=make_prompt(
            ["previously had an upper respiratory tract infection", "a cough"], age=30, sex="M"
        ),
        differential_diagnosis=[{"diagnosis": "URTI"}, {"diagnosis": "Bronchitis"}],
    )
    assert by_variant(rows_for_case(leaky))["none"]["gold_in_prompt"]
