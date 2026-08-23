"""The blinded judge prompts and their unblinding."""

from scripts.analyze_explanation_judging import parse_judgement, score_of
from scripts.make_explanation_judging_cases import (
    CHANNELS,
    channel_order,
    judge_prompt,
    render_channel,
)


def test_blinding_is_a_deterministic_permutation():
    """Rebuilding the file must reproduce the same label assignment, and
    every channel must appear exactly once -- a repeated or dropped channel
    would silently corrupt every downstream mean."""
    order = channel_order("ddxplus_case_0001")
    assert sorted(order) == sorted(CHANNELS)
    assert order == channel_order("ddxplus_case_0001")
    # Different cases shuffle differently at least somewhere, or the
    # "shuffle" is a fixed relabeling and position bias survives blinding.
    others = {tuple(channel_order(f"case_{i}")) for i in range(20)}
    assert len(others) > 1


def test_channels_render_as_content_without_system_names():
    """The judge scores text, not brands: no rendering may say which system
    produced it. The probe renders as a bare class name because that is the
    entirety of what a classifier head outputs."""
    readout = render_channel("readout", "Myocarditis", "chest pain; dyspnea", "", "")
    cot = render_channel("cot", "", "", "The findings suggest pneumonia.", "")
    probe = render_channel("probe", "", "", "", "Pneumonia")
    assert "Myocarditis" in readout and "chest pain" in readout
    assert cot == "The findings suggest pneumonia."
    assert probe == "Pneumonia."
    for text in (readout, cot, probe):
        assert "readout" not in text.lower()
        assert "probe" not in text.lower()
        assert "classifier" not in text.lower()


def test_judge_prompt_carries_all_three_and_demands_json():
    prompt = judge_prompt("A 51-year-old woman presents with fever.",
                          "Bronchitis", ["first", "second", "third"])
    assert "Explanation A:\nfirst" in prompt
    assert "Explanation B:\nsecond" in prompt
    assert "Explanation C:\nthird" in prompt
    assert "The AI's final answer was: Bronchitis" in prompt
    assert '"most_useful"' in prompt


def test_parse_judgement_survives_prose_and_fences():
    obj = {"A": {"grounding": 4, "coherence": 3, "utility": 5},
           "B": {"grounding": 2, "coherence": 2, "utility": 1},
           "C": {"grounding": 5, "coherence": 4, "utility": 4},
           "most_useful": "C"}
    import json

    wrapped = f"Here is my assessment:\n```json\n{json.dumps(obj)}\n```\nDone."
    parsed = parse_judgement(wrapped)
    assert parsed == obj
    assert score_of(parsed, "A", "utility") == 5.0
    # Out-of-range and missing scores are rejected, not clamped.
    assert score_of({"A": {"grounding": 9}}, "A", "grounding") is None
    assert score_of({}, "A", "grounding") is None
    assert parse_judgement("no json here") is None
