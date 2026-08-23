"""Landmark rows for the anchoring trajectory."""

from scripts.make_trajectory_rows import landmark_rows
from src.case_prompts import build_prompt

PRESENTATION = (
    "You are an expert physician. A 51-year-old woman presents with:\n"
    "- coughing up blood\n- a fever\n\nThe referring note suspects Bronchitis."
)


def case(**overrides):
    base = {
        "id": "c1__hint_wrong",
        "base_id": "c1",
        "hint_variant": "wrong",
        "hint_diagnosis_name": "Bronchitis",
        "diagnosis_name": "Tuberculosis",
        "cue_targets": ["coughing up blood", "a fever"],
        "prompt": build_prompt(PRESENTATION, "direct"),
    }
    base.update(overrides)
    return base


def test_wrong_arm_gets_all_six_landmarks_in_order():
    rows, reason = landmark_rows(case())
    assert reason is None
    assert [r["target_role"] for r in rows] == [
        "last_cue",
        "note",
        "question",
        "constraint",
        "format",
        "final",
    ]
    # The last bullet, not the first: the trajectory starts where the chart ends.
    assert rows[0]["target_text"] == "a fever"
    assert rows[-1]["position_mode"] == "last_token"


def test_none_arm_skips_the_note_landmark():
    prompt = build_prompt(PRESENTATION.replace("\n\nThe referring note suspects Bronchitis.", ""), "direct")
    rows, reason = landmark_rows(
        case(hint_variant="none", hint_diagnosis_name=None, prompt=prompt)
    )
    assert reason is None
    assert "note" not in [r["target_role"] for r in rows]


def test_missing_cue_string_is_a_loud_skip_not_a_bad_anchor():
    rows, reason = landmark_rows(case(cue_targets=["not in the prompt"]))
    assert rows == [] and "cue" in reason


def test_last_cue_anchors_on_the_findings_even_as_the_builder_writes_them():
    """The hinted arm as make_hint_injection_cases actually emits it:
    cue_targets overwritten with the note sentence, because that is what the
    note-position readout is aimed at.

    Anchoring last_cue on that field put the two arms' first landmark at
    different positions -- the none arm at the last bullet, the hinted arm at
    the end of the note -- so they disagreed where causal masking makes them
    the same tensor. The findings must come from the none arm.
    """
    as_built = case(
        cue_targets=["The referring note suspects Bronchitis."],
        cue_text="The referring note suspects Bronchitis.",
        target_role="hint",
        target_text="Bronchitis",
    )
    rows, reason = landmark_rows(as_built, ["coughing up blood", "a fever"])
    assert reason is None
    assert rows[0]["target_role"] == "last_cue"
    assert rows[0]["target_text"] == "a fever"

    # And without the none-arm findings it must refuse rather than silently
    # anchor on the note: a wrong position is worse than a missing one.
    rows, reason = landmark_rows(as_built)
    assert rows == [] and reason is not None
