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
