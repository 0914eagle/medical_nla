"""Extraction rows for the referring-note cases."""

import pytest

from scripts.make_hint_position_rows import cue_rows, final_row, hint_row
from src.extract_activations import PASSTHROUGH_FIELDS


def case(**over):
    base = {
        "id": "c1__hint_wrong",
        "base_id": "c1",
        "hint_variant": "wrong",
        "hint_diagnosis_name": "Bronchitis",
        "gold_in_prompt": False,
        "diagnosis_name": "Pneumonia",
        "cue_targets": ["a cough", "a fever"],
        "prompt": (
            "A patient presents with:\n- a cough\n- a fever\n\n"
            "The referring note suspects Bronchitis.\n\nWhat is the diagnosis?"
        ),
    }
    base.update(over)
    return base


def test_the_arm_survives_extraction():
    """The recurring failure in this project: the builder writes which arm a
    row is, the next stage passes through a fixed field list that omits it, and
    the loss is silent until an analysis has nothing to pair on."""
    for field in (
        "base_id",
        "hint_variant",
        "hint_diagnosis_name",
        "gold_in_prompt",
        "diagnosis_alias_in_reasoning",
        "gold_alias_in_reasoning",
    ):
        assert field in PASSTHROUGH_FIELDS
    assert set(final_row(case())) >= {"base_id", "hint_variant", "target_role"}


def test_the_final_position_is_the_token_before_the_answer():
    row = final_row(case())
    assert row["position_mode"] == "last_token"
    assert row["target_role"] == "final"


def test_the_note_is_pointed_at_its_own_mention_not_the_chart_s():
    """15.7% of these charts name the diagnosis themselves, and that mention is
    a finding rather than the suggestion. The note is written last, so the last
    occurrence is the one to read."""
    leaky = case(
        hint_diagnosis_name="Pneumonia",
        prompt=(
            "A patient presents with:\n- ever had Pneumonia\n- a fever\n\n"
            "The referring note suspects Pneumonia.\n\nWhat is the diagnosis?"
        ),
    )
    assert hint_row(leaky, "last_subtoken")["target_text_occurrence"] == 1
    assert hint_row(case(), "last_subtoken")["target_text_occurrence"] == 0


def test_an_arm_with_no_note_has_no_note_position():
    assert hint_row(case(hint_variant="none", hint_diagnosis_name=None), "last_subtoken") is None


def test_cue_rows_resolve_each_mention_in_order():
    rows = cue_rows(case(), "last_subtoken", limit=3)
    assert [r["target_text"] for r in rows] == ["a cough", "a fever"]
    assert all(r["position_mode"] == "target_text" for r in rows)
