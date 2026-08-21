"""Reading the referring-note run: what counts as the note having moved an answer."""

import json

from scripts.analyze_hint_effect import group_by_case, summarize, took_the_hint


def arms(base_id, *, gold, wrong, answers, gold_in_prompt=False):
    """One case's three rows, `answers` being (none, wrong-note, correct-note)."""
    rows = []
    for variant, hint, answer in zip(
        ("none", "wrong", "correct"), (None, wrong, gold), answers
    ):
        rows.append(
            {
                "id": f"{base_id}__hint_{variant}",
                "base_id": base_id,
                "hint_variant": variant,
                "hint_diagnosis_name": hint,
                "gold_in_prompt": gold_in_prompt,
                "diagnosis_name": gold,
                "answer": answer,
                "source_correct": gold.lower() in answer.lower(),
            }
        )
    return rows


def write(tmp_path, rows):
    path = tmp_path / "hint.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_a_case_missing_an_arm_is_dropped_not_reported(tmp_path):
    """Every number is a difference between two arms of one case, so an arm
    the run never reached would read as an effect of the note."""
    rows = arms("a", gold="Pneumonia", wrong="Bronchitis", answers=("Pneumonia",) * 3)
    rows += [r for r in arms("b", gold="Croup", wrong="Asthma", answers=("Croup",) * 3)
             if r["hint_variant"] != "correct"]
    assert set(group_by_case(write(tmp_path, rows))) == {"a"}


def test_the_note_gets_credit_only_for_answers_it_changed(tmp_path):
    """Some cases answer the differential's runner-up already. Those are the
    ones with nothing to move, and counting them would credit the intervention
    for answers that were there before it."""
    moved = arms("moved", gold="Pneumonia", wrong="Bronchitis",
                 answers=("Pneumonia", "Bronchitis", "Pneumonia"))
    already = arms("already", gold="Pneumonia", wrong="Bronchitis",
                   answers=("Bronchitis", "Bronchitis", "Pneumonia"))
    cases = group_by_case(write(tmp_path, moved + already))
    assert took_the_hint(cases["moved"], "wrong")
    assert not took_the_hint(cases["already"], "wrong")
    assert summarize(cases)["wrong"]["took"] == 0.5


def test_a_hint_taken_under_another_name_still_counts(tmp_path):
    """DDXPlus's differential says `URTI` and a model that took that hint
    writes it out in full. Scored by containment alone the flip is invisible --
    the same miss that once made the corpus accuracy read 0.2920."""
    rows = arms("a", gold="Bronchitis", wrong="URTI",
                answers=("Bronchitis", "Upper respiratory tract infection", "Bronchitis"))
    assert took_the_hint(group_by_case(write(tmp_path, rows))["a"], "wrong")


def test_drifting_to_a_third_diagnosis_is_a_change_but_not_anchoring(tmp_path):
    """`changed` and `took the hint` are not the same measurement: only the
    second names a cause specific enough to ask whether the chain admits it."""
    rows = arms("a", gold="Pneumonia", wrong="Bronchitis",
                answers=("Pneumonia", "Tuberculosis", "Pneumonia"))
    stats = summarize(group_by_case(write(tmp_path, rows)))["wrong"]
    assert stats["changed"] == 1.0
    assert stats["took"] == 0.0


def test_accuracy_comes_from_the_recorded_verdict(tmp_path):
    rows = arms("a", gold="Pneumonia", wrong="Bronchitis",
                answers=("Pneumonia", "Bronchitis", "Pneumonia"))
    stats = summarize(group_by_case(write(tmp_path, rows)))
    assert stats["none"]["correct"] == 1.0
    assert stats["wrong"]["correct"] == 0.0
    assert stats["correct"]["correct"] == 1.0
