from scripts.summarize_slide20_robustness import canonical_clean_ids, metrics, restrict


def case(none_correct, wrong_correct, *, clean=True, adopted=False):
    return {
        "none": {
            "source_correct": none_correct,
            "gold_in_prompt": not clean,
            "answer": "Gold" if none_correct else "Other",
        },
        "wrong": {
            "source_correct": wrong_correct,
            "answer": "Hint" if adopted else ("Gold" if wrong_correct else "Other"),
            "hint_diagnosis_name": "Hint",
        },
    }


def test_reference_defines_one_canonical_clean_cohort():
    cases = {
        "keep": case(True, False),
        "no_note_wrong": case(False, False),
        "gold_named": case(True, False, clean=False),
    }
    assert canonical_clean_ids(cases) == {"keep"}


def test_cot_is_restricted_without_selecting_on_cot_correctness():
    eligible = {"a", "b"}
    cot = {
        "a": case(False, False),
        "b": case(True, True),
        "outside": case(True, True),
    }
    row = metrics(restrict(cot, eligible))
    assert row["n"] == 2
    assert row["none_accuracy"] == 0.5
    assert row["wrong_accuracy"] == 0.5
