from scripts.make_specificity_position_variants import expanded_rows


def test_expanded_rows_accepts_v2_case_schema():
    case = {
        "base_id": "shift_x",
        "category": "neuro",
        "nonspecific_prompt": "A patient has headache.",
        "specific_prompt": "A patient has headache, fever, neck stiffness, and confusion.",
        "nonspecific_target": "headache",
        "specific_cue_1": "fever",
        "specific_cue_2": "neck stiffness",
        "specific_cue_3": "confusion",
        "nonspecific_expected": "broad headache context",
        "specific_expected": "meningitis",
        "specific_aliases": ["meningitis"],
        "nonspecific_aliases": ["headache"],
        "diagnosis_aliases": ["meningitis"],
    }

    rows = expanded_rows(case)

    assert len(rows) == 7
    assert rows[0]["id"] == "shift_x__nonspecific_alone_cue"
    assert rows[0]["base_id"] == "shift_x"
    assert rows[0]["category"] == "neuro"
    assert rows[0]["specific_targets"] == ["fever", "neck stiffness", "confusion"]
    assert rows[0]["specific_aliases"] == ["meningitis"]
    assert rows[5]["variant"] == "specific_full_specific_cue_3"
    assert rows[5]["target_text"] == "confusion"
    assert rows[6]["position_mode"] == "last_token"
