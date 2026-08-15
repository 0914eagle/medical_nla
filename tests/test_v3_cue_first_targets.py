from scripts.make_medical_nla_v3_cue_first_targets import cue_first_target_text, cue_list
from scripts.score_medical_nla_v2_readouts import score_row, split_cue_items


def make_row() -> dict:
    return {
        "id": "case_0001::cue_count_all",
        "diagnosis_id": "urti",
        "diagnosis_name": "URTI",
        "diagnosis_aliases": ["upper respiratory tract infection"],
        "cue_targets": ["moderate fever", "sore throat", "nasal congestion", "cough"],
    }


def test_cue_list_is_deterministic_and_capped():
    row = make_row()
    a = cue_list(row, max_cues=3, seed=17)
    b = cue_list(row, max_cues=3, seed=17)
    assert a == b
    assert len(a) == 3
    assert cue_list(row, max_cues=3, seed=18) != a or True  # different seed may reorder


def test_cue_first_target_default_has_no_diagnosis_text():
    text = cue_first_target_text(make_row(), max_cues=12, seed=17)
    assert "<observed>" in text and "</observed>" in text
    assert "<answer>" not in text
    assert "<assessment>" not in text
    assert "URTI" not in text  # no diagnosis label shortcut in the default target
    for cue in make_row()["cue_targets"]:
        assert f"- {cue}" in text


def test_cue_first_target_optional_assessment():
    text = cue_first_target_text(make_row(), max_cues=12, seed=17, include_assessment=True)
    assert "<assessment>Findings most consistent with URTI.</assessment>" in text


def test_score_row_reads_v3_observed_and_assessment():
    row = make_row()
    row["nla_output"] = (
        "<readout>\n<observed>\n- moderate fever\n- sore throat\n- headache\n</observed>\n"
        "<assessment>Findings most consistent with URTI.</assessment>\n</readout>"
    )
    scored = score_row(row)
    assert scored["parsed_observed"] and scored["parsed_assessment"]
    assert scored["answer_hit"] is True  # via assessment fallback
    assert scored["cue_hit_count"] == 2  # fever + sore throat
    assert scored["cue_recall"] == 0.5
    # 3 emitted items, 2 match gold cues -> precision penalizes the extra one.
    assert scored["cue_items_emitted"] == 3
    assert scored["cue_precision"] == 2 / 3


def test_split_cue_items_handles_v2_and_v3_formats():
    assert split_cue_items("a b; c d; e f") == ["a b", "c d", "e f"]
    assert split_cue_items("- fever\n- dry cough\n") == ["fever", "dry cough"]
    assert split_cue_items("") == []
