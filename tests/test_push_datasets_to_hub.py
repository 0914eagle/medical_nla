import json

from scripts.push_datasets_to_hub import build_card, describe_artifact


def write_cases(tmp_path, name, rows):
    path = tmp_path / name
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    return path


ROW = {
    "id": "ddxplus_pneumonia_0000001__cues_all",
    "prompt": "A patient presents with the following findings:\n- a cough\n",
    "cue_targets": ["a cough", "a fever"],
    "cue_polarities": ["positive", "negative"],
    "clean_cues": True,
    "negative_cues": True,
    "prefer_symptoms": False,
}


def test_describe_artifact_reports_the_flags_recorded_on_the_rows(tmp_path):
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW, ROW])
    described = describe_artifact(path)
    assert described["rows"] == 2
    assert described["provenance"] == {
        "clean_cues": True,
        "negative_cues": True,
        "prefer_symptoms": False,
    }
    assert described["mean_cues"] == 2.0
    assert described["cue_polarity"] == {"positive": 2, "negative": 2}


def test_describe_artifact_counts_every_row_not_just_the_sample(tmp_path):
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW] * 50)
    described = describe_artifact(path, sample=10)
    assert described["rows"] == 50
    assert described["sampled"] == 10


def test_card_records_provenance_and_attribution(tmp_path):
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW])
    card = build_card("acct/medical-nla-cases", [describe_artifact(path)], private=True)
    assert "acct/medical-nla-cases" in card
    assert "`negative_cues`: `True`" in card
    assert "DDXPlus" in card
    assert "CC BY 4.0" in card
    assert "private" in card.lower()


def test_card_attributes_each_source_present(tmp_path):
    ddx = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW])
    mcr = write_cases(tmp_path, "mcr_cases_train.jsonl", [ROW])
    card = build_card("a/b", [describe_artifact(ddx), describe_artifact(mcr)], private=False)
    assert "DDXPlus" in card
    assert "MedCaseReasoning" in card
    assert "private" not in card.lower()


def test_card_explains_why_reasoning_quotes_are_not_the_cues(tmp_path):
    # The measurement that rejected the first ingestion belongs with the data.
    path = write_cases(tmp_path, "mcr_cases_train.jsonl", [ROW])
    card = build_card("a/b", [describe_artifact(path)], private=True)
    assert "1.7%" in card
    assert "diagnostic_reasoning" in card


def test_card_marks_a_statistic_computed_from_a_sample(tmp_path):
    # The row count is exact; the cue statistics are not, when the file is
    # longer than the sample, and the card must not present them as if they were.
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW] * 50)
    card = build_card("a/b", [describe_artifact(path, sample=10)], private=True)
    assert "rows: 50" in card
    assert "(first 10 rows)" in card


def test_card_does_not_add_a_note_when_every_row_was_read(tmp_path):
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW] * 5)
    card = build_card("a/b", [describe_artifact(path, sample=1000)], private=True)
    assert "first" not in card
