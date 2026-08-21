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


def test_describe_artifact_summarizes_every_row(tmp_path):
    """A prefix understated DDXPlus at mean 6.59 / max 15 against a true 6.79 /
    21, because the file is grouped by diagnosis and its first rows are a
    corner of the label space."""
    rows = [{**ROW, "cue_targets": ["a"]}] * 40 + [{**ROW, "cue_targets": ["a", "b", "c"]}] * 10
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", rows)
    described = describe_artifact(path)
    assert described["rows"] == 50
    assert described["sampled"] == 50
    assert described["max_cues"] == 3
    assert described["mean_cues"] == 1.4


def test_the_card_claims_no_sampling(tmp_path):
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", [ROW] * 50)
    card = build_card("a/b", [describe_artifact(path)], private=True)
    assert "rows: 50" in card
    assert "first" not in card


def test_the_example_prompt_is_drawn_at_random_and_reproducibly(tmp_path):
    """Row 0 of a file grouped by diagnosis is not a sample of it."""
    rows = [{**ROW, "prompt": f"prompt {i}"} for i in range(50)]
    path = write_cases(tmp_path, "ddxplus_cases.jsonl", rows)
    first = describe_artifact(path, example_seed=17)
    assert first["example_prompt"] == describe_artifact(path, example_seed=17)["example_prompt"]
    assert first["example_prompt"] != describe_artifact(path, example_seed=99)["example_prompt"]
