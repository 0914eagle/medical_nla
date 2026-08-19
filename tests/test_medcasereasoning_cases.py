from scripts.make_medcasereasoning_cases import (
    case_row,
    drop_contained,
    extract_quotes,
    normalize_quote,
    split_statements,
)


def test_normalize_folds_typography_and_whitespace():
    assert normalize_quote("chest  pain\neven at rest") == "chest pain even at rest"
    assert normalize_quote("the patient’s cough") == "the patient's cough"
    assert normalize_quote("“fever”") == '"fever"'


def test_split_statements_handles_numbering_styles():
    reasoning = "1. First point.\n2) Second point.\nStep 3: Third point."
    assert split_statements(reasoning) == ["First point.", "Second point.", "Third point."]


def test_split_statements_falls_back_to_whole_text():
    assert split_statements("Unnumbered reasoning.") == ["Unnumbered reasoning."]


def test_extract_quotes_reading_order_and_dedup():
    reasoning = (
        '1. The note records "persistent chest pain at rest" which is concerning.\n'
        '2. It also mentions "shortness of breath on exertion" here.\n'
        '3. Again "persistent chest pain at rest" is repeated.'
    )
    assert extract_quotes(reasoning) == [
        "persistent chest pain at rest",
        "shortness of breath on exertion",
    ]


def test_extract_quotes_skips_short_fragments():
    # Below the 12-char floor: too short to be a reliable span.
    assert extract_quotes('1. The word "fever" appears.') == []


def test_drop_contained_removes_nested_spans_keeps_order():
    quotes = ["chest pain at rest", "chest pain", "shortness of breath"]
    assert drop_contained(quotes) == ["chest pain at rest", "shortness of breath"]


def test_case_row_keeps_only_quotes_present_in_prompt():
    record = {
        "case_prompt": (
            "A 62-year-old man reports persistent chest pain at rest and "
            "shortness of breath on exertion over three weeks."
        ),
        "diagnostic_reasoning": (
            '1. The history notes "persistent chest pain at rest".\n'
            '2. There is also "shortness of breath on exertion".\n'
            '3. The imaging showed "a large pericardial effusion" (not in the prompt).'
        ),
        "final_diagnosis": "Unstable angina",
    }
    row, stats = case_row(record, index=7, min_cues=2, max_cues=None)
    assert row is not None
    assert row["cue_targets"] == [
        "persistent chest pain at rest",
        "shortness of breath on exertion",
    ]
    # every emitted cue is locatable in the prompt -> a well-defined span
    assert all(cue in row["prompt"] for cue in row["cue_targets"])
    assert row["diagnosis_id"] == "unstable_angina"
    assert row["source"] == "medcasereasoning"
    assert row["variant"] == "cue_count_all"
    assert row["id"].endswith("__cues_all")
    assert stats == {"quotes": 3, "matched": 2, "unmatched": 1}


def test_case_row_drops_case_below_min_cues():
    record = {
        "case_prompt": "A patient with fever of unknown origin.",
        "diagnostic_reasoning": '1. The note says "fever of unknown origin".',
        "final_diagnosis": "Sepsis",
    }
    row, stats = case_row(record, index=0, min_cues=3, max_cues=None)
    assert row is None
    assert stats["matched"] == 1


def test_case_row_requires_all_fields():
    row, _ = case_row({"case_prompt": "text"}, index=0, min_cues=1, max_cues=None)
    assert row is None


def test_case_row_respects_max_cues():
    record = {
        "case_prompt": "Findings include alpha marker one, beta marker two, gamma marker three.",
        "diagnostic_reasoning": (
            '1. See "alpha marker one".\n2. See "beta marker two".\n3. See "gamma marker three".'
        ),
        "final_diagnosis": "Test condition",
    }
    row, _ = case_row(record, index=1, min_cues=2, max_cues=2)
    assert row["cue_count"] == 2
    # available_cue_count records what was matched before the cap
    assert row["available_cue_count"] == 3
