from scripts.make_clinical_span_cases import (
    case_row,
    clean_span,
    ends_with_abbreviation,
    normalize_text,
    segment_cues,
    split_clauses,
    split_sentences,
    strip_leading_connective,
)

CASE_TEXT = (
    "A 58-year-old woman presented with a three-month history of involuntary weight loss. "
    "She reported a persistent cough productive of blood-streaked sputum. "
    "Examination revealed dullness to percussion at the right upper lobe."
)


def test_normalize_collapses_whitespace_and_typography():
    assert normalize_text("chest  pain\neven at rest") == "chest pain even at rest"
    assert normalize_text("the patient’s cough") == "the patient's cough"


def test_split_sentences_basic():
    assert len(split_sentences(CASE_TEXT)) == 3


def test_ends_with_abbreviation_detects_medical_abbreviations():
    assert ends_with_abbreviation("The patient saw Dr.")
    assert ends_with_abbreviation("A dose of 5 mg.")
    assert not ends_with_abbreviation("She reported a cough.")


def test_split_sentences_does_not_break_on_abbreviation():
    text = "The patient was seen by Dr. Kim. She reported a persistent cough."
    sentences = split_sentences(text)
    assert len(sentences) == 2
    assert sentences[0] == "The patient was seen by Dr. Kim."


def test_split_clauses_leaves_short_sentences_intact():
    sentence = "She reported a persistent cough."
    assert split_clauses(sentence, max_words=25) == [sentence]


def test_split_clauses_breaks_long_sentences_on_clause_markers():
    sentence = (
        "The patient had a persistent cough productive of blood-streaked sputum; "
        "she also had night sweats, with weight loss over three months and no fever at any point"
    )
    parts = split_clauses(sentence, max_words=10)
    assert len(parts) > 1
    assert any("persistent cough" in part for part in parts)


def test_clean_span_trims_trailing_punctuation():
    assert clean_span("  a persistent cough.  ") == "a persistent cough"


def test_segment_cues_returns_verbatim_slices_of_the_text():
    text = normalize_text(CASE_TEXT)
    cues = segment_cues(text, min_words=4, max_words=25, max_cues=None)
    assert cues
    # the property the extraction pipeline depends on
    for cue in cues:
        assert cue in text


def test_segment_cues_respects_word_bounds():
    text = normalize_text(CASE_TEXT)
    cues = segment_cues(text, min_words=4, max_words=8, max_cues=None)
    assert all(4 <= len(cue.split()) <= 8 for cue in cues)


def test_segment_cues_drops_nested_spans():
    text = "Fever and chills were present. Fever and chills were present again later on."
    cues = segment_cues(text, min_words=3, max_words=25, max_cues=None)
    lowered = [cue.lower() for cue in cues]
    for i, cue in enumerate(lowered):
        for j, other in enumerate(lowered):
            if i != j:
                assert cue not in other


def test_segment_cues_honors_max_cues():
    text = normalize_text(CASE_TEXT)
    assert len(segment_cues(text, min_words=4, max_words=25, max_cues=2)) == 2


def test_case_row_builds_our_schema_with_resolvable_spans():
    record = {"case_prompt": CASE_TEXT, "final_diagnosis": "Pulmonary tuberculosis", "pmcid": "PMC1"}
    row, cues = case_row(
        record,
        index=3,
        text_field="case_prompt",
        label_field="final_diagnosis",
        source="mcr",
        min_cues=2,
        min_words=4,
        max_words=25,
        max_cues=None,
    )
    assert row is not None
    assert row["variant"] == "cue_count_all"
    assert row["diagnosis_id"] == "pulmonary_tuberculosis"
    assert row["case_id"] == "PMC1"
    assert row["cue_count"] == len(cues)
    assert all(cue in row["prompt"] for cue in row["cue_targets"])


def test_case_row_drops_case_below_min_cues():
    record = {"case_prompt": "Short note.", "final_diagnosis": "Anemia"}
    row, _ = case_row(
        record,
        index=0,
        text_field="case_prompt",
        label_field="final_diagnosis",
        source="mcr",
        min_cues=3,
        min_words=4,
        max_words=25,
        max_cues=None,
    )
    assert row is None


def test_case_row_requires_label():
    record = {"case_prompt": CASE_TEXT, "final_diagnosis": ""}
    row, _ = case_row(
        record,
        index=0,
        text_field="case_prompt",
        label_field="final_diagnosis",
        source="mcr",
        min_cues=1,
        min_words=4,
        max_words=25,
        max_cues=None,
    )
    assert row is None


def test_strip_leading_connective_removes_stranded_conjunctions():
    assert (
        strip_leading_connective("and swelling on the left side of his palate", min_words=3)
        == "swelling on the left side of his palate"
    )
    assert (
        strip_leading_connective("with no instability or weakness", min_words=3)
        == "no instability or weakness"
    )
    assert (
        strip_leading_connective("but hypotension persisted after fluids", min_words=3)
        == "hypotension persisted after fluids"
    )


def test_strip_leading_connective_preserves_negations():
    # Stripping a negation would invert the finding — the one unacceptable error.
    for span in (
        "without evidence of metastasis on imaging",
        "no instability or weakness was found",
        "not responsive to oral antibiotics",
        "denied any history of trauma",
        "negative for Tinel's sign bilaterally",
    ):
        assert strip_leading_connective(span, min_words=3) == span


def test_strip_leading_connective_stops_before_going_too_short():
    # Better a slightly awkward span than one below the usable length.
    assert strip_leading_connective("and fever", min_words=2) == "and fever"


def test_stripped_span_is_still_verbatim_in_the_text():
    text = normalize_text(
        "On examination, both elbows had a full range of motion, "
        "with no instability or weakness."
    )
    cues = segment_cues(text, min_words=4, max_words=10, max_cues=None)
    assert cues
    assert any(cue.startswith("no instability") for cue in cues)
    for cue in cues:
        assert cue in text


def test_no_cue_starts_with_a_stranded_connective():
    text = normalize_text(
        "A 30-year-old man presented with several months of pain, tenderness, "
        "and swelling on the left side of his palate, which was tender."
    )
    cues = segment_cues(text, min_words=4, max_words=12, max_cues=None)
    for cue in cues:
        assert cue.split()[0].lower() not in {"and", "but", "or", "with", "which"}


def test_leak_filter_catches_a_diagnosis_stored_as_an_identifier():
    """The bug that let contaminated cases through: the underscored label was
    compared against prose, where it can never appear."""
    from scripts.make_clinical_span_cases import mentions_diagnosis

    text = "A 45-year-old woman developed scleroderma renal crisis after prednisone."
    assert not mentions_diagnosis(text, "Scleroderma_renal_crisis__wrong")
    assert mentions_diagnosis(text, ["Scleroderma renal crisis", "Scleroderma_renal_crisis"])


def test_leak_filter_folds_typography_on_both_sides():
    from scripts.make_clinical_span_cases import mentions_diagnosis

    assert mentions_diagnosis("Findings of Guillain-Barre syndrome.", ["Guillain-Barré syndrome"])
