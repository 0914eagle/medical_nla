from scripts.audit_ddxplus_cue_rendering import (
    audit_cases,
    audit_vocabulary,
    suspicious_flags,
    vocabulary_rows,
)

EVIDENCE_META = {
    "E_COUGH": {"question_en": "Do you have a cough?", "is_antecedent": False},
    "E_TRAVEL": {
        "question_en": "Have you traveled out of the country in the last 4 weeks?",
        "value_meaning": {"N": {"en": "no"}, "Y": {"en": "yes"}},
        "is_antecedent": True,
    },
}


def test_vocabulary_rows_covers_every_question_value_pair():
    rows = vocabulary_rows(EVIDENCE_META, negative_cues=True)
    entries = {row["entry"] for row in rows}
    # Y/N are included for every item, labelled or not, since that is how
    # negatives are recorded and they are what most needs review.
    assert {"E_COUGH", "E_COUGH_@_N", "E_TRAVEL_@_N", "E_TRAVEL_@_Y"} <= entries


def test_audit_vocabulary_passes_a_clean_questionnaire():
    failures, summary = audit_vocabulary(vocabulary_rows(EVIDENCE_META, negative_cues=True))
    assert failures == []
    assert summary["renderings"] >= 3
    assert summary["kept"] >= 2


def test_audit_vocabulary_fails_a_kept_but_malformed_rendering():
    rows = [
        {
            "entry": "E_X",
            "question": "Is the rash swollen?",
            "value_label": None,
            "polarity": "positive",
            "excluded": False,
            "reason": "",
            "cue_text": "is the rash swollen",
        }
    ]
    failures, _ = audit_vocabulary(rows)
    assert len(failures) == 1
    assert "malformed" in failures[0]


def test_audit_vocabulary_fails_a_kept_opaque_value():
    rows = [
        {
            "entry": "E_X_@_V_29",
            "question": "Where is the pain located?",
            "value_label": "V_29",
            "polarity": "positive",
            "excluded": False,
            "reason": "",
            "cue_text": "the pain is located in the V_29",
        }
    ]
    failures, _ = audit_vocabulary(rows)
    assert any("opaque" in failure for failure in failures)


def test_suspicious_flags_are_soft_signals():
    assert "leftover_question_mark" in suspicious_flags("a cough?")
    assert "second_person" in suspicious_flags("do you have a cough")
    assert "double_subject" in suspicious_flags("the patient feels the patient is unwell")
    assert suspicious_flags("a persistent cough") == []


def test_audit_cases_catches_a_cue_missing_from_its_own_prompt():
    cases = [
        {
            "id": "c1",
            "prompt": "A patient presents with a cough.",
            "cue_targets": ["a fever"],
            "cue_polarities": ["positive"],
        }
    ]
    failures, _ = audit_cases(cases)
    assert any("not verbatim" in failure for failure in failures)


def test_audit_cases_catches_duplicate_cues():
    cases = [
        {
            "id": "c1",
            "prompt": "A patient presents with a cough and a cough.",
            "cue_targets": ["a cough", "a cough"],
            "cue_polarities": ["positive", "positive"],
        }
    ]
    failures, _ = audit_cases(cases)
    assert any("duplicate" in failure for failure in failures)


def test_audit_cases_reports_distributions_over_every_case():
    cases = [
        {
            "id": f"c{i}",
            "prompt": "A patient presents with a cough and a fever.",
            "cue_targets": ["a cough", "a fever"],
            "cue_polarities": ["positive", "negative"],
        }
        for i in range(10)
    ]
    failures, summary = audit_cases(cases)
    assert failures == []
    assert summary["cases"] == 10
    assert summary["cue_count"]["max"] == 2
    assert summary["cue_polarity"] == {"positive": 10, "negative": 10}


def test_vocabulary_rows_covers_negatives_without_a_value_meaning_entry():
    # value_meaning rarely lists N, and negatives are what most needs review.
    meta = {"E_TRAVEL": {"question_en": "Have you traveled recently?", "is_antecedent": True}}
    rows = vocabulary_rows(meta, negative_cues=True)
    assert any(row["entry"] == "E_TRAVEL_@_N" for row in rows)
    negative = next(row for row in rows if row["entry"] == "E_TRAVEL_@_N")
    assert negative["polarity"] == "negative"


def test_vocabulary_rows_uses_the_entries_the_data_contains_when_given():
    meta = {
        "E_A": {"question_en": "Do you have a cough?", "is_antecedent": False},
        "E_B": {"question_en": "Do you have a fever?", "is_antecedent": False},
    }
    rows = vocabulary_rows(meta, negative_cues=True, entries_used={"E_A", "E_B_@_N"})
    assert {row["entry"] for row in rows} == {"E_A", "E_B_@_N"}


def test_entries_in_use_reads_the_patient_file(tmp_path):
    from scripts.audit_ddxplus_cue_rendering import entries_in_use

    path = tmp_path / "patients.csv"
    path.write_text(
        "PATHOLOGY,EVIDENCES\nPneumonia,\"['E_A', 'E_B_@_N']\"\nAnemia,\"['E_A']\"\n",
        encoding="utf-8",
    )
    assert entries_in_use(path) == {"E_A", "E_B_@_N"}


def test_known_abbreviations_are_not_flagged_as_codes():
    # Nine real conditions were flagged, which buries a genuine unresolved code.
    assert suspicious_flags("a chronic obstructive pulmonary disease (COPD)") == []
    assert suspicious_flags("infected with the human immunodeficiency virus (HIV)") == []
    assert "uppercase_code" in suspicious_flags("the pain is located in the V29 region")
    assert "uppercase_code" in suspicious_flags("the pain is located in the V_29 region")


def test_audit_cases_reports_cue_length_and_nesting():
    cases = [
        {
            "id": "c1",
            "prompt": "A patient presents with a cough that produces sputum and a cough.",
            "cue_targets": ["a cough that produces sputum", "a cough"],
            "cue_polarities": ["positive", "positive"],
        }
    ]
    _, summary = audit_cases(cases)
    assert summary["cases_with_nested_cues"] == 1
    assert summary["cue_words"]["max"] == 5


def test_audit_cases_counts_no_nesting_for_distinct_cues():
    cases = [
        {
            "id": "c1",
            "prompt": "A patient presents with a fever and night sweats.",
            "cue_targets": ["a fever", "night sweats"],
            "cue_polarities": ["positive", "positive"],
        }
    ]
    _, summary = audit_cases(cases)
    assert summary["cases_with_nested_cues"] == 0


def test_adult_onset_antecedents_in_a_child_are_reported():
    """DDXPlus samples antecedents without conditioning on age; verified not to
    be a rendering fault, so it is counted rather than failed."""
    from scripts.audit_ddxplus_cue_rendering import adult_onset_antecedents_in_a_child

    child = {"age": 4, "cue_targets": ["a cough", "parkinson's disease", "heart failure"]}
    assert adult_onset_antecedents_in_a_child(child) == [
        "parkinson's disease",
        "heart failure",
    ]


def test_the_same_history_in_an_adult_is_not_reported():
    from scripts.audit_ddxplus_cue_rendering import adult_onset_antecedents_in_a_child

    adult = {"age": 71, "cue_targets": ["parkinson's disease", "heart failure"]}
    assert adult_onset_antecedents_in_a_child(adult) == []
    assert adult_onset_antecedents_in_a_child({"cue_targets": ["heart failure"]}) == []


def test_soft_flags_report_what_raised_them():
    """A flag firing on a tenth of a prose corpus is only usable if it says
    which strings it matched."""
    from scripts.audit_ddxplus_cue_rendering import suspicious_flag_matches

    matches = dict(suspicious_flag_matches("the pain is located in the V_29 region"))
    assert matches["uppercase_code"] == ["V_29"]
    assert "uppercase_code" not in dict(suspicious_flag_matches("a COPD exacerbation"))
