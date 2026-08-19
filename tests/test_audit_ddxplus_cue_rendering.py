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
    # one bare binary plus both values of the categorical
    assert len(rows) == 3
    assert {row["entry"] for row in rows} == {"E_COUGH", "E_TRAVEL_@_N", "E_TRAVEL_@_Y"}


def test_audit_vocabulary_passes_a_clean_questionnaire():
    failures, summary = audit_vocabulary(vocabulary_rows(EVIDENCE_META, negative_cues=True))
    assert failures == []
    assert summary["renderings"] == 3
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
