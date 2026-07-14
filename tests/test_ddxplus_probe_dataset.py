import json
import random

from scripts.make_ddxplus_probe_dataset import make_case, strip_question_to_phrase, variant_rows


def test_make_ddxplus_probe_case_prefers_symptoms_and_variants():
    evidence_meta = {
        "E_DYSPNEA": {"question_en": "Do you have dyspnea?", "is_antecedent": False},
        "E_CHEST_PAIN": {
            "question_en": "Do you have pleuritic chest pain?",
            "is_antecedent": False,
        },
        "E_TACHYCARDIA": {"question_en": "Do you have tachycardia?", "is_antecedent": False},
        "E_SURGERY": {"question_en": "Have you had recent surgery?", "is_antecedent": True},
    }
    row = {
        "PATHOLOGY": "Pulmonary embolism",
        "EVIDENCES": json.dumps(
            ["E_DYSPNEA", "E_CHEST_PAIN", "E_TACHYCARDIA", "E_SURGERY"]
        ),
        "id": "patient_001",
    }

    case = make_case(
        row,
        row_index=0,
        evidence_meta=evidence_meta,
        rng=random.Random(0),
        prefer_symptoms=True,
        max_cues=3,
    )

    assert case is not None
    assert case["source"] == "ddxplus"
    assert case["diagnosis_id"] == "pulmonary_embolism"
    assert case["cue_types"] == ["symptom", "symptom", "symptom"]
    rows = variant_rows(case)
    assert [row["variant"] for row in rows] == [
        "single_cue",
        "single_format",
        "multi_cue_1",
        "multi_cue_2",
        "multi_cue_3",
        "multi_format",
    ]
    assert rows[0]["position_mode"] == "target_text"
    assert rows[0]["target_text_strategy"] == "span_mean"
    assert rows[-1]["position_mode"] == "last_token"
    assert rows[-1]["diagnosis_aliases"] == ["Pulmonary embolism"]


def test_make_ddxplus_probe_case_parses_ddxplus_literal_and_value_meaning():
    evidence_meta = {
        "E_53": {"question_en": "Do you have pain somewhere?", "is_antecedent": False},
        "E_55": {
            "question_en": "Where is the pain located?",
            "is_antecedent": False,
            "value_meaning": {"V_29": {"en": "chest"}},
            "possible-values": [],
        },
        "E_91": {
            "question_en": "Do you have a fever?",
            "is_antecedent": False,
            "value_meaning": {},
            "possible-values": [],
        },
    }
    row = {
        "PATHOLOGY": "Pneumonia",
        "EVIDENCES": "['E_53', 'E_55_@_V_29', 'E_91']",
    }

    case = make_case(
        row,
        row_index=1,
        evidence_meta=evidence_meta,
        rng=random.Random(1),
        prefer_symptoms=True,
        max_cues=3,
    )

    assert case is not None
    assert "E_55" in case["cue_evidence_ids"]
    assert not any(target.startswith("'") or target.startswith("[") for target in case["cue_targets"])
    assert any("chest" in target for target in case["cue_targets"])


def test_strip_question_to_phrase_removes_second_person_fragments():
    assert (
        strip_question_to_phrase("Have you noticed a wheezing sound when you exhale?")
        == "a wheezing sound when exhaling"
    )
