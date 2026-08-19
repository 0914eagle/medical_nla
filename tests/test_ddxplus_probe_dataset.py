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

    # "Do you have pain somewhere?" is a generic screening question that
    # clean_cues drops, so only two usable cues remain here; ask for two.
    case = make_case(
        row,
        row_index=1,
        evidence_meta=evidence_meta,
        rng=random.Random(1),
        prefer_symptoms=True,
        max_cues=2,
    )

    assert case is not None
    assert "E_55" in case["cue_evidence_ids"]
    assert not any(target.startswith("'") or target.startswith("[") for target in case["cue_targets"])
    assert any("chest" in target for target in case["cue_targets"])
    # the generic cue is excluded rather than silently reworded
    assert not any("somewhere" in target for target in case["cue_targets"])


def test_make_ddxplus_probe_case_returns_none_when_usable_cues_fall_short():
    # Guards the branch the previous test used to hit by accident: when
    # filtering leaves fewer cues than requested, the case is dropped rather
    # than emitted with too few.
    evidence_meta = {
        "E_53": {"question_en": "Do you have pain somewhere?", "is_antecedent": False},
        "E_91": {"question_en": "Do you have a fever?", "is_antecedent": False},
    }
    row = {"PATHOLOGY": "Pneumonia", "EVIDENCES": "['E_53', 'E_91']"}

    case = make_case(
        row,
        row_index=1,
        evidence_meta=evidence_meta,
        rng=random.Random(1),
        prefer_symptoms=True,
        max_cues=2,
    )

    assert case is None


def test_make_ddxplus_probe_case_filters_negative_and_generic_cues():
    evidence_meta = {
        "E_TRAVEL": {
            "question_en": "Have you traveled out of the country in the last 4 weeks?",
            "is_antecedent": True,
            "value_meaning": {"N": {"en": "N"}},
        },
        "E_RADIATE": {
            "question_en": "Does the pain radiate to another location?",
            "is_antecedent": False,
            "value_meaning": {"V_NONE": {"en": "nowhere"}},
        },
        "E_PRECISE": {
            "question_en": "How precisely is the pain located?",
            "is_antecedent": False,
        },
        "E_DYSPNEA": {
            "question_en": "Do you have shortness of breath?",
            "is_antecedent": False,
        },
        "E_WHEEZE": {
            "question_en": "Have you noticed a wheezing sound when you exhale?",
            "is_antecedent": False,
        },
        "E_SPUTUM": {
            "question_en": "Do you have increased sputum?",
            "is_antecedent": False,
        },
    }
    row = {
        "PATHOLOGY": "Acute COPD exacerbation / infection",
        "EVIDENCES": json.dumps(
            [
                "E_TRAVEL_@_N",
                "E_RADIATE_@_V_NONE",
                "E_PRECISE",
                "E_DYSPNEA",
                "E_WHEEZE",
                "E_SPUTUM",
            ]
        ),
    }

    case = make_case(
        row,
        row_index=2,
        evidence_meta=evidence_meta,
        rng=random.Random(2),
        prefer_symptoms=True,
        max_cues=3,
        clean_cues=True,
    )

    assert case is not None
    assert case["excluded_cue_count"] == 3
    assert all(" N" not in target for target in case["cue_targets"])
    assert all("nowhere" not in target for target in case["cue_targets"])
    assert all("how precisely is the pain located" not in target for target in case["cue_targets"])
    assert sorted(case["cue_evidence_ids"]) == ["E_DYSPNEA", "E_SPUTUM", "E_WHEEZE"]


def test_strip_question_to_phrase_removes_second_person_fragments():
    assert (
        strip_question_to_phrase("Have you noticed a wheezing sound when you exhale?")
        == "a wheezing sound when exhaling"
    )


def test_render_negative_phrase_negates_the_auxiliary():
    from scripts.make_ddxplus_probe_dataset import render_negative_phrase

    assert (
        render_negative_phrase("Have you traveled out of the country in the last 4 weeks?")
        == "has not traveled out of the country in the last 4 weeks"
    )
    assert render_negative_phrase("Do you have a cough?") == "does not have a cough"
    assert (
        render_negative_phrase("Are you experiencing shortness of breath?")
        == "is not experiencing shortness of breath"
    )
    assert render_negative_phrase("Did you lose consciousness?") == "did not lose consciousness"
    assert render_negative_phrase("Does the patient have a fever?") == "does not have a fever"


def test_render_negative_phrase_rewrites_second_person():
    from scripts.make_ddxplus_probe_dataset import render_negative_phrase

    assert (
        render_negative_phrase("Have you noticed a wheezing sound when you exhale?")
        == "has not noticed a wheezing sound when exhaling"
    )


def test_render_negative_phrase_returns_none_for_unknown_openings():
    # Better to drop the cue than emit something ungrammatical.
    from scripts.make_ddxplus_probe_dataset import render_negative_phrase

    assert render_negative_phrase("Where is the pain located?") is None


def test_cue_from_entry_renders_negatives_only_when_asked():
    from scripts.make_ddxplus_probe_dataset import cue_from_entry

    meta = {
        "E_TRAVEL": {
            "question_en": "Have you traveled out of the country in the last 4 weeks?",
            "value_meaning": {"N": {"en": "no"}},
            "is_antecedent": True,
        }
    }
    dropped = cue_from_entry("E_TRAVEL_@_N", meta, clean_cues=True)
    assert dropped["excluded"]
    assert dropped["exclusion_reason"] == "negative_or_low_information_value"
    assert dropped["cue_polarity"] == "positive"

    kept = cue_from_entry("E_TRAVEL_@_N", meta, clean_cues=True, negative_cues=True)
    assert not kept["excluded"]
    assert kept["cue_polarity"] == "negative"
    assert kept["cue_text"] == "has not traveled out of the country in the last 4 weeks"
    # The old rendering appended the value and read as an affirmative.
    assert not kept["cue_text"].endswith(" no")


def test_cue_from_entry_keeps_positive_rendering_unchanged():
    from scripts.make_ddxplus_probe_dataset import cue_from_entry

    meta = {
        "E_55": {
            "question_en": "Where is the pain located?",
            "value_meaning": {"V_29": {"en": "chest"}},
            "is_antecedent": False,
        },
        "E_COUGH": {"question_en": "Do you have a cough?", "is_antecedent": False},
    }
    # The answer is folded into the statement rather than pasted after the question.
    valued = cue_from_entry("E_55_@_V_29", meta, clean_cues=True, negative_cues=True)
    assert valued["cue_text"] == "the pain is located in the chest"
    assert valued["cue_polarity"] == "positive"

    bare = cue_from_entry("E_COUGH", meta, clean_cues=True, negative_cues=True)
    assert bare["cue_text"] == "a cough"
    assert bare["cue_polarity"] == "positive"


def test_cue_from_entry_drops_unrenderable_negatives():
    from scripts.make_ddxplus_probe_dataset import cue_from_entry

    meta = {
        "E_RADIATE": {
            "question_en": "Where does the pain radiate to?",
            "value_meaning": {"V_NONE": {"en": "nowhere"}},
            "is_antecedent": False,
        }
    }
    cue = cue_from_entry("E_RADIATE_@_V_NONE", meta, clean_cues=True, negative_cues=True)
    assert cue["excluded"]
    assert cue["exclusion_reason"] == "negative_value_unrenderable"


def test_cue_from_entry_separates_missing_answers_from_negative_ones():
    from scripts.make_ddxplus_probe_dataset import cue_from_entry

    meta = {
        "E_SMOKE": {
            "question_en": "Do you smoke?",
            "value_meaning": {"U": {"en": "unknown"}},
            "is_antecedent": True,
        }
    }
    cue = cue_from_entry("E_SMOKE_@_U", meta, clean_cues=True, negative_cues=True)
    assert cue["excluded"]
    assert cue["exclusion_reason"] == "uninformative_value"


def _cue(question, value_meaning=None, value_id=None, **kwargs):
    from scripts.make_ddxplus_probe_dataset import cue_from_entry

    meta = {"E": {"question_en": question, "is_antecedent": False}}
    if value_meaning:
        meta["E"]["value_meaning"] = value_meaning
    entry = f"E_@_{value_id}" if value_id else "E"
    return cue_from_entry(entry, meta, clean_cues=True, **kwargs)


def test_yes_no_questions_are_uninverted_into_findings():
    # "Is the rash swollen?" used to survive as a question inside the prompt.
    assert _cue("Is the rash swollen?")["cue_text"] == "the rash is swollen"
    assert (
        _cue("Are your symptoms more prominent at night?")["cue_text"]
        == "their symptoms are more prominent at night"
    )
    assert (
        _cue("Does the person have a whooping cough?")["cue_text"]
        == "the person does have a whooping cough"
    )
    assert _cue("Did your cheeks suddenly turn red?")["cue_text"] == (
        "their cheeks did suddenly turn red"
    )


def test_second_person_questions_keep_the_shorter_noun_phrase():
    assert _cue("Do you have a cough?")["cue_text"] == "a cough"


def test_affirmative_value_is_not_appended():
    # "Y" was missing from the affirmative set, so cues read "peel off Y".
    cue = _cue("Do your lesions peel off?", {"Y": {"en": "Y"}}, "Y")
    assert cue["cue_text"] == "their lesions do peel off"


def test_long_subject_falls_back_to_dropping_do_support():
    cue = _cue("Do any members of your immediate family have a psychiatric illness?")
    assert cue["cue_text"] == "any members of their immediate family have a psychiatric illness"


def test_restating_parenthetical_is_removed_before_uninverting():
    cue = _cue("Is the lesion (or are the lesions) larger than 1cm?", {"Y": {"en": "Y"}}, "Y")
    assert cue["cue_text"] == "the lesion is larger than 1cm"


def test_wh_answers_are_folded_into_the_statement():
    located = _cue("Where is the swelling located?", {"V1": {"en": "iliac wing(R)"}}, "V1")
    assert located["cue_text"] == "the swelling is located in the iliac wing(R)"
    colored = _cue("What color is the rash?", {"V2": {"en": "pink"}}, "V2")
    assert colored["cue_text"] == "the rash color is pink"
    rated = _cue("How severe is the itching?", {"V7": {"en": "7"}}, "V7")
    assert rated["cue_text"] == "the itching is severe: 7"


def test_missing_answer_values_are_dropped():
    cue = _cue("What color is the rash?", {"V1": {"en": "NA"}}, "V1")
    assert cue["excluded"]
    assert cue["exclusion_reason"] == "uninformative_value"


def test_compound_questions_use_the_override_table():
    positive = _cue("Is your nose or the back of your throat itchy?")
    assert positive["cue_text"] == "an itchy nose or an itchy back of the throat"
    negative = _cue(
        "Is your nose or the back of your throat itchy?",
        {"N": {"en": "no"}},
        "N",
        negative_cues=True,
    )
    assert negative["cue_polarity"] == "negative"
    assert negative["cue_text"] == "no itching of the nose or the back of the throat"


def test_unrenderable_questions_are_dropped_not_emitted():
    from scripts.make_ddxplus_probe_dataset import is_malformed_cue

    cue = _cue("Was it a sudden onset or was it gradual?")
    assert cue["excluded"]
    assert cue["exclusion_reason"] == "unrenderable_question"
    # The gate itself: anything still shaped like a question must be caught.
    assert is_malformed_cue("is the rash swollen")
    assert is_malformed_cue("where is the pain located chest")
    assert is_malformed_cue("their nose is or the back of their throat itchy")
    assert not is_malformed_cue("the rash is swollen")
    assert not is_malformed_cue("has not traveled out of the country")
