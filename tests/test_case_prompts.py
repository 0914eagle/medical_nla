import pytest

from src.case_prompts import (
    build_prompt,
    early_answer_prompt,
    findings_prefix,
    parse_answer,
    prose_prefix,
    truncate_chain,
)


def test_conditions_share_a_byte_identical_prefix():
    # The property the whole extraction plan rests on: cue positions cannot see
    # the instruction, so one extraction serves both arms.
    prefix = findings_prefix(["a cough", "a fever"])
    direct = build_prompt(prefix, "direct")
    cot = build_prompt(prefix, "cot")
    assert direct.startswith(prefix)
    assert cot.startswith(prefix)
    assert direct != cot


def test_cues_appear_verbatim_in_the_prompt():
    # Extraction resolves a cue by substring, so this must hold by construction.
    cues = ["chest pain even at rest", "the rash is swollen"]
    prompt = build_prompt(findings_prefix(cues), "direct")
    for cue in cues:
        assert cue in prompt


def test_prose_prefix_keeps_the_presentation_intact():
    text = "A 58-year-old woman presented with weight loss. Examination was normal."
    prompt = build_prompt(prose_prefix(text), "cot")
    assert text in prompt


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="Unknown condition"):
        build_prompt(findings_prefix(["a cough"]), "freeform")


def test_parse_answer_reads_the_fixed_closing_string():
    assert parse_answer("...so the answer is Pneumonia.") == "Pneumonia"
    assert parse_answer("The answer is acute otitis media.") == "acute otitis media"
    # Tolerates a response that stops without the period.
    assert parse_answer("The answer is Anemia") == "Anemia"
    assert parse_answer("I am not sure.") is None


def test_truncate_chain_cuts_at_a_sentence_boundary():
    chain = "First the fever matters. Then the cough matters. Finally the rash matters."
    half = truncate_chain(chain, 0.5)
    assert half.endswith(".")
    assert "Then the cough" not in half or half.endswith("matters.")
    assert chain.startswith(half)


def test_truncate_chain_endpoints():
    chain = "One sentence. Two sentence."
    assert truncate_chain(chain, 1.0) == chain
    assert truncate_chain(chain, 0.0) == ""


def test_truncate_chain_rejects_a_fraction_outside_the_unit_interval():
    with pytest.raises(ValueError):
        truncate_chain("a chain", 1.5)


def test_early_answer_prompt_forces_an_answer_after_the_truncated_chain():
    prefix = findings_prefix(["a cough"])
    chain = "The cough suggests infection. The fever supports that."
    prompt = early_answer_prompt(prefix, chain, 0.5)
    assert prompt.startswith(prefix)
    assert prompt.rstrip().endswith("The answer is")
    assert "The fever supports that." not in prompt


def test_patient_descriptor_reads_age_and_sex():
    from src.case_prompts import patient_descriptor

    assert patient_descriptor(34, "F") == "A 34-year-old woman"
    assert patient_descriptor(70, "M") == "A 70-year-old man"
    # Paediatric cases matter: several DDXPlus pathologies are age-specific.
    assert patient_descriptor(3, "M") == "A 3-year-old boy"
    assert patient_descriptor(11, "F") == "A 11-year-old girl"


def test_patient_descriptor_degrades_without_demographics():
    from src.case_prompts import patient_descriptor

    assert patient_descriptor(None, None) == "A patient"
    assert patient_descriptor(55, None) == "A 55-year-old patient"
    assert patient_descriptor("", "") == "A patient"


def test_demographics_head_the_sentence_and_are_not_cues():
    prefix = findings_prefix(["a cough", "a fever"], age=3, sex="M")
    assert prefix.startswith("You are an expert physician. A 3-year-old boy presents with")
    # They are context for reading the findings, not a finding to read back.
    assert "- A 3-year-old boy" not in prefix


def test_conditions_still_share_a_prefix_with_demographics():
    prefix = findings_prefix(["a cough"], age=34, sex="F")
    assert build_prompt(prefix, "direct").startswith(prefix)
    assert build_prompt(prefix, "cot").startswith(prefix)
