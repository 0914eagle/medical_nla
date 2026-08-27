from scripts.make_medical_nla_v2_sft_splits import structured_target_text
from scripts.score_medical_nla_v2_readouts import score_row


def test_structured_target_text_contains_answer_and_cues():
    row = {
        "diagnosis_name": "Pulmonary embolism",
        "cue_targets": ["pleuritic chest pain", "tachycardia", "recent surgery"],
    }

    target = structured_target_text(row, max_cues=3)

    assert target.startswith("<explanation>")
    assert "<readout>" in target
    assert "<task_type>diagnosis</task_type>" in target
    assert "<answer>Pulmonary embolism</answer>" in target
    assert "pleuritic chest pain; tachycardia; recent surgery" in target


def test_score_row_parses_answer_and_cue_recall():
    row = {
        "id": "case_001__multi_format",
        "diagnosis_id": "pulmonary_embolism",
        "diagnosis_name": "Pulmonary embolism",
        "diagnosis_aliases": ["Pulmonary embolism", "PE"],
        "cue_targets": ["pleuritic chest pain", "tachycardia", "recent surgery"],
        "nla_output": """
        <readout>
          <task_type>diagnosis</task_type>
          <answer>PE</answer>
          <supporting_cues>pleuritic chest pain; recent surgery</supporting_cues>
        </readout>
        """,
    }

    scored = score_row(row)

    assert scored["parsed_readout"]
    assert scored["parsed_answer"]
    assert scored["answer_hit"]
    assert scored["answer_hits"] == ["PE"]
    assert scored["cue_hit_count"] == 2
    assert scored["cue_recall"] == 2 / 3


def test_direct_canonical_pdd_is_a_diagnosis_leakage_alias():
    row = {
        "id": "direct_case",
        "canonical_pdd": "Unstable angina",
        "cue_targets": ["chest pain at rest"],
        "nla_output": (
            "<readout><observed>- Unstable angina</observed></readout>"
        ),
    }
    scored = score_row(row)
    assert scored["output_answer_hit"] is True
