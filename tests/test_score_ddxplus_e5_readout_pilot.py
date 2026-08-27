from scripts.score_ddxplus_e5_readout_pilot import score_method


def row(base, variant, output, cues, **extra):
    return {
        "id": f"{base}__{variant}",
        "base_id": base,
        "variant": variant,
        "nla_output": output,
        "parsed_explanation_tag": True,
        "cue_targets": cues,
        **extra,
    }


def test_paired_counterfactual_metrics():
    rows = [
        row("a", "original", "chest pain and fever", ["chest pain", "fever"]),
        row(
            "a",
            "cue_deleted",
            "fever",
            ["fever"],
            cf_original_cue="chest pain",
        ),
        row(
            "a",
            "value_edited",
            "back pain and fever",
            ["back pain", "fever"],
            cf_original_cue="chest pain",
            cf_replacement_cue="back pain",
        ),
        row("b", "original", "cough and nausea", ["cough", "nausea"]),
        row(
            "b",
            "cue_deleted",
            "cough and nausea",
            ["nausea"],
            cf_original_cue="cough",
        ),
    ]
    result = score_method(rows, threshold=0.5)
    assert result["n"] == 5
    assert result["base_cases"] == 2
    assert result["deletion"]["pairs"] == 2
    assert result["deletion"]["deleted_target_phantom_rate"] == 0.5
    assert result["deletion"]["removal_success_given_original_hit"] == 0.5
    assert result["value_edit"]["pairs"] == 1
    assert result["value_edit"]["replacement_hit_rate"] == 1.0
    assert result["value_edit"]["original_persistence_rate"] == 0.0
    assert result["value_edit"]["clean_switch_rate"] == 1.0
    assert result["value_edit"]["untouched"]["preservation_given_original_hit"] == 1.0
