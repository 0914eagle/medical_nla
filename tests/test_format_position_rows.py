from scripts.make_format_position_rows import format_row


def test_format_row_sets_last_token_and_carries_fields():
    src = {
        "id": "case1__cues_all",
        "base_id": "case1",
        "variant": "cue_count_all",
        "prompt": "A patient presents with fever. What diagnosis is most likely?",
        "diagnosis_id": "pneumonia",
        "diagnosis_name": "Pneumonia",
        "cue_targets": ["fever", "cough"],
        "activation_path": "/stale/L32.pt",
        "position": 40,
    }
    out = format_row(src, variant="cue_count_all")
    assert out["position_mode"] == "last_token"
    assert out["target_role"] == "format"
    assert out["target_text"] is None
    assert out["prompt"] == src["prompt"]
    assert out["cue_targets"] == ["fever", "cough"]
    assert out["diagnosis_id"] == "pneumonia"
    # stale extraction outputs from the source manifest are dropped
    assert "activation_path" not in out
    assert "position" not in out
    # missing optional fields are simply omitted, not set to None
    assert "patient_id" not in out
