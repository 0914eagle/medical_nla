from src.reconstruction_scoring import (
    confab_regex,
    high_norm_flag,
    infer_domain,
    summarize_scored_rows,
)


def test_infer_domain_from_filename():
    assert infer_domain("/tmp/pilot_medical_v3.jsonl") == "medical"
    assert infer_domain("/tmp/pilot_general_v3.jsonl") == "general"


def test_confab_regex_requires_missing_prompt_term():
    prompt = "A patient has cough."
    explanation = "This vector refers to pneumonia and fever."
    assert confab_regex(prompt, explanation)
    assert not confab_regex("The prompt mentions pneumonia.", explanation)


def test_high_norm_flag_threshold():
    assert high_norm_flag({"activation_norm": 12001})
    assert not high_norm_flag({"activation_norm": 12000})


def test_summary_excludes_high_norm_rows():
    rows = [
        {"domain": "medical", "confab_regex": True, "recon_mse": 0.3, "high_norm_flag": False},
        {"domain": "medical", "confab_regex": False, "recon_mse": 0.4, "high_norm_flag": False},
        {"domain": "medical", "confab_regex": True, "recon_mse": 3.0, "high_norm_flag": True},
    ]
    summary = summarize_scored_rows(rows)
    assert "High-norm rows excluded from summary statistics: 1" in summary
    assert "| medical | True | 1 | 0.3000 |" in summary
