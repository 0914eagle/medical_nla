"""The judge-read comparison between a tuned and an untuned readout."""

from scripts.analyze_readout_extraction import verdict_of
from scripts.make_readout_extraction_cases import extraction_prompt


def test_the_prompt_hides_everything_that_could_grade_toward_an_answer():
    """The judge must not know the gold, the referral's suspicion, or which
    system wrote the text -- otherwise it can score the channel it thinks it
    is looking at."""
    rambling = (
        "This appears to be a medical forum question. The phrase 'chest pain' "
        "suggests the model is considering Myocarditis."
    )
    prompt = extraction_prompt(rambling)
    assert rambling in prompt
    assert "MULTIPLE" in prompt and "NONE" in prompt
    for leak in ("gold", "correct answer", "referral", "adapter", "LoRA", "vanilla"):
        assert leak.lower() not in prompt.lower()


def test_verdict_comes_from_the_last_line():
    """Judges preface. The name is what they end on."""
    assert verdict_of("Myocarditis") == "Myocarditis"
    assert verdict_of("Based on the text, the condition is:\nPneumonia") == "Pneumonia"
    assert verdict_of("**Bronchitis**") == "Bronchitis"
    assert verdict_of("NONE") == "NONE"
    assert verdict_of("") == ""
