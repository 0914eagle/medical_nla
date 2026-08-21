from pathlib import Path

TEMPLATE = Path("prompt_templates/cue_position_readout.txt")


def test_the_template_can_carry_the_vector():
    """Without the placeholder the activation is never injected and the run
    silently reads out nothing."""
    assert "{injection_char}" in TEMPLATE.read_text(encoding="utf-8")


def test_the_template_asks_for_the_schema_the_target_actually_uses():
    """The pilot's prompt demanded <answer>/<supporting_cues> while the
    supervised target contained <observed> findings; the adapter then learned a
    format nothing had asked for."""
    text = TEMPLATE.read_text(encoding="utf-8")
    for tag in ("<explanation>", "<readout>", "<observed>"):
        assert tag in text, tag
    for absent in ("<answer>", "<supporting_cues>", "<task_type>"):
        assert absent not in text, absent


def test_the_template_does_not_ask_for_a_diagnosis():
    """Cue-position targets contain no diagnosis, so asking for one points the
    readout away from what it is scored against."""
    assert "Do not name a diagnosis" in TEMPLATE.read_text(encoding="utf-8")
