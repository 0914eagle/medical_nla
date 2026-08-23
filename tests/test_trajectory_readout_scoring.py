"""Lenient scoring, which exists so the untuned checkpoint can be compared."""

from scripts.analyze_trajectory_readouts import mentions_name, names_mentioned

VOCAB = ["Myocarditis", "Pneumonia", "Bronchitis", "Pulmonary embolism"]


def test_a_short_alias_does_not_match_inside_a_word():
    """"PE" lives inside "appears", and the untuned readout is hundreds of
    words long. Bare containment would score it a perfect reader of any
    vector at all."""
    prose = "This appears to be a perfectly ordinary specimen."
    assert not mentions_name(prose, "Pulmonary embolism", ["PE"])
    assert mentions_name("Findings suggest PE, likely acute.", "Pulmonary embolism", ["PE"])


def test_a_rambling_readout_is_caught_naming_everything():
    """Containment is generous on purpose; this counter is what keeps a
    channel from scoring well by listing half the differential."""
    focused = "<readout><answer>Myocarditis</answer></readout>"
    shotgun = (
        "The vector might represent Myocarditis, though Pneumonia and "
        "Bronchitis are also plausible readings of this activation."
    )
    assert names_mentioned(focused, VOCAB) == 1
    assert names_mentioned(shotgun, VOCAB) == 3


def test_a_readout_naming_nothing_scores_zero():
    meta = "This appears to be a medical forum question about token prediction."
    assert names_mentioned(meta, VOCAB) == 0
