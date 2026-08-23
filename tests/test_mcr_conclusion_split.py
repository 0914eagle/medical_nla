"""Splits for the MCR conclusion readout."""

from scripts.make_mcr_conclusion_split import split_of, target_text


def test_target_carries_the_conclusion_and_its_grounds():
    t = target_text("Osteoid osteoma", ["night pain", "lumbar tenderness"])
    assert "<answer>Osteoid osteoma</answer>" in t
    assert "<supporting_cues>night pain; lumbar tenderness</supporting_cues>" in t


def test_a_case_with_no_cues_still_produces_wellformed_xml():
    """A malformed target teaches the adapter to emit malformed XML, which is
    worse than dropping the row -- the caller drops it, and this only has to
    not corrupt the schema if one slips through."""
    t = target_text("Danon disease", [])
    assert "<supporting_cues></supporting_cues>" in t


def test_the_corpus_test_half_never_becomes_training_data():
    """The adapter trains, so unlike the behavioural runs it has to respect
    MCR's own split: train on train, read out on test."""
    for i in range(50):
        assert split_of(f"case_{i}", "test", 0.1, 17) == "test"


def test_validation_is_carved_out_of_train_and_is_stable():
    assignments = [split_of(f"case_{i}", "train", 0.2, 17) for i in range(300)]
    assert set(assignments) == {"train", "val"}
    assert 0.10 < assignments.count("val") / len(assignments) < 0.32
    # Same seed, same case, same side -- a case must not drift between builds.
    assert assignments == [split_of(f"case_{i}", "train", 0.2, 17) for i in range(300)]
