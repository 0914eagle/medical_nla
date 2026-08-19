from scripts.make_prose_cue_position_splits import (
    cue_of,
    norm_cue,
    normalize_manifest_row,
    split_cases,
)


def test_norm_cue_folds_case_and_spacing():
    assert norm_cue("  He was   OTHERWISE healthy ") == "he was otherwise healthy"


def test_cue_of_prefers_cue_text_then_target_then_targets():
    assert cue_of({"cue_text": "a", "target_text": "b"}) == "a"
    assert cue_of({"target_text": "b", "cue_targets": ["c"]}) == "b"
    assert cue_of({"cue_targets": ["c"]}) == "c"
    assert cue_of({"cue_targets": []}) is None


def test_normalize_manifest_row_requires_activation_and_cue():
    assert normalize_manifest_row({"cue_text": "fever and chills"}) is None
    assert normalize_manifest_row({"activation_path": "/a.pt"}) is None
    row = normalize_manifest_row({"activation_path": "/a.pt", "cue_targets": ["fever and chills"]})
    assert row["cue_text"] == "fever and chills"


def test_split_cases_partitions_every_case_exactly_once():
    cases = [f"case{i}" for i in range(100)]
    pools = split_cases(cases, seed=17, train_frac=0.7, val_frac=0.1)
    assert set(pools) == set(cases)
    counts = {pool: sum(1 for v in pools.values() if v == pool) for pool in ("train", "val", "test")}
    assert counts["train"] == 70
    assert counts["val"] == 10
    assert counts["test"] == 20


def test_split_cases_is_deterministic_for_a_seed():
    cases = [f"case{i}" for i in range(50)]
    assert split_cases(cases, seed=3, train_frac=0.7, val_frac=0.1) == split_cases(
        cases, seed=3, train_frac=0.7, val_frac=0.1
    )


def test_split_cases_keeps_a_case_whole():
    # The same case id appearing in many rows must land in one pool only.
    cases = ["caseA"] * 5 + ["caseB"] * 5
    pools = split_cases(cases, seed=1, train_frac=0.5, val_frac=0.0)
    assert set(pools) == {"caseA", "caseB"}
    assert len(set(pools.values())) <= 2
