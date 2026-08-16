import random

import pytest

from scripts.make_ddxplus_cue_position_rows import cue_rows_for_case, cue_spans_in_prompt
from scripts.make_medical_nla_v4_cue_position_splits import (
    assign_split,
    split_cases,
    split_cue_strings,
)


def make_case(case_id: str, cues: list[str]) -> dict:
    joined = ", ".join(cues[:-1]) + f", and {cues[-1]}" if len(cues) > 2 else " and ".join(cues)
    return {
        "id": f"{case_id}__cues_all",
        "base_id": case_id,
        "variant": "cue_count_all",
        "diagnosis_id": "urti",
        "prompt": f"A patient presents with {joined}. What diagnosis is most likely?",
        "cue_targets": cues,
    }


def test_cue_spans_handle_substring_and_repeated_cues():
    prompt = "A patient presents with a productive cough, a cough, and a cough. Diagnosis?"
    spans = cue_spans_in_prompt(prompt, ["a productive cough", "a cough", "a cough"])
    assert all(span is not None for span in spans)
    starts = [span[0] for span in spans]
    assert starts[0] < starts[1] < starts[2]
    # The bare "a cough" spans must not sit inside "a productive cough".
    assert prompt[spans[1][0] : spans[1][1]] == "a cough"
    assert spans[1][0] > spans[0][1] - len("a cough")


def test_cue_rows_for_case_fields_and_occurrence():
    case = make_case("case1", ["fever", "a cough", "sore throat"])
    rows, skipped = cue_rows_for_case(
        case, max_cues_per_case=None, strategy="last_subtoken", rng=random.Random(0)
    )
    assert skipped == 0
    assert [row["cue_index"] for row in rows] == [0, 1, 2]
    for row in rows:
        assert row["position_mode"] == "target_text"
        assert row["target_text"] == row["cue_text"]
        assert row["cue_targets"] == [row["cue_text"]]
        assert row["target_text_occurrence"] == 0
        assert row["id"].startswith("case1__cuepos")


def test_cue_rows_respects_max_cues_and_missing_cue():
    case = make_case("case2", ["fever", "a cough", "sore throat", "headache"])
    case["cue_targets"] = case["cue_targets"] + ["not in prompt"]
    rows, skipped = cue_rows_for_case(
        case, max_cues_per_case=2, strategy="last_subtoken", rng=random.Random(0)
    )
    assert skipped == 1
    assert len(rows) == 2


def test_split_cue_strings_disjoint_and_deterministic():
    cues = [f"cue {i}" for i in range(20)]
    train_a, held_a = split_cue_strings(cues, seed=17, heldout_frac=0.25)
    train_b, held_b = split_cue_strings(cues, seed=17, heldout_frac=0.25)
    assert (train_a, held_a) == (train_b, held_b)
    assert not train_a & held_a
    assert len(held_a) == 5


def test_assign_split_drops_heldout_cues_in_train_cases():
    heldout = {"fever"}
    row_heldout = {"cue_text": "Fever"}
    row_seen = {"cue_text": "a cough"}
    assert assign_split(row_heldout, "train", heldout) is None
    assert assign_split(row_heldout, "val", heldout) is None
    assert assign_split(row_heldout, "test", heldout) == "test_heldout_cue"
    assert assign_split(row_seen, "train", heldout) == "train"
    assert assign_split(row_seen, "test", heldout) == "test_seen_cue"


def test_split_cases_disjoint_pools():
    pools = split_cases([f"case{i}" for i in range(30)], seed=17, train_frac=0.7, val_frac=0.1)
    counts = {pool: list(pools.values()).count(pool) for pool in ("train", "val", "test")}
    assert counts["train"] == 21 and counts["val"] == 3 and counts["test"] == 6
    with pytest.raises(ValueError):
        split_cases(["a", "b"], seed=17, train_frac=0.9, val_frac=0.2)
