import random

from scripts.evaluate_cue_counterfactuals import evaluate
from scripts.make_cue_counterfactual_rows import counterfactual_rows_for_case, make_prompt


def make_case(cues: list[str]) -> dict:
    return {
        "id": "case1__cues_all",
        "base_id": "case1",
        "diagnosis_id": "urti",
        "cue_targets": cues,
        "prompt": make_prompt(cues),
    }


CUES = ["a moderate fever", "a sore throat", "nasal congestion", "a dry cough"]
VOCAB = CUES + ["swelling of the ankle", "black stools", "intense itching"]


def test_counterfactual_rows_structure():
    case = make_case(CUES)
    rows = counterfactual_rows_for_case(case, vocab=VOCAB, rng=random.Random(3), strategy="last_subtoken")
    assert rows is not None
    # 2 swapped-slot rows (orig+swap) + 2 retained slots x 3 variants = 8 rows
    assert len(rows) == 8
    assert len({row["id"] for row in rows}) == 8

    swap_rows = [r for r in rows if r["cf_variant"] == "swap"]
    orig_rows = [r for r in rows if r["cf_variant"] == "orig"]
    removed_rows = [r for r in rows if r["cf_variant"] == "removed"]
    assert len(orig_rows) == 3 and len(swap_rows) == 3 and len(removed_rows) == 2

    slot_row = next(r for r in swap_rows if r["cf_role"] == "swapped_slot")
    original, replacement = slot_row["cf_original_cue"], slot_row["cf_replacement_cue"]
    assert replacement not in CUES
    assert replacement in slot_row["prompt"] and original not in slot_row["prompt"]
    assert slot_row["target_text"] == replacement

    for r in removed_rows:
        assert original not in r["prompt"]
        assert r["cue_text"] in r["prompt"]
    for r in rows:
        assert r["cue_targets"] == [r["cue_text"]]
        assert r["target_text"] == r["cue_text"]
        assert r["position_mode"] == "target_text"


def test_counterfactual_skips_nonstandard_prompt_and_small_cases():
    bad = make_case(CUES)
    bad["prompt"] = "A different template entirely."
    assert counterfactual_rows_for_case(bad, vocab=VOCAB, rng=random.Random(0), strategy="last_subtoken") is None
    small = make_case(CUES[:2])
    assert counterfactual_rows_for_case(small, vocab=VOCAB, rng=random.Random(0), strategy="last_subtoken") is None


def scored_row(
    variant, role, gold, emitted, *, base="c1", slot=0,
    original="black stools", replacement="intense itching",
):
    return {
        "base_id": base,
        "cf_variant": variant,
        "cf_role": role,
        "cf_slot": slot,
        "cue_text": gold,
        "cue_targets": [gold],
        "cf_original_cue": original,
        "cf_replacement_cue": replacement,
        "cf_removed_cue": original,
        "observed_readout": f"- {emitted}",
    }


def test_evaluate_counterfactuals_math():
    rows = [
        # Swapped slot: orig reads original; swap tracks replacement (faithful case).
        scored_row("orig", "swapped_slot", "black stools", "black stools"),
        scored_row("swap", "swapped_slot", "intense itching", "intense itching"),
        # Second case: swap FAILS to track and keeps reading the original.
        scored_row("orig", "swapped_slot", "black stools", "black stools", base="c2"),
        scored_row("swap", "swapped_slot", "intense itching", "black stools", base="c2"),
        # Retained slot: stable under orig+swap, lost under removal, phantom appears.
        scored_row("orig", "retained", "sore throat", "sore throat", slot=1),
        scored_row("swap", "retained", "sore throat", "sore throat", slot=1),
        scored_row("removed", "retained", "sore throat", "black stools", slot=1),
    ]
    result = evaluate(rows, threshold=0.5)
    assert result["orig_reads_original"] == 1.0
    assert result["swap_reads_replacement"] == 0.5
    assert result["swap_still_reads_original"] == 0.5
    assert result["retained_read_rate_orig"] == 1.0
    assert result["retained_read_rate_swap"] == 1.0
    assert result["retained_read_rate_removed"] == 0.0
    assert result["retained_degraded_under_removal"] == 1.0
    assert result["retained_degraded_under_swap"] == 0.0
    assert result["phantom_rate_removed_cue"] == 1.0


def corpus_case(cues: list[str], age=41, sex="M") -> dict:
    """A case built the way the corpus builder builds one.

    The tests above make their case with this script's own make_prompt, so they
    pass whatever format that is -- which is how the script came to hold an
    inline "A patient presents with X, Y and Z" sentence long after the corpus
    moved to a findings list. Every real case then failed the round-trip check
    and the script produced no rows at all, silently, because producing none is
    what the check is supposed to do when a prompt cannot be rebuilt.
    """
    from scripts.make_ddxplus_cue_count_cases import make_prompt as corpus_make_prompt

    return {
        "id": "case1__cue_count_all",
        "base_id": "case1",
        "diagnosis_id": "urti",
        "cue_targets": cues,
        "age": age,
        "sex": sex,
        "prompt": corpus_make_prompt(cues, age=age, sex=sex),
        "prompt_cot": corpus_make_prompt(cues, condition="cot", age=age, sex=sex),
    }


def test_a_case_from_the_corpus_builder_round_trips():
    case = corpus_case(CUES)
    rows = counterfactual_rows_for_case(
        case, vocab=VOCAB, rng=random.Random(3), strategy="last_subtoken"
    )
    assert rows is not None and len(rows) == 8


def test_age_and_sex_are_carried_into_the_rebuilt_prompt():
    """They head the presentation, so dropping them rebuilds a different case."""
    case = corpus_case(CUES, age=7, sex="F")
    rows = counterfactual_rows_for_case(
        case, vocab=VOCAB, rng=random.Random(3), strategy="last_subtoken"
    )
    assert rows is not None
    orig = next(r for r in rows if r["cf_variant"] == "orig")
    assert orig["prompt"] == case["prompt"]


def test_every_row_carries_a_chain_of_thought_prompt():
    """run_source_answers reads prompt_cot in the cot condition, and hypothesis
    1 is a comparison against that arm."""
    from src.case_prompts import COT_INSTRUCTION, DIRECT_INSTRUCTION

    rows = counterfactual_rows_for_case(
        corpus_case(CUES), vocab=VOCAB, rng=random.Random(3), strategy="last_subtoken"
    )
    for row in rows:
        assert row["prompt_cot"], row["id"]
        assert row["prompt_cot"].endswith(COT_INSTRUCTION)
        assert row["prompt"].endswith(DIRECT_INSTRUCTION)
        # The two differ only after the presentation.
        assert row["prompt"].split(DIRECT_INSTRUCTION)[0] == (
            row["prompt_cot"].split(COT_INSTRUCTION)[0]
        )
