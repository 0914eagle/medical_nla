import random

from scripts.make_span_counterfactual_rows import (
    counterfactual_rows_for_case,
    cue_spans,
    remove_span,
    spans_overlap,
    substitute,
)

CASE = {
    "id": "mcr_case_0000001__cues_all",
    "base_id": "mcr_case_0000001",
    "source": "medcasereasoning",
    "diagnosis_id": "unstable_angina",
    "prompt": (
        "A 62-year-old man reports persistent chest pain at rest, "
        "shortness of breath on exertion, and swelling of both ankles."
    ),
    "cue_targets": [
        "persistent chest pain at rest",
        "shortness of breath on exertion",
        "swelling of both ankles",
    ],
    "variant": "cue_count_all",
}


def test_cue_spans_resolve_in_order():
    spans = cue_spans(CASE["prompt"], CASE["cue_targets"])
    assert all(span is not None for span in spans)
    for cue, span in zip(CASE["cue_targets"], spans):
        assert CASE["prompt"][span[0] : span[1]] == cue


def test_spans_overlap_detects_nesting():
    assert not spans_overlap([(0, 5), (6, 10)])
    assert spans_overlap([(0, 8), (5, 12)])


def test_substitute_changes_only_the_target_span():
    prompt = CASE["prompt"]
    span = cue_spans(prompt, CASE["cue_targets"])[1]
    out = substitute(prompt, span, "severe nausea after meals")
    assert "shortness of breath on exertion" not in out
    assert "severe nausea after meals" in out
    # everything outside the span is byte-identical
    assert out[: span[0]] == prompt[: span[0]]
    assert out[span[0] + len("severe nausea after meals") :] == prompt[span[1] :]


def test_remove_span_tidies_leftover_punctuation():
    prompt = "A patient with fever , cough, and rash."
    out = remove_span(prompt, (16, 22))
    assert "  " not in out
    assert " ," not in out


def test_counterfactual_rows_cover_variants_and_roles():
    rows = counterfactual_rows_for_case(
        CASE,
        vocab=["a new unrelated finding", "another unrelated finding"],
        rng=random.Random(0),
        strategy="last_subtoken",
        swap_slots=1,
    )
    assert rows is not None
    variants = {row["cf_variant"] for row in rows}
    assert variants == {"orig", "swap", "removed"}
    roles = {row["cf_role"] for row in rows}
    assert roles == {"swapped_slot", "retained"}
    # one orig/swap pair on the swapped slot
    swapped = [row for row in rows if row["cf_role"] == "swapped_slot"]
    assert len(swapped) == 2
    assert {row["cf_variant"] for row in swapped} == {"orig", "swap"}


def test_swap_row_targets_replacement_and_orig_targets_original():
    rows = counterfactual_rows_for_case(
        CASE,
        vocab=["a new unrelated finding"],
        rng=random.Random(0),
        strategy="last_subtoken",
        swap_slots=1,
    )
    swapped = {row["cf_variant"]: row for row in rows if row["cf_role"] == "swapped_slot"}
    assert swapped["swap"]["target_text"] == "a new unrelated finding"
    assert swapped["orig"]["target_text"] == swapped["orig"]["cf_original_cue"]
    # the swap prompt no longer contains the original cue
    assert swapped["orig"]["cf_original_cue"] not in swapped["swap"]["prompt"]


def test_every_row_target_is_locatable_in_its_own_prompt():
    rows = counterfactual_rows_for_case(
        CASE,
        vocab=["a new unrelated finding"],
        rng=random.Random(1),
        strategy="last_subtoken",
        swap_slots=1,
    )
    for row in rows:
        assert row["target_text"] in row["prompt"]


def test_removed_variant_drops_only_the_removed_cue():
    rows = counterfactual_rows_for_case(
        CASE,
        vocab=["a new unrelated finding"],
        rng=random.Random(0),
        strategy="last_subtoken",
        swap_slots=1,
    )
    removed = [row for row in rows if row["cf_variant"] == "removed"]
    assert removed
    for row in removed:
        assert row["cf_removed_cue"] not in row["prompt"]
        # the retained cue this row reads is still present
        assert row["cue_text"] in row["prompt"]


def test_case_with_too_few_cues_is_skipped():
    case = dict(CASE, cue_targets=["persistent chest pain at rest", "swelling of both ankles"])
    assert (
        counterfactual_rows_for_case(
            case,
            vocab=["x y z finding"],
            rng=random.Random(0),
            strategy="last_subtoken",
            swap_slots=1,
        )
        is None
    )


def test_case_with_unlocatable_cue_is_skipped():
    case = dict(CASE, cue_targets=CASE["cue_targets"] + ["a cue absent from the prompt"])
    assert (
        counterfactual_rows_for_case(
            case,
            vocab=["x y z finding"],
            rng=random.Random(0),
            strategy="last_subtoken",
            swap_slots=1,
        )
        is None
    )
