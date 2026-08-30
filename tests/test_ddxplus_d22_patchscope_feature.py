from __future__ import annotations

from scripts.calibrate_ddxplus_d22_patchscope_feature import (
    CLINICAL_PROMPTS,
    choose_control_cell,
    keyword_hit,
    summarize_control_rows,
)


def control_row(
    family: str,
    layer: int,
    hit: bool,
    no_patch_hit: bool = False,
    changed: bool = True,
    source_layer: int = 32,
) -> dict[str, object]:
    return {
        "family": family,
        "source_layer": source_layer,
        "target_layer": layer,
        "keyword_hit": hit,
        "no_patch_keyword_hit": no_patch_hit,
        "differs_from_no_patch": changed,
        "first_token_kl_to_no_patch": 1.0,
    }


def test_keyword_hit_is_case_insensitive() -> None:
    assert keyword_hit("Paris is the capital", ("PARIS",))
    assert not keyword_hit("Tokyo", ("Paris",))
    assert not keyword_hit("This is a string", ("ring",))


def test_choose_control_cell_uses_only_eligible_cells() -> None:
    rows = []
    rows.extend(
        control_row("entity_description", 16, index < 2)
        for index in range(5)
    )
    rows.extend(
        control_row("relation_specific", 24, index < 4)
        for index in range(5)
    )
    selected = choose_control_cell(summarize_control_rows(rows))
    assert selected is not None
    assert selected["family"] == "relation_specific"
    assert selected["target_layer"] == 24


def test_choose_control_cell_prefers_hs32_then_relation_on_ties() -> None:
    rows = []
    for family in ("entity_description", "relation_specific"):
        for layer in (16, 32):
            rows.extend(control_row(family, layer, True) for _ in range(5))
    selected = choose_control_cell(summarize_control_rows(rows))
    assert selected is not None
    assert selected["target_layer"] == 32
    assert selected["family"] == "relation_specific"


def test_summary_keeps_source_layers_separate() -> None:
    rows = [
        control_row("entity_description", 24, True, source_layer=16),
        control_row("entity_description", 24, False, source_layer=24),
    ]
    summaries = summarize_control_rows(rows)
    assert len(summaries) == 2
    assert {(row["source_layer"], row["keyword_hits"]) for row in summaries} == {
        (16, 1),
        (24, 0),
    }


def test_clinical_prompts_end_at_final_marker() -> None:
    assert set(CLINICAL_PROMPTS) == {"entity_description", "relation_specific"}
    assert all(prompt.endswith("foo") for prompt in CLINICAL_PROMPTS.values())
