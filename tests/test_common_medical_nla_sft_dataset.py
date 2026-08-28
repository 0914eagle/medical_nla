from pathlib import Path

from scripts.make_common_medical_nla_sft_dataset import normalize_row, stratified_sample


def make_row(identifier: str, diagnosis: str, *, source: str = "ddxplus") -> dict:
    return {
        "id": identifier,
        "base_id": identifier,
        "variant": "original" if source == "ddxplus" else None,
        "position_family": "P0",
        "layer": 32,
        "activation_path": f"/{identifier}.pt",
        "diagnosis_id": diagnosis,
        "cue_targets": ["fever", "dry cough"],
        "cue_value_ids": [None, None],
    }


def test_common_target_omits_diagnosis_and_answer() -> None:
    row = make_row("case1", "pneumonia")
    normalized = normalize_row(
        row,
        source_dataset="ddxplus",
        split="train",
        max_cues=12,
        seed=17,
        require_activation_file=False,
    )
    assert normalized is not None
    assert "- fever" in normalized["target_text"]
    assert "pneumonia" not in normalized["target_text"].casefold()
    assert "<answer>" not in normalized["target_text"]


def test_common_target_can_preserve_source_cue_order() -> None:
    row = make_row("case1", "pneumonia")
    row["cue_targets"] = ["third", "first", "second", "FIRST"]
    normalized = normalize_row(
        row,
        source_dataset="ddxplus",
        split="train",
        max_cues=12,
        seed=17,
        require_activation_file=False,
        cue_order="source",
    )
    assert normalized is not None
    text = normalized["target_text"]
    assert text.index("- third") < text.index("- first") < text.index("- second")
    assert text.casefold().count("- first") == 1
    assert normalized["target_style"].endswith("source_order")


def test_ddxplus_counterfactual_rows_are_not_training_rows() -> None:
    row = make_row("case1", "pneumonia")
    row["variant"] = "cue_deleted"
    assert (
        normalize_row(
            row,
            source_dataset="ddxplus",
            split="val",
            max_cues=12,
            seed=17,
            require_activation_file=False,
        )
        is None
    )


def test_stratified_sample_covers_each_diagnosis_before_repeating() -> None:
    rows = [make_row(f"a{i}", "a") for i in range(5)]
    rows += [make_row(f"b{i}", "b") for i in range(5)]
    rows += [make_row(f"c{i}", "c") for i in range(5)]
    selected = stratified_sample(
        rows, cap=3, seed=17, source_dataset="ddxplus", split="train"
    )
    assert {row["diagnosis_id"] for row in selected} == {"a", "b", "c"}


def test_normalizer_can_require_activation_file(tmp_path: Path) -> None:
    activation = tmp_path / "a.pt"
    activation.write_bytes(b"tensor")
    row = make_row("case1", "pneumonia")
    row["activation_path"] = str(activation)
    assert normalize_row(
        row,
        source_dataset="ddxplus",
        split="train",
        max_cues=12,
        seed=17,
        require_activation_file=True,
    )
