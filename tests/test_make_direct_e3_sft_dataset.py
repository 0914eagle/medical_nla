from scripts.make_direct_e3_sft_dataset import (
    build_split,
    grounded_observations,
    target_text,
)


def test_grounded_observations_excludes_nonverbatim_and_deduplicates() -> None:
    row = {
        "id": "case",
        "gold_deductions": [
            {"observation": "Chest pain at rest", "observation_exact_in_note": True},
            {"observation": "chest pain at rest", "observation_exact_in_note": True},
            {"observation": "Inferred ischemia", "observation_exact_in_note": False},
        ],
    }
    assert grounded_observations(row, max_observations=12, seed=17) == [
        "Chest pain at rest"
    ]


def test_target_uses_source_answer_without_claiming_it_is_gold() -> None:
    result = target_text(["Chest pain at rest"], "Unstable angina")
    assert "- Chest pain at rest" in result
    assert "<answer>Unstable angina</answer>" in result
    assert "gold" not in result.casefold()


def test_primary_builder_excludes_gold_label_in_note(tmp_path) -> None:
    activation = tmp_path / "activation.pt"
    activation.write_bytes(b"tensor placeholder")
    clinical = {
        "id": "case",
        "gold_label_exact_in_note": True,
        "gold_deductions": [
            {"observation": "Finding", "observation_exact_in_note": True}
        ],
    }
    activation_row = {
        "id": "case__p0",
        "base_id": "case",
        "position_family": "P0",
        "layer": 32,
        "activation_path": str(activation),
    }
    rows, counts = build_split(
        split="train",
        split_rows=[clinical],
        activation_rows=[activation_row],
        source_answers={"case": {"answer": "Diagnosis"}},
        max_observations=12,
        seed=17,
        include_gold_label_in_note=False,
    )
    assert rows == []
    assert counts["gold_label_exact_in_note"] == 1
