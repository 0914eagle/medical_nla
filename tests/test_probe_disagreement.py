"""The probe baseline's pure logic, testable without torch or tensors."""

from scripts.evaluate_probe_disagreement import (
    class_of_answer,
    final_activation_paths,
    fold_of,
)
from src.jsonl import write_jsonl


def test_fold_assignment_is_deterministic_and_binary():
    ids = [f"case_{i}" for i in range(50)]
    folds = [fold_of(i) for i in ids]
    assert set(folds) <= {0, 1}
    assert folds == [fold_of(i) for i in ids]
    # Both folds must actually occur, or cross-fitting silently trains on all.
    assert len(set(folds)) == 2


def test_class_matching_uses_containment_not_equality():
    classes = ["Anemia", "Pulmonary embolism"]
    assert class_of_answer("Anemia of chronic disease", classes) == 0
    assert class_of_answer("likely pulmonary embolism", classes) == 1
    assert class_of_answer("Bronchitis", classes) is None


def test_manifest_join_keeps_final_rows_only(tmp_path):
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "base_id": "c1",
                "hint_variant": "wrong",
                "target_role": "final",
                "activation_path": "/x/c1.pt",
            },
            {
                "base_id": "c1",
                "hint_variant": "wrong",
                "target_role": "hint",
                "activation_path": "/x/c1_hint.pt",
            },
        ],
    )
    paths = final_activation_paths([str(manifest)])
    assert paths == {("c1", "wrong"): "/x/c1.pt"}
