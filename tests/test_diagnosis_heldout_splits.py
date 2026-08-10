import pytest

from scripts.make_medical_nla_diagnosis_heldout_splits import (
    assign_splits,
    eligible_diagnoses,
    split_diagnosis_classes,
)


def make_row(diagnosis_id: str, case: int, *, hit: bool) -> dict:
    base_id = f"{diagnosis_id}_case{case}"
    return {
        "id": f"{base_id}::cue_count_all",
        "base_id": base_id,
        "diagnosis_id": diagnosis_id,
        "source_diagnosis_hit": hit,
    }


def make_rows() -> list[dict]:
    rows = []
    for diagnosis_id in ("dx_a", "dx_b", "dx_c", "dx_d"):
        for case in range(10):
            rows.append(make_row(diagnosis_id, case, hit=case < 6))
    return rows


def test_eligible_diagnoses_threshold():
    rows = make_rows() + [make_row("dx_rare", 0, hit=True)]
    assert eligible_diagnoses(rows, min_source_correct_per_diagnosis=6) == [
        "dx_a",
        "dx_b",
        "dx_c",
        "dx_d",
    ]
    assert "dx_rare" in eligible_diagnoses(rows, min_source_correct_per_diagnosis=1)


def test_split_diagnosis_classes_disjoint_and_deterministic():
    pool = [f"dx_{i}" for i in range(10)]
    train_a, heldout_a = split_diagnosis_classes(
        pool, seed=17, heldout_frac=0.3, num_heldout=None, heldout_diagnoses=None
    )
    train_b, heldout_b = split_diagnosis_classes(
        pool, seed=17, heldout_frac=0.3, num_heldout=None, heldout_diagnoses=None
    )
    assert (train_a, heldout_a) == (train_b, heldout_b)
    assert not set(train_a) & set(heldout_a)
    assert sorted(train_a + heldout_a) == sorted(pool)
    assert len(heldout_a) == 3


def test_split_diagnosis_classes_explicit_and_errors():
    pool = ["dx_a", "dx_b", "dx_c"]
    train, heldout = split_diagnosis_classes(
        pool, seed=17, heldout_frac=0.3, num_heldout=None, heldout_diagnoses=["dx_b"]
    )
    assert heldout == ["dx_b"]
    assert train == ["dx_a", "dx_c"]
    with pytest.raises(ValueError):
        split_diagnosis_classes(
            pool, seed=17, heldout_frac=0.3, num_heldout=None, heldout_diagnoses=["dx_z"]
        )
    with pytest.raises(ValueError):
        split_diagnosis_classes(
            pool, seed=17, heldout_frac=0.0, num_heldout=None, heldout_diagnoses=None
        )
    with pytest.raises(ValueError):
        split_diagnosis_classes(
            pool, seed=17, heldout_frac=1.0, num_heldout=None, heldout_diagnoses=None
        )


def test_assign_splits_pools_and_source_correct_rule():
    rows = make_rows()
    split_map = assign_splits(
        rows,
        train_diagnoses=["dx_a", "dx_b", "dx_c"],
        heldout_diagnoses=["dx_d"],
        seed=17,
        train_frac=0.6,
        val_frac=0.2,
        train_source_correct_only=True,
    )
    by_split: dict[str, set[str]] = {}
    for row in rows:
        split = split_map[row["base_id"]]
        by_split.setdefault(split, set()).add(row["base_id"])

    assert {row["base_id"] for row in rows if row["diagnosis_id"] == "dx_d"} == by_split[
        "test_heldout"
    ]
    hit_by_base = {row["base_id"]: row["source_diagnosis_hit"] for row in rows}
    for split in ("train", "val"):
        assert all(hit_by_base[base_id] for base_id in by_split.get(split, set()))
    seen_bases = by_split.get("test_seen", set())
    assert any(not hit_by_base[base_id] for base_id in seen_bases)
    assert any(hit_by_base[base_id] for base_id in seen_bases), (
        "test_seen must keep some source-correct rows for a fair seen/heldout comparison"
    )
    train_class_bases = {row["base_id"] for row in rows if row["diagnosis_id"] != "dx_d"}
    assert train_class_bases == (
        by_split.get("train", set()) | by_split.get("val", set()) | seen_bases
    )
    assert by_split.get("val"), "val must not be empty with val_frac=0.2"


def test_assign_splits_rejects_full_fit_pool_consumption():
    with pytest.raises(ValueError, match="test_seen"):
        assign_splits(
            make_rows(),
            train_diagnoses=["dx_a", "dx_b", "dx_c"],
            heldout_diagnoses=["dx_d"],
            seed=17,
            train_frac=0.85,
            val_frac=0.15,
            train_source_correct_only=True,
        )


def test_assign_splits_leakage_guard():
    rows = [
        make_row("dx_a", 0, hit=True),
        make_row("dx_a", 1, hit=True),
    ]
    leaked = dict(rows[1])
    leaked["diagnosis_id"] = "dx_d"
    with pytest.raises(ValueError, match="Leakage"):
        assign_splits(
            rows + [leaked],
            train_diagnoses=["dx_a"],
            heldout_diagnoses=["dx_d"],
            seed=17,
            train_frac=0.5,
            val_frac=0.2,
            train_source_correct_only=True,
        )
    with pytest.raises(ValueError, match="overlap"):
        assign_splits(
            rows,
            train_diagnoses=["dx_a"],
            heldout_diagnoses=["dx_a"],
            seed=17,
            train_frac=0.5,
            val_frac=0.2,
            train_source_correct_only=True,
        )
