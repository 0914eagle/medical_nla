from scripts.train_medical_nla_contrastive import (
    balanced_pairs,
    build_disjoint_pairs,
    crossed_rows,
    symmetric_pair_objective,
)

import torch


def row(identifier: str, source: str, group: str, target: str) -> dict:
    result = {
        "id": identifier,
        "base_id": identifier,
        "source_dataset": source,
        "target_text": target,
        "activation_path": f"/{identifier}.pt",
    }
    if source == "ddxplus":
        result["diagnosis_id"] = group
    else:
        result["disease_category"] = group
    return result


def test_disjoint_pairs_stay_within_source_and_stratum() -> None:
    rows = [
        row("d1", "direct", "heart", "one"),
        row("d2", "direct", "heart", "two"),
        row("x1", "ddxplus", "dx", "three"),
        row("x2", "ddxplus", "dx", "four"),
    ]
    pairs = build_disjoint_pairs(rows, seed=17, epoch=1)
    assert set(pairs) == {"direct", "ddxplus"}
    for source_name, items in pairs.items():
        assert len(items) == 1
        assert all(item["source_dataset"] == source_name for pair in items for item in pair)


def test_balanced_pairs_caps_each_source_equally() -> None:
    rows = []
    for source in ("direct", "ddxplus"):
        for index in range(6):
            rows.append(row(f"{source}-{index}", source, "same", f"target-{index}"))
    pairs = balanced_pairs(rows, seed=17, epoch=1, max_pairs_per_source=2)
    assert len(pairs) == 4
    assert sum(pair[0]["source_dataset"] == "direct" for pair in pairs) == 2
    assert sum(pair[0]["source_dataset"] == "ddxplus" for pair in pairs) == 2


def test_crossed_rows_swap_targets_not_activations() -> None:
    first = row("a", "direct", "heart", "one")
    second = row("b", "direct", "heart", "two")
    own_a, own_b, cross_ab, cross_ba = crossed_rows(first, second)
    assert own_a is first and own_b is second
    assert cross_ab["activation_path"] == first["activation_path"]
    assert cross_ab["target_text"] == second["target_text"]
    assert cross_ba["activation_path"] == second["activation_path"]
    assert cross_ba["target_text"] == first["target_text"]


def test_symmetric_objective_rewards_crossed_nll_above_matched() -> None:
    good_loss, good_gap = symmetric_pair_objective(
        torch.tensor([1.0, 1.2, 2.0, 2.2]), n_pairs=1, temperature=0.1
    )
    bad_loss, bad_gap = symmetric_pair_objective(
        torch.tensor([2.0, 2.2, 1.0, 1.2]), n_pairs=1, temperature=0.1
    )
    assert good_gap.item() > 0
    assert bad_gap.item() < 0
    assert good_loss.item() < bad_loss.item()
