from __future__ import annotations

import torch

from scripts.audit_medical_nla_d22_geometry import (
    average_rank,
    cluster_bootstrap_ci,
    remove_mean_direction,
    row_bootstrap_ci,
)


def test_remove_mean_direction_removes_only_the_mean_axis() -> None:
    result = remove_mean_direction(
        torch.tensor([3.0, 4.0]), torch.tensor([2.0, 0.0])
    )
    assert torch.allclose(result, torch.tensor([0.0, 4.0]))


def test_average_rank_handles_ties() -> None:
    assert average_rank([0.9, 0.7, 0.2], 0) == 1.0
    assert average_rank([0.9, 0.9, 0.2], 0) == 1.5
    assert average_rank([0.9, 0.9, 0.2], 2) == 3.0


def test_bootstraps_are_deterministic() -> None:
    values = [1.0, 2.0, 3.0, 4.0]
    clusters = ["a", "a", "b", "b"]
    assert row_bootstrap_ci(values, replicates=100) == row_bootstrap_ci(
        values, replicates=100
    )
    assert cluster_bootstrap_ci(
        values, clusters, replicates=100
    ) == cluster_bootstrap_ci(values, clusters, replicates=100)
