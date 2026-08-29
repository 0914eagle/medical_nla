import pytest
import torch

from scripts.freeze_medical_nla_bottleneck_effect_floor import effect_floor
from scripts.train_medical_nla_soft_bottleneck import (
    hash_order,
    initialize_auxiliary_head,
    ordered_id_sha256,
)
from src.nla_bottleneck import (
    NlaBottleneckProjector,
    canonicalize_basis_signs,
    fit_source_balanced_pca,
    sha256_state_dict,
)


def test_projector_is_a_no_skip_centered_bottleneck() -> None:
    mean = torch.zeros(4)
    basis = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    projector = NlaBottleneckProjector(mean, basis)
    activation = torch.tensor([0.0, 0.0, 3.0, 4.0])
    reconstructed, latent = projector(activation)
    assert latent.shape == (2,)
    assert torch.equal(latent, torch.zeros(2))
    assert torch.equal(reconstructed, torch.zeros(4))


def test_source_balanced_pca_is_deterministic_and_sign_canonical() -> None:
    ddxplus = torch.tensor(
        [[4.0, 0.0, 0.0], [3.0, 1.0, 0.0], [4.0, -1.0, 0.0]]
    )
    direct = torch.tensor(
        [[0.0, 4.0, 0.0], [1.0, 3.0, 0.0], [-1.0, 4.0, 0.0]]
    )
    first = fit_source_balanced_pca(ddxplus, direct, d_z=2)
    second = fit_source_balanced_pca(ddxplus, direct, d_z=2)
    assert torch.equal(first[0], second[0])
    assert torch.equal(first[1], second[1])
    pivots = first[1].abs().argmax(dim=1)
    assert torch.all(first[1][torch.arange(2), pivots] > 0)


def test_basis_sign_canonicalization_preserves_zero_safe_sign() -> None:
    basis = torch.tensor([[-2.0, 1.0], [0.0, -3.0]])
    canonical = canonicalize_basis_signs(basis)
    assert torch.equal(canonical, torch.tensor([[2.0, -1.0], [-0.0, 3.0]]))


def test_auxiliary_head_is_identical_for_every_proposed_seed() -> None:
    first = initialize_auxiliary_head(256, 91)
    torch.manual_seed(999)
    second = initialize_auxiliary_head(256, 91)
    assert sha256_state_dict(first.state_dict()) == sha256_state_dict(second.state_dict())


def test_smoke_order_is_seeded_and_hashable() -> None:
    rows = [{"base_id": str(index)} for index in range(20)]
    order17 = hash_order(rows, seed=17)
    repeated = hash_order(rows, seed=17)
    order29 = hash_order(rows, seed=29)
    assert order17 == repeated
    assert order17 != order29
    assert ordered_id_sha256(order17) == ordered_id_sha256(repeated)


def test_effect_floor_uses_spread_and_absolute_minimum() -> None:
    assert effect_floor([0.001, 0.002, 0.003]) == pytest.approx((0.002, 0.005))
    assert effect_floor([0.001, 0.006, 0.011]) == pytest.approx((0.010, 0.020))
    with pytest.raises(ValueError):
        effect_floor([0.1, 0.2])
