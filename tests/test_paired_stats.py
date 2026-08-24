"""The note-cost table's statistics, checked against cases with known answers."""

from __future__ import annotations

import random

from src.paired_stats import (
    group_difference_ci,
    paired_delta_ci,
    spearman,
    trend_ci,
)


def test_paired_delta_recovers_a_constant_shift():
    pairs = [(0.5 + i * 0.01 - 0.2, 0.5 + i * 0.01) for i in range(60)]
    out = paired_delta_ci(pairs, draws=400, seed=3)
    assert out["n"] == 60
    assert abs(out["delta"] + 0.2) < 1e-9
    # A constant shift has no spread, so the interval collapses onto it.
    assert abs(out["lo"] + 0.2) < 1e-9
    assert abs(out["hi"] + 0.2) < 1e-9
    assert out["excludes_zero"]


def test_pairing_is_what_makes_the_interval_tight():
    """The same deltas, hidden inside noisy levels, must stay detectable.

    This is the reason the trajectory analyzer had to keep case pairs instead
    of two flat lists: the between-case spread here is 100x the effect, so an
    unpaired comparison of means drowns it while the paired one does not.
    """
    rng = random.Random(11)
    pairs = []
    for _ in range(400):
        level = rng.uniform(0.0, 1.0)
        pairs.append((level - 0.01, level))
    out = paired_delta_ci(pairs, draws=600, seed=5)
    assert abs(out["delta"] + 0.01) < 1e-9
    assert out["excludes_zero"]


def test_paired_delta_on_no_effect_includes_zero():
    rng = random.Random(7)
    pairs = [(rng.gauss(0.5, 0.1), rng.gauss(0.5, 0.1)) for _ in range(300)]
    out = paired_delta_ci(pairs, draws=600, seed=5)
    assert out["lo"] < 0 < out["hi"]
    assert not out["excludes_zero"]


def test_empty_input_is_reported_not_crashed():
    out = paired_delta_ci([], draws=10)
    assert out["n"] == 0
    assert out["delta"] != out["delta"]  # nan


def test_group_difference_separates_two_effect_sizes():
    small = [(0.5 - 0.01, 0.5) for _ in range(200)]
    large = [(0.5 - 0.20, 0.5) for _ in range(200)]
    out = group_difference_ci(large, small, draws=400, seed=3)
    assert abs(out["diff"] + 0.19) < 1e-9
    assert out["excludes_zero"]
    assert out["hi"] < 0  # large loses strictly more


def test_group_difference_on_equal_groups_includes_zero():
    rng = random.Random(19)
    a = [(rng.gauss(0.4, 0.1), rng.gauss(0.5, 0.1)) for _ in range(250)]
    b = [(rng.gauss(0.4, 0.1), rng.gauss(0.5, 0.1)) for _ in range(250)]
    out = group_difference_ci(a, b, draws=600, seed=2)
    assert out["lo"] < 0 < out["hi"]


def test_spearman_direction_and_ties():
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    # A constant column has no ranks to correlate; nan rather than a fake 0.
    assert spearman([1, 1, 1, 1], [1, 2, 3, 4]) != spearman([1, 1, 1, 1], [1, 2, 3, 4])


def test_trend_finds_the_monotone_dose_response():
    """Groups ordered kept / third-diagnosis / adopted, losing more each time."""
    rng = random.Random(23)
    kept = [(0.8 - 0.007 + rng.gauss(0, 0.01), 0.8) for _ in range(400)]
    third = [(0.8 - 0.055 + rng.gauss(0, 0.01), 0.8) for _ in range(230)]
    adopted = [(0.8 - 0.195 + rng.gauss(0, 0.01), 0.8) for _ in range(91)]
    out = trend_ci(
        [("kept", kept), ("third", third), ("adopted", adopted)],
        draws=400,
        seed=13,
    )
    assert out["rho"] < 0  # later groups lose more
    assert out["hi"] < 0  # and the interval says so
    assert out["excludes_zero"]
    assert out["groups"] == ["kept", "third", "adopted"]


def test_trend_on_flat_groups_includes_zero():
    rng = random.Random(29)
    flat = lambda n: [(0.8 - 0.05 + rng.gauss(0, 0.05), 0.8) for _ in range(n)]
    out = trend_ci(
        [("a", flat(200)), ("b", flat(200)), ("c", flat(200))], draws=400, seed=13
    )
    assert out["lo"] < 0 < out["hi"]
    assert not out["excludes_zero"]


def test_trend_needs_at_least_two_groups():
    out = trend_ci([("only", [(0.1, 0.2)])], draws=10)
    assert out["rho"] != out["rho"]  # nan


def test_bootstrap_is_deterministic_under_a_seed():
    rng = random.Random(31)
    pairs = [(rng.gauss(0.4, 0.1), rng.gauss(0.5, 0.1)) for _ in range(120)]
    first = paired_delta_ci(pairs, draws=300, seed=99)
    second = paired_delta_ci(pairs, draws=300, seed=99)
    assert first == second
