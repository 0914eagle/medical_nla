"""Paired bootstrap and trend tests for the note-cost table.

Table 3 reports the note's cost to p(gold) for three behaviour groups: cases
that kept their answer (-.007), cases that moved to a third diagnosis (-.055),
and cases that adopted the suggestion (-.195). Ordered like that the numbers
look like a dose-response, and the paper wants to say so. Three point estimates
cannot support that sentence -- .007 and .055 are small numbers on group sizes
that differ by an order of magnitude, and nothing so far says the ordering
survives resampling.

Two different questions live in that table and they need different resampling:

  within a group   the wrong-arm and none-arm readings are the SAME case, so
                   the case is the unit and the delta is computed per case
                   before averaging. Resampling cases keeps the pairing; the
                   difference of two independently resampled means throws away
                   the very correlation that makes the paired estimate tight.

  between groups   different cases, so the groups resample independently.

The trend test asks the third question directly -- whether the per-case delta
falls monotonically across the ordered groups -- and answers it with a rank
correlation, which needs no assumption about the shape of the deltas.

Nothing here imports torch, so it runs and is tested without a GPU.
"""

from __future__ import annotations

import random
from typing import Iterable, Sequence

Pair = tuple[float, float]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _percentiles(values: list[float], alpha: float) -> tuple[float, float]:
    """Empirical interval, with indices clamped inside the sample."""
    if not values:
        return float("nan"), float("nan")
    ordered = sorted(values)
    n = len(ordered)
    lo = int((alpha / 2) * n)
    hi = int((1 - alpha / 2) * n) - 1
    return ordered[max(0, min(lo, n - 1))], ordered[max(0, min(hi, n - 1))]


def paired_delta_ci(
    pairs: Sequence[Pair],
    draws: int = 2000,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, float]:
    """Mean of (treated - control) over cases, with a paired case bootstrap.

    `pairs` is one (treated, control) reading per case. Both readings move
    together under resampling because the case, not the reading, is drawn.
    """
    deltas = [t - c for t, c in pairs]
    if not deltas:
        return {"n": 0, "delta": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(draws):
        means.append(_mean([deltas[rng.randrange(n)] for _ in range(n)]))
    lo, hi = _percentiles(means, alpha)
    return {
        "n": n,
        "delta": _mean(deltas),
        "lo": lo,
        "hi": hi,
        "excludes_zero": (lo > 0) or (hi < 0),
    }


def group_difference_ci(
    a: Sequence[Pair],
    b: Sequence[Pair],
    draws: int = 2000,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, float]:
    """CI on (mean paired delta of a) - (mean paired delta of b).

    The two groups hold different cases, so they are resampled independently.
    Pairing is still preserved inside each group.
    """
    da = [t - c for t, c in a]
    db = [t - c for t, c in b]
    if not da or not db:
        return {"diff": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = random.Random(seed)
    na, nb = len(da), len(db)
    diffs = []
    for _ in range(draws):
        ma = _mean([da[rng.randrange(na)] for _ in range(na)])
        mb = _mean([db[rng.randrange(nb)] for _ in range(nb)])
        diffs.append(ma - mb)
    lo, hi = _percentiles(diffs, alpha)
    return {
        "n_a": na,
        "n_b": nb,
        "diff": _mean(da) - _mean(db),
        "lo": lo,
        "hi": hi,
        "excludes_zero": (lo > 0) or (hi < 0),
    }


def _ranks(values: Sequence[float]) -> list[float]:
    """Average ranks, so ties do not invent an ordering."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Rank correlation. Returns nan when either side is constant."""
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = _mean(rx), _mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx)
    dy = sum((b - my) ** 2 for b in ry)
    if dx <= 0 or dy <= 0:
        return float("nan")
    return num / (dx * dy) ** 0.5


def trend_ci(
    ordered_groups: Iterable[tuple[str, Sequence[Pair]]],
    draws: int = 2000,
    seed: int = 17,
    alpha: float = 0.05,
) -> dict[str, object]:
    """Does the per-case delta fall monotonically across the ordered groups?

    Groups arrive in the order the dose-response claim asserts. Each case
    contributes its own delta against its group's rank, and the correlation
    between the two is bootstrapped by resampling cases within each group --
    the group sizes are part of the design, not of the sampling, so they are
    held fixed.

    A negative rho means later groups lose more gold mass, which is the
    direction the table claims.
    """
    groups = [(name, [t - c for t, c in pairs]) for name, pairs in ordered_groups]
    groups = [(name, deltas) for name, deltas in groups if deltas]
    if len(groups) < 2:
        return {"rho": float("nan"), "lo": float("nan"), "hi": float("nan"),
                "groups": [name for name, _ in groups]}

    xs: list[float] = []
    ys: list[float] = []
    for rank, (_, deltas) in enumerate(groups):
        xs.extend([float(rank)] * len(deltas))
        ys.extend(deltas)

    rng = random.Random(seed)
    rhos = []
    for _ in range(draws):
        bx: list[float] = []
        by: list[float] = []
        for rank, (_, deltas) in enumerate(groups):
            n = len(deltas)
            bx.extend([float(rank)] * n)
            by.extend(deltas[rng.randrange(n)] for _ in range(n))
        rho = spearman(bx, by)
        if rho == rho:  # skip nan draws
            rhos.append(rho)
    lo, hi = _percentiles(rhos, alpha)
    return {
        "rho": spearman(xs, ys),
        "lo": lo,
        "hi": hi,
        "n": len(ys),
        "groups": [name for name, _ in groups],
        "excludes_zero": (lo > 0) or (hi < 0),
    }
