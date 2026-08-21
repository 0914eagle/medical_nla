"""Taking part of a corpus without taking a corner of it.

Every case file in this project is grouped by diagnosis -- a hundred cases of
one label, then a hundred of the next -- so `rows[:n]` is not a smaller version
of the corpus, it is a few of its labels. That has now been found and undone in
six places: a validation set, a dataset card, an appendix table, readout pool
selection, and twice in candidate scoring, where `--limit 200` left two labels
in play and a two-way choice reported top1 200/200 against a source model that
answers these cases at 0.3724.

The fix is one line at each site, which is exactly why it kept being written
the other way. It lives here so the next `--limit` gets it by default.
"""

from __future__ import annotations

import random
from typing import Any, TypeVar

T = TypeVar("T")


def sample_rows(
    rows: list[T],
    limit: int | None,
    *,
    seed: int = 17,
    label: str = "rows",
    announce: bool = True,
) -> list[T]:
    """A random `limit` of `rows`, seeded, or all of them.

    Seeded so a re-run scores the same subset, and announced so the size and
    the seed appear in the log beside the numbers they produced.
    """
    if limit is None or len(rows) <= limit:
        return list(rows)
    available = len(rows)
    sampled = random.Random(seed).sample(list(rows), limit)
    if announce:
        print(
            f"[sample] {limit:,} of {available:,} {label}, seed {seed}",
            flush=True,
        )
    return sampled


def distinct_values(rows: list[dict[str, Any]], key: str) -> int:
    """How many distinct values of `key` a row set covers.

    Used to say out loud how much of the label space a sample touches, which
    is the check that would have caught the two-way candidate set at a glance.
    """
    return len({row.get(key) for row in rows if row.get(key) is not None})
