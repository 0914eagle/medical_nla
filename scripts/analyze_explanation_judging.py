"""Unblind and aggregate the explanation-quality judgements.

The judge saw explanations labeled A/B/C in a per-case shuffled order; each
row carries `channel_map` to reverse that. The output is per-channel mean
scores on the three axes, the most-useful share, and the moved/kept split --
the moved rows are the ones where an explanation had something real to
surface.

A judge that answers in prose instead of JSON is a real outcome, not a bug to
hide: parse failures are counted and reported, and rows that fail to parse
contribute nothing to the means.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl

AXES = ("grounding", "coherence", "utility")


def parse_judgement(response: str) -> dict[str, Any] | None:
    """The first JSON object in the response that carries all three labels.

    Judges wrap JSON in prose or code fences often enough that requiring a
    bare object would throw away valid judgements. raw_decode from each
    opening brace handles the nesting that a regex over braces cannot;
    requiring A, B and C guards against grabbing some other object in the
    text.
    """
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", response):
        try:
            obj, _ = decoder.raw_decode(response, match.start())
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and all(label in obj for label in "ABC"):
            return obj
    return None


def score_of(obj: dict[str, Any], label: str, axis: str) -> float | None:
    value = (obj.get(label) or {}).get(axis)
    if isinstance(value, (int, float)) and 1 <= value <= 5:
        return float(value)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judgements", required=True,
                        help="run_source_answers output over the judge prompts.")
    args = parser.parse_args()

    rows = list(read_jsonl(args.judgements))
    parsed = 0
    failures = 0
    # (group, channel, axis) -> scores; group "all" accumulates everything.
    scores: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    most_useful: dict[str, Counter] = defaultdict(Counter)

    for row in rows:
        obj = parse_judgement(str(row.get("response") or ""))
        channel_map = row.get("channel_map") or {}
        if obj is None or not channel_map:
            failures += 1
            continue
        parsed += 1
        group = str(row.get("group") or "?")
        for label, channel in channel_map.items():
            for axis in AXES:
                value = score_of(obj, label, axis)
                if value is not None:
                    scores[(group, channel, axis)].append(value)
                    scores[("all", channel, axis)].append(value)
        pick = str(obj.get("most_useful") or "").strip().upper()[:1]
        if pick in channel_map:
            most_useful[group][channel_map[pick]] += 1
            most_useful["all"][channel_map[pick]] += 1

    print(f"judgements {len(rows):,} | parsed {parsed:,} | unparseable {failures:,}")
    for group in ("all", "moved", "kept"):
        if not any(g == group for g, _, _ in scores):
            continue
        print(f"\n[{group}]")
        header = "  channel   " + "".join(f"{axis:>12}" for axis in AXES) + "   most-useful"
        print(header)
        total_picks = sum(most_useful[group].values()) or 1
        for channel in ("readout", "cot", "probe"):
            cells = []
            for axis in AXES:
                values = scores.get((group, channel, axis), [])
                cells.append(f"{mean(values):>12.3f}" if values else f"{'-':>12}")
            share = most_useful[group].get(channel, 0) / total_picks
            print(f"  {channel:<10}" + "".join(cells)
                  + f"   {most_useful[group].get(channel, 0):>4} ({share:.3f})")


if __name__ == "__main__":
    main()
