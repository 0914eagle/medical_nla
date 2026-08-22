"""Extraction rows for the referring-note cases, at the positions that can differ.

The design fixes what the cue positions hold: the note sits after the findings,
so under causal attention every cue-position activation is bit-identical
between the no-note and wrong-note arms. Reading them is a demonstration, not a
measurement -- whatever the readout says there, it says twice.

The answer differs anyway. So the divergence is downstream of the cues, and
this builds rows at the two places it can be:

**hint** -- the suspicion itself, in the arms that carry one. What the note was
encoded as.

**final** -- the last prompt token, where the model is about to answer. The
layer sweep found evidence at cue positions and an internal *conclusion* here,
which is the quantity that has to move if the note moved the answer.

Both arms get a `final` row; only the hinted arms get a `hint` row. Cue rows
are available too, off by default, for the invariance figure.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl
from src.sampling import sample_rows

CARRY_FIELDS = (
    "prompt",
    "base_id",
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "hint_variant",
    "hint_diagnosis_name",
    "gold_in_prompt",
    "source",
    "patient_id",
)


def final_row(case: dict[str, Any]) -> dict[str, Any]:
    """The last prompt token: what the model concluded before it spoke.

    `last_token` is resolved by src.extract_activations against the chat text,
    so it lands after the instruction and the generation prompt, wherever the
    template puts them.
    """
    row = {field: case.get(field) for field in CARRY_FIELDS}
    row.update(
        {
            "id": f"{case['id']}__final",
            "case_id": str(case["id"]),
            "variant": "hint_final",
            "target_role": "final",
            "position_mode": "last_token",
        }
    )
    return row


def hint_row(case: dict[str, Any], strategy: str) -> dict[str, Any] | None:
    """The suspected diagnosis where the note names it."""
    hinted = str(case.get("hint_diagnosis_name") or "").strip()
    prompt = str(case.get("prompt") or "")
    if not hinted or hinted not in prompt:
        return None
    row = {field: case.get(field) for field in CARRY_FIELDS}
    row.update(
        {
            "id": f"{case['id']}__hintpos",
            "case_id": str(case["id"]),
            "variant": "hint_position",
            "target_role": "hint",
            "cue_text": hinted,
            "cue_targets": [hinted],
            "position_mode": "target_text",
            "target_text": hinted,
            "target_text_strategy": strategy,
            # The note is the last place the name appears -- it is written into
            # the chart in 15.7% of these cases, and that mention is a finding,
            # not the suggestion we are pointing at.
            "target_text_occurrence": prompt.count(hinted) - 1,
        }
    )
    return row


def cue_rows(case: dict[str, Any], strategy: str, limit: int) -> list[dict[str, Any]]:
    prompt = str(case.get("prompt") or "")
    out: list[dict[str, Any]] = []
    search_from = 0
    for index, cue in enumerate(case.get("cue_targets") or []):
        cue = str(cue)
        pos = prompt.find(cue, search_from)
        if pos == -1:
            continue
        row = {field: case.get(field) for field in CARRY_FIELDS}
        row.update(
            {
                "id": f"{case['id']}__cuepos{index:02d}",
                "case_id": str(case["id"]),
                "variant": "hint_cue_position",
                "target_role": "cue",
                "cue_index": index,
                "cue_text": cue,
                "cue_targets": [cue],
                "position_mode": "target_text",
                "target_text": cue,
                "target_text_strategy": strategy,
                "target_text_occurrence": prompt[:pos].count(cue),
            }
        )
        out.append(row)
        search_from = pos + len(cue)
        if len(out) >= limit:
            break
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="make_hint_injection_cases output.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["none", "wrong"],
        help="Which hint arms to extract. The correct-note arm is not part of "
        "the attribution question and doubles the extraction for nothing.",
    )
    parser.add_argument(
        "--positions",
        nargs="+",
        default=["final", "hint"],
        choices=["final", "hint", "cue"],
        help="cue positions are identical across arms by construction; they are "
        "for the invariance figure, not for a measurement.",
    )
    parser.add_argument("--max-cues-per-case", type=int, default=3)
    parser.add_argument(
        "--target-text-strategy", default="last_subtoken", choices=["last_subtoken", "span_mean"]
    )
    parser.add_argument("--limit", type=int, default=None, help="Cases, not rows.")
    parser.add_argument("--sample-seed", type=int, default=17)
    args = parser.parse_args()

    arms = set(args.arms)
    positions = set(args.positions)
    cases = [row for row in read_jsonl(args.cases) if row.get("hint_variant") in arms]
    if not cases:
        raise SystemExit(f"no rows in {args.cases} with hint_variant in {sorted(arms)}")

    # Sampled by case, so an arm is never kept without its counterpart: every
    # number downstream is a difference between two arms of one case.
    base_ids = sorted({str(row.get("base_id")) for row in cases})
    kept = set(sample_rows(base_ids, args.limit, seed=args.sample_seed, label="cases"))
    cases = [row for row in cases if str(row.get("base_id")) in kept]

    rows: list[dict[str, Any]] = []
    made = Counter()
    for case in cases:
        if "final" in positions:
            rows.append(final_row(case))
            made["final"] += 1
        if "hint" in positions:
            row = hint_row(case, args.target_text_strategy)
            if row is not None:
                rows.append(row)
                made["hint"] += 1
            elif case.get("hint_diagnosis_name"):
                made["hint not found in prompt"] += 1
        if "cue" in positions:
            built = cue_rows(case, args.target_text_strategy, args.max_cues_per_case)
            rows.extend(built)
            made["cue"] += len(built)

    seen: set[str] = set()
    for row in rows:
        if row["id"] in seen:
            raise SystemExit(f"duplicate extraction row id: {row['id']}")
        seen.add(row["id"])

    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows):,} rows over {len(kept):,} cases to {args.output}")
    for role, count in sorted(made.items()):
        print(f"  {role:<24} {count:,}")


if __name__ == "__main__":
    main()
