"""Extraction rows for the anchoring trajectory: where does the answer change?

Table 4 asked *whether* a single run betrays the pushed cases, and a
supervised probe answered it better than any readout on this closed corpus.
The question that survives -- the one no classifier head or behavioral study
answers -- is positional and narrative: **up to which point does the state
still hold the original answer, and where, in the span the note can reach,
does it flip?**

The prompt has a fixed skeleton, so the trajectory gets landmarks instead of
arbitrary token offsets, and every landmark is a literal substring the
extractor can anchor on:

    [findings bullets]                <- last_cue (last rendered bullet)
    [The referring note suspects X.]  <- note (wrong arm only)
    What is the single most likely…   <- question
    Give the diagnosis only. …        <- constraint
    You MUST end your response …      <- format
    <last prompt token>               <- final

Everything before the note is bit-identical across arms under causal
attention (the design result), so the none-arm curve is the counterfactual
baseline and the wrong-arm curve can only depart from it at the note or
later. Both arms are built at every shared landmark; the note landmark
exists only where a note does.

One forward pass serves all of a prompt's landmarks (the extractor groups
rows by prompt), so the whole trajectory costs the same GPU time as the
original two-position run.

Downstream: per-landmark probes draw the quantitative flip curve
(p(gold) vs p(suggestion) along the prompt, moved against not-moved), and
the verbalizer narrates the flip region on sampled cases -- the probe
locates, the readout explains.
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

# Fixed instruction substrings, each occurring exactly once in a direct prompt.
INSTRUCTION_LANDMARKS = (
    ("question", "What is the single most likely diagnosis?"),
    ("constraint", "Do not explain your reasoning."),
    ("format", 'You MUST end your response with exactly "The answer is <diagnosis>."'),
)


def landmark_rows(case: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    """All landmark rows for one arm's prompt, or a reason it was skipped."""
    prompt = str(case.get("prompt") or "")
    carry = {field: case.get(field) for field in CARRY_FIELDS}
    case_id = str(case["id"])

    def anchored(role: str, target: str) -> dict[str, Any]:
        return {
            **carry,
            "id": f"{case_id}__traj_{role}",
            "case_id": case_id,
            "variant": "trajectory",
            "target_role": role,
            "position_mode": "target_text",
            "target_text": target,
            "target_text_strategy": "last_subtoken",
            "target_text_occurrence": prompt.count(target) - 1,
        }

    rows: list[dict[str, Any]] = []

    cues = [str(c) for c in (case.get("cue_targets") or [])]
    last_cue = next((c for c in reversed(cues) if c and c in prompt), None)
    if last_cue is None:
        return [], "no cue string found verbatim in the prompt"
    rows.append(anchored("last_cue", last_cue))

    hinted = str(case.get("hint_diagnosis_name") or "").strip()
    if hinted and hinted in prompt:
        rows.append(anchored("note", hinted))

    for role, target in INSTRUCTION_LANDMARKS:
        if target not in prompt:
            return [], f"instruction landmark missing: {role}"
        rows.append(anchored(role, target))

    rows.append(
        {
            **carry,
            "id": f"{case_id}__traj_final",
            "case_id": case_id,
            "variant": "trajectory",
            "target_role": "final",
            "position_mode": "last_token",
        }
    )
    return rows, None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Hint case file (v2, four arms).")
    parser.add_argument(
        "--arms",
        nargs="+",
        default=["none", "wrong"],
        help="Which arms to build; none is the counterfactual baseline curve.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    skipped = Counter()
    n_cases = 0
    for case in read_jsonl(args.cases):
        if str(case.get("hint_variant") or "") not in set(args.arms):
            continue
        n_cases += 1
        built, reason = landmark_rows(case)
        if reason:
            skipped[reason] += 1
            continue
        rows.extend(built)
    if not rows:
        raise SystemExit("no rows built; check --cases and --arms")

    write_jsonl(Path(args.output), rows)
    per_role = Counter(r["target_role"] for r in rows)
    print(f"wrote {len(rows):,} rows over {n_cases:,} arm-cases to {args.output}")
    for role, count in per_role.most_common():
        print(f"  {role:<12} {count:,}")
    for reason, count in skipped.items():
        print(f"skipped {count:,}: {reason}")


if __name__ == "__main__":
    main()
