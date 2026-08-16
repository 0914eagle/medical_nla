"""Explode DDXPlus all-cue cases into per-cue activation extraction rows.

The v3 result showed the layer-32 format-position activation keeps only
theme-level content. This generator prepares the positive-control run: one
extraction row per (case, cue) at the cue's own token span, where the cue's
information is guaranteed present. `src.extract_activations` resolves
`target_text` via chat-template substring matching; `target_text_occurrence`
is computed here so cues whose text repeats (or appears inside an earlier
cue) resolve to the intended span.

Input rows need `prompt` and `cue_targets` (cue-count case files and the
all-cue format extraction manifest both work).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

CARRY_FIELDS = (
    "prompt",
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "source",
    "patient_id",
    "cue_count",
    "cue_count_condition",
)


def cue_spans_in_prompt(prompt: str, cues: list[str]) -> list[tuple[int, int] | None]:
    """Char span of each cue, scanning left-to-right in cue order.

    Sequential search keeps repeated cue strings and cues that are substrings
    of an earlier cue from resolving to the wrong mention.
    """
    spans: list[tuple[int, int] | None] = []
    search_from = 0
    for cue in cues:
        pos = prompt.find(cue, search_from)
        if pos == -1:
            pos = prompt.find(cue)
        if pos == -1:
            spans.append(None)
            continue
        spans.append((pos, pos + len(cue)))
        search_from = pos + len(cue)
    return spans


def cue_rows_for_case(
    row: dict[str, Any],
    *,
    max_cues_per_case: int | None,
    strategy: str,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], int]:
    prompt = str(row.get("prompt") or "")
    cues = [str(cue) for cue in (row.get("cue_targets") or []) if str(cue).strip()]
    if not prompt or not cues:
        return [], len(cues)
    spans = cue_spans_in_prompt(prompt, cues)
    indexed = [(idx, cue, span) for idx, (cue, span) in enumerate(zip(cues, spans))]
    resolved = [(idx, cue, span) for idx, cue, span in indexed if span is not None]
    skipped = len(indexed) - len(resolved)
    if max_cues_per_case is not None and len(resolved) > max_cues_per_case:
        resolved = rng.sample(resolved, max_cues_per_case)
        resolved.sort(key=lambda item: item[0])

    base_id = str(row.get("base_id") or row["id"])
    out_rows = []
    for idx, cue, span in resolved:
        occurrence = prompt[: span[0]].count(cue)
        out = {field: row.get(field) for field in CARRY_FIELDS}
        out["id"] = f"{base_id}__cuepos{idx:02d}"
        out["base_id"] = base_id
        out["case_id"] = str(row["id"])
        out["variant"] = "cue_position"
        out["target_role"] = "cue"
        out["cue_index"] = idx
        out["cue_text"] = cue
        out["cue_targets"] = [cue]
        out["position_mode"] = "target_text"
        out["target_text"] = cue
        out["target_text_strategy"] = strategy
        out["target_text_occurrence"] = occurrence
        out_rows.append(out)
    return out_rows, skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Case JSONL with prompt + cue_targets.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["cue_count_all"],
        help="Input variants to keep; pass 'any' to keep all rows.",
    )
    parser.add_argument("--max-cues-per-case", type=int, default=4)
    parser.add_argument(
        "--target-text-strategy", default="last_subtoken", choices=["last_subtoken", "span_mean"]
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-cases", type=int, default=None)
    args = parser.parse_args()

    variants = set(args.variants)
    keep_all = "any" in variants
    rng = random.Random(args.seed)
    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    n_cases = 0
    total_skipped = 0
    for row in read_jsonl(args.input):
        if not keep_all and row.get("variant") not in variants:
            continue
        n_cases += 1
        case_rows, skipped = cue_rows_for_case(
            row,
            max_cues_per_case=args.max_cues_per_case,
            strategy=args.target_text_strategy,
            rng=rng,
        )
        total_skipped += skipped
        for out in case_rows:
            if out["id"] in seen_ids:
                raise ValueError(f"Duplicate cue-position row id: {out['id']}")
            seen_ids.add(out["id"])
            rows.append(out)
        if args.limit_cases is not None and n_cases >= args.limit_cases:
            break
    if not rows:
        raise ValueError("No cue-position rows produced. Check --input and --variants.")

    write_jsonl(Path(args.output), rows)
    print(
        f"[done] wrote {len(rows)} cue-position rows from {n_cases} cases to {args.output} "
        f"(cues not found in prompt: {total_skipped})"
    )


if __name__ == "__main__":
    main()
