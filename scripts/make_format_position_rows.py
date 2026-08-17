"""Emit format-position (last-token) extraction rows from a case/manifest file.

The cue-position sweep mapped where individual cue detail lives across
layers (inverted U, peak L24). This is the other half: re-extract the
FORMAT position (the prompt's final token, where the model's integrated
answer state forms) at several layers, so the v3 cue-first readout and a
diagnosis probe can be run per layer and overlaid on the cue-detail curve.

Input is any JSONL carrying `prompt` (a cue-count case file or the existing
all-cue format manifest both work). Output rows set position_mode=last_token
so `src.extract_activations` reads the final token regardless of the source
layer, carrying the fields the diagnosis-heldout split and v3 targets need.
"""

from __future__ import annotations

import argparse
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
    "cue_targets",
    "cue_types",
    "cue_evidence_ids",
    "cue_evidence_entries",
    "source",
    "patient_id",
)


def format_row(row: dict[str, Any], *, variant: str) -> dict[str, Any]:
    out = {field: row.get(field) for field in CARRY_FIELDS if row.get(field) is not None}
    out["id"] = str(row["id"])
    out["base_id"] = str(row.get("base_id") or row["id"])
    out["variant"] = variant
    out["target_role"] = "format"
    out["position_mode"] = "last_token"
    out["target_text"] = None
    out["target_text_strategy"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Case/manifest JSONL with prompt + cue_targets.")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["cue_count_all"],
        help="Input variants to keep; pass 'any' to keep all rows.",
    )
    parser.add_argument(
        "--out-variant",
        default="cue_count_all",
        help="Variant label to stamp on output rows (matches downstream --variants).",
    )
    args = parser.parse_args()

    keep = set(args.variants)
    keep_all = "any" in keep
    rows = []
    seen: set[str] = set()
    for row in read_jsonl(args.input):
        if not keep_all and row.get("variant") not in keep:
            continue
        if not row.get("prompt"):
            continue
        row_id = str(row["id"])
        if row_id in seen:
            continue
        seen.add(row_id)
        rows.append(format_row(row, variant=args.out_variant))
    if not rows:
        raise ValueError("No rows produced. Check --input and --variants.")

    write_jsonl(Path(args.output), rows)
    print(f"[done] wrote {len(rows)} format-position rows to {args.output}")


if __name__ == "__main__":
    main()
