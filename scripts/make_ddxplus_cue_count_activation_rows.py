"""Convert DDXPlus cue-count case prompts into activation extraction rows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def keep_condition(row: dict[str, Any], condition: str) -> bool:
    if row.get("cue_count_condition") == condition:
        return True
    variant = str(row.get("variant") or "")
    return variant == f"cue_count_{condition}" or variant.endswith(f"_cues_{condition}")


def activation_row(row: dict[str, Any], *, condition: str) -> dict[str, Any]:
    out = dict(row)
    out["id"] = str(row["id"])
    out["base_id"] = str(row.get("base_id") or row["id"])
    out["variant"] = f"cue_count_{condition}"
    out["condition"] = condition
    out["target_role"] = "format"
    out["cue_index"] = None
    out["position_mode"] = "last_token"
    out["target_text"] = None
    out["target_text_strategy"] = None
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Cue-count case JSONL.")
    parser.add_argument("--output", required=True, help="Activation extraction JSONL.")
    parser.add_argument("--condition", default="all", help="Cue-count condition to keep.")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = []
    seen_ids: set[str] = set()
    for row in read_jsonl(args.input):
        if not keep_condition(row, args.condition):
            continue
        row_id = str(row["id"])
        if row_id in seen_ids:
            raise ValueError(f"Duplicate row id after filtering: {row_id}")
        if not row.get("prompt"):
            raise ValueError(f"Row {row_id} has no prompt.")
        rows.append(activation_row(row, condition=args.condition))
        seen_ids.add(row_id)
        if args.limit is not None and len(rows) >= args.limit:
            break
    if not rows:
        raise ValueError("No rows selected. Check --input and --condition.")

    write_jsonl(Path(args.output), rows)
    print(f"[done] wrote {len(rows)} activation rows to {args.output}")


if __name__ == "__main__":
    main()
