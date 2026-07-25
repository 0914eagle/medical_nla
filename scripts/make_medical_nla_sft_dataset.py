"""Build SFT rows for a diagnosis-preserving Medical-NLA adapter.

Rows point to existing activation tensors and provide a target AV explanation
that explicitly preserves the gold diagnosis. This is intentionally separate
from training so the dataset can be inspected before fitting a LoRA adapter.
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


def clean_text(text: Any) -> str:
    return " ".join(str(text or "").split())


def diagnosis_name(row: dict[str, Any]) -> str:
    name = clean_text(row.get("diagnosis_name"))
    if name:
        return name
    aliases = row.get("diagnosis_aliases") or []
    if aliases:
        return clean_text(aliases[0])
    return clean_text(row.get("diagnosis_id"))


def cue_summary(row: dict[str, Any], *, max_cues: int) -> str:
    cues = row.get("cue_targets") or row.get("specific_targets") or []
    if isinstance(cues, str):
        cues = [cues]
    cues = [clean_text(cue) for cue in cues if clean_text(cue)]
    if row.get("target_text"):
        target = clean_text(row.get("target_text"))
        if target and target not in cues:
            cues.insert(0, target)
    cues = cues[:max_cues]
    if not cues:
        return "the clinical cues in the prompt"
    if len(cues) == 1:
        return cues[0]
    return ", ".join(cues[:-1]) + f", and {cues[-1]}"


def target_text(row: dict[str, Any], *, style: str, max_cues: int) -> str:
    dx = diagnosis_name(row)
    cues = cue_summary(row, max_cues=max_cues)
    if style == "diagnosis_first":
        body = (
            f"The activation represents an integrated clinical diagnosis state for {dx}. "
            f"It combines the patient cues, including {cues}, toward that diagnosis."
        )
    elif style == "cue_then_diagnosis":
        body = (
            f"The activation reflects clinical cues such as {cues}. "
            f"Together, these cues support the most likely diagnosis: {dx}."
        )
    else:
        raise ValueError(f"Unsupported target style: {style}")
    return f"<explanation>{body}</explanation>"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["multi_format"],
        help="Manifest variants to use for SFT. Use multi_format for integrated diagnosis readout.",
    )
    parser.add_argument(
        "--style",
        choices=["diagnosis_first", "cue_then_diagnosis"],
        default="diagnosis_first",
    )
    parser.add_argument("--max-cues", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    variants = set(args.variants)
    out_rows = []
    for row in read_jsonl(args.manifest):
        if variants and row.get("variant") not in variants:
            continue
        if not row.get("activation_path"):
            continue
        if not row.get("diagnosis_id") and not row.get("diagnosis_name"):
            continue
        out = {
            "id": row["id"],
            "base_id": row.get("base_id", row["id"]),
            "activation_path": row["activation_path"],
            "prompt": row.get("prompt"),
            "variant": row.get("variant"),
            "diagnosis_id": row.get("diagnosis_id"),
            "diagnosis_name": diagnosis_name(row),
            "diagnosis_aliases": row.get("diagnosis_aliases"),
            "target_text": target_text(row, style=args.style, max_cues=args.max_cues),
            "target_style": args.style,
            "cue_targets": row.get("cue_targets"),
            "cue_types": row.get("cue_types"),
            "source": row.get("source"),
            "patient_id": row.get("patient_id"),
            "layer": row.get("layer"),
            "position": row.get("position"),
            "position_family": row.get("position_family"),
            "position_mode": row.get("position_mode"),
        }
        out_rows.append(out)
        if args.limit is not None and len(out_rows) >= args.limit:
            break
    if not out_rows:
        raise ValueError("No SFT rows produced. Check --manifest and --variants.")
    write_jsonl(Path(args.output), out_rows)
    print(f"wrote {len(out_rows)} SFT rows to {args.output}")


if __name__ == "__main__":
    main()
