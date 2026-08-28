"""Build diagnosis-free SFT targets for complete DDXPlus intervention families."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_medical_nla_v3_cue_first_targets import cue_first_target_text, cue_list
from src.jsonl import read_jsonl, write_jsonl


TARGET_STYLE = "ddxplus_counterfactual_observed_only_v1_source_order"
EXPECTED_VARIANTS = {"original", "cue_deleted"}


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize_manifest_row(
    row: dict[str, Any], *, split: str, max_cues: int, require_activation_file: bool
) -> dict[str, Any]:
    row_id = clean(row.get("id"))
    base_id = clean(row.get("base_id"))
    variant = clean(row.get("variant") or "original")
    if not row_id or not base_id:
        raise ValueError("Manifest row has no id/base_id")
    if variant not in {"original", "cue_deleted", "value_edited"}:
        raise ValueError(f"Unsupported DDXPlus variant {variant!r} for {row_id}")
    if str(row.get("official_split")) != split:
        raise ValueError(f"{row_id}: official_split={row.get('official_split')!r}, expected {split!r}")
    if str(row.get("position_family")) != "P0" or int(row.get("layer", -1)) != 32:
        raise ValueError(f"{row_id}: expected CoT-P0/HS32")
    activation_path = Path(str(row.get("activation_path") or ""))
    if require_activation_file and not activation_path.is_file():
        raise FileNotFoundError(activation_path)
    all_cues = cue_list(row, max_cues=1_000_000, seed=17, order="source")
    if not all_cues:
        raise ValueError(f"{row_id}: no cue targets")
    if len(all_cues) > max_cues:
        raise ValueError(
            f"{row_id}: {len(all_cues)} cues exceed --max-cues={max_cues}; "
            "refusing to hide an intervention through target truncation"
        )
    out = dict(row)
    out.update(
        {
            "id": row_id,
            "base_id": base_id,
            "split": split,
            "source_dataset": "ddxplus",
            "variant": variant,
            "activation_path": str(activation_path),
            "target_style": TARGET_STYLE,
            "target_text": cue_first_target_text(
                row,
                max_cues=max_cues,
                seed=17,
                include_assessment=False,
                cue_order="source",
            ),
        }
    )
    return out


def validate_families(rows: list[dict[str, Any]], *, split: str) -> dict[str, int]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["base_id"])].append(row)
    counts: Counter[str] = Counter()
    for base_id, family in grouped.items():
        variants = Counter(str(row["variant"]) for row in family)
        if not EXPECTED_VARIANTS.issubset(variants) or any(value != 1 for value in variants.values()):
            raise ValueError(f"{split}/{base_id}: incomplete or duplicate family {variants}")
        if set(variants) not in (
            {"original", "cue_deleted"},
            {"original", "cue_deleted", "value_edited"},
        ):
            raise ValueError(f"{split}/{base_id}: unexpected family {variants}")
        counts.update(variants)
        original = next(row for row in family if row["variant"] == "original")
        deleted = next(row for row in family if row["variant"] == "cue_deleted")
        removed = clean(deleted.get("cf_original_cue")).casefold()
        deleted_cues = {clean(value).casefold() for value in deleted.get("cue_targets") or []}
        if (
            removed in deleted_cues
            or len(deleted.get("cue_targets") or [])
            != len(original.get("cue_targets") or []) - 1
        ):
            raise ValueError(f"{split}/{base_id}: invalid deletion target")
        edited = next((row for row in family if row["variant"] == "value_edited"), None)
        if edited is not None:
            replacement = clean(edited.get("cf_replacement_cue")).casefold()
            edited_cues = {clean(value).casefold() for value in edited.get("cue_targets") or []}
            if replacement not in edited_cues or removed in edited_cues:
                raise ValueError(f"{split}/{base_id}: invalid value-edit target")
    counts["families"] = len(grouped)
    return dict(counts)


def load_split(
    paths: list[Path], *, split: str, max_cues: int, require_activation_file: bool
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        for raw in read_jsonl(path):
            row = normalize_manifest_row(
                raw,
                split=split,
                max_cues=max_cues,
                require_activation_file=require_activation_file,
            )
            if row["id"] in seen:
                raise ValueError(f"Duplicate {split} row ID {row['id']}")
            seen.add(row["id"])
            rows.append(row)
    rows.sort(key=lambda row: (str(row["base_id"]), str(row["variant"])))
    return rows, validate_families(rows, split=split)


def id_hash(values: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def write_summary(path: Path, protocol: dict[str, Any]) -> None:
    lines = [
        "# DDXPlus Counterfactual Medical-NLA SFT Dataset",
        "",
        "Diagnosis-free, source-order finding targets at CoT-P0/HS32.",
        "",
        "| split | families | original | deletion | value edit | rows |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation"):
        item = protocol["splits"][split]
        lines.append(
            f"| {split} | {item['families']} | {item['original']} | "
            f"{item['cue_deleted']} | {item.get('value_edited', 0)} | {item['rows']} |"
        )
    lines.extend(
        [
            "",
            f"- train/validation base-ID overlap: **{protocol['base_id_overlap']}**",
            f"- maximum target cues: **{protocol['max_cues']}** (overflow is an error)",
            "- diagnosis text supervision: **none**",
            "- locked DDXPlus test use: **none**",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", action="append", required=True, type=Path)
    parser.add_argument("--validation-manifest", action="append", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-cues", type=int, default=64)
    parser.add_argument(
        "--require-activation-file", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.max_cues <= 0:
        raise ValueError("--max-cues must be positive")

    train_rows, train_counts = load_split(
        args.train_manifest,
        split="train",
        max_cues=args.max_cues,
        require_activation_file=args.require_activation_file,
    )
    val_rows, val_counts = load_split(
        args.validation_manifest,
        split="validation",
        max_cues=args.max_cues,
        require_activation_file=args.require_activation_file,
    )
    train_bases = {str(row["base_id"]) for row in train_rows}
    val_bases = {str(row["base_id"]) for row in val_rows}
    overlap = train_bases & val_bases
    if overlap:
        raise ValueError(f"Train/validation base-ID overlap: {len(overlap)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "sft_train.jsonl", train_rows)
    write_jsonl(args.out_dir / "sft_validation.jsonl", val_rows)
    protocol = {
        "schema_version": 1,
        "target_style": TARGET_STYLE,
        "primary_hidden_state": "CoT-P0/HS32/last_token",
        "max_cues": args.max_cues,
        "base_id_overlap": len(overlap),
        "splits": {
            "train": {**train_counts, "rows": len(train_rows), "base_id_sha256": id_hash(train_bases)},
            "validation": {**val_counts, "rows": len(val_rows), "base_id_sha256": id_hash(val_bases)},
        },
        "train_manifests": [str(path) for path in args.train_manifest],
        "validation_manifests": [str(path) for path in args.validation_manifest],
        "locked_test_read": False,
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", protocol)
    print(
        f"[dataset] train={len(train_rows)} validation={len(val_rows)} "
        f"train_families={train_counts['families']} val_families={val_counts['families']}",
        flush=True,
    )
    print(f"[out] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
