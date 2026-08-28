"""Build a source-balanced, diagnosis-free Medical-NLA SFT corpus.

DiReCT and DDXPlus use different annotation schemes, but both expose concrete
clinical findings paired with CoT-P0/HS32 activations.  This builder maps both
to one output contract and deliberately omits diagnosis supervision.  The
result is a readout task, not a closed-set diagnostic classifier.

The development pilot keeps the two sources equally weighted by selecting the
same number of rows from each source.  Selection is deterministic and cycles
over diagnosis strata before taking a second row from any stratum.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.make_medical_nla_v3_cue_first_targets import cue_first_target_text
from src.jsonl import read_jsonl, write_jsonl


TARGET_STYLE = "common_cot_p0_observed_only_v1"
CANONICAL_TARGET_STYLE = "common_cot_p0_observed_only_v2_source_order"


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def diagnosis_stratum(row: dict[str, Any]) -> str:
    for field in ("diagnosis_id", "canonical_pdd", "disease_category", "diagnosis_name"):
        value = clean_text(row.get(field))
        if value:
            return value.casefold()
    return "<unknown>"


def normalize_row(
    row: dict[str, Any],
    *,
    source_dataset: str,
    split: str,
    max_cues: int,
    seed: int,
    require_activation_file: bool,
    cue_order: str = "shuffled",
) -> dict[str, Any] | None:
    identifier = row_id(row)
    if not identifier:
        raise ValueError(f"{source_dataset}/{split} row has no id")
    if source_dataset == "ddxplus" and row.get("variant") not in (None, "original"):
        return None
    if str(row.get("position_family")) != "P0":
        raise ValueError(f"{source_dataset}/{identifier} is not P0")
    if int(row.get("layer", -1)) != 32:
        raise ValueError(f"{source_dataset}/{identifier} is not HS32")
    raw_activation_path = str(row.get("activation_path") or "")
    if not raw_activation_path:
        raise ValueError(f"{source_dataset}/{identifier} has no activation path")
    activation_path = Path(raw_activation_path)
    if require_activation_file and not activation_path.is_file():
        raise FileNotFoundError(activation_path)

    cues = [clean_text(value) for value in (row.get("cue_targets") or [])]
    cues = [value for value in cues if value]
    if not cues:
        return None
    normalized = {
        "id": f"{source_dataset}::{identifier}",
        "base_id": identifier,
        "split": split,
        "source_dataset": source_dataset,
        "activation_path": str(activation_path),
        "position_family": "P0",
        "layer": 32,
        "cue_targets": cues,
        "cue_evidence_ids": row.get("cue_evidence_ids"),
        "cue_value_ids": row.get("cue_value_ids"),
        "cue_value_labels": row.get("cue_value_labels"),
        "diagnosis_id": row.get("diagnosis_id"),
        "diagnosis_name": row.get("diagnosis_name"),
        "canonical_pdd": row.get("canonical_pdd"),
        "disease_category": row.get("disease_category"),
        "target_style": (
            CANONICAL_TARGET_STYLE if cue_order == "source" else TARGET_STYLE
        ),
    }
    normalized["target_text"] = cue_first_target_text(
        normalized,
        max_cues=max_cues,
        seed=seed,
        include_assessment=False,
        cue_order=cue_order,
    )
    return normalized


def stratified_sample(
    rows: list[dict[str, Any]], *, cap: int, seed: int, source_dataset: str, split: str
) -> list[dict[str, Any]]:
    if cap <= 0:
        raise ValueError("source cap must be positive")
    if len(rows) <= cap:
        return sorted(rows, key=lambda row: str(row["id"]))
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[diagnosis_stratum(row)].append(row)
    rng = random.Random(f"{seed}:{source_dataset}:{split}")
    for bucket in buckets.values():
        rng.shuffle(bucket)
    labels = sorted(buckets)
    rng.shuffle(labels)
    selected: list[dict[str, Any]] = []
    while len(selected) < cap:
        added = False
        for label in labels:
            if buckets[label]:
                selected.append(buckets[label].pop())
                added = True
                if len(selected) == cap:
                    break
        if not added:
            break
    if len(selected) != cap:
        raise AssertionError(f"Selected {len(selected)} rows, expected {cap}")
    return selected


def hash_scoped_ids(rows: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(str(row["id"]) for row in rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def load_source(
    spec: str,
    *,
    split: str,
    max_cues: int,
    seed: int,
    require_activation_file: bool,
    cue_order: str,
) -> tuple[str, list[dict[str, Any]], Counter[str]]:
    if "=" not in spec:
        raise ValueError(f"Expected SOURCE=PATH, got {spec!r}")
    source_dataset, raw_path = spec.split("=", 1)
    source_dataset = source_dataset.strip()
    if source_dataset not in {"direct", "ddxplus"}:
        raise ValueError(f"Unsupported source {source_dataset!r}")
    counts: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in read_jsonl(raw_path):
        counts["input"] += 1
        normalized = normalize_row(
            raw,
            source_dataset=source_dataset,
            split=split,
            max_cues=max_cues,
            seed=seed,
            require_activation_file=require_activation_file,
            cue_order=cue_order,
        )
        if normalized is None:
            counts["filtered"] += 1
            continue
        if normalized["id"] in seen:
            raise ValueError(f"Duplicate scoped ID {normalized['id']}")
        seen.add(normalized["id"])
        rows.append(normalized)
    counts["eligible"] = len(rows)
    return source_dataset, rows, counts


def source_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    valued = 0
    cue_total = 0
    for row in rows:
        cues = list(row.get("cue_targets") or [])
        values = list(row.get("cue_value_ids") or [])
        cue_total += len(cues)
        valued += sum(bool(value) for value in values)
    return {
        "rows": len(rows),
        "strata": len({diagnosis_stratum(row) for row in rows}),
        "cues": cue_total,
        "value_bearing_cues": valued,
        "id_sha256": hash_scoped_ids(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--val", action="append", required=True, metavar="SOURCE=PATH")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--train-per-source", type=int, default=None)
    parser.add_argument("--val-per-source", type=int, default=None)
    parser.add_argument("--max-cues", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--cue-order",
        choices=("shuffled", "source"),
        default="shuffled",
        help=(
            "Order of deduplicated findings in the target. 'shuffled' preserves "
            "the v1 pilot; 'source' removes arbitrary sequence-label noise."
        ),
    )
    parser.add_argument(
        "--use-all-train-rows",
        action="store_true",
        help="Keep every eligible row from both training sources instead of equal caps.",
    )
    parser.add_argument(
        "--no-require-activation-file",
        action="store_true",
        help="Tests only: do not require activation_path to exist.",
    )
    args = parser.parse_args()
    if args.max_cues <= 0:
        raise ValueError("--max-cues must be positive")

    loaded: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    audits: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for split, specs in (("train", args.train), ("val", args.val)):
        for spec in specs:
            source, rows, counts = load_source(
                spec,
                split=split,
                max_cues=args.max_cues,
                seed=args.seed,
                require_activation_file=not args.no_require_activation_file,
                cue_order=args.cue_order,
            )
            if source in loaded[split]:
                raise ValueError(f"Duplicate {split} source {source}")
            loaded[split][source] = rows
            audits[split][source] = dict(counts)
    expected = {"direct", "ddxplus"}
    for split in ("train", "val"):
        if set(loaded[split]) != expected:
            raise ValueError(f"{split} sources are {set(loaded[split])}, expected {expected}")

    train_cap = args.train_per_source or min(len(rows) for rows in loaded["train"].values())
    val_cap = args.val_per_source or min(len(rows) for rows in loaded["val"].values())
    selected: dict[str, list[dict[str, Any]]] = {}
    for split, cap in (("train", train_cap), ("val", val_cap)):
        selected[split] = []
        for source, rows in sorted(loaded[split].items()):
            if split == "train" and args.use_all_train_rows:
                selected[split].extend(sorted(rows, key=lambda row: str(row["id"])))
            else:
                selected[split].extend(
                    stratified_sample(
                        rows,
                        cap=cap,
                        seed=args.seed,
                        source_dataset=source,
                        split=split,
                    )
                )
        random.Random(f"{args.seed}:{split}:mixed").shuffle(selected[split])

    train_ids = {(row["source_dataset"], row["base_id"]) for row in selected["train"]}
    val_ids = {(row["source_dataset"], row["base_id"]) for row in selected["val"]}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError(f"Train/validation overlap: {len(overlap)} scoped cases")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "sft_train.jsonl", selected["train"])
    write_jsonl(args.out_dir / "sft_val.jsonl", selected["val"])
    protocol = {
        "schema_version": 1,
        "target_style": (
            CANONICAL_TARGET_STYLE if args.cue_order == "source" else TARGET_STYLE
        ),
        "position": "CoT-P0/HS32/last_token",
        "diagnosis_supervision": False,
        "ddxplus_rendered_values_included": True,
        "seed": args.seed,
        "max_cues": args.max_cues,
        "train_per_source": None if args.use_all_train_rows else train_cap,
        "use_all_train_rows": args.use_all_train_rows,
        "val_per_source": val_cap,
        "input_audit": audits,
        "selected": {
            split: {
                source: source_stats(
                    [row for row in selected[split] if row["source_dataset"] == source]
                )
                for source in sorted(expected)
            }
            for split in ("train", "val")
        },
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Common Medical-NLA SFT Pilot",
        "",
        "CoT-P0/HS32 only. Both sources use the same diagnosis-free `<observed>` target.",
        "DDXPlus rendered values remain in cue text for this pilot; value-edit response is a separate gate.",
        "",
        "| split | source | eligible | selected | strata | cues | value-bearing cues | ID hash |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for split in ("train", "val"):
        for source in sorted(expected):
            stats = protocol["selected"][split][source]
            lines.append(
                f"| {split} | {source} | {audits[split][source]['eligible']} | "
                f"{stats['rows']} | {stats['strata']} | {stats['cues']} | "
                f"{stats['value_bearing_cues']} | `{stats['id_sha256']}` |"
            )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[dataset] train={len(selected['train'])} val={len(selected['val'])} "
        f"out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
