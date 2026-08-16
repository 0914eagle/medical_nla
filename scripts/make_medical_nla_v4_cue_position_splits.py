"""Create cue-heldout SFT splits for the cue-position (v4) experiment.

Positive-control design: each row's activation comes from a cue's own token
span, so the cue's information is guaranteed present. The OOD unit here is
the CUE STRING, not the diagnosis — DDXPlus cues are shared across
diagnoses, so a diagnosis-heldout split would not keep cue text unseen.

Pools (cue set x case set, both disjoint):

- train:            train cues, train cases
- val:              train cues, val cases
- test_seen_cue:    train cues, test cases   (unseen case, seen cue text)
- test_heldout_cue: heldout cues, test cases (cue text never supervised)

Heldout-cue rows falling in train/val cases are dropped entirely (never
supervised, not tested from seen prompts); the count is recorded. A heldout
cue may still appear inside a training row's PROMPT as context — only the
supervised target never contains it.

Reading the outcome: high test_heldout_cue read rate means the AV can carry
case-specific detail from a position that has it, so the v3 format-position
failure was about the position; failure here implicates the single-vector
NLA readout mechanism itself.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_medical_nla_v3_cue_first_targets import cue_first_target_text
from src.jsonl import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test_seen_cue", "test_heldout_cue")
TARGET_STYLE = "structured_readout_v4_cue_position"


def norm_cue(cue: str) -> str:
    return " ".join(str(cue or "").split()).lower()


def normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Recover cue fields from an extraction manifest row.

    `src.extract_activations` passes through only a fixed field list, which
    drops `cue_text`; the same string survives as `target_text` (and inside
    `cue_targets`), so rebuild from those.
    """
    if not row.get("activation_path"):
        return None
    cue = row.get("cue_text") or row.get("target_text")
    if not cue:
        cue_targets = row.get("cue_targets") or []
        cue = cue_targets[0] if cue_targets else None
    if not cue or not str(cue).strip():
        return None
    out = dict(row)
    out["cue_text"] = str(cue)
    out["cue_targets"] = [str(cue)]
    return out


def split_cue_strings(
    cues: list[str],
    *,
    seed: int,
    heldout_frac: float,
) -> tuple[set[str], set[str]]:
    pool = sorted({norm_cue(cue) for cue in cues if norm_cue(cue)})
    n_heldout = int(round(len(pool) * heldout_frac))
    if not 0 < n_heldout < len(pool):
        raise ValueError(
            f"Heldout cue count {n_heldout} must be in (0, {len(pool)}); "
            "adjust --heldout-cue-frac."
        )
    shuffled = list(pool)
    random.Random(seed).shuffle(shuffled)
    heldout = set(shuffled[:n_heldout])
    return set(pool) - heldout, heldout


def split_cases(
    base_ids: list[str],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    if train_frac + val_frac >= 1.0:
        raise ValueError("train_frac + val_frac must be < 1 so test cases remain.")
    ids = sorted(set(base_ids))
    random.Random(seed + 1).shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))
    if n_train + n_val >= n:
        raise ValueError(f"Case split leaves no test cases (n={n}).")
    pools = {}
    for base_id in ids[:n_train]:
        pools[base_id] = "train"
    for base_id in ids[n_train : n_train + n_val]:
        pools[base_id] = "val"
    for base_id in ids[n_train + n_val :]:
        pools[base_id] = "test"
    return pools


def assign_split(row: dict[str, Any], case_pool: str, heldout_cues: set[str]) -> str | None:
    cue_is_heldout = norm_cue(row.get("cue_text")) in heldout_cues
    if case_pool == "train":
        return None if cue_is_heldout else "train"
    if case_pool == "val":
        return None if cue_is_heldout else "val"
    return "test_heldout_cue" if cue_is_heldout else "test_seen_cue"


def write_summary(
    path: Path,
    *,
    counts: Counter,
    dropped: int,
    n_train_cues: int,
    n_heldout_cues: int,
    heldout_examples: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Medical-NLA v4 Cue-Position Split Summary\n\n")
        f.write(
            "Cue-string-level heldout on cue-position activations. "
            "test_heldout_cue targets were never supervised in any row.\n\n"
        )
        f.write(f"- target_style: {TARGET_STYLE}\n")
        f.write(f"- train_cue_strings: {n_train_cues}\n")
        f.write(f"- heldout_cue_strings: {n_heldout_cues}\n")
        for split in SPLITS:
            f.write(f"- {split}_rows: {counts[split]}\n")
        f.write(f"- dropped_heldout_cue_rows_in_train_val_cases: {dropped}\n")
        f.write("\n## Heldout Cue Examples\n\n")
        for cue in heldout_examples[:20]:
            f.write(f"- {cue}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Cue-position extraction manifest.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--heldout-cue-frac", type=float, default=0.25)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    rows = []
    for raw_row in read_jsonl(args.manifest):
        row = normalize_manifest_row(raw_row)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError(
            "No usable rows: need activation_path plus cue_text/target_text/cue_targets per row."
        )

    train_cues, heldout_cues = split_cue_strings(
        [row["cue_text"] for row in rows],
        seed=args.seed,
        heldout_frac=args.heldout_cue_frac,
    )
    case_pools = split_cases(
        [str(row.get("base_id") or row["id"]) for row in rows],
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    dropped = 0
    for row in rows:
        case_pool = case_pools[str(row.get("base_id") or row["id"])]
        split = assign_split(row, case_pool, heldout_cues)
        if split is None:
            dropped += 1
            continue
        out = dict(row)
        out["split"] = split
        out["cue_pool"] = "heldout" if norm_cue(row["cue_text"]) in heldout_cues else "train"
        out["target_text"] = cue_first_target_text(out, max_cues=1, seed=args.seed)
        out["target_style"] = TARGET_STYLE
        split_rows[split].append(out)

    for split in ("train", "val"):
        for row in split_rows[split]:
            if norm_cue(row["cue_text"]) in heldout_cues:
                raise ValueError(f"Leakage: heldout cue in {split}: {row['id']}")
    if not split_rows["train"] or not split_rows["test_heldout_cue"]:
        raise ValueError("Empty train or test_heldout_cue pool; adjust fractions/seed.")

    counts: Counter = Counter()
    for split in SPLITS:
        counts[split] = len(split_rows[split])
        write_jsonl(out_dir / f"manifest_{split}.jsonl", split_rows[split])
        write_jsonl(out_dir / f"sft_{split}.jsonl", split_rows[split])

    heldout_sorted = sorted(heldout_cues)
    metadata = {
        "manifest": args.manifest,
        "target_style": TARGET_STYLE,
        "heldout_cue_frac": args.heldout_cue_frac,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "n_rows_in": len(rows),
        "n_train_cue_strings": len(train_cues),
        "n_heldout_cue_strings": len(heldout_cues),
        "dropped_heldout_cue_rows_in_train_val_cases": dropped,
        "heldout_cue_strings": heldout_sorted,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "summary.md",
        counts=counts,
        dropped=dropped,
        n_train_cues=len(train_cues),
        n_heldout_cues=len(heldout_cues),
        heldout_examples=heldout_sorted,
    )
    print(
        f"[done] wrote v4 cue-position splits to {out_dir} "
        f"(train={counts['train']} val={counts['val']} "
        f"seen={counts['test_seen_cue']} heldout={counts['test_heldout_cue']} dropped={dropped})"
    )


if __name__ == "__main__":
    main()
