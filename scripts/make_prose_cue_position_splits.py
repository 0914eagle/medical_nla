"""Splits for prose corpora, where unseen cues occur naturally.

DDXPlus draws its cues from a fixed questionnaire, so the same ~164 strings
recur across cases and an unseen-cue pool has to be manufactured by holding
some strings out of training. Case-report prose is the opposite: 93% of
MedCaseReasoning cue spans occur exactly once in the corpus, so splitting by
case already leaves most evaluation cues unseen.

That makes the honest split simple — partition by case, then label each test
cue by whether its string happens to appear in the training vocabulary. The
unseen pool carries the generalization claim; the seen pool (boilerplate like
"he was otherwise healthy") is the in-distribution reference. Neither is
imposed; both are measured, and the ratio is reported.

Writes the same `manifest_{split}.jsonl` / `sft_{split}.jsonl` layout as the
DDXPlus split maker, so training and readout consume it unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

SPLITS = ("train", "val", "test_seen_cue", "test_heldout_cue")
TARGET_STYLE = "cue_first"


def norm_cue(text: str) -> str:
    """Compare cue strings ignoring case and spacing, so 'seen' is not undercounted."""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def cue_of(row: dict[str, Any]) -> str | None:
    for field in ("cue_text", "target_text"):
        value = row.get(field)
        if value and str(value).strip():
            return str(value).strip()
    targets = row.get("cue_targets") or []
    if targets and str(targets[0]).strip():
        return str(targets[0]).strip()
    return None


def normalize_manifest_row(row: dict[str, Any]) -> dict[str, Any] | None:
    """Keep rows that carry an activation and a cue; recover cue_text if dropped."""
    if not row.get("activation_path"):
        return None
    cue = cue_of(row)
    if cue is None:
        return None
    out = dict(row)
    out["cue_text"] = cue
    return out


def split_cases(
    case_ids: list[str], *, seed: int, train_frac: float, val_frac: float
) -> dict[str, str]:
    """Assign whole cases to train/val/test so no case spans two pools."""
    unique = sorted(set(case_ids))
    rng = random.Random(seed)
    rng.shuffle(unique)
    n = len(unique)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    pools: dict[str, str] = {}
    for index, case_id in enumerate(unique):
        if index < n_train:
            pools[case_id] = "train"
        elif index < n_train + n_val:
            pools[case_id] = "val"
        else:
            pools[case_id] = "test"
    return pools


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Cue-position extraction manifest.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--val-frac", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--max-rows-per-case",
        type=int,
        default=None,
        help="Cap rows kept per case, so long cases do not dominate training.",
    )
    args = parser.parse_args()

    rows = []
    for raw_row in read_jsonl(args.manifest):
        row = normalize_manifest_row(raw_row)
        if row is not None:
            rows.append(row)
    if not rows:
        raise ValueError("No usable rows: need activation_path plus a cue string per row.")

    case_ids = [str(row.get("base_id") or row["id"]) for row in rows]
    case_pools = split_cases(
        case_ids, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac
    )

    if args.max_rows_per_case is not None:
        rng = random.Random(args.seed)
        by_case: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_case.setdefault(str(row.get("base_id") or row["id"]), []).append(row)
        capped: list[dict[str, Any]] = []
        for case_rows in by_case.values():
            if len(case_rows) > args.max_rows_per_case:
                case_rows = rng.sample(case_rows, args.max_rows_per_case)
            capped.extend(case_rows)
        rows = capped

    # Training vocabulary decides what "seen" means; build it before labelling.
    train_cue_vocab = {
        norm_cue(row["cue_text"])
        for row in rows
        if case_pools[str(row.get("base_id") or row["id"])] == "train"
    }

    split_rows: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    for row in rows:
        pool = case_pools[str(row.get("base_id") or row["id"])]
        seen_cue = norm_cue(row["cue_text"]) in train_cue_vocab
        if pool == "train":
            split = "train"
        elif pool == "val":
            split = "val"
        else:
            split = "test_seen_cue" if seen_cue else "test_heldout_cue"
        out = dict(row)
        out["split"] = split
        out["cue_pool"] = "train" if seen_cue else "heldout"
        out["target_text"] = row["cue_text"]
        out["target_style"] = TARGET_STYLE
        split_rows[split].append(out)

    if not split_rows["train"] or not split_rows["test_heldout_cue"]:
        raise ValueError("Empty train or test_heldout_cue pool; adjust fractions/seed.")

    # A case must not appear in two pools, or a test cue could be memorized
    # from its own case's other spans.
    cases_by_split = {
        split: {str(row.get("base_id") or row["id"]) for row in split_rows[split]}
        for split in SPLITS
    }
    train_cases = cases_by_split["train"] | cases_by_split["val"]
    test_cases = cases_by_split["test_seen_cue"] | cases_by_split["test_heldout_cue"]
    overlap = train_cases & test_cases
    if overlap:
        raise ValueError(f"Case leakage across pools: {sorted(overlap)[:3]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter = Counter()
    for split in SPLITS:
        counts[split] = len(split_rows[split])
        write_jsonl(out_dir / f"manifest_{split}.jsonl", split_rows[split])
        write_jsonl(out_dir / f"sft_{split}.jsonl", split_rows[split])

    n_test = counts["test_seen_cue"] + counts["test_heldout_cue"]
    metadata = {
        "manifest": args.manifest,
        "target_style": TARGET_STYLE,
        "train_frac": args.train_frac,
        "val_frac": args.val_frac,
        "seed": args.seed,
        "max_rows_per_case": args.max_rows_per_case,
        "n_rows_in": len(rows),
        "n_cases": len(set(case_pools)),
        "n_train_cue_strings": len(train_cue_vocab),
        "counts": dict(counts),
        "unseen_cue_rate_in_test": round(counts["test_heldout_cue"] / n_test, 4) if n_test else 0.0,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    print(f"[done] wrote splits to {out_dir}")


if __name__ == "__main__":
    main()
