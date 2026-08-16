"""Summarize the cue-position (v4) positive-control readouts.

Inputs are scored rows (scripts/score_medical_nla_v2_readouts.py) for the
test_seen_cue and test_heldout_cue pools. Each row has exactly one gold cue,
so per-row cue_recall is 0/1 and the pool mean is a read rate.

Decision this summary supports:

- heldout-cue read rate high: the AV mechanism can verbalize case-specific
  detail from a position that contains it; the v3 format-position failure
  was positional (detail compressed away at the answer position).
- heldout-cue read rate low while seen-cue is high: seen-cue reading was
  memorization; the single-vector NLA readout does not carry unseen detail,
  implicating the mechanism, not the position.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

# Function words excluded from soft matching so it scores clinical content,
# not phrasing. DDXPlus cues are verbose questions; strict substring matching
# counts semantically correct paraphrases ("pain that increases with
# movement" vs gold "pain that is increased with movement") as misses.
STOPWORDS = frozenset(
    "a an the of or and is are am do does did their they them her his she he it its this that "
    "these those with in on at to for from by as was were be been being have has had feel feels "
    "felt like patient patients".split()
)


def content_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9()]+", str(text or "").lower())
        if token not in STOPWORDS
    }


def gold_token_recall(gold_cue: str, emitted_text: str) -> float | None:
    gold = content_tokens(gold_cue)
    if not gold:
        return None
    return len(gold & content_tokens(emitted_text)) / len(gold)


def row_cue(row: dict[str, Any]) -> str:
    """cue_text is dropped by run_nla passthrough; recover from surviving fields."""
    cue = row.get("cue_text") or row.get("target_text")
    if not cue:
        for field in ("gold_cue_targets", "cue_targets"):
            values = row.get(field) or []
            if values:
                cue = values[0]
                break
    return " ".join(str(cue or "").split())


def normalize_item(text: str) -> str:
    text = " ".join(str(text or "").split()).strip("-•* ").strip()
    return text.rstrip(".").strip().lower()


def pool_metrics(rows: list[dict[str, Any]], *, soft_threshold: float) -> dict[str, Any]:
    n = len(rows)
    read = [bool(row.get("cue_recall")) for row in rows]
    precisions = [
        float(row["cue_precision"]) for row in rows if row.get("cue_precision") is not None
    ]
    exact = 0
    soft = 0
    token_recalls = []
    for row in rows:
        gold = row_cue(row)
        emitted = str(row.get("observed_readout") or "")
        if gold and normalize_item(emitted) == normalize_item(gold):
            exact += 1
        recall = gold_token_recall(gold, emitted)
        if recall is not None:
            token_recalls.append(recall)
            if recall >= soft_threshold:
                soft += 1
    return {
        "n": n,
        "parsed_readout": sum(bool(row.get("parsed_readout")) for row in rows),
        "read_rate_strict": sum(read) / n if n else 0.0,
        "read_rate_soft": soft / n if n else 0.0,
        "mean_gold_token_recall": mean(token_recalls) if token_recalls else 0.0,
        "mean_cue_precision": mean(precisions) if precisions else 0.0,
        "exact_match_rate": exact / n if n else 0.0,
        "mean_items_emitted": mean(float(row.get("cue_items_emitted") or 0) for row in rows)
        if n
        else 0.0,
    }


def write_summary(
    path: Path,
    *,
    seen: dict[str, Any],
    heldout: dict[str, Any],
    heldout_rows: list[dict[str, Any]],
    soft_threshold: float,
) -> None:
    by_cue: dict[str, list[float]] = {}
    paraphrases = []
    for row in heldout_rows:
        cue = row_cue(row)
        emitted = str(row.get("observed_readout") or "")
        recall = gold_token_recall(cue, emitted)
        by_cue.setdefault(cue.lower(), []).append(recall if recall is not None else 0.0)
        if not row.get("cue_recall") and recall is not None and recall >= soft_threshold:
            paraphrases.append((recall, cue, " ".join(emitted.split())))
    missed_outputs = Counter(
        " ".join(str(row.get("observed_readout") or "-").split())[:120]
        for row in heldout_rows
        if not row.get("cue_recall")
        and (gold_token_recall(row_cue(row), str(row.get("observed_readout") or "")) or 0.0)
        < soft_threshold
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cue-Position (v4) Readout Summary\n\n")
        f.write(
            "Single-gold-cue rows. read_rate_strict = exact-substring hit; "
            f"read_rate_soft = gold content-token recall >= {soft_threshold} "
            "(counts paraphrases of unseen cue phrasings as reads).\n\n"
        )
        f.write(
            "| pool | n | parsed | read_rate_strict | read_rate_soft "
            "| mean_gold_token_recall | exact_match_rate | mean_precision | mean_items |\n"
        )
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for label, metrics in (("test_seen_cue", seen), ("test_heldout_cue", heldout)):
            f.write(
                f"| {label} | {metrics['n']} | {metrics['parsed_readout']} | "
                f"{metrics['read_rate_strict']:.4f} | {metrics['read_rate_soft']:.4f} | "
                f"{metrics['mean_gold_token_recall']:.4f} | {metrics['exact_match_rate']:.4f} | "
                f"{metrics['mean_cue_precision']:.4f} | {metrics['mean_items_emitted']:.2f} |\n"
            )
        f.write(
            f"\n- seen_minus_heldout_soft: "
            f"{seen['read_rate_soft'] - heldout['read_rate_soft']:+.4f}\n"
        )

        f.write("\n## Paraphrase Reads on Heldout (strict miss, soft hit)\n\n")
        f.write("| token_recall | gold cue | emitted |\n")
        f.write("|---:|---|---|\n")
        for recall, cue, emitted in sorted(paraphrases, reverse=True)[:20]:
            f.write(f"| {recall:.2f} | {cue[:90]} | {emitted[:90]} |\n")

        f.write("\n## Heldout Cues by Mean Token Recall\n\n")
        f.write("| cue | n | mean_token_recall |\n")
        f.write("|---|---:|---:|\n")
        ranked = sorted(
            ((mean(recalls), cue, len(recalls)) for cue, recalls in by_cue.items()),
            reverse=True,
        )
        for rate, cue, count in ranked[:12]:
            f.write(f"| {cue[:100]} | {count} | {rate:.4f} |\n")
        f.write("| ... | | |\n")
        for rate, cue, count in ranked[-12:]:
            f.write(f"| {cue[:100]} | {count} | {rate:.4f} |\n")

        f.write("\n## Most Common Outputs on Hard-Missed Heldout Cues (soft miss too)\n\n")
        f.write("| output (truncated) | count |\n")
        f.write("|---|---:|\n")
        for output, count in missed_outputs.most_common(15):
            f.write(f"| {output} | {count} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seen-scored", required=True)
    parser.add_argument("--heldout-scored", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--soft-threshold", type=float, default=0.5)
    args = parser.parse_args()

    seen_rows = list(read_jsonl(args.seen_scored))
    heldout_rows = list(read_jsonl(args.heldout_scored))
    if not seen_rows or not heldout_rows:
        raise ValueError("Both scored pools are required.")

    seen = pool_metrics(seen_rows, soft_threshold=args.soft_threshold)
    heldout = pool_metrics(heldout_rows, soft_threshold=args.soft_threshold)
    write_jsonl(
        Path(args.output_jsonl),
        [
            {"pool": "test_seen_cue", **seen},
            {"pool": "test_heldout_cue", **heldout},
        ],
    )
    write_summary(
        Path(args.summary_md),
        seen=seen,
        heldout=heldout,
        heldout_rows=heldout_rows,
        soft_threshold=args.soft_threshold,
    )
    print(
        f"[done] seen strict={seen['read_rate_strict']:.4f} soft={seen['read_rate_soft']:.4f} | "
        f"heldout strict={heldout['read_rate_strict']:.4f} "
        f"soft={heldout['read_rate_soft']:.4f} -> {args.summary_md}"
    )


if __name__ == "__main__":
    main()
