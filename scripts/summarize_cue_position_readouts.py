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
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def pool_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    read = [bool(row.get("cue_recall")) for row in rows]
    precisions = [
        float(row["cue_precision"]) for row in rows if row.get("cue_precision") is not None
    ]
    exact = 0
    for row in rows:
        emitted = " ".join(str(row.get("observed_readout") or "").split()).lower()
        gold = " ".join(str(row.get("cue_text") or "").split()).lower()
        if gold and emitted.strip("- ").strip() == gold:
            exact += 1
    return {
        "n": n,
        "parsed_readout": sum(bool(row.get("parsed_readout")) for row in rows),
        "read_rate": sum(read) / n if n else 0.0,
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
) -> None:
    by_cue: dict[str, list[bool]] = {}
    for row in heldout_rows:
        cue = " ".join(str(row.get("cue_text") or "").split()).lower()
        by_cue.setdefault(cue, []).append(bool(row.get("cue_recall")))
    missed_outputs = Counter(
        " ".join(str(row.get("observed_readout") or "-").split())[:120]
        for row in heldout_rows
        if not row.get("cue_recall")
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cue-Position (v4) Readout Summary\n\n")
        f.write("Single-gold-cue rows; read_rate = fraction of rows whose cue was emitted.\n\n")
        f.write("| pool | n | parsed | read_rate | exact_match_rate | mean_precision | mean_items |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|\n")
        for label, metrics in (("test_seen_cue", seen), ("test_heldout_cue", heldout)):
            f.write(
                f"| {label} | {metrics['n']} | {metrics['parsed_readout']} | "
                f"{metrics['read_rate']:.4f} | {metrics['exact_match_rate']:.4f} | "
                f"{metrics['mean_cue_precision']:.4f} | {metrics['mean_items_emitted']:.2f} |\n"
            )
        f.write(
            f"\n- seen_minus_heldout_read_rate: {seen['read_rate'] - heldout['read_rate']:+.4f}\n"
        )

        f.write("\n## Heldout Cues by Read Rate\n\n")
        f.write("| cue | n | read_rate |\n")
        f.write("|---|---:|---:|\n")
        ranked = sorted(
            ((sum(hits) / len(hits), cue, len(hits)) for cue, hits in by_cue.items()),
            reverse=True,
        )
        for rate, cue, count in ranked[:15]:
            f.write(f"| {cue[:100]} | {count} | {rate:.4f} |\n")
        f.write("| ... | | |\n")
        for rate, cue, count in ranked[-15:]:
            f.write(f"| {cue[:100]} | {count} | {rate:.4f} |\n")

        f.write("\n## Most Common Outputs on Missed Heldout Cues\n\n")
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
    args = parser.parse_args()

    seen_rows = list(read_jsonl(args.seen_scored))
    heldout_rows = list(read_jsonl(args.heldout_scored))
    if not seen_rows or not heldout_rows:
        raise ValueError("Both scored pools are required.")

    seen = pool_metrics(seen_rows)
    heldout = pool_metrics(heldout_rows)
    write_jsonl(
        Path(args.output_jsonl),
        [
            {"pool": "test_seen_cue", **seen},
            {"pool": "test_heldout_cue", **heldout},
        ],
    )
    write_summary(Path(args.summary_md), seen=seen, heldout=heldout, heldout_rows=heldout_rows)
    print(
        f"[done] seen read_rate={seen['read_rate']:.4f} "
        f"heldout read_rate={heldout['read_rate']:.4f} -> {args.summary_md}"
    )


if __name__ == "__main__":
    main()
