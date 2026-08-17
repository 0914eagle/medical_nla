"""Evaluate cue-swap / cue-removal counterfactual readouts.

Input: rows scored by scripts/score_medical_nla_v2_readouts.py for the
counterfactual extraction (each row carries cf_variant / cf_role /
cf_original_cue / cf_replacement_cue). Judgments use soft content-token
recall (threshold configurable), the same instrument as the v4/v5 pools.

Reported:

1. Swap sensitivity (swapped slot):
   - orig variant reads the original cue (baseline)
   - swap variant reads the REPLACEMENT cue (tracks content)
   - swap variant still reads the ORIGINAL cue (context memorization signal)
2. Retained stability: read rate of retained slots under orig vs swap vs
   removed prompts (context-perturbation robustness), paired by slot.
3. Phantom rate: removed variant retained-slot readouts that mention the
   removed cue.

A faithful reader shows: high swap->replacement tracking, low
swap->original persistence, stable retained reads, near-zero phantoms.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cue_position_readouts import content_tokens
from src.jsonl import read_jsonl, write_jsonl


def token_recall(gold: str, text: str) -> float:
    gold_toks = content_tokens(gold)
    if not gold_toks:
        return 0.0
    return len(gold_toks & content_tokens(text)) / len(gold_toks)


def emitted_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("observed_readout") or "").split())


def rate(hits: int, n: int) -> float:
    return hits / n if n else 0.0


def evaluate(rows: list[dict[str, Any]], *, threshold: float) -> dict[str, Any]:
    swapped = [row for row in rows if row.get("cf_role") == "swapped_slot"]
    retained = [row for row in rows if row.get("cf_role") == "retained"]

    swap_stats = {"orig_reads_original": [0, 0], "swap_reads_replacement": [0, 0],
                  "swap_still_reads_original": [0, 0]}
    for row in swapped:
        text = emitted_text(row)
        original = str(row.get("cf_original_cue") or "")
        replacement = str(row.get("cf_replacement_cue") or "")
        if row.get("cf_variant") == "orig":
            swap_stats["orig_reads_original"][1] += 1
            swap_stats["orig_reads_original"][0] += token_recall(original, text) >= threshold
        elif row.get("cf_variant") == "swap":
            swap_stats["swap_reads_replacement"][1] += 1
            swap_stats["swap_reads_replacement"][0] += token_recall(replacement, text) >= threshold
            swap_stats["swap_still_reads_original"][1] += 1
            swap_stats["swap_still_reads_original"][0] += token_recall(original, text) >= threshold

    retained_by_variant: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    phantom = [0, 0]
    paired: dict[tuple[str, int], dict[str, bool]] = defaultdict(dict)
    for row in retained:
        text = emitted_text(row)
        gold = str(row.get("cue_text") or (row.get("cue_targets") or [""])[0])
        variant = str(row.get("cf_variant"))
        ok = token_recall(gold, text) >= threshold
        retained_by_variant[variant][1] += 1
        retained_by_variant[variant][0] += ok
        paired[(str(row.get("base_id")), int(row.get("cf_slot") or -1))][variant] = ok
        if variant == "removed":
            phantom[1] += 1
            removed = str(row.get("cf_removed_cue") or "")
            phantom[0] += token_recall(removed, text) >= threshold

    degraded_swap = degraded_removed = pair_n = 0
    for marks in paired.values():
        if "orig" not in marks:
            continue
        pair_n += 1
        if marks["orig"] and "swap" in marks and not marks["swap"]:
            degraded_swap += 1
        if marks["orig"] and "removed" in marks and not marks["removed"]:
            degraded_removed += 1

    return {
        "threshold": threshold,
        "n_swapped_rows": len(swapped),
        "n_retained_rows": len(retained),
        "orig_reads_original": rate(*swap_stats["orig_reads_original"]),
        "swap_reads_replacement": rate(*swap_stats["swap_reads_replacement"]),
        "swap_still_reads_original": rate(*swap_stats["swap_still_reads_original"]),
        "retained_read_rate_orig": rate(*retained_by_variant["orig"]),
        "retained_read_rate_swap": rate(*retained_by_variant["swap"]),
        "retained_read_rate_removed": rate(*retained_by_variant["removed"]),
        "retained_pairs": pair_n,
        "retained_degraded_under_swap": rate(degraded_swap, pair_n),
        "retained_degraded_under_removal": rate(degraded_removed, pair_n),
        "phantom_rate_removed_cue": rate(*phantom),
    }


def write_summary(path: Path, result: dict[str, Any], examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Cue Counterfactual Faithfulness Summary\n\n")
        f.write(f"Soft token-recall threshold: {result['threshold']}\n\n")
        f.write("## Swap Sensitivity (swapped slot)\n\n")
        f.write(f"- orig_reads_original (baseline): {result['orig_reads_original']:.4f}\n")
        f.write(f"- swap_reads_replacement (tracks content): {result['swap_reads_replacement']:.4f}\n")
        f.write(
            f"- swap_still_reads_original (context-memorization signal): "
            f"{result['swap_still_reads_original']:.4f}\n"
        )
        f.write("\n## Retained-Slot Stability\n\n")
        f.write(f"- read rate under orig: {result['retained_read_rate_orig']:.4f}\n")
        f.write(f"- read rate under swap: {result['retained_read_rate_swap']:.4f}\n")
        f.write(f"- read rate under removal: {result['retained_read_rate_removed']:.4f}\n")
        f.write(f"- paired slots: {result['retained_pairs']}\n")
        f.write(
            f"- degraded (read under orig, lost under swap): "
            f"{result['retained_degraded_under_swap']:.4f}\n"
        )
        f.write(
            f"- degraded (read under orig, lost under removal): "
            f"{result['retained_degraded_under_removal']:.4f}\n"
        )
        f.write("\n## Phantom Check\n\n")
        f.write(
            f"- removed cue mentioned at retained slots: "
            f"{result['phantom_rate_removed_cue']:.4f}\n"
        )
        f.write("\n## Swap Examples (original -> replacement | readout)\n\n")
        f.write("| tracks? | original cue | replacement cue | swap readout |\n")
        f.write("|---|---|---|---|\n")
        for ex in examples[:25]:
            f.write(
                f"| {'Y' if ex['tracks'] else 'N'} | {ex['original'][:60]} | "
                f"{ex['replacement'][:60]} | {ex['readout'][:80]} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", required=True, help="Scored counterfactual readouts.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    rows = list(read_jsonl(args.scored))
    if not rows:
        raise ValueError("No scored rows.")
    result = evaluate(rows, threshold=args.threshold)

    examples = []
    for row in rows:
        if row.get("cf_role") == "swapped_slot" and row.get("cf_variant") == "swap":
            text = emitted_text(row)
            examples.append(
                {
                    "tracks": token_recall(str(row.get("cf_replacement_cue") or ""), text)
                    >= args.threshold,
                    "original": str(row.get("cf_original_cue") or ""),
                    "replacement": str(row.get("cf_replacement_cue") or ""),
                    "readout": text,
                }
            )
    examples.sort(key=lambda ex: ex["tracks"])

    write_jsonl(Path(args.output_jsonl), [result])
    write_summary(Path(args.summary_md), result, examples)
    print(
        f"[done] swap_reads_replacement={result['swap_reads_replacement']:.4f} "
        f"swap_still_reads_original={result['swap_still_reads_original']:.4f} "
        f"phantom={result['phantom_rate_removed_cue']:.4f} -> {args.summary_md}"
    )


if __name__ == "__main__":
    main()
