"""Rate a pool of cue-position readouts, and dump rows for hand labelling.

Prints the read rate two ways -- literal containment and content-word overlap
at several thresholds -- because on paraphrased cues the two disagree by a lot
and the gap is the thing to look at, not either number alone. The design
question these pools answer is whether a heldout cue reads as well as a seen
one, so pass both and they are reported side by side with their difference.

    python scripts/score_cue_position_readouts.py \
        --heldout $ART/results/readout_L24_s17_heldout_ep1.jsonl \
        --seen    $ART/results/readout_L24_s17_seen_ep1.jsonl \
        --dump-sample 60 --dump $ART/reports/readout_L24_s17_sample.tsv
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.cue_readout_scoring import score_readout
from src.jsonl import read_jsonl
from src.nla import extract_explanation

THRESHOLDS = (0.3, 0.5, 0.67, 0.8, 1.0)


def gold_cue(row: dict[str, Any]) -> str:
    cues = row.get("cue_targets")
    if isinstance(cues, list) and cues:
        return str(cues[0])
    if isinstance(cues, str) and cues:
        return cues
    return str(row.get("cue_text") or row.get("target_text") or "")


def readout_body(row: dict[str, Any]) -> str:
    """The text between <observed> tags, or the whole output if untagged."""
    raw = str(row.get("nla_output") or row.get("raw_nla_output") or "")
    for tag in ("observed", "readout"):
        opened, closed = f"<{tag}>", f"</{tag}>"
        if opened in raw and closed in raw:
            return raw.split(opened, 1)[1].split(closed, 1)[0]
    body, _ = extract_explanation(raw)
    return body or raw


def load_pool(path: str) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        gold = gold_cue(row)
        if not gold:
            continue
        rows.append(
            {
                "id": row.get("id"),
                "gold": gold,
                "emitted": readout_body(row),
                "parsed": bool(row.get("parsed_explanation_tag")),
                "cjk": float(row.get("cjk_fraction") or 0.0),
                "diagnosis_name": row.get("diagnosis_name"),
            }
        )
    return rows


def score_pool(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for row in rows:
        row.update(score_readout(row["emitted"], row["gold"]))
    return rows


def report(name: str, rows: list[dict[str, Any]]) -> dict[str, float]:
    n = len(rows)
    if not n:
        print(f"{name}: no rows")
        return {}
    out = {
        "n": n,
        "parsed": sum(r["parsed"] for r in rows) / n,
        "cjk": sum(r["cjk"] > 0.05 for r in rows) / n,
        "exact": sum(r["exact"] for r in rows) / n,
    }
    for threshold in THRESHOLDS:
        out[f"f1>={threshold}"] = sum(r["f1"] >= threshold for r in rows) / n
    out["mean_f1"] = sum(r["f1"] for r in rows) / n
    print(f"\n{name}  (n={n:,})")
    print(f"  parsed the tag        {out['parsed']:.4f}")
    print(f"  non-latin output      {out['cjk']:.4f}")
    print(f"  literal containment   {out['exact']:.4f}   <- the v2 rule")
    for threshold in THRESHOLDS:
        print(f"  overlap f1 >= {threshold:.2f}    {out[f'f1>={threshold}']:.4f}")
    print(f"  mean overlap f1       {out['mean_f1']:.4f}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--seen")
    parser.add_argument("--dump", help="TSV of sampled rows, with a blank verdict column.")
    parser.add_argument("--dump-sample", type=int, default=0)
    parser.add_argument("--dump-seed", type=int, default=17)
    args = parser.parse_args()

    pools = {"heldout": load_pool(args.heldout)}
    if args.seen:
        pools["seen"] = load_pool(args.seen)

    summaries = {}
    for name, rows in pools.items():
        summaries[name] = report(name, score_pool(rows))

    if "seen" in summaries and summaries["seen"] and summaries["heldout"]:
        print("\nheldout minus seen (the number the design turns on):")
        for key in ("exact", "f1>=0.5", "f1>=0.8", "mean_f1"):
            gap = summaries["heldout"][key] - summaries["seen"][key]
            print(f"  {key:<14} {gap:+.4f}")
        print(
            "\n  A gap near zero means the readout is not reciting supervised "
            "strings.\n  A large negative gap means the seen-cue rate was "
            "memorization."
        )

    if args.dump and args.dump_sample:
        rows = [dict(row, pool=name) for name, pool in pools.items() for row in pool]
        sample = random.Random(args.dump_seed).sample(rows, min(args.dump_sample, len(rows)))
        sample.sort(key=lambda row: row["f1"])
        path = Path(args.dump)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("verdict\tpool\tf1\texact\tgold\tread\tid\n")
            for row in sample:
                gold = " ".join(str(row["gold"]).split())
                read = " ".join(str(row["best_item"]).split())
                f.write(
                    f"\t{row['pool']}\t{row['f1']:.2f}\t{int(row['exact'])}\t"
                    f"{gold}\t{read}\t{row['id']}\n"
                )
        print(
            f"\nwrote {len(sample)} rows to {path}\n"
            "  Fill the verdict column with A (same finding), B (partly), "
            "C (wrong), D (empty/refused).\n"
            "  Sorted by overlap so the threshold can be read off where A stops."
        )


if __name__ == "__main__":
    main()
