"""Keep only the manifest rows belonging to one SFT split, and say what they are.

An activation extraction covers the whole corpus, because the split is decided
downstream when the SFT rows are built. Reading out train rows with an adapter
trained on them and reporting a description rate measures memorisation, which
is the failure a held-out split exists to prevent -- and nothing in the file
names would have shown it.

Also reports whether the kept prompts carry a referring note, because that
decides which question the readouts can answer: note-free states speak to the
instrument (does the readout describe real clinical prose), note-bearing ones
to attribution (what did the note do). A manifest cannot be repurposed between
the two, so the answer is printed rather than assumed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Extraction manifest, all splits.")
    parser.add_argument("--split", required=True, help="sft_{train,val,test}.jsonl to keep.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--key", default="base_id", help="Field joining the two files.")
    args = parser.parse_args()

    keep = {str(r.get(args.key) or "") for r in read_jsonl(args.split)} - {""}
    if not keep:
        raise SystemExit(f"no '{args.key}' values in {args.split}")

    rows = [r for r in read_jsonl(args.manifest) if str(r.get(args.key) or "") in keep]
    if not rows:
        raise SystemExit(
            f"no manifest row matched a '{args.key}' from the split -- are the two "
            "files from the same corpus build?"
        )
    write_jsonl(Path(args.output), rows)

    arms = Counter(str(r.get("hint_variant") or "-") for r in rows)
    noted = sum(1 for r in rows
                if "referring note" in str(r.get("prompt") or "").lower())
    print(f"[split] {args.key}s in split {len(keep):,}   manifest rows kept "
          f"{len(rows):,} -> {args.output}")
    print(f"[split] arms={dict(arms)}   prompts naming a referring note: {noted:,}")
    if noted == 0:
        print("[split] NOTE-FREE -> instrument question only. Unlocks Table 1's "
              "MCR description-rate row.")
        print("[split] Does NOT unlock Table 3b's MCR readout cell or the MCR r5 "
              "rung: those need answer-position activations extracted over the "
              "MCR hint cases' wrong arm, which is a separate run.")
    else:
        print("[split] note-bearing -> attribution is available too.")


if __name__ == "__main__":
    main()
