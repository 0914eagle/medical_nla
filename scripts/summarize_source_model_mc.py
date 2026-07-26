"""Summarize one or more source-model multiple-choice output JSONL files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_source_model_mc import write_summary
from src.jsonl import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output-jsonl", default=None)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(read_jsonl(path))
    rows.sort(key=lambda row: str(row.get("id") or ""))
    if args.output_jsonl:
        write_jsonl(args.output_jsonl, rows)
    write_summary(Path(args.summary_md), rows)
    print(f"[done] summarized {len(rows)} rows", flush=True)


if __name__ == "__main__":
    main()
