"""Expand distractor-pair prompts into extraction rows.

Each pair is expanded into four rows:
- original_primary: original prompt at the primary diagnostic entity.
- distractor_primary: distractor prompt at the same primary diagnostic entity.
- distractor_entity: distractor prompt at the distractor cue.
- distractor_format: distractor prompt at the final/format token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def read_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def expanded_rows(pair: dict) -> list[dict]:
    common = {
        "base_id": pair["id"],
        "source_id": pair.get("source_id"),
        "primary_target": pair["primary_target"],
        "distractor_target": pair["distractor_target"],
        "correct_dx": pair["correct_dx"],
        "distractor_dx": pair["distractor_dx"],
        "distractor_position": pair.get("distractor_position"),
        "distractor_strength": pair.get("distractor_strength"),
        "notes": pair.get("notes"),
    }
    specs = [
        (
            "original_primary",
            pair["original_prompt"],
            "target_text",
            pair["primary_target"],
            "span_mean",
        ),
        (
            "distractor_primary",
            pair["distractor_prompt"],
            "target_text",
            pair["primary_target"],
            "span_mean",
        ),
        (
            "distractor_entity",
            pair["distractor_prompt"],
            "target_text",
            pair["distractor_target"],
            "span_mean",
        ),
        ("distractor_format", pair["distractor_prompt"], "last_token", None, None),
    ]

    rows = []
    for variant, prompt, mode, target_text, strategy in specs:
        row = {
            **common,
            "id": f"{pair['id']}__{variant}",
            "variant": variant,
            "prompt": prompt,
            "position_mode": mode,
            "target_text": target_text,
            "target_text_strategy": strategy,
        }
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/prompts_medical_distractor_pairs_v1.jsonl"
    )
    parser.add_argument(
        "--output", default="data/prompts_medical_distractor_variants_v1.jsonl"
    )
    args = parser.parse_args()

    pairs = list(read_jsonl(Path(args.input)))
    rows = [row for pair in pairs for row in expanded_rows(pair)]
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} rows from {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
