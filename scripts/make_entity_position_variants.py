from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/prompts_medical_entities.jsonl")
    parser.add_argument("--output", default="data/prompts_medical_position_variants.jsonl")
    parser.add_argument(
        "--entity-strategies",
        nargs="+",
        default=["first_subtoken", "last_subtoken", "span_mean"],
        choices=["first_subtoken", "last_subtoken", "span_mean"],
    )
    parser.add_argument("--include-format", action="store_true", default=True)
    args = parser.parse_args()

    variants: list[dict] = []
    for row in read_jsonl(args.input):
        base_id = row["id"]
        if args.include_format:
            variants.append(
                {
                    "id": f"{base_id}__format_last",
                    "base_id": base_id,
                    "prompt": row["prompt"],
                    "position_family": "format",
                    "position_mode": "last_token",
                    "target_text": None,
                    "target_text_strategy": None,
                }
            )
        for strategy in args.entity_strategies:
            variants.append(
                {
                    "id": f"{base_id}__entity_{strategy}",
                    "base_id": base_id,
                    "prompt": row["prompt"],
                    "position_family": "entity",
                    "position_mode": "target_text",
                    "target_text": row["target_text"],
                    "target_text_occurrence": row.get("target_text_occurrence", 0),
                    "target_text_strategy": strategy,
                }
            )

    write_jsonl(args.output, variants)
    print(f"wrote {len(variants)} rows to {args.output}")


if __name__ == "__main__":
    main()
