"""Expand distractor-pair prompts into extraction rows.

By default each pair is expanded into after-primary, before-primary, and
before-neutral conditions. The after-primary condition matches the original
v1 distractor run. The before-neutral condition controls for moving the primary
target later in the prompt without adding distractor semantics.

Each pair is expanded into nine rows:
- original_primary: original prompt at the primary diagnostic entity.
- after_distractor_primary: after-primary distractor prompt at primary entity.
- after_distractor_entity: after-primary distractor prompt at distractor cue.
- after_distractor_format: after-primary distractor prompt at final/format token.
- before_distractor_primary: before-primary distractor prompt at primary entity.
- before_distractor_entity: before-primary distractor prompt at distractor cue.
- before_distractor_format: before-primary distractor prompt at final/format token.
- before_neutral_primary: before-neutral prompt at primary entity.
- before_neutral_format: before-neutral prompt at final/format token.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


NEUTRAL_PREFIX = "The patient arrived at the clinic in the morning."


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


def inserted_distractor_sentence(pair: dict) -> str:
    """Return the sentence added in distractor_prompt.

    Pair files may define `distractor_sentence` explicitly. Otherwise this
    assumes the distractor prompt was formed by inserting one sentence before
    the final question sentence, which is how v1 pairs were authored.
    """
    if pair.get("distractor_sentence"):
        return str(pair["distractor_sentence"]).strip()
    original = pair["original_prompt"].strip()
    distractor = pair["distractor_prompt"].strip()
    question = original[original.rfind(".") + 1 :].strip()
    prefix = original[: original.rfind(".") + 1].strip()
    if question and distractor.startswith(prefix) and distractor.endswith(question):
        inserted = distractor[len(prefix) : len(distractor) - len(question)].strip()
        return inserted.strip()
    marker = pair["distractor_target"]
    sentences = [s.strip() for s in distractor.split(".") if s.strip()]
    for sentence in sentences:
        if marker.lower() in sentence.lower():
            return sentence + "."
    raise ValueError(f"Could not infer distractor sentence for {pair['id']}")


def prepend_context(context_sentence: str, prompt: str) -> str:
    sentence = context_sentence.strip()
    if not sentence.endswith((".", "?", "!")):
        sentence += "."
    return sentence + " " + prompt.strip()


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
    before_distractor_prompt = prepend_context(
        inserted_distractor_sentence(pair), pair["original_prompt"]
    )
    before_neutral_prompt = prepend_context(
        pair.get("neutral_sentence", NEUTRAL_PREFIX), pair["original_prompt"]
    )
    specs = [
        ("original_primary", "original", "none", pair["original_prompt"], "target_text", pair["primary_target"], "span_mean"),
        ("after_distractor_primary", "after_distractor", "after_primary", pair["distractor_prompt"], "target_text", pair["primary_target"], "span_mean"),
        ("after_distractor_entity", "after_distractor", "after_primary", pair["distractor_prompt"], "target_text", pair["distractor_target"], "span_mean"),
        ("after_distractor_format", "after_distractor", "after_primary", pair["distractor_prompt"], "last_token", None, None),
        ("before_distractor_primary", "before_distractor", "before_primary", before_distractor_prompt, "target_text", pair["primary_target"], "span_mean"),
        ("before_distractor_entity", "before_distractor", "before_primary", before_distractor_prompt, "target_text", pair["distractor_target"], "span_mean"),
        ("before_distractor_format", "before_distractor", "before_primary", before_distractor_prompt, "last_token", None, None),
        ("before_neutral_primary", "before_neutral", "before_primary", before_neutral_prompt, "target_text", pair["primary_target"], "span_mean"),
        ("before_neutral_format", "before_neutral", "before_primary", before_neutral_prompt, "last_token", None, None),
    ]

    rows = []
    for variant, condition, condition_order, prompt, mode, target_text, strategy in specs:
        row = {
            **common,
            "id": f"{pair['id']}__{variant}",
            "variant": variant,
            "condition": condition,
            "condition_order": condition_order,
            "insertion_type": pair.get("insertion_type", "prepend_sentence"),
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
        "--output", default="data/prompts_medical_distractor_order_variants_v1.jsonl"
    )
    args = parser.parse_args()

    pairs = list(read_jsonl(Path(args.input)))
    rows = [row for pair in pairs for row in expanded_rows(pair)]
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} rows from {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
