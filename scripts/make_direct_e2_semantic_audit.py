"""Build a blinded semantic audit over every DiReCT E2 P0 readout.

The current lexical scorer is intentionally strict and can miss abbreviations
or clinical synonyms. This builder gives a separate judge the readout and
three shuffled targets, but not the patient note or the target roles. A claimed
match must include an exact quote from the readout so unsupported judge guesses
can be rejected downstream.
"""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path
from typing import Any

from scripts.summarize_direct_e2_readouts import parse_named_path
from src.answer_matching import is_correct
from src.jsonl import read_jsonl, write_jsonl


TEMPLATE = """You are auditing a natural-language readout of a medical model's internal state.

Readout:
---
{readout}
---

For each target below, decide whether the readout EXPLICITLY names that target,
an unambiguous abbreviation, or a true clinical synonym. Do not infer a target
only because the described findings would be compatible with it. For a target
marked CATEGORY, a clearly named subtype of that category counts.

{targets}

Return exactly one JSON object with keys {keys}. Each value must have:
- "match": true or false
- "evidence": an exact quote from the readout when true, otherwise ""

No markdown and no explanation outside the JSON object."""


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def shuffled_targets(
    row: dict[str, Any], source: dict[str, Any], *, arm: str, seed: int
) -> list[dict[str, str]]:
    targets = [
        {"role": "source_answer", "kind": "DIAGNOSIS", "text": str(source.get("answer") or "")},
        {
            "role": "gold_pdd",
            "kind": "DIAGNOSIS",
            "text": str(row.get("canonical_pdd") or row.get("diagnosis_name") or ""),
        },
        {"role": "category", "kind": "CATEGORY", "text": str(row.get("disease_category") or "")},
    ]
    if any(not target["text"].strip() for target in targets):
        raise ValueError(f"Missing semantic target for {case_id(row)}")
    digest = hashlib.sha256(f"{seed}:{arm}:{case_id(row)}".encode()).digest()
    random.Random(int.from_bytes(digest[:8], "big")).shuffle(targets)
    for label, target in zip(("A", "B", "C"), targets):
        target["label"] = label
    return targets


def build_prompt(readout: str, targets: list[dict[str, str]]) -> str:
    rendered = "\n".join(
        f"- {target['label']} ({target['kind']}): {target['text']}" for target in targets
    )
    keys = ", ".join(f'"{target["label"]}"' for target in targets)
    return TEMPLATE.format(readout=readout.strip(), targets=rendered, keys=keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--source-answers", nargs="+", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    for path in args.source_answers:
        for row in read_jsonl(path):
            identifier = case_id(row)
            if identifier in sources:
                raise ValueError(f"Duplicate source answer: {identifier}")
            sources[identifier] = row

    requests: list[dict[str, str]] = []
    index: list[dict[str, Any]] = []
    expected_ids: set[str] | None = None
    for value in args.readout:
        arm, path = parse_named_path(value)
        rows = list(read_jsonl(path))
        ids = {case_id(row) for row in rows}
        if len(ids) != len(rows):
            raise ValueError(f"Missing or duplicate case IDs in {path}")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Readout arm {arm} does not contain the same case IDs")
        for row in rows:
            identifier = case_id(row)
            if identifier not in sources:
                raise ValueError(f"No source answer for {identifier}")
            readout = str(row.get("nla_output") or "").strip()
            if not readout:
                raise ValueError(f"Empty readout for {arm}/{identifier}")
            targets = shuffled_targets(row, sources[identifier], arm=arm, seed=args.seed)
            request_id = f"{identifier}__{arm}__semantic_audit"
            requests.append({"id": request_id, "prompt": build_prompt(readout, targets)})
            index.append(
                {
                    "id": request_id,
                    "base_id": identifier,
                    "arm": arm,
                    "readout": readout,
                    "targets": targets,
                    "lexical": {
                        target["role"]: is_correct(
                            readout,
                            target["text"],
                            list(row.get("diagnosis_aliases") or [])
                            if target["role"] == "gold_pdd"
                            else [],
                        )
                        for target in targets
                    },
                }
            )

    write_jsonl(args.requests, requests)
    write_jsonl(args.index, index)
    print(f"[audit] arms={len(args.readout)} cases={len(expected_ids or set())} requests={len(requests)}")
    print(f"[requests] {args.requests}")
    print(f"[index] {args.index}")


if __name__ == "__main__":
    main()
