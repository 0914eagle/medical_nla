"""Expand specificity-shift cases into NLA extraction rows.

The experiment asks whether NLA output shifts from a nonspecific cue meaning
to a case-specific diagnosis after discriminating cues are added, and where
that shift appears.
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


def case_id(case: dict) -> str:
    return case.get("id") or case["base_id"]


def specific_targets(case: dict) -> list[str]:
    if "specific_targets" in case:
        return list(case["specific_targets"])
    return [case["specific_cue_1"], case["specific_cue_2"], case["specific_cue_3"]]


def diagnostic_shift(case: dict) -> str:
    if "diagnostic_shift" in case:
        return case["diagnostic_shift"]
    return f"{case['nonspecific_target'].replace(' ', '_')}_to_{case['specific_expected'].replace(' ', '_')}"


def common_fields(case: dict) -> dict:
    fields = {
        "base_id": case_id(case),
        "category": case.get("category"),
        "nonspecific_target": case["nonspecific_target"],
        "specific_targets": specific_targets(case),
        "nonspecific_expected": case["nonspecific_expected"],
        "specific_expected": case["specific_expected"],
        "diagnostic_shift": diagnostic_shift(case),
        "notes": case.get("notes"),
    }
    for key in ("specific_aliases", "nonspecific_aliases", "diagnosis_aliases"):
        if key in case:
            fields[key] = case[key]
    return fields


def extraction_row(
    case: dict,
    *,
    variant: str,
    condition: str,
    prompt: str,
    target_text: str | None,
    target_role: str,
    cue_index: int | None = None,
    specific_target: str | None = None,
) -> dict:
    return {
        **common_fields(case),
        "id": f"{case_id(case)}__{variant}",
        "variant": variant,
        "condition": condition,
        "target_role": target_role,
        "cue_index": cue_index,
        "specific_target": specific_target,
        "prompt": prompt,
        "position_mode": "last_token" if target_text is None else "target_text",
        "target_text": target_text,
        "target_text_strategy": None if target_text is None else "span_mean",
    }


def expanded_rows(case: dict) -> list[dict]:
    rows = [
        extraction_row(
            case,
            variant="nonspecific_alone_cue",
            condition="nonspecific_alone",
            prompt=case["nonspecific_prompt"],
            target_text=case["nonspecific_target"],
            target_role="nonspecific_cue",
        ),
        extraction_row(
            case,
            variant="nonspecific_alone_format",
            condition="nonspecific_alone",
            prompt=case["nonspecific_prompt"],
            target_text=None,
            target_role="format",
        ),
        extraction_row(
            case,
            variant="specific_full_nonspecific_cue",
            condition="specific_full",
            prompt=case["specific_prompt"],
            target_text=case["nonspecific_target"],
            target_role="nonspecific_cue",
        ),
    ]
    for idx, target in enumerate(specific_targets(case), start=1):
        rows.append(
            extraction_row(
                case,
                variant=f"specific_full_specific_cue_{idx}",
                condition="specific_full",
                prompt=case["specific_prompt"],
                target_text=target,
                target_role="specific_cue",
                cue_index=idx,
                specific_target=target,
            )
        )
    rows.append(
        extraction_row(
            case,
            variant="specific_full_format",
            condition="specific_full",
            prompt=case["specific_prompt"],
            target_text=None,
            target_role="format",
        )
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/prompts_medical_specificity_cases_v1.jsonl"
    )
    parser.add_argument(
        "--output", default="data/prompts_medical_specificity_variants_v1.jsonl"
    )
    args = parser.parse_args()

    cases = list(read_jsonl(Path(args.input)))
    rows = [row for case in cases for row in expanded_rows(case)]
    write_jsonl(Path(args.output), rows)
    print(f"wrote {len(rows)} rows from {len(cases)} cases to {args.output}")


if __name__ == "__main__":
    main()
