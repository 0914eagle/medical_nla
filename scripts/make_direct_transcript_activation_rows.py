"""Build P0/P1/P2 extraction rows from one DiReCT source CoT run.

P0 is the final token of the source prompt. P1 is the final subtoken of the
assistant's last ``The answer is`` marker, before the diagnosis. P2 is the
final subtoken of the parsed diagnosis. P1/P2 teacher-force the exact generated
assistant response, so all positions belong to one source run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.case_prompts import ANSWER_CUE


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-answers", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    extraction_rows: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for source in read_jsonl(args.source_answers):
        prompt = str(source.get("prompt") or "")
        response = str(source.get("response") or "")
        answer = str(source.get("answer") or "").strip()
        if not prompt or not response:
            skipped["missing_prompt_or_response"] += 1
            continue
        if not answer:
            skipped["unparsed_answer"] += 1
            continue
        if ANSWER_CUE.casefold() not in response.casefold():
            skipped["missing_answer_boundary"] += 1
            continue

        base_id = str(source.get("base_id") or source["id"])
        common = {
            "base_id": base_id,
            "prompt": prompt,
            "split": source.get("split"),
            "source": "direct",
            "disease_category": source.get("disease_category"),
            "canonical_pdd": source.get("canonical_pdd") or source.get("diagnosis_name"),
            "diagnosis_name": source.get("diagnosis_name"),
            "diagnosis_aliases": source.get("diagnosis_aliases") or [],
            "patient_group": source.get("patient_group"),
            "source_correct": source.get("source_correct"),
            "answer_forced": source.get("answer_forced"),
            "diagnosis_alias_in_reasoning": source.get(
                "diagnosis_alias_in_reasoning"
            ),
            "gold_alias_in_reasoning": source.get("gold_alias_in_reasoning"),
        }
        extraction_rows.append(
            {
                **common,
                "id": f"{base_id}__p0_prompt_boundary",
                "position_label": "P0_prompt_boundary",
                "position_family": "P0",
                "position_mode": "last_token",
            }
        )
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
        extraction_rows.append(
            {
                **common,
                "id": f"{base_id}__p1_answer_boundary",
                "chat_messages": messages,
                "position_label": "P1_answer_boundary",
                "position_family": "P1",
                "position_mode": "target_text",
                "target_text": ANSWER_CUE,
                "target_text_occurrence": -1,
                "target_text_strategy": "last_subtoken",
            }
        )
        extraction_rows.append(
            {
                **common,
                "id": f"{base_id}__p2_diagnosis_token",
                "chat_messages": messages,
                "position_label": "P2_diagnosis_token",
                "position_family": "P2",
                "position_mode": "target_text",
                "target_text": answer,
                "target_text_occurrence": -1,
                "target_text_strategy": "last_subtoken",
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for row in extraction_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    n_cases = len(extraction_rows) // 3
    lines = [
        "# DiReCT E1 Transcript Activation Row Summary",
        "",
        "Aggregate-only summary. The row JSONL contains restricted text.",
        "",
        f"- source rows converted: **{n_cases}**",
        f"- extraction rows: **{len(extraction_rows)}**",
        f"- skipped: `{dict(skipped)}`",
        "- positions per converted source row: **P0, P1, P2**",
    ]
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"[rows] cases={n_cases} extraction_rows={len(extraction_rows)} "
        f"skipped={dict(skipped)}"
    )
    print(f"[output] {args.output_jsonl}")


if __name__ == "__main__":
    main()
