"""Aggregate blinded semantic judgements for DiReCT E2 readouts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.answer_matching import normalize
from src.jsonl import read_jsonl, write_jsonl


def parse_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    for index, char in enumerate(str(text or "")):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        return value if isinstance(value, dict) else None
    return None


def supported_match(value: Any, readout: str) -> tuple[bool, bool, str]:
    if not isinstance(value, dict):
        return False, False, ""
    match = value.get("match") is True
    evidence = str(value.get("evidence") or "").strip()
    evidence_supported = bool(evidence) and normalize(evidence) in normalize(readout)
    return match and evidence_supported, evidence_supported, evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--judgements", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--manual-primary-jsonl", type=Path, required=True)
    parser.add_argument("--primary-arm", default="default_HS32")
    args = parser.parse_args()

    index_rows = list(read_jsonl(args.index))
    judgement_rows = list(read_jsonl(args.judgements))
    index = {str(row["id"]): row for row in index_rows}
    judged = {str(row["id"]): row for row in judgement_rows}
    if len(index) != len(index_rows):
        raise ValueError("Duplicate index IDs")
    if len(judged) != len(judgement_rows):
        raise ValueError("Duplicate judgement IDs")

    audit: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    parsed_rows = 0
    for identifier, source in index.items():
        judgement = judged.get(identifier)
        obj = parse_object(str(judgement.get("response") or "")) if judgement else None
        parsed_rows += obj is not None
        verdicts: dict[str, Any] = {}
        for target in source["targets"]:
            label = target["label"]
            accepted, evidence_supported, evidence = supported_match(
                obj.get(label) if obj else None, source["readout"]
            )
            verdict = {
                "semantic_match": accepted,
                "evidence_supported": evidence_supported,
                "evidence": evidence,
                "lexical_match": bool(source["lexical"][target["role"]]),
                "target_text": target["text"],
            }
            verdicts[target["role"]] = verdict
            if judgement is not None:
                aggregates[(source["arm"], target["role"])].append(verdict)
        audit.append(
            {
                **source,
                "judge_parsed": obj is not None,
                "judge_model": judgement.get("judge_model") if judgement else None,
                "verdicts": verdicts,
            }
        )

    write_jsonl(args.audit_jsonl, audit)
    primary = [
        {**row, "human_source": None, "human_gold": None, "human_category": None}
        for row in audit
        if row["arm"] == args.primary_arm and row["id"] in judged
    ]
    write_jsonl(args.manual_primary_jsonl, primary)

    lines = [
        "# DiReCT E2 Semantic Readout Audit",
        "",
        f"- indexed requests: **{len(index)}**",
        f"- judgement rows: **{len(judged)}**",
        f"- parsed responses: **{parsed_rows}/{len(judged)}** judged",
        f"- primary manual-audit rows: **{len(primary)}** (`{args.primary_arm}`)",
        "",
        "A semantic match is accepted only when the judge supplies a quote found in the readout.",
        "",
        "| arm | target | n | lexical | semantic | semantic gain |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (arm, role), rows in sorted(aggregates.items()):
        n = len(rows)
        lexical = sum(row["lexical_match"] for row in rows) / n
        semantic = sum(row["semantic_match"] for row in rows) / n
        lines.append(
            f"| {arm} | {role} | {n} | {lexical:.4f} | {semantic:.4f} | {semantic - lexical:+.4f} |"
        )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary] {args.summary_md}")
    print(f"[manual primary] {args.manual_primary_jsonl}")


if __name__ == "__main__":
    main()
