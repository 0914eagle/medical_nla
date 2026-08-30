"""Prepare and summarize a private 20-case audit of zero Vanilla mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ddxplus_semantic_mapping import extract_json_object
from src.jsonl import read_jsonl, write_jsonl

CATEGORIES = {
    "no_clinical_content",
    "generic_clinical_only",
    "diagnosis_only",
    "unrelated_clinical_content",
    "expected_finding_paraphrase_missed",
    "expected_value_paraphrase_missed",
    "malformed_or_empty",
}


def stable_key(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{value}".encode()).hexdigest()


def readout_text(row: dict[str, Any]) -> str:
    return str(row.get("nla_output") or row.get("observed") or row.get("response") or "")


def prepare(args: argparse.Namespace) -> None:
    manifest = {
        str(row["id"]): row
        for row in read_jsonl(args.manifest)
        if str(row.get("variant") or "original") == "original"
    }
    readouts = {str(row["id"]): row for row in read_jsonl(args.readouts)}
    if set(manifest) - set(readouts):
        raise ValueError("Original manifest rows are missing Vanilla readouts")
    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row_id, row in manifest.items():
        by_diagnosis[str(row.get("diagnosis_id") or "<missing>")].append(row)
    if len(by_diagnosis) < args.cases:
        raise ValueError("Not enough diagnoses for one-case-per-diagnosis sampling")

    diagnoses = sorted(by_diagnosis, key=lambda value: stable_key(value, args.seed))
    selected = []
    for diagnosis in diagnoses[: args.cases]:
        rows = sorted(
            by_diagnosis[diagnosis],
            key=lambda row: stable_key(str(row["id"]), args.seed),
        )
        selected.append(rows[0])

    requests = []
    private = []
    for index, row in enumerate(selected):
        row_id = str(row["id"])
        output = readout_text(readouts[row_id])
        evidence = [str(value) for value in (row.get("cue_evidence_ids") or [])]
        values = [
            None if value is None or str(value).strip() == "" else str(value)
            for value in (row.get("cue_value_ids") or [])
        ]
        texts = [str(value) for value in (row.get("cue_targets") or [])]
        if not (len(evidence) == len(values) == len(texts)):
            raise ValueError(f"Misaligned expected cues for {row_id}")
        expected = [
            {"evidence_id": e, "value_id": v, "reference_phrase": text}
            for e, v, text in zip(evidence, values, texts, strict=True)
        ]
        audit_id = f"vanilla_zero_audit_{index:02d}_{stable_key(row_id, args.seed)[:12]}"
        prompt = f"""You are performing a post-hoc measurement audit. This does not change the frozen score.

Determine why the Vanilla NLA readout below received no DDXPlus ontology mapping. Compare only the readout with the expected cue references. Do not infer an unmentioned finding. A paraphrase match requires an exact continuous supporting quote copied from the readout.

READOUT:
{output}

EXPECTED CUE REFERENCES:
{json.dumps(expected, ensure_ascii=False, indent=2)}

Return JSON only:
{{
  "category": "one of: no_clinical_content, generic_clinical_only, diagnosis_only, unrelated_clinical_content, expected_finding_paraphrase_missed, expected_value_paraphrase_missed, malformed_or_empty",
  "matched_expected_evidence_ids": ["IDs only when supported by a readout quote"],
  "supporting_quotes": ["exact continuous readout quotes"],
  "reason": "brief explanation"
}}

Use expected_finding_paraphrase_missed or expected_value_paraphrase_missed only when the frozen mapper should reasonably have mapped an explicit readout phrase. Otherwise select the dominant failure mode."""
        requests.append({"id": audit_id, "prompt": prompt})
        private.append(
            {
                "id": audit_id,
                "source_id": row_id,
                "diagnosis_id": row.get("diagnosis_id"),
                "readout": output,
                "expected": expected,
                "readout_characters": len(output),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "requests.jsonl", requests)
    write_jsonl(args.out_dir / "private_index.jsonl", private)
    report = {
        "schema_version": 1,
        "cases": len(selected),
        "diagnoses": len({row.get("diagnosis_id") for row in selected}),
        "seed": args.seed,
        "sampling": (
            f"one stable-hash case from each of {args.cases} stable-hash diagnoses"
        ),
        "locked_score_changed": False,
    }
    (args.out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[prepare-zero-audit] cases={len(selected)} out={args.out_dir}")


def finalize(args: argparse.Namespace) -> None:
    private = {str(row["id"]): row for row in read_jsonl(args.private_index)}
    judgements = {str(row["id"]): row for row in read_jsonl(args.judgements)}
    if set(private) != set(judgements):
        raise ValueError(
            f"Audit population mismatch: missing={len(set(private)-set(judgements))} "
            f"extra={len(set(judgements)-set(private))}"
        )
    categories = Counter()
    quote_valid = 0
    matched_cases = 0
    private_rows = []
    for row_id in sorted(private):
        source = private[row_id]
        parsed = extract_json_object(judgements[row_id].get("response"))
        category = str(parsed.get("category") or "")
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category for {row_id}: {category!r}")
        expected_ids = {str(item["evidence_id"]) for item in source["expected"]}
        matched = {str(value) for value in parsed.get("matched_expected_evidence_ids") or []}
        if not matched <= expected_ids:
            raise ValueError(f"Auditor emitted non-reference evidence for {row_id}")
        quotes = [str(value).strip() for value in parsed.get("supporting_quotes") or []]
        normalized_output = " ".join(str(source["readout"]).split()).casefold()
        valid_quotes = [
            quote for quote in quotes
            if quote and " ".join(quote.split()).casefold() in normalized_output
        ]
        if len(valid_quotes) != len(quotes):
            raise ValueError(f"Auditor emitted a non-verbatim quote for {row_id}")
        if matched and not valid_quotes:
            raise ValueError(f"Auditor matched evidence without a quote for {row_id}")
        if category.startswith("expected_") and not matched:
            raise ValueError(f"Mapper-miss category has no matched evidence for {row_id}")
        categories[category] += 1
        quote_valid += bool(valid_quotes)
        matched_cases += bool(matched)
        private_rows.append(
            {
                **source,
                "audit_category": category,
                "matched_expected_evidence_ids": sorted(matched),
                "supporting_quotes": valid_quotes,
                "reason": str(parsed.get("reason") or ""),
                "auditor_model": judgements[row_id].get("judge_model"),
            }
        )
    write_jsonl(args.out_dir / "private_case_audit.jsonl", private_rows)
    lengths = [int(row["readout_characters"]) for row in private.values()]
    diagnosis_count = len({str(row.get("diagnosis_id")) for row in private.values()})
    mapper_miss_cases = sum(
        categories[name]
        for name in (
            "expected_finding_paraphrase_missed",
            "expected_value_paraphrase_missed",
        )
    )
    report = {
        "schema_version": 1,
        "cases": len(private),
        "diagnoses": diagnosis_count,
        "categories": dict(sorted(categories.items())),
        "mapper_miss_cases": mapper_miss_cases,
        "mapper_miss_rate": mapper_miss_cases / len(private),
        "expected_cue_match_cases": matched_cases,
        "expected_cue_match_rate": matched_cases / len(private),
        "cases_with_valid_supporting_quote": quote_valid,
        "readout_character_median": statistics.median(lengths),
        "auditor_models": sorted(
            {str(row.get("judge_model") or "") for row in judgements.values()}
        ),
        "locked_score_changed": False,
        "clinical_text_emitted_in_summary": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DDXPlus Vanilla Zero-Score Private Sample Audit",
        "",
        "Post-hoc diagnosis-stratified 20-case audit. No clinical text is emitted.",
        "The frozen locked score is not changed by this audit.",
        "",
        f"- cases / diagnoses: **{report['cases']} / {report['diagnoses']}**",
        f"- auditor model: `{', '.join(report['auditor_models'])}`",
        f"- possible frozen-mapper misses: **{mapper_miss_cases}/{len(private)} "
        f"({mapper_miss_cases/len(private):.4f})**",
        f"- expected-cue paraphrase match: **{matched_cases}/{len(private)} "
        f"({matched_cases/len(private):.4f})**",
        f"- median readout characters: **{report['readout_character_median']}**",
        "",
        "| failure category | n | proportion |",
        "|---|---:|---:|",
    ]
    for category in sorted(CATEGORIES):
        count = categories[category]
        lines.append(f"| {category} | {count} | {count/len(private):.4f} |")
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--manifest", required=True, type=Path)
    prep.add_argument("--readouts", required=True, type=Path)
    prep.add_argument("--out-dir", required=True, type=Path)
    prep.add_argument("--cases", type=int, default=20)
    prep.add_argument("--seed", type=int, default=17)
    final = sub.add_parser("finalize")
    final.add_argument("--private-index", required=True, type=Path)
    final.add_argument("--judgements", required=True, type=Path)
    final.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        finalize(args)


if __name__ == "__main__":
    main()
