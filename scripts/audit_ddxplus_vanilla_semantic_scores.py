"""Aggregate-only audit of frozen DDXPlus Vanilla semantic scoring."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.ddxplus_semantic_mapping import extract_json_object
from src.jsonl import read_jsonl


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit(
    prepared_path: Path,
    judgements_path: Path,
    decisions_path: Path,
    mapped_path: Path,
    results_path: Path,
) -> dict[str, Any]:
    prepared = list(read_jsonl(prepared_path))
    judgements = list(read_jsonl(judgements_path))
    decisions = list(read_jsonl(decisions_path))
    mapped = list(read_jsonl(mapped_path))
    results = json.loads(results_path.read_text(encoding="utf-8"))

    total_claims = lexical_claims = lexical_mappings = 0
    for row in prepared:
        for claim in row.get("claims") or []:
            total_claims += 1
            mappings = claim.get("lexical_mappings") or []
            lexical_claims += bool(mappings)
            lexical_mappings += len(mappings)

    raw_result_rows = raw_nonempty_rows = raw_mappings = parse_errors = 0
    raw_evidence = Counter()
    for judgement in judgements:
        try:
            payload = extract_json_object(judgement.get("response"))
        except Exception:  # noqa: BLE001 - aggregate malformed count only
            parse_errors += 1
            continue
        for row in payload.get("results") or []:
            if not isinstance(row, dict):
                continue
            raw_result_rows += 1
            mappings = [item for item in row.get("mappings") or [] if isinstance(item, dict)]
            raw_nonempty_rows += bool(mappings)
            raw_mappings += len(mappings)
            for item in mappings:
                evidence = str(item.get("evidence_id") or "").strip()
                if evidence:
                    raw_evidence[evidence] += 1

    accepted_semantic_claims = 0
    accepted_semantic_mappings = 0
    accepted_evidence = Counter()
    for row in decisions:
        mappings = row.get("mappings") or []
        accepted_semantic_claims += bool(mappings)
        accepted_semantic_mappings += len(mappings)
        for item in mappings:
            accepted_evidence[str(item["evidence_id"])] += 1

    emitted_rows = emitted_claims = 0
    variants = Counter()
    for row in mapped:
        selected = row.get("selected_claims") or []
        emitted_rows += bool(selected)
        emitted_claims += len(selected)
        variants[str(row.get("variant") or "original")] += 1

    if raw_mappings and not (accepted_semantic_mappings or lexical_mappings):
        interpretation = "raw_mappings_rejected_do_not_report_metrics"
    elif not (raw_mappings or lexical_mappings):
        interpretation = "mapper_confirmed_no_ontology_claims"
    else:
        interpretation = "nonzero_mappings_scored"

    return {
        "schema_version": 1,
        "population": results.get("population"),
        "prepared_rows": len(prepared),
        "mapped_rows": len(mapped),
        "judgement_requests": len(judgements),
        "semantic_decision_claims": len(decisions),
        "total_claim_occurrences": total_claims,
        "lexical_claim_occurrences": lexical_claims,
        "lexical_mappings": lexical_mappings,
        "raw_mapper_result_rows": raw_result_rows,
        "raw_mapper_nonempty_rows": raw_nonempty_rows,
        "raw_mapper_mappings": raw_mappings,
        "raw_mapper_parse_errors": parse_errors,
        "accepted_semantic_claims": accepted_semantic_claims,
        "accepted_semantic_mappings": accepted_semantic_mappings,
        "semantic_acceptance_given_raw": ratio(accepted_semantic_mappings, raw_mappings),
        "rows_with_emitted_claims": emitted_rows,
        "emitted_claims": emitted_claims,
        "variants": dict(sorted(variants.items())),
        "unique_raw_evidence_ids": len(raw_evidence),
        "unique_accepted_evidence_ids": len(accepted_evidence),
        "interpretation": interpretation,
        "frozen_metrics": results.get("metrics"),
        "clinical_text_emitted_by_audit": False,
    }


def render(report: dict[str, Any]) -> str:
    lines = [
        "# DDXPlus Vanilla NLA Semantic-Score Audit",
        "",
        "Aggregate-only audit; no generated claim or clinical text is emitted.",
        "",
        f"- readout rows: **{report['mapped_rows']}**",
        f"- mapper requests: **{report['judgement_requests']}**",
        f"- residual semantic decisions: **{report['semantic_decision_claims']}**",
        f"- lexical mappings: **{report['lexical_mappings']}**",
        f"- raw AI mappings: **{report['raw_mapper_mappings']}**",
        f"- accepted AI mappings: **{report['accepted_semantic_mappings']}**",
        f"- emitted claims after case-level deduplication: **{report['emitted_claims']}**",
        f"- rows with at least one emitted claim: **{report['rows_with_emitted_claims']}**",
        f"- interpretation: `{report['interpretation']}`",
        "",
    ]
    if report["interpretation"] == "raw_mappings_rejected_do_not_report_metrics":
        lines.append(
            "Raw mapper decisions were removed by frozen validation. Do not report the "
            "all-zero downstream metrics until the rejection mechanism is audited."
        )
    elif report["interpretation"] == "mapper_confirmed_no_ontology_claims":
        lines.append(
            "Both lexical and method-blind semantic stages selected no frozen ontology "
            "claim. The all-zero Vanilla row is mechanically supported."
        )
    else:
        lines.append(
            "At least one ontology claim survived frozen mapping; use results.json for "
            "the final method metrics."
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--judgements", required=True, type=Path)
    parser.add_argument("--decisions", required=True, type=Path)
    parser.add_argument("--mapped", required=True, type=Path)
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    report = audit(
        args.prepared,
        args.judgements,
        args.decisions,
        args.mapped,
        args.results,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    args.summary_md.write_text(render(report), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
