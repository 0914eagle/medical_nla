"""Convert quote-constrained E4 extraction responses to official predictions.

Predictions and audit JSONL contain generated clinical text and must remain in
private storage. Invalid, invented, or unquoted claims are rejected rather than
silently repaired.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl, write_jsonl


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def extract_json_object(value: Any) -> dict[str, Any]:
    text = str(value or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Extraction response is not a JSON object")
    return parsed


def quoted(quote: Any, source: str) -> str | None:
    candidate = clean_text(quote)
    if not candidate:
        return None
    normalized_source = clean_text(source)
    if candidate not in normalized_source:
        return None
    return candidate


def ontology(candidate_rows: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, list[str]]]:
    names: dict[str, str] = {}
    chains_by_label: dict[str, Counter[tuple[str, ...]]] = defaultdict(Counter)
    for row in candidate_rows:
        label = clean_text(row.get("canonical_pdd"))
        if not label or label == "<unresolved>":
            continue
        names.setdefault(label.casefold(), label)
        annotation_chain = [
            clean_text(item) for item in row.get("annotation_chain") or [] if clean_text(item)
        ]
        if not annotation_chain:
            continue
        chain = (
            clean_text(row.get("disease_category")),
            *reversed(annotation_chain),
        )
        chains_by_label[label.casefold()][tuple(chain)] += 1
    chains = {
        key: list(counts.most_common(1)[0][0])
        for key, counts in chains_by_label.items()
        if counts
    }
    return names, chains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-index", required=True, type=Path)
    parser.add_argument("--judgements", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--audit-jsonl", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--expected-cases", type=int, default=50)
    args = parser.parse_args()

    index_rows = list(read_jsonl(args.private_index))
    index = {str(row["id"]): row for row in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("Duplicate private-index request IDs")
    judgement_rows = list(read_jsonl(args.judgements))
    judgements = {str(row["id"]): row for row in judgement_rows}
    if len(judgements) != len(judgement_rows):
        raise ValueError("Duplicate judgement request IDs")
    missing = set(index) - set(judgements)
    extra = set(judgements) - set(index)
    if missing or extra:
        raise ValueError(
            f"Judgement population mismatch: missing={len(missing)} extra={len(extra)}"
        )

    names, chains = ontology(list(read_jsonl(args.candidate_manifest)))
    methods = sorted({str(row["method"]) for row in index_rows})
    method_counts = Counter(str(row["method"]) for row in index_rows)
    wrong_counts = {
        method: count for method, count in method_counts.items() if count != args.expected_cases
    }
    if wrong_counts:
        raise ValueError(f"Unexpected method populations: {wrong_counts}")

    stats: dict[str, Counter[str]] = {method: Counter() for method in methods}
    extractor_backends: Counter[str] = Counter()
    extractor_models: Counter[str] = Counter()
    audit_rows: list[dict[str, Any]] = []
    seen_paths: dict[str, set[str]] = {method: set() for method in methods}
    for request_id in sorted(index):
        item = index[request_id]
        method = str(item["method"])
        source = str(item.get("method_output") or "")
        judgement = judgements[request_id]
        extractor_backend = clean_text(judgement.get("judge_backend")) or "unknown"
        extractor_model = clean_text(judgement.get("judge_model")) or "unknown"
        extractor_backends[extractor_backend] += 1
        extractor_models[extractor_model] += 1
        relative_path = str(item["source_relative_path"])
        if relative_path in seen_paths[method]:
            raise ValueError(f"Duplicate prediction path for {method}: {relative_path}")
        seen_paths[method].add(relative_path)

        prediction: dict[str, Any] = {}
        accepted_claims: list[dict[str, Any]] = []
        rejected_claims = 0
        parse_error: str | None = None
        diagnosis_label: str | None = None
        diagnosis_quote: str | None = None
        try:
            parsed = extract_json_object(judgement.get("response"))
            raw_label = clean_text(parsed.get("diagnosis_label"))
            raw_quote = quoted(parsed.get("diagnosis_quote"), source)
            canonical_label = names.get(raw_label.casefold()) if raw_label else None
            if canonical_label and raw_quote and canonical_label.casefold() in chains:
                diagnosis_label = canonical_label
                diagnosis_quote = raw_quote
                stats[method]["diagnosis_accepted"] += 1
            elif raw_label or parsed.get("diagnosis_quote"):
                stats[method]["diagnosis_rejected"] += 1

            claims = parsed.get("claims") or []
            if not isinstance(claims, list):
                raise ValueError("claims is not a list")
            used_observations: set[str] = set()
            for claim in claims[:12]:
                if not isinstance(claim, dict):
                    rejected_claims += 1
                    continue
                observation = quoted(claim.get("observation_quote"), source)
                rationale = quoted(claim.get("rationale_quote"), source)
                if not observation or observation.casefold() in used_observations:
                    rejected_claims += 1
                    continue
                used_observations.add(observation.casefold())
                prediction[observation] = [rationale, None, diagnosis_label]
                accepted_claims.append(
                    {"observation": observation, "rationale": rationale}
                )
                stats[method]["observations_accepted"] += 1
                if rationale:
                    stats[method]["rationales_accepted"] += 1
            stats[method]["claims_rejected"] += rejected_claims
        except Exception as exc:
            parse_error = f"{type(exc).__name__}: {exc}"
            stats[method]["parse_errors"] += 1

        prediction["chain"] = chains.get(
            diagnosis_label.casefold() if diagnosis_label else "", [""]
        )
        output_path = args.prediction_root / method / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(prediction, ensure_ascii=False), encoding="utf-8"
        )
        stats[method]["rows"] += 1
        if accepted_claims:
            stats[method]["rows_with_observation"] += 1
        if any(claim["rationale"] for claim in accepted_claims):
            stats[method]["rows_with_rationale"] += 1
        audit_rows.append(
            {
                "id": request_id,
                "base_id": item["base_id"],
                "method": method,
                "source_relative_path": relative_path,
                "diagnosis_label": diagnosis_label,
                "diagnosis_quote": diagnosis_quote,
                "accepted_claims": accepted_claims,
                "rejected_claims": rejected_claims,
                "parse_error": parse_error,
                "extractor_backend": extractor_backend,
                "extractor_model": extractor_model,
            }
        )

    write_jsonl(args.audit_jsonl, audit_rows)
    lines = [
        "# DiReCT E4 Quote-Constrained Extraction",
        "",
        "Private audit: accepted text remains under the restricted data root.",
        "",
        f"- extractor backends: `{dict(extractor_backends)}`",
        f"- extractor models: `{dict(extractor_models)}`",
        "",
        "| method | rows | diagnosis | rows with observation | observations | "
        "rows with rationale | rationales | parse errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method in methods:
        values = stats[method]
        lines.append(
            f"| {method} | {values['rows']} | {values['diagnosis_accepted']} | "
            f"{values['rows_with_observation']} | {values['observations_accepted']} | "
            f"{values['rows_with_rationale']} | {values['rationales_accepted']} | "
            f"{values['parse_errors']} |"
        )
    lines.extend(
        [
            "",
            "Only exact quotes from each method output are retained. The extractor sees no note",
            "or gold annotation. Current Medical-NLA targets contain observations and an answer,",
            "not rationales; Expcom/Expall are therefore exploratory rather than primary metrics.",
            "",
        ]
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[apply] methods={len(methods)} rows={len(audit_rows)} "
        f"prediction_root={args.prediction_root}"
    )
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
