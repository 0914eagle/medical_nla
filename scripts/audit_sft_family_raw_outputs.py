"""Build and summarize private raw-output audits for Medical-NLA SFT families.

The DiReCT bundle contains restricted clinical annotations and must never leave
the private data root.  The DDXPlus bundle is public-data derived, but follows
the same local-bundle/aggregate-summary convention to prevent handling errors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ddxplus_semantic_mapping import extract_json_object
from src.jsonl import read_jsonl, write_jsonl
from scripts.score_direct_official_eval import score_record


DIRECT_FIELDS = (
    "physician_observation_supported",
    "disease_template_only",
    "boilerplate_or_format_only",
    "unsupported_patient_claim",
    "extractor_missed_explicit_aligned_claim",
)
DELETION_FIELDS = (
    "original_target_mentioned",
    "deleted_target_phantom",
    "untouched_finding_retained",
    "unsupported_patient_claim",
)
VALUE_FIELDS = (
    "old_value_mentioned_original",
    "new_value_mentioned_after_edit",
    "old_value_persists_after_edit",
    "clean_value_switch",
    "untouched_finding_retained",
    "unsupported_patient_claim",
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def stable_key(namespace: str, value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_values(values: list[str] | set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def sha256_tree(root: Path) -> str:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"No files under {root}")
    payload = "\n".join(
        f"{path.relative_to(root)}\0{sha256_file(path)}" for path in files
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def require_restricted_direct(path: Path) -> None:
    parts = path.resolve().parts
    if not any(
        parts[index : index + 2] == ("restricted", "direct")
        for index in range(len(parts) - 1)
    ):
        raise ValueError(
            f"DiReCT private output must be under a restricted/direct path: {path}"
        )


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def output_text(row: dict[str, Any]) -> str:
    return str(
        row.get("nla_output")
        or row.get("raw_nla_output")
        or row.get("response")
        or ""
    ).strip()


def source_text(row: dict[str, Any]) -> str:
    response = str(row.get("response") or "").strip()
    if response:
        return response
    reasoning = str(row.get("reasoning") or "").strip()
    answer = clean(row.get("answer"))
    if reasoning and answer:
        return f"{reasoning}\nThe answer is: {answer}"
    return reasoning or answer


def index_unique(
    rows: list[dict[str, Any]], label: str, *, key: Callable[[dict[str, Any]], str] = base_id
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = key(row)
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate {label} ID: {identifier!r}")
        result[identifier] = row
    return result


def parse_direct_method(value: str) -> dict[str, Any]:
    parts = value.split("|")
    if len(parts) != 5:
        raise argparse.ArgumentTypeError(
            "Expected LABEL|READOUT_OR_DASH|SEMANTIC_ROOT|SEMANTIC_METHOD|SOURCE_FILTER_OR_DASH"
        )
    label, readout, semantic_root, semantic_method, source_filter = parts
    if not all((label, semantic_root, semantic_method)):
        raise argparse.ArgumentTypeError("Direct method fields cannot be empty")
    return {
        "label": label,
        "readout": None if readout == "-" else Path(readout),
        "semantic_root": Path(semantic_root),
        "semantic_method": semantic_method,
        "source_filter": None if source_filter == "-" else source_filter,
    }


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label or not raw_path:
        raise argparse.ArgumentTypeError("Expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def load_direct_semantic(
    root: Path, method: str
) -> dict[str, dict[str, Any]]:
    private_index = list(read_jsonl(root / "private_index.jsonl"))
    extraction_audit = {
        str(row["id"]): row for row in read_jsonl(root / "private_extraction_audit.jsonl")
    }
    judgements = {
        str(row["id"]): row for row in read_jsonl(root / "extraction_judgements.jsonl")
    }
    result: dict[str, dict[str, Any]] = {}
    for item in private_index:
        if str(item.get("method")) != method:
            continue
        request_id = str(item["id"])
        identifier = str(item["base_id"])
        if identifier in result:
            raise ValueError(f"Duplicate semantic row {root}/{method}/{identifier}")
        relative = Path(str(item["source_relative_path"]))
        evaluation_path = root / "evaluations" / method / relative
        result[identifier] = {
            "request_id": request_id,
            "method_output": str(item.get("method_output") or ""),
            "accepted_extraction": extraction_audit[request_id],
            "raw_extractor_response": str(judgements[request_id].get("response") or ""),
            "extractor_model": judgements[request_id].get("judge_model"),
            "official_evaluation": json.loads(evaluation_path.read_text(encoding="utf-8")),
        }
    if not result:
        raise ValueError(f"No semantic rows for {method} under {root}")
    return result


def direct_prompt(
    observations: list[str], opaque_methods: list[dict[str, Any]]
) -> str:
    rendered = []
    for item in opaque_methods:
        claims = item["accepted_extraction"].get("accepted_claims") or []
        rendered.append(
            f"<output id=\"{item['opaque_id']}\">\n{item['method_output']}\n</output>\n"
            f"<existing_extractor_claims>\n{json.dumps(claims, ensure_ascii=False)}\n"
            "</existing_extractor_claims>"
        )
    return f"""You are auditing already-frozen Medical-NLA validation outputs. This is exploratory and cannot change the frozen semantic scores.

Compare each opaque method output only with the physician-observation references below. Do not infer findings absent from the output. Every positive finding must be supported by an exact contiguous quote from that output. Method identities are intentionally hidden.

<physician_observation_references>
{json.dumps(observations, ensure_ascii=False, indent=2)}
</physician_observation_references>

{chr(10).join(rendered)}

Return one JSON object only:
{{
  "items": [
    {{
      "opaque_id": "copy exactly",
      "physician_observation_supported": true,
      "disease_template_only": false,
      "boilerplate_or_format_only": false,
      "unsupported_patient_claim": false,
      "extractor_missed_explicit_aligned_claim": false,
      "supporting_quotes": {{"positive_field_name": ["exact contiguous quote"]}},
      "reason": "brief"
    }}
  ]
}}

Definitions:
- physician_observation_supported: at least one patient-specific output claim matches a reference.
- disease_template_only: clinical prose is generic disease knowledge rather than this patient's referenced findings.
- boilerplate_or_format_only: output mainly discusses reasoning format, task mechanics, or reusable scaffolding.
- unsupported_patient_claim: output asserts a patient-specific clinical fact unsupported by all references.
- extractor_missed_explicit_aligned_claim: an explicit reference-aligned claim appears in the output but not in existing_extractor_claims.

Include a supporting quote for every true field. Return exactly one item for every opaque output."""


def prepare_direct(args: argparse.Namespace) -> None:
    require_restricted_direct(args.out_dir)
    cohort_rows = list(read_jsonl(args.cohort))
    cohort = index_unique(cohort_rows, "DiReCT cohort")
    sources = index_unique(
        [row for path in args.source_answers for row in read_jsonl(path)],
        "source answer",
    )
    methods: dict[str, dict[str, Any]] = {}
    source_files = [args.cohort, *args.source_answers]
    for spec in args.method:
        label = str(spec["label"])
        if label in methods:
            raise ValueError(f"Duplicate method label {label}")
        semantic = load_direct_semantic(spec["semantic_root"], spec["semantic_method"])
        if spec["readout"] is None:
            rows = {identifier: sources[identifier] for identifier in cohort if identifier in sources}
            texts = {identifier: source_text(row) for identifier, row in rows.items()}
        else:
            source_files.append(spec["readout"])
            raw = list(read_jsonl(spec["readout"]))
            if spec["source_filter"]:
                raw = [
                    row for row in raw
                    if str(row.get("source_dataset") or "") == spec["source_filter"]
                ]
            rows = index_unique(raw, label)
            texts = {identifier: output_text(row) for identifier, row in rows.items()}
        available = set(texts) & set(semantic)
        methods[label] = {
            "texts": texts,
            "semantic": semantic,
            "available": available,
            "semantic_root": spec["semantic_root"],
            "semantic_method": spec["semantic_method"],
        }

    intersection = set(cohort)
    for item in methods.values():
        intersection &= item["available"]
    if len(intersection) < args.cases:
        raise ValueError(
            f"DiReCT method intersection has {len(intersection)} cases; need {args.cases}"
        )
    selected = sorted(intersection, key=lambda value: stable_key("direct", value, args.seed))
    if len(intersection) == args.cases:
        selected = sorted(intersection)
        sampling = "complete common population"
    else:
        selected = selected[: args.cases]
        sampling = f"stable-hash first {args.cases} from method intersection"

    if args.methods_per_request <= 0:
        raise ValueError("--methods-per-request must be positive")
    requests: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    for identifier in selected:
        all_methods = []
        for index, label in enumerate(
            sorted(methods, key=lambda value: stable_key(identifier, value, args.seed)),
            start=1,
        ):
            semantic = methods[label]["semantic"][identifier]
            text = methods[label]["texts"][identifier]
            if clean(text) != clean(semantic["method_output"]):
                raise ValueError(f"Output/semantic text mismatch for {label}/{identifier}")
            all_methods.append(
                {
                    "opaque_id": f"M{index:02d}",
                    "method": label,
                    "method_output": text,
                    **semantic,
                }
            )
        observations = [str(value) for value in cohort[identifier].get("cue_targets") or []]
        if not observations:
            raise ValueError(f"No physician observations for {identifier}")
        for chunk_index, start in enumerate(
            range(0, len(all_methods), args.methods_per_request)
        ):
            opaque_methods = all_methods[start : start + args.methods_per_request]
            request_id = (
                f"direct_sft_raw_{stable_key('request', identifier, args.seed)[:20]}_"
                f"c{chunk_index:02d}"
            )
            requests.append(
                {"id": request_id, "prompt": direct_prompt(observations, opaque_methods)}
            )
            private_rows.append(
                {
                    "id": request_id,
                    "base_id": identifier,
                    "physician_observations": observations,
                    "methods": opaque_methods,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "requests.jsonl", requests)
    write_jsonl(args.out_dir / "private_bundle.jsonl", private_rows)
    protocol = {
        "schema_version": 1,
        "audit": "DiReCT SFT raw-output audit",
        "exploratory": True,
        "frozen_scores_changed": False,
        "restricted_text": True,
        "external_api_allowed": False,
        "cases": len(selected),
        "intersection_cases": len(intersection),
        "methods": list(methods),
        "methods_per_request": args.methods_per_request,
        "requests": len(requests),
        "sampling": sampling,
        "seed": args.seed,
        "base_id_sha256": sha256_values(set(selected)),
        "source_sha256": {str(path): sha256_file(path) for path in source_files},
        "semantic_artifact_sha256": {
            label: {
                "root": str(item["semantic_root"]),
                "method": item["semantic_method"],
                "private_index": sha256_file(item["semantic_root"] / "private_index.jsonl"),
                "extraction_audit": sha256_file(
                    item["semantic_root"] / "private_extraction_audit.jsonl"
                ),
                "extraction_judgements": sha256_file(
                    item["semantic_root"] / "extraction_judgements.jsonl"
                ),
                "evaluation_tree": sha256_tree(
                    item["semantic_root"] / "evaluations" / item["semantic_method"]
                ),
            }
            for label, item in methods.items()
        },
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[direct-preflight] cohort={len(cohort)} intersection={len(intersection)} "
        f"selected={len(selected)} methods={len(methods)} requests={len(requests)}"
    )
    print(f"[private] {args.out_dir}")


def stratified_select(
    candidates: list[dict[str, Any]], *, cases: int, seed: int, namespace: str, stratum: str
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        groups[str(item.get(stratum) or "<missing>")].append(item)
    selected: list[dict[str, Any]] = []
    for group in sorted(groups, key=lambda value: stable_key(namespace, value, seed)):
        rows = sorted(
            groups[group],
            key=lambda row: stable_key(namespace, str(row["base_id"]), seed),
        )
        selected.append(rows[0])
        if len(selected) == cases:
            return selected
    used = {str(row["base_id"]) for row in selected}
    remaining = sorted(
        [row for row in candidates if str(row["base_id"]) not in used],
        key=lambda row: stable_key(namespace, str(row["base_id"]), seed),
    )
    selected.extend(remaining[: cases - len(selected)])
    if len(selected) != cases:
        raise ValueError(f"{namespace}: selected {len(selected)} of requested {cases}")
    return selected


def ddx_prompt(cohort: str, expected: dict[str, Any], methods: list[dict[str, Any]]) -> str:
    rendered = []
    for method in methods:
        outputs = "\n".join(
            f"<{variant}>\n{text}\n</{variant}>"
            for variant, text in method["outputs"].items()
        )
        rendered.append(f"<method id=\"{method['opaque_id']}\">\n{outputs}\n</method>")
    if cohort == "deletion":
        fields = DELETION_FIELDS
        definitions = """- original_target_mentioned: the removed target is explicitly stated in original.
- deleted_target_phantom: the removed target is still explicitly stated after deletion.
- untouched_finding_retained: at least one unchanged expected finding is explicit in both outputs.
- unsupported_patient_claim: either output asserts a patient fact outside its expected cue list."""
    else:
        fields = VALUE_FIELDS
        definitions = """- old_value_mentioned_original: the old native value is explicit in original.
- new_value_mentioned_after_edit: the replacement native value is explicit after edit.
- old_value_persists_after_edit: the old value remains explicit after edit.
- clean_value_switch: new value is explicit and old value is absent after edit.
- untouched_finding_retained: at least one unchanged expected finding is explicit in both outputs.
- unsupported_patient_claim: either output asserts a patient fact outside its expected cue list."""
    schema = ",\n      ".join(f'"{field}": false' for field in fields)
    return f"""You are auditing frozen DDXPlus Medical-NLA validation outputs. This is exploratory and cannot change frozen scores.

Use only the expected cues and exact output text. Do not infer an unmentioned finding. Every true field must have an exact contiguous supporting quote from the relevant output. Method identities are hidden.

<audit_cohort>{cohort}</audit_cohort>
<expected>
{json.dumps(expected, ensure_ascii=False, indent=2)}
</expected>

{chr(10).join(rendered)}

Return JSON only:
{{"items": [{{"opaque_id": "copy exactly", {schema}, "supporting_quotes": {{"positive_field_name": ["exact quote"]}}, "reason": "brief"}}]}}

{definitions}
Return exactly one item for every opaque method."""


def prepare_ddxplus(args: argparse.Namespace) -> None:
    method_rows: dict[str, dict[str, dict[str, Any]]] = {}
    source_hashes: dict[str, str] = {}
    for label, path in args.readout:
        if label in method_rows:
            raise ValueError(f"Duplicate DDXPlus method {label}")
        rows = index_unique(list(read_jsonl(path)), label, key=lambda row: str(row.get("id") or ""))
        method_rows[label] = rows
        source_hashes[str(path)] = sha256_file(path)
    common_ids = set.intersection(*(set(rows) for rows in method_rows.values()))
    if not common_ids:
        raise ValueError("DDXPlus methods have no common row IDs")

    canonical_label = next(iter(method_rows))
    canonical = method_rows[canonical_label]
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row_id in common_ids:
        reference = canonical[row_id]
        identifier = str(reference.get("base_id") or "")
        variant = str(reference.get("variant") or "original")
        if not identifier or variant in grouped[identifier]:
            raise ValueError(f"Invalid DDXPlus family row {row_id}")
        for label, rows in method_rows.items():
            other = rows[row_id]
            for field in ("base_id", "variant", "diagnosis_id", "cf_original_cue", "cf_replacement_cue"):
                if clean(other.get(field)) != clean(reference.get(field)):
                    raise ValueError(f"Metadata mismatch {label}/{row_id}/{field}")
        grouped[identifier][variant] = reference

    deletion_candidates = []
    value_candidates = []
    for identifier, family in grouped.items():
        original = family.get("original")
        deleted = family.get("cue_deleted")
        edited = family.get("value_edited")
        if original is not None and deleted is not None:
            deletion_candidates.append(
                {
                    "base_id": identifier,
                    "diagnosis_id": original.get("diagnosis_id"),
                    "family": {"original": original, "cue_deleted": deleted},
                }
            )
        if original is not None and edited is not None:
            value_candidates.append(
                {
                    "base_id": identifier,
                    "diagnosis_id": original.get("diagnosis_id"),
                    "evidence_id": edited.get("cf_original_evidence_id"),
                    "family": {"original": original, "value_edited": edited},
                }
            )
    if len(deletion_candidates) < args.cases or len(value_candidates) < args.cases:
        raise ValueError(
            f"Insufficient cohorts: deletion={len(deletion_candidates)} "
            f"value_edit={len(value_candidates)} requested={args.cases}"
        )
    selected = {
        "deletion": stratified_select(
            deletion_candidates,
            cases=args.cases,
            seed=args.seed,
            namespace="ddx-deletion",
            stratum="diagnosis_id",
        ),
        "value_edit": stratified_select(
            value_candidates,
            cases=args.cases,
            seed=args.seed,
            namespace="ddx-value-edit",
            stratum="evidence_id",
        ),
    }

    requests = []
    private_rows = []
    for cohort, cases in selected.items():
        variants = ("original", "cue_deleted") if cohort == "deletion" else ("original", "value_edited")
        for case in cases:
            identifier = str(case["base_id"])
            family = case["family"]
            opaque_methods = []
            for index, label in enumerate(
                sorted(method_rows, key=lambda value: stable_key(identifier, value, args.seed)),
                start=1,
            ):
                outputs = {
                    variant: output_text(method_rows[label][str(family[variant]["id"])])
                    for variant in variants
                }
                opaque_methods.append(
                    {"opaque_id": f"M{index:02d}", "method": label, "outputs": outputs}
                )
            original_cues = [str(value) for value in family["original"].get("cue_targets") or []]
            derived = family[variants[1]]
            expected = {
                "original_cues": original_cues,
                "derived_cues": [str(value) for value in derived.get("cue_targets") or []],
                "changed_original_cue": derived.get("cf_original_cue"),
                "changed_replacement_cue": derived.get("cf_replacement_cue"),
                "diagnosis_id_for_sampling_only": case.get("diagnosis_id"),
                "evidence_id_for_sampling_only": case.get("evidence_id"),
            }
            request_id = f"ddx_{cohort}_{stable_key(cohort, identifier, args.seed)[:20]}"
            requests.append({"id": request_id, "prompt": ddx_prompt(cohort, expected, opaque_methods)})
            private_rows.append(
                {
                    "id": request_id,
                    "cohort": cohort,
                    "base_id": identifier,
                    "expected": expected,
                    "methods": opaque_methods,
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "requests.jsonl", requests)
    write_jsonl(args.out_dir / "private_bundle.jsonl", private_rows)
    deletion_ids = {str(row["base_id"]) for row in selected["deletion"]}
    value_ids = {str(row["base_id"]) for row in selected["value_edit"]}
    protocol = {
        "schema_version": 1,
        "audit": "DDXPlus SFT counterfactual raw-output audit",
        "exploratory": True,
        "frozen_scores_changed": False,
        "cases_per_cohort": args.cases,
        "candidate_counts": {
            "deletion": len(deletion_candidates),
            "value_edit": len(value_candidates),
        },
        "selected_counts": {name: len(rows) for name, rows in selected.items()},
        "cohort_overlap": len(deletion_ids & value_ids),
        "deletion_base_id_sha256": sha256_values(deletion_ids),
        "value_edit_base_id_sha256": sha256_values(value_ids),
        "seed": args.seed,
        "methods": list(method_rows),
        "source_sha256": source_hashes,
    }
    (args.out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[ddx-preflight] common_rows={len(common_ids)} deletion={len(deletion_candidates)} "
        f"value_edit={len(value_candidates)} selected={args.cases}+{args.cases}"
    )
    print(f"[private] {args.out_dir}")


def validate_quotes(texts: list[str], quotes: list[str], context: str) -> None:
    normalized = [clean(text).casefold() for text in texts]
    for quote in quotes:
        candidate = clean(quote).casefold()
        if not candidate or not any(candidate in text for text in normalized):
            raise ValueError(f"Non-verbatim quote for {context}: {quote!r}")


def method_similarity(private_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    outputs: dict[str, list[str]] = defaultdict(list)
    for row in private_rows:
        for method in row["methods"]:
            outputs[str(method["method"])].append(clean(method["method_output"]).casefold())
    result = {}
    for label, texts in outputs.items():
        exact_duplicate_rows = len(texts) - len(set(texts))
        maxima = []
        token_sets = [set(text.split()) for text in texts]
        for index, own in enumerate(token_sets):
            similarities = []
            for other_index, other in enumerate(token_sets):
                if index == other_index:
                    continue
                union = own | other
                similarities.append(len(own & other) / len(union) if union else 1.0)
            maxima.append(max(similarities) if similarities else 0.0)
        result[label] = {
            "exact_duplicate_rows": exact_duplicate_rows,
            "median_max_word_jaccard": statistics.median(maxima) if maxima else None,
        }
    return result


def audit_direct_judgements(args: argparse.Namespace) -> dict[str, Any]:
    private = {str(row["id"]): row for row in read_jsonl(args.private_bundle)}
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    judgements = {str(row["id"]): row for row in read_jsonl(args.judgements)}
    if set(private) != set(requests):
        raise ValueError("Direct request population does not match private bundle")
    invalid = []
    retry = []
    for request_id in sorted(private):
        source = private[request_id]
        judgement = judgements.get(request_id)
        error = None
        try:
            if judgement is None:
                raise ValueError("missing judgement")
            parsed = extract_json_object(judgement.get("response"))
            items = {str(item.get("opaque_id")): item for item in parsed.get("items") or []}
            expected = {str(item["opaque_id"]): item for item in source["methods"]}
            if set(items) != set(expected):
                raise ValueError("opaque method population mismatch")
            for opaque_id, method in expected.items():
                item = items[opaque_id]
                quote_map = item.get("supporting_quotes") or {}
                for field in DIRECT_FIELDS:
                    value = bool(item.get(field))
                    quotes = [str(quote) for quote in quote_map.get(field) or []]
                    if value and not quotes:
                        raise ValueError(f"true {field} lacks a quote for {opaque_id}")
                    validate_quotes(
                        [method["method_output"]],
                        quotes,
                        f"{request_id}/{opaque_id}/{field}",
                    )
        except Exception as exc:  # noqa: BLE001 - enumerate every invalid response
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            continue
        invalid.append({"id": request_id, "error": error})
        previous = "" if judgement is None else str(judgement.get("response") or "")
        retry_prompt = str(requests[request_id]["prompt"]) + f"""

<retry_correction>
The previous response was invalid: {error}
Previous response:
{previous}

Return a corrected JSON object. Every true field must include at least one exact
contiguous quote under supporting_quotes for that same field. If no valid quote
exists, set the field to false. Preserve every required opaque_id exactly.
</retry_correction>"""
        retry.append({"id": request_id, "prompt": retry_prompt})
    write_jsonl(args.retry_requests, retry)
    report = {
        "schema_version": 1,
        "requests": len(requests),
        "judgements": len(judgements),
        "valid": len(requests) - len(invalid),
        "invalid": len(invalid),
        "invalid_requests": invalid,
        "retry_requests": str(args.retry_requests),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[audit-direct-judgements] valid={report['valid']} "
        f"invalid={report['invalid']} retry={args.retry_requests}"
    )
    return report


def finalize_direct(args: argparse.Namespace) -> None:
    require_restricted_direct(args.out_dir)
    private = {str(row["id"]): row for row in read_jsonl(args.private_bundle)}
    judgements = {str(row["id"]): row for row in read_jsonl(args.judgements)}
    if set(private) != set(judgements):
        raise ValueError("Direct judgement population does not match private bundle")
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    private_results = []
    for request_id in sorted(private):
        source = private[request_id]
        parsed = extract_json_object(judgements[request_id].get("response"))
        items = {str(item.get("opaque_id")): item for item in parsed.get("items") or []}
        expected = {str(item["opaque_id"]): item for item in source["methods"]}
        if set(items) != set(expected):
            raise ValueError(f"Opaque method mismatch for {request_id}")
        decisions = []
        for opaque_id, method in expected.items():
            item = items[opaque_id]
            flags = {field: bool(item.get(field)) for field in DIRECT_FIELDS}
            quote_map = item.get("supporting_quotes") or {}
            for field, value in flags.items():
                quotes = [str(quote) for quote in quote_map.get(field) or []]
                if value and not quotes:
                    raise ValueError(f"True {field} lacks quote for {request_id}/{opaque_id}")
                validate_quotes([method["method_output"]], quotes, f"{request_id}/{opaque_id}/{field}")
                counts[str(method["method"])][field] += value
            counts[str(method["method"])]["rows"] += 1
            decisions.append(
                {
                    "opaque_id": opaque_id,
                    "method": method["method"],
                    **flags,
                    "supporting_quotes": quote_map,
                    "reason": str(item.get("reason") or ""),
                }
            )
        private_results.append({"id": request_id, "base_id": source["base_id"], "items": decisions})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private_adjudications.jsonl", private_results)
    similarities = method_similarity(list(private.values()))
    case_count = len({str(row["base_id"]) for row in private.values()})
    report = {
        "schema_version": 1,
        "audit": "DiReCT SFT raw-output audit",
        "exploratory": True,
        "frozen_scores_changed": False,
        "restricted_text_emitted": False,
        "cases": case_count,
        "requests": len(private),
        "counts": {label: dict(values) for label, values in sorted(counts.items())},
        "cross_case_similarity": similarities,
        "judge_models": sorted({str(row.get("judge_model") or "") for row in judgements.values()}),
    }
    (args.out_dir / "aggregate_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DiReCT SFT Raw-Output Audit",
        "",
        "Exploratory validation audit. Restricted clinical text is omitted and frozen scores are unchanged.",
        "",
        f"- cases / requests: **{case_count} / {len(private)}**",
        f"- judge models: `{report['judge_models']}`",
        "",
        "| method | n | physician finding | disease template only | boilerplate/format | unsupported claim | extractor miss | duplicate rows | median max Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(counts):
        values = counts[label]
        sim = similarities[label]
        lines.append(
            f"| {label} | {values['rows']} | {values['physician_observation_supported']} | "
            f"{values['disease_template_only']} | {values['boilerplate_or_format_only']} | "
            f"{values['unsupported_patient_claim']} | "
            f"{values['extractor_missed_explicit_aligned_claim']} | "
            f"{sim['exact_duplicate_rows']} | {sim['median_max_word_jaccard']:.4f} |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def finalize_ddxplus(args: argparse.Namespace) -> None:
    private = {str(row["id"]): row for row in read_jsonl(args.private_bundle)}
    judgements = {str(row["id"]): row for row in read_jsonl(args.judgements)}
    if set(private) != set(judgements):
        raise ValueError("DDXPlus judgement population does not match private bundle")
    counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    private_results = []
    for request_id in sorted(private):
        source = private[request_id]
        cohort = str(source["cohort"])
        fields = DELETION_FIELDS if cohort == "deletion" else VALUE_FIELDS
        parsed = extract_json_object(judgements[request_id].get("response"))
        items = {str(item.get("opaque_id")): item for item in parsed.get("items") or []}
        expected = {str(item["opaque_id"]): item for item in source["methods"]}
        if set(items) != set(expected):
            raise ValueError(f"Opaque method mismatch for {request_id}")
        decisions = []
        for opaque_id, method in expected.items():
            item = items[opaque_id]
            flags = {field: bool(item.get(field)) for field in fields}
            quote_map = item.get("supporting_quotes") or {}
            texts = list(method["outputs"].values())
            for field, value in flags.items():
                quotes = [str(quote) for quote in quote_map.get(field) or []]
                if value and not quotes:
                    raise ValueError(f"True {field} lacks quote for {request_id}/{opaque_id}")
                validate_quotes(texts, quotes, f"{request_id}/{opaque_id}/{field}")
                counts[(cohort, str(method["method"]))][field] += value
            counts[(cohort, str(method["method"]))]["rows"] += 1
            decisions.append(
                {
                    "opaque_id": opaque_id,
                    "method": method["method"],
                    **flags,
                    "supporting_quotes": quote_map,
                    "reason": str(item.get("reason") or ""),
                }
            )
        private_results.append(
            {"id": request_id, "cohort": cohort, "base_id": source["base_id"], "items": decisions}
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private_adjudications.jsonl", private_results)
    report = {
        "schema_version": 1,
        "audit": "DDXPlus SFT counterfactual raw-output audit",
        "exploratory": True,
        "frozen_scores_changed": False,
        "clinical_text_emitted": False,
        "counts": {f"{cohort}::{method}": dict(value) for (cohort, method), value in sorted(counts.items())},
        "judge_models": sorted({str(row.get("judge_model") or "") for row in judgements.values()}),
    }
    (args.out_dir / "aggregate_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DDXPlus SFT Counterfactual Raw-Output Audit",
        "",
        "Exploratory validation audit. Clinical text is omitted and frozen scores are unchanged.",
        "",
        f"- judge models: `{report['judge_models']}`",
    ]
    for cohort in ("deletion", "value_edit"):
        fields = DELETION_FIELDS if cohort == "deletion" else VALUE_FIELDS
        labels = sorted(method for (name, method) in counts if name == cohort)
        lines.extend(
            [
                "",
                f"## {cohort.replace('_', ' ').title()}",
                "",
                "| method | n | " + " | ".join(field.replace("_", " ") for field in fields) + " |",
                "|---|---:|" + "---:|" * len(fields),
            ]
        )
        for label in labels:
            value = counts[(cohort, label)]
            lines.append(
                f"| {label} | {value['rows']} | "
                + " | ".join(str(value[field]) for field in fields)
                + " |"
            )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def finalize_direct_deterministic(args: argparse.Namespace) -> None:
    """Summarize the full DiReCT audit without selecting valid AI responses."""
    require_restricted_direct(args.out_dir)
    private_rows = list(read_jsonl(args.private_bundle))
    if not private_rows:
        raise ValueError("Empty DiReCT private bundle")
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    obscomp: dict[str, list[float]] = defaultdict(list)
    private_case_rows = []
    seen: set[tuple[str, str]] = set()
    for request in private_rows:
        identifier = str(request["base_id"])
        for method in request["methods"]:
            label = str(method["method"])
            key = (identifier, label)
            if key in seen:
                raise ValueError(f"Duplicate deterministic audit row {identifier}/{label}")
            seen.add(key)
            evaluation = method["official_evaluation"]
            len_gt = int(evaluation["len_ob_gt"])
            len_pred = int(evaluation["len_ob_pred"])
            paired = len(evaluation.get("ob_record_paired") or {})
            if paired > min(len_gt, len_pred):
                raise ValueError(f"Impossible paired count for {identifier}/{label}")
            accepted = len(
                method["accepted_extraction"].get("accepted_claims") or []
            )
            if accepted != len_pred:
                raise ValueError(
                    f"Extractor/evaluator count mismatch for {identifier}/{label}: "
                    f"accepted={accepted} len_ob_pred={len_pred}"
                )
            score = score_record(evaluation, "official")
            values = stats[label]
            values["rows"] += 1
            values["rows_with_extractable_observation"] += len_pred > 0
            values["rows_with_physician_match"] += paired > 0
            values["rows_with_unmatched_only"] += len_pred > 0 and paired == 0
            values["rows_without_extractable_observation"] += len_pred == 0
            values["predicted_observations"] += len_pred
            values["matched_observations"] += paired
            values["unmatched_predicted_observations"] += len_pred - paired
            values["physician_reference_observations"] += len_gt
            obscomp[label].append(float(score["comp_coverage"]))
            private_case_rows.append(
                {
                    "base_id": identifier,
                    "method": label,
                    "len_ob_gt": len_gt,
                    "len_ob_pred": len_pred,
                    "paired_observations": paired,
                    "unmatched_predicted_observations": len_pred - paired,
                    "obscomp": score["comp_coverage"],
                }
            )
    cases = {str(row["base_id"]) for row in private_rows}
    expected_rows = len(cases)
    for label, values in stats.items():
        if values["rows"] != expected_rows:
            raise ValueError(
                f"Method {label} has {values['rows']} rows; expected {expected_rows}"
            )
    similarities = method_similarity(private_rows)
    ai_instrument = {
        "status": "rejected_measurement_instrument",
        "reason": "local judge failed the frozen JSON/exact-quote contract",
        "selected_valid_subset_used": False,
    }
    if args.ai_audit_report is not None and args.ai_audit_report.is_file():
        audit = json.loads(args.ai_audit_report.read_text(encoding="utf-8"))
        ai_instrument.update(
            {
                "requests": audit.get("requests"),
                "valid": audit.get("valid"),
                "invalid": audit.get("invalid"),
                "audit_report": str(args.ai_audit_report),
                "audit_report_sha256": sha256_file(args.ai_audit_report),
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "private_deterministic_case_audit.jsonl", private_case_rows)
    report = {
        "schema_version": 1,
        "audit": "DiReCT SFT deterministic raw-output audit",
        "exploratory": True,
        "frozen_scores_changed": False,
        "restricted_text_emitted": False,
        "cases": expected_rows,
        "methods": len(stats),
        "extractor_miss": "not assessed",
        "disease_template_classification": "not assessed",
        "ai_checklist_instrument": ai_instrument,
        "counts": {label: dict(values) for label, values in sorted(stats.items())},
        "mean_obscomp": {
            label: statistics.fmean(values) for label, values in sorted(obscomp.items())
        },
        "cross_case_similarity": similarities,
    }
    (args.out_dir / "deterministic_results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# DiReCT SFT Deterministic Raw-Output Audit",
        "",
        "Exploratory 50-case validation census using the frozen exact-quote extractor and",
        "official evaluator artifacts. Restricted clinical text is omitted; frozen scores",
        "are unchanged.",
        "",
        f"- cases / methods: **{expected_rows} / {len(stats)}**",
        "- AI checklist instrument: **rejected**; no valid-response subset is used",
        "- extractor miss: **not assessed**",
        "- disease-template classification: **not assessed**",
        "",
        "| method | n | extractable rows | physician-matched rows | unmatched-only rows | no extractable row | predicted obs. | matched obs. | unmatched obs. | mean Obscomp | duplicate rows | median max Jaccard |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in sorted(stats):
        values = stats[label]
        similarity = similarities[label]
        lines.append(
            f"| {label} | {values['rows']} | "
            f"{values['rows_with_extractable_observation']} | "
            f"{values['rows_with_physician_match']} | "
            f"{values['rows_with_unmatched_only']} | "
            f"{values['rows_without_extractable_observation']} | "
            f"{values['predicted_observations']} | "
            f"{values['matched_observations']} | "
            f"{values['unmatched_predicted_observations']} | "
            f"{statistics.fmean(obscomp[label]):.4f} | "
            f"{similarity['exact_duplicate_rows']} | "
            f"{similarity['median_max_word_jaccard']:.4f} |"
        )
    lines.extend(
        [
            "",
            "`unmatched` means not matched to the available physician reference by the frozen",
            "official evaluator. It is not asserted to be medically false or hallucinated.",
            "",
        ]
    )
    (args.out_dir / "deterministic_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print((args.out_dir / "deterministic_summary.md").read_text(encoding="utf-8"))


def audit_ddxplus_judgements(args: argparse.Namespace) -> dict[str, Any]:
    private = {str(row["id"]): row for row in read_jsonl(args.private_bundle)}
    requests = {str(row["id"]): row for row in read_jsonl(args.requests)}
    judgements = {str(row["id"]): row for row in read_jsonl(args.judgements)}
    if set(private) != set(requests):
        raise ValueError("DDXPlus request population does not match private bundle")
    invalid = []
    retry = []
    for request_id in sorted(private):
        source = private[request_id]
        judgement = judgements.get(request_id)
        error = None
        try:
            if judgement is None:
                raise ValueError("missing judgement")
            cohort = str(source["cohort"])
            fields = DELETION_FIELDS if cohort == "deletion" else VALUE_FIELDS
            parsed = extract_json_object(judgement.get("response"))
            items = {str(item.get("opaque_id")): item for item in parsed.get("items") or []}
            expected = {str(item["opaque_id"]): item for item in source["methods"]}
            if set(items) != set(expected):
                raise ValueError("opaque method population mismatch")
            for opaque_id, method in expected.items():
                item = items[opaque_id]
                quote_map = item.get("supporting_quotes") or {}
                texts = list(method["outputs"].values())
                for field in fields:
                    value = bool(item.get(field))
                    quotes = [str(quote) for quote in quote_map.get(field) or []]
                    if value and not quotes:
                        raise ValueError(f"true {field} lacks a quote for {opaque_id}")
                    validate_quotes(texts, quotes, f"{request_id}/{opaque_id}/{field}")
        except Exception as exc:  # noqa: BLE001 - enumerate every invalid response
            error = f"{type(exc).__name__}: {exc}"
        if error is None:
            continue
        invalid.append({"id": request_id, "error": error})
        previous = "" if judgement is None else str(judgement.get("response") or "")
        retry_prompt = str(requests[request_id]["prompt"]) + f"""

<retry_correction>
The previous response was invalid: {error}
Previous response:
{previous}

Return a corrected JSON object. Copy every supporting quote character-for-character
as one exact contiguous substring of the corresponding output, including spaces and
punctuation. Every true field needs a quote under that same field. If no exact quote
exists, set the field to false. Preserve every required opaque_id exactly.
</retry_correction>"""
        retry.append({"id": request_id, "prompt": retry_prompt})
    write_jsonl(args.retry_requests, retry)
    report = {
        "schema_version": 1,
        "requests": len(requests),
        "judgements": len(judgements),
        "valid": len(requests) - len(invalid),
        "invalid": len(invalid),
        "invalid_requests": invalid,
        "retry_requests": str(args.retry_requests),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[audit-ddxplus-judgements] valid={report['valid']} "
        f"invalid={report['invalid']} retry={args.retry_requests}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    direct = sub.add_parser("prepare-direct")
    direct.add_argument("--cohort", required=True, type=Path)
    direct.add_argument("--source-answers", required=True, nargs="+", type=Path)
    direct.add_argument("--method", required=True, action="append", type=parse_direct_method)
    direct.add_argument("--out-dir", required=True, type=Path)
    direct.add_argument("--cases", type=int, default=50)
    direct.add_argument("--seed", type=int, default=17)
    direct.add_argument("--methods-per-request", type=int, default=2)

    ddx = sub.add_parser("prepare-ddxplus")
    ddx.add_argument("--readout", required=True, action="append", type=parse_named_path)
    ddx.add_argument("--out-dir", required=True, type=Path)
    ddx.add_argument("--cases", type=int, default=50)
    ddx.add_argument("--seed", type=int, default=17)

    final_direct = sub.add_parser("finalize-direct")
    final_direct.add_argument("--private-bundle", required=True, type=Path)
    final_direct.add_argument("--judgements", required=True, type=Path)
    final_direct.add_argument("--out-dir", required=True, type=Path)

    deterministic_direct = sub.add_parser("finalize-direct-deterministic")
    deterministic_direct.add_argument("--private-bundle", required=True, type=Path)
    deterministic_direct.add_argument("--out-dir", required=True, type=Path)
    deterministic_direct.add_argument("--ai-audit-report", type=Path)

    audit_direct = sub.add_parser("audit-direct-judgements")
    audit_direct.add_argument("--private-bundle", required=True, type=Path)
    audit_direct.add_argument("--requests", required=True, type=Path)
    audit_direct.add_argument("--judgements", required=True, type=Path)
    audit_direct.add_argument("--retry-requests", required=True, type=Path)
    audit_direct.add_argument("--report", required=True, type=Path)

    audit_ddx = sub.add_parser("audit-ddxplus-judgements")
    audit_ddx.add_argument("--private-bundle", required=True, type=Path)
    audit_ddx.add_argument("--requests", required=True, type=Path)
    audit_ddx.add_argument("--judgements", required=True, type=Path)
    audit_ddx.add_argument("--retry-requests", required=True, type=Path)
    audit_ddx.add_argument("--report", required=True, type=Path)

    final_ddx = sub.add_parser("finalize-ddxplus")
    final_ddx.add_argument("--private-bundle", required=True, type=Path)
    final_ddx.add_argument("--judgements", required=True, type=Path)
    final_ddx.add_argument("--out-dir", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "prepare-direct":
        prepare_direct(args)
    elif args.command == "prepare-ddxplus":
        prepare_ddxplus(args)
    elif args.command == "finalize-direct":
        finalize_direct(args)
    elif args.command == "finalize-direct-deterministic":
        finalize_direct_deterministic(args)
    elif args.command == "audit-direct-judgements":
        audit_direct_judgements(args)
    elif args.command == "audit-ddxplus-judgements":
        audit_ddxplus_judgements(args)
    else:
        finalize_ddxplus(args)


if __name__ == "__main__":
    main()
