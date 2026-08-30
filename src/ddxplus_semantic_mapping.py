"""Frozen lexical and AI-assisted mapping of DDXPlus readout claims."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


ABBREVIATIONS = ("e.g.", "i.e.", "mr.", "mrs.", "dr.", "vs.", "etc.")
NON_ASSERTIVE = re.compile(
    r"\b(?:may|might|could|possibly|consider|considering|differential|"
    r"recommend|recommended|should|would|if|risk of|rule out|r/o)\b",
    re.IGNORECASE,
)
OBSERVED_RE = re.compile(r"<observed>\s*(?P<body>.*?)\s*</observed>", re.I | re.S)
WORD_RE = re.compile(r"[a-z0-9]+")


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def normalize(value: Any) -> str:
    return " ".join(WORD_RE.findall(clean(value).casefold()))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def claims_from_text(text: str) -> list[str]:
    """Split one readout without using case metadata or labels."""
    source = str(text or "").strip()
    observed = OBSERVED_RE.search(source)
    if observed is not None:
        source = observed.group("body").strip()

    bullet_lines = []
    current: list[str] = []
    saw_bullet = False
    for line in source.splitlines():
        match = re.match(r"^\s*[-*•]\s+(.*)$", line)
        if match:
            saw_bullet = True
            if current:
                bullet_lines.append(clean(" ".join(current)))
            current = [match.group(1)]
        elif saw_bullet and line.strip():
            current.append(line.strip())
    if current:
        bullet_lines.append(clean(" ".join(current)))
    if saw_bullet:
        return [item for item in bullet_lines if item]

    protected = source
    replacements: dict[str, str] = {}
    for index, abbreviation in enumerate(ABBREVIATIONS):
        pattern = re.compile(re.escape(abbreviation), flags=re.I)

        def replace(match: re.Match[str], *, prefix: int = index) -> str:
            token = f"__ABBR_{prefix}_{len(replacements)}__"
            replacements[token] = match.group(0)
            return token

        protected = pattern.sub(replace, protected)
    pieces = re.split(r"(?:\n+|(?<=[.!?])\s+)", protected)
    result = []
    for piece in pieces:
        for token, abbreviation in replacements.items():
            piece = piece.replace(token, abbreviation)
        item = clean(piece).strip(" -•*")
        if item:
            result.append(item)
    return result


def assertion_candidate(claim: str) -> bool:
    text = clean(claim)
    return bool(text) and "?" not in text and NON_ASSERTIVE.search(text) is None


def metadata_text(value: Any) -> str:
    if isinstance(value, dict):
        for key in ("en", "name", "label", "text"):
            if clean(value.get(key)):
                return clean(value[key])
        return ""
    return clean(value)


def question_phrase(question: str) -> str:
    """Conservatively derive an alias from a simple English evidence question."""
    text = clean(question).strip(" ?.")
    patterns = (
        r"^(?:do|does|did) (?:you|the patient) (?:have|experience|report) (?:a |an |any )?(.*)$",
        r"^(?:have|has) (?:you|the patient) (?:had |experienced |noticed )?(.*)$",
        r"^(?:is|are|was|were) (?:your|the patient's) (.*)$",
    )
    for pattern in patterns:
        match = re.match(pattern, text, flags=re.I)
        if match:
            return clean(match.group(1))
    return ""


def build_alias_and_ontology(
    structured_protocol: dict[str, Any], evidence_meta: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = [str(item) for item in structured_protocol["finding_labels"]]
    values_by_evidence = {
        str(key): [str(value) for value in values]
        for key, values in structured_protocol["values_by_evidence"].items()
    }
    lexicon = structured_protocol["lexicon"]
    raw_aliases: dict[str, set[str]] = defaultdict(set)
    raw_values: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    ontology_items = []

    for evidence in labels:
        meta = evidence_meta.get(evidence) or {}
        canonical = clean(lexicon["findings"][evidence]["text"])
        question = clean(meta.get("question_en") or meta.get("question") or "")
        candidates = {canonical, question_phrase(question)}
        candidates = {item for item in candidates if len(normalize(item)) >= 3}
        raw_aliases[evidence].update(candidates)

        value_items = []
        meanings = meta.get("value_meaning") or {}
        for value_id in values_by_evidence.get(evidence, []):
            key = f"{evidence}\0{value_id}"
            rendered = clean((lexicon.get("values") or {}).get(key, {}).get("text"))
            meaning = metadata_text(meanings.get(value_id)) if isinstance(meanings, dict) else ""
            aliases = {item for item in (rendered, meaning) if len(normalize(item)) >= 1}
            raw_values[evidence][value_id].update(aliases)
            if rendered:
                raw_aliases[evidence].add(rendered)
            value_items.append(
                {"value_id": value_id, "labels": sorted(aliases, key=str.casefold)}
            )
        ontology_items.append(
            {
                "evidence_id": evidence,
                "canonical_phrase": canonical,
                "question": question,
                "values": value_items,
            }
        )

    owners: dict[str, set[str]] = defaultdict(set)
    for evidence, aliases in raw_aliases.items():
        for alias in aliases:
            owners[normalize(alias)].add(evidence)
    ambiguous = sorted(alias for alias, evidence in owners.items() if len(evidence) > 1)
    aliases = {
        evidence: sorted(
            [alias for alias in items if len(owners[normalize(alias)]) == 1],
            key=lambda item: (-len(normalize(item)), item.casefold()),
        )
        for evidence, items in raw_aliases.items()
    }
    value_aliases = {
        evidence: {
            value_id: sorted(items, key=lambda item: (-len(normalize(item)), item.casefold()))
            for value_id, items in values.items()
        }
        for evidence, values in raw_values.items()
    }
    alias_table = {
        "schema_version": 1,
        "finding_aliases": aliases,
        "value_aliases": value_aliases,
        "ambiguous_normalized_aliases_excluded": ambiguous,
        "manual_aliases": False,
    }
    ontology = {"schema_version": 1, "evidence": ontology_items}
    return alias_table, ontology


def boundary_contains(haystack: str, needle: str) -> bool:
    normalized_haystack = f" {normalize(haystack)} "
    normalized_needle = normalize(needle)
    return bool(normalized_needle) and f" {normalized_needle} " in normalized_haystack


def exact_quote(haystack: str, needle: str) -> str | None:
    match = re.search(re.escape(clean(needle)), clean(haystack), flags=re.I)
    return match.group(0) if match else None


def lexical_mappings(claim: str, alias_table: dict[str, Any]) -> list[dict[str, Any]]:
    if not assertion_candidate(claim):
        return []
    mappings = []
    for evidence, aliases in alias_table["finding_aliases"].items():
        matches = [
            (alias, exact_quote(claim, alias))
            for alias in aliases
            if boundary_contains(claim, alias) and exact_quote(claim, alias) is not None
        ]
        if not matches:
            continue
        _alias, supporting_quote = max(
            matches, key=lambda item: len(normalize(item[0]))
        )
        matched_values = []
        for candidate, value_aliases in alias_table["value_aliases"].get(evidence, {}).items():
            if any(boundary_contains(claim, alias) for alias in value_aliases):
                matched_values.append(candidate)
        mappings.append(
            {
                "evidence_id": evidence,
                "value_id": matched_values[0] if len(matched_values) == 1 else None,
                "supporting_quote": supporting_quote,
                "mapping_stage": "lexical",
            }
        )
    filtered = []
    for item in mappings:
        quote = normalize(item["supporting_quote"])
        nested = any(
            item["evidence_id"] != other["evidence_id"]
            and quote != normalize(other["supporting_quote"])
            and f" {quote} " in f" {normalize(other['supporting_quote'])} "
            for other in mappings
        )
        if not nested:
            filtered.append(item)
    return sorted(filtered, key=lambda item: item["evidence_id"])


def claim_sha(claim: str) -> str:
    return hashlib.sha256(clean(claim).encode("utf-8")).hexdigest()


def protocol_cache_key(
    claim: str,
    *,
    ontology_sha256: str,
    alias_sha256: str,
    prompt_sha256: str,
    model_id: str,
) -> str:
    payload = {
        "claim": clean(claim),
        "ontology_sha256": ontology_sha256,
        "alias_sha256": alias_sha256,
        "prompt_sha256": prompt_sha256,
        "model_id": clean(model_id),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def make_batch_prompt(
    claims: list[dict[str, str]], ontology: dict[str, Any], template: str
) -> str:
    return template.replace(
        "{{ONTOLOGY_JSON}}", canonical_json(ontology["evidence"])
    ).replace("{{CLAIMS_JSON}}", canonical_json(claims))


def extract_json_object(text: Any) -> dict[str, Any]:
    source = str(text or "").strip()
    if source.startswith("```"):
        source = re.sub(r"^```(?:json)?\s*", "", source, flags=re.I)
        source = re.sub(r"\s*```$", "", source)
    try:
        value = json.loads(source)
    except json.JSONDecodeError:
        start, end = source.find("{"), source.rfind("}")
        if start < 0 or end <= start:
            raise
        value = json.loads(source[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("Mapper response is not a JSON object")
    return value


def validate_mapping(
    mapping: Any,
    *,
    claim: str,
    labels: set[str],
    values_by_evidence: dict[str, set[str]],
) -> dict[str, Any] | None:
    if not isinstance(mapping, dict):
        return None
    evidence = clean(mapping.get("evidence_id"))
    quote = clean(mapping.get("supporting_quote"))
    if evidence not in labels or not quote or quote not in clean(claim):
        return None
    value = clean(mapping.get("value_id")) or None
    if value is not None and value not in values_by_evidence.get(evidence, set()):
        value = None
    return {
        "evidence_id": evidence,
        "value_id": value,
        "supporting_quote": quote,
        "mapping_stage": "semantic",
    }


def parse_batch_response(
    response: Any,
    *,
    expected_claims: dict[str, str],
    ontology: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    parsed = extract_json_object(response)
    rows = parsed.get("results")
    if not isinstance(rows, list):
        raise ValueError("Mapper response has no results list")
    labels = {str(item["evidence_id"]) for item in ontology["evidence"]}
    values = {
        str(item["evidence_id"]): {
            str(value["value_id"]) for value in item.get("values") or []
        }
        for item in ontology["evidence"]
    }
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        identifier = clean(row.get("claim_id"))
        if identifier not in expected_claims or identifier in result:
            continue
        accepted = []
        seen = set()
        for raw in row.get("mappings") or []:
            item = validate_mapping(
                raw,
                claim=expected_claims[identifier],
                labels=labels,
                values_by_evidence=values,
            )
            if item is None or item["evidence_id"] in seen:
                continue
            seen.add(item["evidence_id"])
            accepted.append(item)
        result[identifier] = sorted(accepted, key=lambda item: item["evidence_id"])
    missing = set(expected_claims) - set(result)
    if missing:
        raise ValueError(f"Mapper response misses {len(missing)} claims")
    return result


def prepare_items(
    rows: Iterable[dict[str, Any]], alias_table: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    prepared = []
    residual: dict[str, str] = {}
    for row in rows:
        claims = claims_from_text(str(row.get("text") or ""))
        claim_rows = []
        for claim in claims:
            identifier = claim_sha(claim)
            lexical = lexical_mappings(claim, alias_table)
            if not lexical:
                residual.setdefault(identifier, claim)
            claim_rows.append(
                {"claim_id": identifier, "text": claim, "lexical_mappings": lexical}
            )
        prepared.append({**row, "claims": claim_rows})
    return prepared, residual


def materialize_items(
    prepared: list[dict[str, Any]], semantic: dict[str, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    outputs = []
    for row in prepared:
        selected: dict[str, dict[str, Any]] = {}
        for claim in row["claims"]:
            mappings = claim["lexical_mappings"] or semantic.get(claim["claim_id"], [])
            for mapping in mappings:
                evidence = mapping["evidence_id"]
                selected.setdefault(
                    evidence,
                    {
                        **mapping,
                        "text": claim["text"],
                    },
                )
        outputs.append(
            {
                **{key: value for key, value in row.items() if key != "claims"},
                "selected_claims": [selected[key] for key in sorted(selected)],
            }
        )
    return outputs
