"""Create probe-ready prompt/activation rows from DDXPlus.

The output has two JSONL files:
- cases: one row per selected DDXPlus patient.
- variants: extraction rows compatible with `python -m src.extract_activations`.

The generated prompts intentionally include cue phrases verbatim so target_text
matching is stable.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import random
import re
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any, Iterable


NEGATIVE_VALUE_LABELS = {
    "n",
    "no",
    "false",
    "absent",
    "none",
    "never",
    "not",
    "negative",
    "0",
}

LOW_INFORMATION_VALUE_LABELS = {
    "nowhere",
    "no where",
    "none",
    "unknown",
    "unspecified",
}

GENERIC_CUE_PATTERNS = [
    r"\bpain somewhere\b",
    r"\bhow fast did the pain appear\b",
    r"\bhow intense is the pain\b",
    r"\bhow precisely is the pain located\b",
    r"\bcharacterize their pain\b",
    r"\bdoes the pain radiate\b",
    r"\bpain radiate\b",
]


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def read_patient_rows(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
        return
    if suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            yield from data
            return
        if isinstance(data, dict):
            for key in ("data", "rows", "patients"):
                if isinstance(data.get(key), list):
                    yield from data[key]
                    return
        raise ValueError(f"Could not find patient rows in JSON file {path}")

    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def get_field(row: dict[str, Any], names: list[str], *, required: bool = True) -> Any:
    lowered = {key.lower(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return row[key]
    if required:
        raise KeyError(f"None of fields {names} found in row keys {list(row.keys())}")
    return None


def parse_maybe_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(text)
            except (SyntaxError, ValueError):
                return value
    return value


def parse_evidence_entries(value: Any) -> list[str]:
    parsed = parse_maybe_json(value)
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    if isinstance(parsed, dict):
        return [str(key) for key, present in parsed.items() if present]
    if isinstance(parsed, str):
        for sep in (";", "|"):
            if sep in parsed:
                return [part.strip() for part in parsed.split(sep) if part.strip()]
        if "," in parsed:
            return [part.strip() for part in parsed.split(",") if part.strip()]
        if parsed.strip():
            return [parsed.strip()]
    return []


def evidence_base_and_value(entry: str) -> tuple[str, str | None]:
    for sep in ("_@_", "@", ":"):
        if sep in entry:
            base, value = entry.split(sep, 1)
            return base, value
    return entry, None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_label(text: str) -> str:
    return normalize_space(str(text).strip().strip("\"'")).lower()


def strip_question_to_phrase(text: str) -> str:
    phrase = normalize_space(text)
    phrase = phrase.strip(" ?.")
    phrase = re.sub(
        r"^(do you|did you|are you|were you|have you|has the patient|does the patient)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(
        r"^(have|has|feel|experience|experiencing|suffer from|present with|"
        r"noticed|notice|observed|observe|objectified|felt|measured)\s+",
        "",
        phrase,
        flags=re.I,
    )
    phrase = re.sub(r"\bwhen you exhale\b", "when exhaling", phrase, flags=re.I)
    phrase = re.sub(r"\byour\b", "their", phrase, flags=re.I)
    phrase = re.sub(r"\byou\b", "the patient", phrase, flags=re.I)
    phrase = re.sub(r"\b(yes or no|right now|currently)\b", "", phrase, flags=re.I)
    phrase = normalize_space(phrase.strip(" ?.:;,-"))
    if not phrase:
        phrase = normalize_space(text).strip(" ?.")
    return phrase[:1].lower() + phrase[1:]


def lookup_meta(evidence_meta: dict[str, Any], evidence_id: str) -> dict[str, Any]:
    meta = evidence_meta.get(evidence_id)
    if isinstance(meta, dict):
        return meta
    return {}


def meta_text(meta: dict[str, Any], fallback: str) -> str:
    for key in (
        "question_en",
        "question",
        "name",
        "label",
        "display_name",
        "description",
        "text",
    ):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def possible_value_label(meta: dict[str, Any], value_id: str | None) -> str | None:
    if value_id is None:
        return None
    value_meaning = meta.get("value_meaning")
    if isinstance(value_meaning, dict):
        value_meta = value_meaning.get(value_id)
        if isinstance(value_meta, str):
            return value_meta
        if isinstance(value_meta, dict):
            for label_key in (
                "en",
                "label",
                "name",
                "value",
                "text",
                "meaning",
                "value_meaning",
                "value_label",
            ):
                label = value_meta.get(label_key)
                if isinstance(label, str) and label.strip():
                    return label
    for label_key in ("value_label", "meaning"):
        label = meta.get(label_key)
        if isinstance(label, str) and label.strip():
            return label
    for key in ("possible-values", "possible_values", "values"):
        values = meta.get(key)
        if not isinstance(values, dict):
            continue
        value_meta = values.get(value_id)
        if isinstance(value_meta, str):
            return value_meta
        if isinstance(value_meta, dict):
            for label_key in (
                "en",
                "label",
                "name",
                "value",
                "text",
                "meaning",
                "value_meaning",
                "value_label",
            ):
                label = value_meta.get(label_key)
                if isinstance(label, str) and label.strip():
                    return label
    return None


def is_negative_or_low_information_value(value_label: str | None) -> bool:
    if value_label is None:
        return False
    label = normalize_label(value_label)
    return label in NEGATIVE_VALUE_LABELS or label in LOW_INFORMATION_VALUE_LABELS


def is_generic_cue_text(text: str) -> bool:
    normalized = normalize_label(text)
    return any(re.search(pattern, normalized) for pattern in GENERIC_CUE_PATTERNS)


def is_antecedent(meta: dict[str, Any]) -> bool:
    value = meta.get("is_antecedent")
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return False


def cue_from_entry(
    entry: str,
    evidence_meta: dict[str, Any],
    *,
    clean_cues: bool,
) -> dict[str, Any]:
    base_id, value_id = evidence_base_and_value(entry)
    meta = lookup_meta(evidence_meta, base_id)
    phrase = strip_question_to_phrase(meta_text(meta, base_id))
    value_label = possible_value_label(meta, value_id)
    excluded = False
    exclusion_reason = None
    if clean_cues and is_negative_or_low_information_value(value_label):
        excluded = True
        exclusion_reason = "negative_or_low_information_value"
    if value_label and value_label.lower() not in {"yes", "true", "present"}:
        phrase = normalize_space(f"{phrase} {value_label}")
    if clean_cues and is_generic_cue_text(phrase):
        excluded = True
        exclusion_reason = exclusion_reason or "generic_cue"
    return {
        "evidence_id": base_id,
        "evidence_entry": entry,
        "value_id": value_id,
        "value_label": value_label,
        "cue_text": phrase,
        "cue_type": "antecedent" if is_antecedent(meta) else "symptom",
        "is_antecedent": is_antecedent(meta),
        "excluded": excluded,
        "exclusion_reason": exclusion_reason,
    }


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "unknown"


def join_cues(cues: list[str]) -> str:
    if len(cues) == 1:
        return cues[0]
    if len(cues) == 2:
        return f"{cues[0]} and {cues[1]}"
    return ", ".join(cues[:-1]) + f", and {cues[-1]}"


def make_prompt(cues: list[str]) -> str:
    return f"A patient presents with {join_cues(cues)}. What diagnosis is most likely?"


def make_case(
    row: dict[str, Any],
    *,
    row_index: int,
    evidence_meta: dict[str, Any],
    rng: random.Random,
    prefer_symptoms: bool,
    max_cues: int,
    clean_cues: bool = True,
) -> dict[str, Any] | None:
    pathology = str(get_field(row, ["PATHOLOGY", "pathology", "diagnosis", "label"]))
    patient_id = get_field(row, ["id", "patient_id", "PATIENT", "patient"], required=False)
    if patient_id is None:
        patient_id = f"row_{row_index:07d}"
    entries = parse_evidence_entries(get_field(row, ["EVIDENCES", "evidences", "evidence"]))
    all_cues = [
        cue_from_entry(entry, evidence_meta, clean_cues=clean_cues)
        for entry in entries
    ]
    cues = [
        cue
        for cue in all_cues
        if cue["cue_text"] and cue["cue_text"].lower() != "none" and not cue["excluded"]
    ]
    symptom_cues = [cue for cue in cues if not cue["is_antecedent"]]
    candidate_cues = symptom_cues if prefer_symptoms and len(symptom_cues) >= max_cues else cues
    if len(candidate_cues) < max_cues:
        return None
    selected = rng.sample(candidate_cues, max_cues)
    cue_targets = [cue["cue_text"] for cue in selected]
    diagnosis_id = slug(pathology)
    case_id = f"ddxplus_{diagnosis_id}_{row_index:07d}"
    return {
        "id": case_id,
        "source": "ddxplus",
        "patient_id": str(patient_id),
        "diagnosis_id": diagnosis_id,
        "diagnosis_name": pathology,
        "diagnosis_aliases": [pathology],
        "cue_targets": cue_targets,
        "cue_types": [cue["cue_type"] for cue in selected],
        "cue_evidence_ids": [cue["evidence_id"] for cue in selected],
        "cue_evidence_entries": [cue["evidence_entry"] for cue in selected],
        "cue_value_ids": [cue["value_id"] for cue in selected],
        "cue_value_labels": [cue["value_label"] for cue in selected],
        "clean_cues": clean_cues,
        "excluded_cue_count": sum(1 for cue in all_cues if cue["excluded"]),
        "excluded_cue_entries": [
            {
                "evidence_entry": cue["evidence_entry"],
                "cue_text": cue["cue_text"],
                "exclusion_reason": cue["exclusion_reason"],
            }
            for cue in all_cues
            if cue["excluded"]
        ],
        "single_prompt": make_prompt([cue_targets[0]]),
        "multi_prompt": make_prompt(cue_targets),
    }


def variant_rows(case: dict[str, Any]) -> list[dict[str, Any]]:
    common = {
        "base_id": case["id"],
        "source": case["source"],
        "patient_id": case["patient_id"],
        "diagnosis_id": case["diagnosis_id"],
        "diagnosis_name": case["diagnosis_name"],
        "diagnosis_aliases": case["diagnosis_aliases"],
        "cue_targets": case["cue_targets"],
        "cue_types": case["cue_types"],
        "cue_evidence_ids": case["cue_evidence_ids"],
        "cue_evidence_entries": case["cue_evidence_entries"],
        "cue_value_ids": case["cue_value_ids"],
        "cue_value_labels": case["cue_value_labels"],
        "clean_cues": case["clean_cues"],
        "excluded_cue_count": case["excluded_cue_count"],
        "excluded_cue_entries": case["excluded_cue_entries"],
    }
    rows = [
        {
            **common,
            "id": f"{case['id']}__single_cue",
            "variant": "single_cue",
            "condition": "single",
            "target_role": "cue",
            "cue_index": 1,
            "prompt": case["single_prompt"],
            "position_mode": "target_text",
            "target_text": case["cue_targets"][0],
            "target_text_strategy": "span_mean",
        },
        {
            **common,
            "id": f"{case['id']}__single_format",
            "variant": "single_format",
            "condition": "single",
            "target_role": "format",
            "cue_index": None,
            "prompt": case["single_prompt"],
            "position_mode": "last_token",
            "target_text": None,
            "target_text_strategy": None,
        },
    ]
    for idx, cue in enumerate(case["cue_targets"], start=1):
        rows.append(
            {
                **common,
                "id": f"{case['id']}__multi_cue_{idx}",
                "variant": f"multi_cue_{idx}",
                "condition": "multi",
                "target_role": "cue",
                "cue_index": idx,
                "prompt": case["multi_prompt"],
                "position_mode": "target_text",
                "target_text": cue,
                "target_text_strategy": "span_mean",
            }
        )
    rows.append(
        {
            **common,
            "id": f"{case['id']}__multi_format",
            "variant": "multi_format",
            "condition": "multi",
            "target_role": "format",
            "cue_index": None,
            "prompt": case["multi_prompt"],
            "position_mode": "last_token",
            "target_text": None,
            "target_text_strategy": None,
        }
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True)
    parser.add_argument("--evidences", required=True)
    parser.add_argument("--cases-output", required=True)
    parser.add_argument("--variants-output", required=True)
    parser.add_argument("--max-diagnoses", type=int, default=49)
    parser.add_argument("--examples-per-diagnosis", type=int, default=100)
    parser.add_argument("--max-cues", type=int, default=3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--clean-cues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Filter negative/low-information values and generic pain metadata "
            "before sampling cues. Use --no-clean-cues to reproduce v1 behavior."
        ),
    )
    parser.add_argument(
        "--prefer-symptoms",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Prefer non-antecedent symptom cues when at least --max-cues are available.",
    )
    parser.add_argument("--max-patient-rows", type=int, default=None)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    evidence_meta = read_json(Path(args.evidences))
    if not isinstance(evidence_meta, dict):
        raise ValueError("Evidence metadata must be a JSON object keyed by evidence id.")

    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    patients = read_patient_rows(Path(args.patients))
    if args.max_patient_rows is not None:
        patients = islice(patients, args.max_patient_rows)
    for row_index, row in enumerate(patients):
        case = make_case(
            row,
            row_index=row_index,
            evidence_meta=evidence_meta,
            rng=rng,
            prefer_symptoms=args.prefer_symptoms,
            max_cues=args.max_cues,
            clean_cues=args.clean_cues,
        )
        if case is None:
            continue
        bucket = by_diagnosis[case["diagnosis_id"]]
        if len(bucket) < args.examples_per_diagnosis:
            bucket.append(case)

    selected_diagnoses = [
        diagnosis
        for diagnosis, cases in sorted(
            by_diagnosis.items(), key=lambda item: (-len(item[1]), item[0])
        )
        if len(cases) >= args.examples_per_diagnosis
    ][: args.max_diagnoses]
    cases = [case for diagnosis in selected_diagnoses for case in by_diagnosis[diagnosis]]
    variants = [row for case in cases for row in variant_rows(case)]

    write_jsonl(Path(args.cases_output), cases)
    write_jsonl(Path(args.variants_output), variants)

    print(f"diagnoses_selected: {len(selected_diagnoses)}")
    print(f"cases_written: {len(cases)}")
    print(f"variants_written: {len(variants)}")
    print(f"variants_per_case: {2 + args.max_cues + 1}")
    print(f"clean_cues: {args.clean_cues}")
    print(f"excluded_cues_total: {sum(case['excluded_cue_count'] for case in cases)}")
    print("top_diagnoses:")
    for diagnosis, count in Counter(case["diagnosis_id"] for case in cases).most_common(20):
        print(f"  {diagnosis}: {count}")


if __name__ == "__main__":
    main()
