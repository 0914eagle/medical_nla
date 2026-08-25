"""Audit a restricted DiReCT release without emitting note text or patient IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NODE_RE = re.compile(r"\$(Input|Cause|Intermedia)_(\d+)$")
INPUT_RE = re.compile(r"^input(\d+)$", re.IGNORECASE)
PATIENT_RE = re.compile(r"^(.+?)-DS-", re.IGNORECASE)


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def summarize_numbers(values: Iterable[int]) -> dict[str, float | int | None]:
    xs = list(values)
    if not xs:
        return {"n": 0, "min": None, "median": None, "mean": None, "max": None}
    return {
        "n": len(xs),
        "min": min(xs),
        "median": statistics.median(xs),
        "mean": statistics.fmean(xs),
        "max": max(xs),
    }


def duplicate_summary(hashes: Counter[str]) -> dict[str, int]:
    duplicate_groups = [count for count in hashes.values() if count > 1]
    return {
        "unique": len(hashes),
        "duplicate_groups": len(duplicate_groups),
        "rows_in_duplicate_groups": sum(duplicate_groups),
        "extra_duplicate_rows": sum(count - 1 for count in duplicate_groups),
        "max_group_size": max(duplicate_groups, default=1),
    }


def strip_node_suffix(key: str) -> str:
    return NODE_RE.sub("", key).strip()


def walk_node_keys(value: Any) -> Iterable[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if isinstance(key, str) and NODE_RE.search(key):
                yield key
            yield from walk_node_keys(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_node_keys(child)


def infer_layout(relative: Path, root_keys: list[str]) -> tuple[str, str, str]:
    parts = relative.parts
    bucket = parts[0] if len(parts) >= 2 else "<root>"
    root_diagnosis = strip_node_suffix(root_keys[0]) if len(root_keys) == 1 else "<ambiguous>"

    if len(parts) >= 4:
        return bucket, parts[-3], parts[-2]
    if len(parts) == 3:
        return bucket, parts[-2], root_diagnosis
    return bucket, "<unknown>", root_diagnosis


def audit_samples(samples_root: Path) -> dict[str, Any]:
    files = sorted(samples_root.rglob("*.json"))
    path_depths: Counter[int] = Counter()
    archive_buckets: Counter[str] = Counter()
    categories: Counter[str] = Counter()
    pdd_pairs: Counter[tuple[str, str]] = Counter()
    root_key_counts: Counter[int] = Counter()
    node_type_counts: Counter[str] = Counter()
    node_counts_per_note: list[int] = []
    input_field_presence: Counter[str] = Counter()
    input_field_empty: Counter[str] = Counter()
    input_chars_per_note: list[int] = []
    unknown_top_keys: Counter[str] = Counter()
    file_hashes: Counter[str] = Counter()
    input_hashes: Counter[str] = Counter()
    patient_counts: Counter[str] = Counter()
    unparsed_patient_rows = 0
    invalid_json = 0
    non_dict_rows = 0
    path_root_pdd_mismatch = 0
    path_root_pdd_mismatch_pairs: Counter[tuple[str, str]] = Counter()

    for path in files:
        relative = path.relative_to(samples_root)
        path_depths[len(relative.parts)] += 1
        raw = path.read_bytes()
        file_hashes[digest_bytes(raw)] += 1

        patient_match = PATIENT_RE.match(path.stem)
        if patient_match:
            patient_counts[digest_bytes(patient_match.group(1).encode())] += 1
        else:
            unparsed_patient_rows += 1

        try:
            row = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_json += 1
            continue
        if not isinstance(row, dict):
            non_dict_rows += 1
            continue

        root_keys = [key for key in row if not INPUT_RE.fullmatch(str(key))]
        root_key_counts[len(root_keys)] += 1
        bucket, category, pdd = infer_layout(relative, root_keys)
        archive_buckets[bucket] += 1
        categories[category] += 1
        pdd_pairs[(category, pdd)] += 1

        if len(relative.parts) >= 4 and len(root_keys) == 1:
            path_pdd = relative.parts[-2]
            root_pdd = strip_node_suffix(root_keys[0])
            if path_pdd.casefold() != root_pdd.casefold():
                path_root_pdd_mismatch += 1
                path_root_pdd_mismatch_pairs[(path_pdd, root_pdd)] += 1

        inputs: list[str] = []
        for key, value in row.items():
            match = INPUT_RE.fullmatch(str(key))
            if not match:
                if key not in root_keys:
                    unknown_top_keys[str(key)] += 1
                continue
            canonical = f"input{int(match.group(1))}"
            input_field_presence[canonical] += 1
            text = value if isinstance(value, str) else ""
            if not text.strip():
                input_field_empty[canonical] += 1
            inputs.append(f"{canonical}\0{text}")

        input_chars_per_note.append(sum(len(item.split("\0", 1)[1]) for item in inputs))
        input_hashes[digest_bytes("\n".join(sorted(inputs)).encode("utf-8"))] += 1

        node_keys = list(walk_node_keys(row))
        node_counts_per_note.append(len(node_keys))
        for key in node_keys:
            match = NODE_RE.search(key)
            if match:
                node_type_counts[match.group(1)] += 1

    repeated_patients = [count for count in patient_counts.values() if count > 1]
    return {
        "json_files": len(files),
        "invalid_json": invalid_json,
        "non_dict_rows": non_dict_rows,
        "path_depths": dict(sorted(path_depths.items())),
        "archive_buckets": dict(archive_buckets.most_common()),
        "disease_categories": len(categories),
        "category_counts": dict(categories.most_common()),
        "pdd_pairs": len(pdd_pairs),
        "pdd_pair_counts": {
            f"{category} / {pdd}": count
            for (category, pdd), count in pdd_pairs.most_common()
        },
        "pdd_count_summary": summarize_numbers(pdd_pairs.values()),
        "root_key_counts": dict(sorted(root_key_counts.items())),
        "path_root_pdd_mismatch": path_root_pdd_mismatch,
        "path_root_pdd_mismatch_pairs": {
            f"{path_pdd} -> {root_pdd}": count
            for (path_pdd, root_pdd), count in path_root_pdd_mismatch_pairs.most_common()
        },
        "node_type_counts": dict(node_type_counts.most_common()),
        "nodes_per_note": summarize_numbers(node_counts_per_note),
        "input_field_presence": dict(sorted(input_field_presence.items())),
        "input_field_empty": dict(sorted(input_field_empty.items())),
        "input_chars_per_note": summarize_numbers(input_chars_per_note),
        "unknown_top_keys": dict(unknown_top_keys.most_common()),
        "exact_file_duplicates": duplicate_summary(file_hashes),
        "input_text_duplicates": duplicate_summary(input_hashes),
        "patient_id_groups": {
            "parsed_rows": sum(patient_counts.values()),
            "unparsed_rows": unparsed_patient_rows,
            "unique_hashed_patients": len(patient_counts),
            "repeated_patient_groups": len(repeated_patients),
            "rows_from_repeated_patients": sum(repeated_patients),
            "max_notes_per_patient": max(repeated_patients, default=1),
        },
    }


def audit_kg(kg_root: Path) -> dict[str, Any]:
    files = sorted(kg_root.rglob("*.json"))
    top_key_sets: Counter[tuple[str, ...]] = Counter()
    invalid_json = 0
    diagnostic_present = 0
    knowledge_present = 0

    for path in files:
        try:
            row = read_json(path)
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_json += 1
            continue
        if not isinstance(row, dict):
            top_key_sets[(f"<{type(row).__name__}>",)] += 1
            continue
        keys = tuple(sorted(str(key) for key in row))
        top_key_sets[keys] += 1
        diagnostic_present += int("diagnostic" in row)
        knowledge_present += int("knowledge" in row)

    return {
        "json_files": len(files),
        "category_names": sorted(path.stem for path in files),
        "invalid_json": invalid_json,
        "diagnostic_key_present": diagnostic_present,
        "knowledge_key_present": knowledge_present,
        "top_key_sets": {" | ".join(keys): count for keys, count in top_key_sets.items()},
    }


def markdown_summary(result: dict[str, Any]) -> str:
    samples = result["samples"]
    kg = result["knowledge_graphs"]
    lines = [
        "# DiReCT Restricted Release Audit",
        "",
        "Aggregate-only audit. No note text, patient identifier, or file name is emitted.",
        "",
        "## Release",
        "",
        f"- sample JSON files: **{samples['json_files']}**",
        f"- knowledge-graph JSON files: **{kg['json_files']}**",
        f"- invalid sample JSON: **{samples['invalid_json']}**",
        f"- disease categories: **{samples['disease_categories']}**",
        f"- category/PDD pairs: **{samples['pdd_pairs']}**",
        f"- PDD-pair count summary: `{samples['pdd_count_summary']}`",
        f"- path depths: `{samples['path_depths']}`",
        "",
        "## Structure",
        "",
        f"- root diagnosis-node counts per file: `{samples['root_key_counts']}`",
        f"- node types: `{samples['node_type_counts']}`",
        f"- nodes per note: `{samples['nodes_per_note']}`",
        f"- input field presence: `{samples['input_field_presence']}`",
        f"- empty input fields: `{samples['input_field_empty']}`",
        f"- path/root PDD mismatches: **{samples['path_root_pdd_mismatch']}**",
        f"- mismatch label pairs: `{samples['path_root_pdd_mismatch_pairs']}`",
        "",
        "## Leakage And Duplication Audit",
        "",
        f"- exact JSON duplicates: `{samples['exact_file_duplicates']}`",
        f"- identical input-text groups: `{samples['input_text_duplicates']}`",
        f"- patient grouping: `{samples['patient_id_groups']}`",
        "",
        "Patient-disjoint splitting is required when repeated patient groups are nonzero.",
        "",
        "## Category Counts",
        "",
        "| category | notes |",
        "|---|---:|",
    ]
    lines.extend(f"| {category} | {count} |" for category, count in samples["category_counts"].items())
    lines.extend(
        [
            "",
            "## Knowledge Graphs",
            "",
            f"- `diagnostic` key present: **{kg['diagnostic_key_present']}/{kg['json_files']}**",
            f"- `knowledge` key present: **{kg['knowledge_key_present']}/{kg['json_files']}**",
            f"- top-level key sets: `{kg['top_key_sets']}`",
            f"- sample-only categories: `{result['category_alignment']['sample_only']}`",
            f"- KG-only categories: `{result['category_alignment']['kg_only']}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", required=True, type=Path)
    parser.add_argument("--kg-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--expected-notes", type=int, default=511)
    args = parser.parse_args()

    result = {
        "samples": audit_samples(args.samples_root),
        "knowledge_graphs": audit_kg(args.kg_root),
    }
    sample_categories = set(result["samples"]["category_counts"])
    kg_categories = set(result["knowledge_graphs"]["category_names"])
    result["category_alignment"] = {
        "matched": sorted(sample_categories & kg_categories),
        "sample_only": sorted(sample_categories - kg_categories),
        "kg_only": sorted(kg_categories - sample_categories),
    }
    if result["samples"]["json_files"] != args.expected_notes:
        result["warning"] = (
            f"Expected {args.expected_notes} sample JSON files, found "
            f"{result['samples']['json_files']}"
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.summary_md.write_text(markdown_summary(result), encoding="utf-8")

    print(f"[audit] samples={result['samples']['json_files']} kg={result['knowledge_graphs']['json_files']}")
    print(f"[audit] categories={result['samples']['disease_categories']} pdd_pairs={result['samples']['pdd_pairs']}")
    print(f"[audit] output_json={args.output_json}")
    print(f"[audit] summary_md={args.summary_md}")


if __name__ == "__main__":
    main()
