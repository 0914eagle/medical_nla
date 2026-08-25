"""Audit a restricted DiReCT release without emitting note text or patient IDs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


NODE_RE = re.compile(r"\$(Input|Cause|Intermedia)_?(\d+)$")
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


def canonical_json_digest(path: Path) -> str:
    payload = read_json(path)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return digest_bytes(canonical)


def compare_kg_roots(release_root: Path, reference_root: Path) -> dict[str, Any]:
    release = {path.stem: path for path in release_root.rglob("*.json")}
    reference = {path.stem: path for path in reference_root.rglob("*.json")}
    common = sorted(set(release) & set(reference))
    equal: list[str] = []
    different: list[str] = []
    invalid: list[str] = []
    for name in common:
        try:
            is_equal = canonical_json_digest(release[name]) == canonical_json_digest(
                reference[name]
            )
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid.append(name)
            continue
        (equal if is_equal else different).append(name)
    return {
        "release_files": len(release),
        "reference_files": len(reference),
        "common_files": len(common),
        "semantic_hash_equal": len(equal),
        "semantic_hash_different": different,
        "invalid_common_files": invalid,
        "release_only": sorted(set(release) - set(reference)),
        "reference_only": sorted(set(reference) - set(release)),
    }


def normalize_data_root(value: str) -> str:
    parts = [part for part in value.replace("\\", "/").split("/") if part not in {"", "."}]
    return "/".join(parts)


def data_root_candidates(value: str) -> list[str]:
    normalized = normalize_data_root(value)
    parts = normalized.split("/") if normalized else []
    candidates = [normalized]
    if parts and parts[0].casefold() == "samples":
        candidates.append("/".join(parts[1:]))
    else:
        candidates.append(f"samples/{normalized}")
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def resolve_release_path(value: str, release_paths: dict[str, Path]) -> str | None:
    candidates = data_root_candidates(value)
    for candidate in candidates:
        if candidate in release_paths:
            return candidate

    suffix_matches = {
        release_relative
        for release_relative in release_paths
        for candidate in candidates
        if release_relative.endswith(f"/{candidate}")
        or candidate.endswith(f"/{release_relative}")
    }
    if len(suffix_matches) == 1:
        return next(iter(suffix_matches))
    return None


def directory_group(value: str, category_dirs: set[str]) -> str | None:
    parts = normalize_data_root(value).split("/")
    parent_parts = parts[:-1]
    category_lookup = {category.casefold() for category in category_dirs}
    for index, part in enumerate(parent_parts):
        if part.casefold() in category_lookup:
            return "/".join(parent_parts[index:]).casefold()
    return None


def audit_data_list(samples_root: Path, data_list_path: Path) -> dict[str, Any]:
    with data_list_path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    required = {"Disease Category", "PDD", "Data Root", "Whether Amended"}
    fields = set(rows[0]) if rows else set()
    missing_fields = sorted(required - fields)
    if missing_fields:
        raise ValueError(f"Data list is missing required columns: {missing_fields}")

    release_paths = {
        path.relative_to(samples_root).as_posix(): path
        for path in samples_root.rglob("*.json")
    }
    listed_category_dirs = {
        parts[1]
        for row in rows
        if len(parts := normalize_data_root(row["Data Root"]).split("/")) >= 3
        and parts[0].casefold() == "samples"
    }
    release_groups = Counter(
        group
        for relative in release_paths
        if (group := directory_group(relative, listed_category_dirs)) is not None
    )
    listed_groups = Counter(
        group
        for row in rows
        if (
            group := directory_group(row["Data Root"], listed_category_dirs)
        )
        is not None
    )
    release_basenames = Counter(Path(path).name for path in release_paths)
    listed_basenames = Counter(
        Path(normalize_data_root(row["Data Root"])).name for row in rows
    )
    listed_paths: Counter[str] = Counter()
    matched_release_paths: set[str] = set()
    categories: Counter[str] = Counter()
    pdd_pairs: Counter[tuple[str, str]] = Counter()
    amended: Counter[str] = Counter()
    path_category_mismatches: Counter[tuple[str, str]] = Counter()
    path_pdd_mismatches: Counter[tuple[str, str]] = Counter()
    root_pdd_mismatches: Counter[tuple[str, str]] = Counter()
    matched = 0
    invalid_matched_json = 0

    for row in rows:
        listed_relative = normalize_data_root(row["Data Root"])
        listed_paths[listed_relative] += 1
        listed_category = row["Disease Category"].strip()
        listed_pdd = row["PDD"].strip()
        categories[listed_category] += 1
        pdd_pairs[(listed_category, listed_pdd)] += 1
        amended[row["Whether Amended"].strip() or "<empty>"] += 1

        matched_relative = resolve_release_path(row["Data Root"], release_paths)
        path = release_paths.get(matched_relative) if matched_relative else None
        if path is None:
            continue
        matched += 1
        matched_release_paths.add(matched_relative)
        parts = Path(matched_relative).parts
        if len(parts) >= 4:
            path_category, path_pdd = parts[-3], parts[-2]
        elif len(parts) == 3:
            path_category, path_pdd = parts[-2], None
        else:
            path_category, path_pdd = None, None

        if path_category and path_category.casefold() != listed_category.casefold():
            path_category_mismatches[(path_category, listed_category)] += 1
        if path_pdd and path_pdd.casefold() != listed_pdd.casefold():
            path_pdd_mismatches[(path_pdd, listed_pdd)] += 1

        try:
            payload = read_json(path)
        except (UnicodeDecodeError, json.JSONDecodeError):
            invalid_matched_json += 1
            continue
        if not isinstance(payload, dict):
            invalid_matched_json += 1
            continue
        root_keys = [key for key in payload if not INPUT_RE.fullmatch(str(key))]
        if len(root_keys) == 1:
            root_pdd = strip_node_suffix(root_keys[0])
            if root_pdd.casefold() != listed_pdd.casefold():
                root_pdd_mismatches[(listed_pdd, root_pdd)] += 1

    duplicate_roots = [count for count in listed_paths.values() if count > 1]
    return {
        "rows": len(rows),
        "unique_data_roots": len(listed_paths),
        "duplicate_data_roots": sum(count - 1 for count in duplicate_roots),
        "matched_release_files": matched,
        "listed_paths_missing_from_release": sum(
            1
            for row in rows
            if resolve_release_path(row["Data Root"], release_paths) is None
        ),
        "release_files_missing_from_list": len(set(release_paths) - matched_release_paths),
        "release_path_depths": dict(
            sorted(Counter(len(Path(path).parts) for path in release_paths).items())
        ),
        "listed_path_depths": dict(
            sorted(
                Counter(
                    len(Path(normalize_data_root(row["Data Root"])).parts)
                    for row in rows
                ).items()
            )
        ),
        "directory_groups_release": len(release_groups),
        "directory_groups_listed": len(listed_groups),
        "directory_groups_shared": len(set(release_groups) & set(listed_groups)),
        "directory_groups_with_equal_counts": sum(
            release_groups[group] == listed_groups[group]
            for group in set(release_groups) & set(listed_groups)
        ),
        "unique_release_basenames": len(release_basenames),
        "unique_listed_basenames": len(listed_basenames),
        "exact_basename_overlap": len(set(release_basenames) & set(listed_basenames)),
        "row_identity_available": (
            len(set(release_basenames) & set(listed_basenames)) == len(rows)
        ),
        "invalid_matched_json": invalid_matched_json,
        "disease_categories": len(categories),
        "category_counts": dict(categories.most_common()),
        "pdd_pairs": len(pdd_pairs),
        "pdd_count_summary": summarize_numbers(pdd_pairs.values()),
        "amendment_counts": dict(amended.most_common()),
        "path_category_mismatch_pairs": {
            f"{path_category} -> {listed_category}": count
            for (path_category, listed_category), count in path_category_mismatches.most_common()
        },
        "path_pdd_mismatch_pairs": {
            f"{path_pdd} -> {listed_pdd}": count
            for (path_pdd, listed_pdd), count in path_pdd_mismatches.most_common()
        },
        "listed_pdd_vs_root_mismatch_pairs": {
            f"{listed_pdd} -> {root_pdd}": count
            for (listed_pdd, root_pdd), count in root_pdd_mismatches.most_common()
        },
        "listed_pdd_vs_root_mismatches": (
            sum(root_pdd_mismatches.values()) if matched else None
        ),
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
    data_list = result.get("official_data_list")
    if data_list:
        lines.extend(
            [
                "## Official Data List Alignment",
                "",
                f"- rows / unique roots: **{data_list['rows']} / {data_list['unique_data_roots']}**",
                f"- matched release files: **{data_list['matched_release_files']}**",
                f"- listed paths missing from release: **{data_list['listed_paths_missing_from_release']}**",
                f"- release files missing from list: **{data_list['release_files_missing_from_list']}**",
                f"- release/listed path depths: `{data_list['release_path_depths']}` / `{data_list['listed_path_depths']}`",
                f"- shared directory groups: **{data_list['directory_groups_shared']} / {data_list['directory_groups_listed']}**",
                f"- directory groups with equal row counts: **{data_list['directory_groups_with_equal_counts']} / {data_list['directory_groups_listed']}**",
                f"- exact basename overlap: **{data_list['exact_basename_overlap']}**",
                f"- row identity available from path: **{data_list['row_identity_available']}**",
                f"- disease categories / category-PDD pairs: **{data_list['disease_categories']} / {data_list['pdd_pairs']}**",
                f"- PDD-pair count summary: `{data_list['pdd_count_summary']}`",
                f"- amendment counts: `{data_list['amendment_counts']}`",
                f"- path category label mappings: `{data_list['path_category_mismatch_pairs']}`",
                f"- path PDD mismatches: `{data_list['path_pdd_mismatch_pairs']}`",
                f"- listed PDD vs annotation-root mismatches: **{data_list['listed_pdd_vs_root_mismatches'] if data_list['listed_pdd_vs_root_mismatches'] is not None else 'N/A (no row identity)'}**",
                f"- listed PDD -> annotation-root mappings: `{data_list['listed_pdd_vs_root_mismatch_pairs']}`",
                "",
                "When directory groups align but basename overlap is zero, the restricted release has renamed files. The public data list remains valid for aggregate vocabulary and counts, but its row-level amendment flags cannot be joined by path.",
                "",
                "The official evaluator derives diagnosis accuracy from the annotation chain root, not the folder label. Report a sensitivity analysis excluding list/root mismatches.",
                "",
            ]
        )
    kg_comparison = result.get("reference_kg_comparison")
    if kg_comparison:
        lines.extend(
            [
                "## Reference KG Comparison",
                "",
                f"- release / reference files: **{kg_comparison['release_files']} / {kg_comparison['reference_files']}**",
                f"- common files: **{kg_comparison['common_files']}**",
                f"- canonical JSON hash equal: **{kg_comparison['semantic_hash_equal']}**",
                f"- canonical JSON hash different: `{kg_comparison['semantic_hash_different']}`",
                f"- release-only categories: `{kg_comparison['release_only']}`",
                f"- reference-only categories: `{kg_comparison['reference_only']}`",
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
    parser.add_argument(
        "--data-list",
        type=Path,
        help="Optional public DiReCT data_list.csv used for aggregate release alignment.",
    )
    parser.add_argument(
        "--reference-kg-root",
        type=Path,
        help="Optional public DiReCT KG root used for canonical JSON hash comparison.",
    )
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
    if args.data_list:
        result["official_data_list"] = audit_data_list(args.samples_root, args.data_list)
    if args.reference_kg_root:
        result["reference_kg_comparison"] = compare_kg_roots(
            args.kg_root, args.reference_kg_root
        )
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
