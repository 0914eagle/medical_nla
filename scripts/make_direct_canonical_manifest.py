"""Build a private canonical manifest from the restricted DiReCT release.

The JSONL output contains restricted note text and must stay under the private
data root. The Markdown summary is aggregate-only and is safe to share.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


NODE_RE = re.compile(r"\$(Input|Cause|Intermedia)_?(\d+)$")
INPUT_RE = re.compile(r"^input(\d+)$", re.IGNORECASE)
PATIENT_RE = re.compile(r"^(.+?)-DS-", re.IGNORECASE)
SECTION_NAMES = {
    "input1": "Chief Complaint",
    "input2": "History of Present Illness",
    "input3": "Past Medical History",
    "input4": "Family History",
    "input5": "Physical Exam",
    "input6": "Pertinent Results",
}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strip_node_suffix(key: str) -> str:
    return NODE_RE.sub("", key).strip()


def node_type(key: str) -> str | None:
    match = NODE_RE.search(key)
    return match.group(1) if match else None


def read_official_vocab(path: Path | None) -> tuple[dict[str, str], set[str]]:
    if path is None:
        return {}, set()
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if rows and "PDD" not in rows[0]:
        raise ValueError("Official data list has no PDD column")
    pdds = sorted({row["PDD"].strip() for row in rows if row.get("PDD", "").strip()})
    by_casefold: dict[str, str] = {}
    for pdd in pdds:
        key = pdd.casefold()
        if key in by_casefold and by_casefold[key] != pdd:
            raise ValueError(f"Ambiguous case-insensitive PDD labels: {by_casefold[key]!r}, {pdd!r}")
        by_casefold[key] = pdd
    category_dirs: set[str] = set()
    for row in rows:
        parts = [
            part
            for part in row["Data Root"].replace("\\", "/").split("/")
            if part not in {"", "."}
        ]
        if len(parts) >= 3 and parts[0].casefold() == "samples":
            category_dirs.add(parts[1])
    return by_casefold, category_dirs


def infer_path_labels(
    relative: Path, category_dirs: set[str]
) -> tuple[str, str | None]:
    parts = relative.parts
    if category_dirs:
        category_lookup = {category.casefold() for category in category_dirs}
        for index, part in enumerate(parts[:-1]):
            if part.casefold() in category_lookup:
                folder_pdd = parts[index + 1] if index + 1 < len(parts) - 1 else None
                return part, folder_pdd
    if len(parts) >= 4:
        return parts[-3], parts[-2]
    if len(parts) == 3:
        return parts[-2], None
    raise ValueError(f"Unexpected restricted sample path depth: {len(parts)}")


def extract_annotation(
    payload: dict[str, Any], note_sections: dict[str, str]
) -> tuple[str, list[str], list[dict[str, Any]], int]:
    root_keys = [key for key in payload if not INPUT_RE.fullmatch(str(key))]
    if len(root_keys) != 1:
        raise ValueError(f"Expected one annotation root, found {len(root_keys)}")

    annotation_root = strip_node_suffix(root_keys[0])
    chain_preorder: list[str] = []
    deductions: list[dict[str, Any]] = []
    node_count = 0

    def visit(key: str, children: Any, ancestors: list[tuple[str, str]]) -> None:
        nonlocal node_count
        kind = node_type(key)
        if kind is None:
            raise ValueError(f"Annotation key has no node suffix: {key!r}")
        node_count += 1
        content = strip_node_suffix(key)
        if kind == "Intermedia":
            chain_preorder.append(content)
        if kind == "Input":
            input_match = NODE_RE.search(key)
            annotated_section = (
                f"input{int(input_match.group(2))}" if input_match else None
            )
            nearest_cause = next(
                (text for ancestor_kind, text in reversed(ancestors) if ancestor_kind == "Cause"),
                None,
            )
            nearest_diagnosis = next(
                (
                    text
                    for ancestor_kind, text in reversed(ancestors)
                    if ancestor_kind == "Intermedia"
                ),
                None,
            )
            normalized_observation = normalize_text(content)
            source_sections = [
                section
                for section, text in note_sections.items()
                if normalized_observation and normalized_observation in normalize_text(text)
            ]
            deductions.append(
                {
                    "observation": content,
                    "rationale": nearest_cause,
                    "diagnosis": nearest_diagnosis,
                    "annotated_source_section": annotated_section,
                    "source_sections_exact": source_sections,
                    "observation_exact_in_note": bool(source_sections),
                    "observation_exact_in_annotated_section": bool(
                        annotated_section in source_sections
                    ),
                }
            )
        if isinstance(children, dict):
            next_ancestors = [*ancestors, (kind, content)]
            for child_key, grand_children in children.items():
                visit(str(child_key), grand_children, next_ancestors)

    visit(root_keys[0], payload[root_keys[0]], [])
    return annotation_root, list(reversed(chain_preorder)), deductions, node_count


def build_row(
    path: Path,
    samples_root: Path,
    official_pdds: dict[str, str],
    category_dirs: set[str],
) -> dict[str, Any]:
    relative = path.relative_to(samples_root)
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Sample JSON is not an object")

    note_sections: dict[str, str] = {}
    for index in range(1, 7):
        key = f"input{index}"
        value = payload.get(key, "")
        note_sections[key] = value.replace("\ufeff", "") if isinstance(value, str) else ""
    note_text = "\n".join(
        f"{SECTION_NAMES[key]}:\n{note_sections[key]}" for key in SECTION_NAMES
    )

    category, folder_pdd = infer_path_labels(relative, category_dirs)
    annotation_root, chain, deductions, node_count = extract_annotation(
        payload, note_sections
    )
    canonical_pdd = official_pdds.get(annotation_root.casefold()) if official_pdds else annotation_root

    patient_match = PATIENT_RE.match(path.stem)
    patient_group = (
        f"patient_{digest(patient_match.group(1).encode())[:16]}"
        if patient_match
        else "patient_unparsed_shared"
    )
    input_digest = digest(
        "\n".join(f"{key}\0{note_sections[key]}" for key in sorted(note_sections)).encode(
            "utf-8"
        )
    )
    return {
        "id": f"direct_{digest(relative.as_posix().encode())[:16]}",
        "source_path": str(path),
        "patient_group": patient_group,
        "patient_id_parsed": patient_match is not None,
        "disease_category": category,
        "folder_pdd": folder_pdd,
        "annotation_root_diagnosis": annotation_root,
        "canonical_pdd": canonical_pdd,
        "canonical_pdd_resolved": canonical_pdd is not None,
        "folder_root_conflict": bool(
            folder_pdd and folder_pdd.casefold() != annotation_root.casefold()
        ),
        "annotation_chain": chain,
        "gold_deductions": deductions,
        "note_sections": note_sections,
        "note_text": note_text,
        "node_count": node_count,
        "input_digest": input_digest,
        "json_digest": digest(raw),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: Path, rows: list[dict[str, Any]], failures: int) -> None:
    canonical_counts = Counter(row["canonical_pdd"] or "<unresolved>" for row in rows)
    duplicate_counts = Counter(row["input_digest"] for row in rows)
    duplicate_rows = sum(count for count in duplicate_counts.values() if count > 1)
    deductions = [item for row in rows for item in row["gold_deductions"]]
    exact_grounded = sum(item["observation_exact_in_note"] for item in deductions)
    exact_in_annotated_section = sum(
        item["observation_exact_in_annotated_section"] for item in deductions
    )
    missing_rationale = sum(item["rationale"] is None for item in deductions)
    missing_diagnosis = sum(item["diagnosis"] is None for item in deductions)
    deduction_counts = [len(row["gold_deductions"]) for row in rows]
    lines = [
        "# DiReCT Canonical Private Manifest",
        "",
        "Aggregate-only summary. The JSONL manifest contains restricted note text and must not be committed or shared.",
        "",
        f"- rows: **{len(rows)}**",
        f"- parse failures: **{failures}**",
        f"- patient groups: **{len({row['patient_group'] for row in rows})}**",
        f"- rows with unparsed patient ID: **{sum(not row['patient_id_parsed'] for row in rows)}**",
        f"- folder/root conflicts: **{sum(row['folder_root_conflict'] for row in rows)}**",
        f"- canonical PDD resolved: **{sum(row['canonical_pdd_resolved'] for row in rows)}/{len(rows)}**",
        f"- unresolved canonical PDD: **{sum(not row['canonical_pdd_resolved'] for row in rows)}**",
        f"- unique canonical PDD labels: **{len(canonical_counts) - int('<unresolved>' in canonical_counts)}**",
        f"- duplicate input rows: **{duplicate_rows}**",
        f"- deductions: **{len(deductions)}**",
        f"- deductions missing rationale / diagnosis: **{missing_rationale} / {missing_diagnosis}**",
        f"- deductions per row: **min {min(deduction_counts, default=0)}, median {statistics.median(deduction_counts) if deduction_counts else 0}, max {max(deduction_counts, default=0)}**",
        f"- observation exact-substring grounding: **{exact_grounded}/{len(deductions)} ({exact_grounded / len(deductions) if deductions else 0:.4f})**",
        f"- exact grounding in annotated section: **{exact_in_annotated_section}/{len(deductions)} ({exact_in_annotated_section / len(deductions) if deductions else 0:.4f})**",
        "",
        "## Canonical PDD Counts",
        "",
        "| PDD | n |",
        "|---|---:|",
    ]
    lines.extend(f"| {label} | {count} |" for label, count in canonical_counts.most_common())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-root", required=True, type=Path)
    parser.add_argument("--data-list", type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument(
        "--allow-failures",
        action="store_true",
        help="Write partial output instead of exiting nonzero when a sample fails to parse.",
    )
    args = parser.parse_args()

    official_pdds, category_dirs = read_official_vocab(args.data_list)
    rows: list[dict[str, Any]] = []
    failures = 0
    for path in sorted(args.samples_root.rglob("*.json")):
        try:
            rows.append(
                build_row(path, args.samples_root, official_pdds, category_dirs)
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            failures += 1

    write_jsonl(args.output_jsonl, rows)
    write_summary(args.summary_md, rows, failures)
    print(f"[manifest] rows={len(rows)} failures={failures}")
    print(f"[manifest] private_jsonl={args.output_jsonl}")
    print(f"[manifest] aggregate_summary={args.summary_md}")
    if failures and not args.allow_failures:
        raise SystemExit(
            f"Refusing to accept a partial manifest: {failures} sample(s) failed to parse"
        )


if __name__ == "__main__":
    main()
