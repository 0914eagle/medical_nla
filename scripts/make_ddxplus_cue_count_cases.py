"""Create DDXPlus case prompts with different cue counts for source baselines."""

from __future__ import annotations

import argparse
import random
import sys
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_ddxplus_probe_dataset import (
    cue_from_entry,
    merge_multivalue_cues,
    get_field,
    parse_evidence_entries,
    read_json,
    make_prompt as probe_make_prompt,
    read_patient_rows,
    slug,
    write_jsonl,
)


def parse_cue_count(value: str) -> int | None:
    if value.strip().lower() in {"all", "full", "max"}:
        return None
    count = int(value)
    if count < 1:
        raise ValueError("--cue-counts must be positive integers or 'all'.")
    return count


def make_prompt(cues: list[str]) -> str:
    """Same findings-list frame as the probe generator; see it for the rationale."""
    return probe_make_prompt(cues)


def cue_pool(
    row: dict[str, Any],
    *,
    evidence_meta: dict[str, Any],
    prefer_symptoms: bool,
    clean_cues: bool,
    negative_cues: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    entries = parse_evidence_entries(get_field(row, ["EVIDENCES", "evidences", "evidence"]))
    all_cues = [
        cue_from_entry(
            entry, evidence_meta, clean_cues=clean_cues, negative_cues=negative_cues
        )
        for entry in entries
    ]
    cues = merge_multivalue_cues(
        [
            cue
            for cue in all_cues
            if cue["cue_text"] and cue["cue_text"].lower() != "none" and not cue["excluded"]
        ]
    )
    symptom_cues = [cue for cue in cues if not cue["is_antecedent"]]
    if prefer_symptoms and symptom_cues:
        return symptom_cues, all_cues
    return cues, all_cues


def make_rows_for_patient(
    row: dict[str, Any],
    *,
    row_index: int,
    evidence_meta: dict[str, Any],
    rng: random.Random,
    cue_counts: list[int | None],
    prefer_symptoms: bool,
    clean_cues: bool,
    negative_cues: bool = False,
) -> list[dict[str, Any]]:
    pathology = str(get_field(row, ["PATHOLOGY", "pathology", "diagnosis", "label"]))
    patient_id = get_field(row, ["id", "patient_id", "PATIENT", "patient"], required=False)
    if patient_id is None:
        patient_id = f"row_{row_index:07d}"
    cues, all_cues = cue_pool(
        row,
        evidence_meta=evidence_meta,
        prefer_symptoms=prefer_symptoms,
        clean_cues=clean_cues,
        negative_cues=negative_cues,
    )
    if not cues:
        return []
    diagnosis_id = slug(pathology)
    base_id = f"ddxplus_{diagnosis_id}_{row_index:07d}"
    rows = []
    for cue_count in cue_counts:
        selected_count = len(cues) if cue_count is None else cue_count
        if len(cues) < selected_count:
            continue
        selected = list(cues) if cue_count is None else rng.sample(cues, selected_count)
        cue_targets = [cue["cue_text"] for cue in selected]
        cue_count_label = "all" if cue_count is None else str(cue_count)
        rows.append(
            {
                "id": f"{base_id}__cues_{cue_count_label}",
                "base_id": base_id,
                "source": "ddxplus",
                "patient_id": str(patient_id),
                "diagnosis_id": diagnosis_id,
                "diagnosis_name": pathology,
                "diagnosis_aliases": [pathology],
                "cue_count_condition": cue_count_label,
                "cue_count": len(cue_targets),
                "available_cue_count": len(cues),
                "cue_targets": cue_targets,
                "cue_types": [cue["cue_type"] for cue in selected],
                "cue_polarities": [cue["cue_polarity"] for cue in selected],
                "cue_evidence_ids": [cue["evidence_id"] for cue in selected],
                "cue_evidence_entries": [cue["evidence_entry"] for cue in selected],
                "cue_value_ids": [cue["value_id"] for cue in selected],
                "cue_value_labels": [cue["value_label"] for cue in selected],
                "clean_cues": clean_cues,
                "negative_cues": negative_cues,
                "prefer_symptoms": prefer_symptoms,
                "excluded_cue_count": sum(1 for cue in all_cues if cue["excluded"]),
                "prompt": make_prompt(cue_targets),
                "variant": f"cue_count_{cue_count_label}",
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", required=True)
    parser.add_argument("--evidences", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cue-counts", nargs="+", default=["3", "5", "all"])
    parser.add_argument("--max-diagnoses", type=int, default=49)
    parser.add_argument("--examples-per-diagnosis", type=int, default=100)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--clean-cues", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--negative-cues",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep negatively-answered evidences as cues, rendered by negating the "
            "question's auxiliary ('has not traveled out of the country'). Without "
            "this they are dropped, so prompts carry positive findings only."
        ),
    )
    parser.add_argument("--prefer-symptoms", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--stop-when-full",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Stop scanning once every diagnosis seen has its quota and there are "
            "at least --max-diagnoses of them. Output is identical whenever the "
            "corpus holds exactly that many diagnoses, which DDXPlus does (49); "
            "with more, later rows could still change which ones are selected."
        ),
    )
    parser.add_argument("--max-patient-rows", type=int, default=None)
    args = parser.parse_args()

    cue_counts = [parse_cue_count(value) for value in args.cue_counts]
    rng = random.Random(args.seed)
    evidence_meta = read_json(Path(args.evidences))
    if not isinstance(evidence_meta, dict):
        raise ValueError("Evidence metadata must be a JSON object keyed by evidence id.")

    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    patients = read_patient_rows(Path(args.patients))
    if args.max_patient_rows is not None:
        patients = islice(patients, args.max_patient_rows)
    # "all" imposes no minimum of its own: it takes whatever cues the case has.
    # Only the explicit counts do, so with `--cue-counts all` there is no floor.
    explicit_counts = [count for count in cue_counts if count is not None]
    min_required = max(explicit_counts) if explicit_counts else 1
    # Nothing is written until the whole release is scanned, so report
    # progress rather than looking hung on the million-row CSV.
    for row_index, row in enumerate(patients):
        if row_index and row_index % 100_000 == 0:
            kept = sum(len(bucket) for bucket in by_diagnosis.values())
            print(
                f"[scan] {row_index:,} rows read | {len(by_diagnosis)} diagnoses "
                f"| {kept:,} cases kept",
                flush=True,
            )
        rows = make_rows_for_patient(
            row,
            row_index=row_index,
            evidence_meta=evidence_meta,
            rng=rng,
            cue_counts=cue_counts,
            prefer_symptoms=args.prefer_symptoms,
            clean_cues=args.clean_cues,
            negative_cues=args.negative_cues,
        )
        if not rows:
            continue
        diagnosis_id = rows[0]["diagnosis_id"]
        if max(row["available_cue_count"] for row in rows) < min_required:
            continue
        bucket = by_diagnosis[diagnosis_id]
        if len(bucket) < args.examples_per_diagnosis:
            bucket.append(rows)

        if args.stop_when_full:
            full = sum(
                1 for cases in by_diagnosis.values() if len(cases) >= args.examples_per_diagnosis
            )
            if full >= args.max_diagnoses and full == len(by_diagnosis):
                print(
                    f"[scan] stopping at row {row_index:,}: all {full} diagnoses full",
                    flush=True,
                )
                break

    selected_diagnoses = [
        diagnosis
        for diagnosis, cases in sorted(by_diagnosis.items(), key=lambda item: (-len(item[1]), item[0]))
        if len(cases) >= args.examples_per_diagnosis
    ][: args.max_diagnoses]
    output_rows = [
        row
        for diagnosis in selected_diagnoses
        for patient_rows in by_diagnosis[diagnosis]
        for row in patient_rows
    ]
    write_jsonl(Path(args.output), output_rows)

    print(f"diagnoses_selected: {len(selected_diagnoses)}")
    print(f"patients_selected: {sum(len(by_diagnosis[diagnosis]) for diagnosis in selected_diagnoses)}")
    print(f"rows_written: {len(output_rows)}")
    print("by_cue_count:")
    for condition, count in Counter(row["cue_count_condition"] for row in output_rows).most_common():
        print(f"  {condition}: {count}")
    print("top_diagnoses:")
    for diagnosis, count in Counter(row["diagnosis_id"] for row in output_rows).most_common(20):
        print(f"  {diagnosis}: {count}")


if __name__ == "__main__":
    main()
