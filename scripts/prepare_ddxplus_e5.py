"""Build the frozen DDXPlus population for E5 activation-grounding tests.

This builder deliberately starts from the official CSV splits. It never pools
validation and test rows, and it keeps every derived prompt from a base case in
the same split. The primary population is diagnosis-balanced and excludes
presentations that literally name the gold diagnosis.

For each selected case it emits:

* the original all-cue prompt;
* a paired prompt with one cue deleted;
* when possible, a paired prompt where that cue's value is changed to another
  value declared for the same DDXPlus evidence ID; and
* a deterministic hard-shuffle donor from the same diagnosis with similar cue
  count and prompt length.

Primary activation rows use the CoT-instructed prompt boundary (CoT-P0), which
matches the DiReCT activation distribution used to train Medical-NLA. Direct-P0
rows are emitted separately for the validation base cases only; they are an
instruction-sensitivity control, not part of the locked primary population.

No generated model output or activation is read here. Source-answer agreement
is measured later but is not an eligibility condition: holding the diagnosis
fixed while changing the evidence is the point of the hard negative.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_ddxplus_cue_count_cases import make_rows_for_patient
from scripts.make_ddxplus_probe_dataset import (
    cue_from_entry,
    make_prompt,
    read_json,
    read_patient_rows,
)
from src.answer_matching import normalize
from src.jsonl import write_jsonl


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    name, raw_path = value.split("=", 1)
    name = name.strip().lower()
    if not name or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty NAME=PATH")
    return name, Path(raw_path).expanduser()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_values(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def declared_value_ids(meta: dict[str, Any]) -> list[str]:
    """Return only values explicitly declared by the DDXPlus release."""
    values: set[str] = set()
    for key in ("value_meaning", "possible-values", "possible_values", "values"):
        raw = meta.get(key)
        if isinstance(raw, dict):
            values.update(str(item) for item in raw)
        elif isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    value = item.get("id") or item.get("value") or item.get("key")
                    if value is not None:
                        values.add(str(value))
                elif item is not None:
                    values.add(str(item))
    return sorted(values)


def stable_order(items: Iterable[str], *, seed: int, namespace: str) -> list[str]:
    def key(value: str) -> str:
        return hashlib.sha256(f"{seed}:{namespace}:{value}".encode()).hexdigest()

    return sorted(items, key=key)


def gold_named(case: dict[str, Any]) -> bool:
    labels = [
        str(label)
        for label in [case.get("diagnosis_name"), *(case.get("diagnosis_aliases") or [])]
        if label
    ]
    haystack = f" {normalize(str(case.get('prompt') or ''))} "
    return any(
        normalized and f" {normalized} " in haystack
        for normalized in (normalize(label) for label in labels)
    )


def canonicalize_case(case: dict[str, Any], *, split: str, row_index: int) -> dict[str, Any]:
    out = dict(case)
    diagnosis_id = str(case["diagnosis_id"])
    base_id = f"ddxplus_{split}_{diagnosis_id}_{row_index:07d}"
    source_patient_id = str(case.get("patient_id") or f"row_{row_index:07d}")
    fallback_id = source_patient_id == f"row_{row_index:07d}"
    out.update(
        {
            "id": f"{base_id}__original",
            "base_id": base_id,
            "official_split": split,
            "variant": "original",
            "cue_count_condition": "all",
            "gold_named_in_prompt": gold_named(case),
            "source_row_index": row_index,
            "source_patient_id": source_patient_id,
            "patient_id_source": "row_index_fallback" if fallback_id else "provided",
            # Official split is part of the identity whenever the release does
            # not expose a reusable patient identifier.
            "patient_id": f"{split}:{source_patient_id}",
        }
    )
    return out


def sample_split(
    path: Path,
    *,
    split: str,
    evidence_meta: dict[str, Any],
    seed: int,
    quota: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reservoir-sample up to ``quota`` eligible cases per diagnosis."""
    rng = random.Random(f"ddxplus-e5:{seed}:{split}")
    reservoirs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    eligible_seen: Counter[str] = Counter()
    counters: Counter[str] = Counter()

    for row_index, patient in enumerate(read_patient_rows(path)):
        counters["rows_scanned"] += 1
        rows = make_rows_for_patient(
            patient,
            row_index=row_index,
            evidence_meta=evidence_meta,
            rng=rng,
            cue_counts=[None],
            prefer_symptoms=True,
            clean_cues=True,
            negative_cues=False,
        )
        if not rows:
            counters["no_rendered_cue"] += 1
            continue
        case = canonicalize_case(rows[0], split=split, row_index=row_index)
        if int(case.get("cue_count") or 0) < 3:
            counters["fewer_than_three_cues"] += 1
            continue
        if case["gold_named_in_prompt"]:
            counters["gold_named_in_prompt"] += 1
            continue

        diagnosis = str(case["diagnosis_id"])
        eligible_seen[diagnosis] += 1
        seen = eligible_seen[diagnosis]
        bucket = reservoirs[diagnosis]
        if len(bucket) < quota:
            bucket.append(case)
        else:
            replacement = rng.randrange(seen)
            if replacement < quota:
                bucket[replacement] = case

        if row_index and row_index % 100_000 == 0:
            print(
                f"[{split}] scanned={row_index:,} diagnoses={len(reservoirs)} "
                f"eligible={sum(eligible_seen.values()):,}",
                flush=True,
            )

    selected_diagnoses = sorted(reservoirs)
    cases = [
        row
        for label in sorted(selected_diagnoses)
        for row in sorted(reservoirs[label], key=lambda item: str(item["base_id"]))
    ]
    summary = {
        **dict(counters),
        "eligible_rows": sum(eligible_seen.values()),
        "diagnoses_observed": len(reservoirs),
        "diagnoses_with_eligible_cases": len(selected_diagnoses),
        "cases_sampled_before_common_filter": len(cases),
        "examples_per_diagnosis_cap": quota,
        "eligible_counts_by_diagnosis": dict(sorted(eligible_seen.items())),
        "sampled_counts_by_diagnosis": {
            label: len(reservoirs[label]) for label in selected_diagnoses
        },
        "short_diagnoses": {
            label: len(reservoirs[label])
            for label in selected_diagnoses
            if len(reservoirs[label]) < quota
        },
    }
    return cases, summary


def common_diagnosis_support(
    sampled_by_split: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Return diagnosis labels represented by an eligible case in every split."""
    if not sampled_by_split:
        return []
    supports = [
        {str(row["diagnosis_id"]) for row in rows}
        for rows in sampled_by_split.values()
    ]
    return sorted(set.intersection(*supports))


def make_activation_row(
    case: dict[str, Any], *, condition: str = "cot"
) -> dict[str, Any]:
    if condition not in {"cot", "direct"}:
        raise ValueError(f"Unsupported P0 condition: {condition!r}")
    prompt_field = "prompt_cot" if condition == "cot" else "prompt"
    prompt = case.get(prompt_field)
    if not prompt:
        raise ValueError(
            f"Case {case.get('id')} has no {prompt_field!r} for {condition}-P0."
        )

    out = dict(case)
    out.update(
        {
            "id": f"{case['id']}__{condition}_p0",
            "prompt": prompt,
            "condition": condition,
            "target_role": "format",
            "cue_index": None,
            "position_label": f"{condition}_P0_prompt_boundary",
            "position_family": "P0",
            "position_mode": "last_token",
            "target_text": None,
            "target_text_strategy": None,
        }
    )
    return out


def editable_alternatives(
    case: dict[str, Any], evidence_meta: dict[str, Any], *, seed: int
) -> list[tuple[int, dict[str, Any]]]:
    alternatives: list[tuple[int, dict[str, Any]]] = []
    cue_ids = list(case.get("cue_evidence_ids") or [])
    value_ids = list(case.get("cue_value_ids") or [])
    cue_texts = {str(value).casefold() for value in case.get("cue_targets") or []}
    for index, (evidence_id, current_value) in enumerate(zip(cue_ids, value_ids)):
        current = str(current_value or "")
        if not current or "," in current:
            continue
        meta = evidence_meta.get(str(evidence_id))
        if not isinstance(meta, dict):
            continue
        candidates = [value for value in declared_value_ids(meta) if value != current]
        for alternate in stable_order(
            candidates,
            seed=seed,
            namespace=f"{case['base_id']}:{evidence_id}",
        ):
            rendered = cue_from_entry(
                f"{evidence_id}_@_{alternate}",
                evidence_meta,
                clean_cues=True,
                negative_cues=True,
            )
            text = str(rendered.get("cue_text") or "").strip()
            if rendered.get("excluded") or not text:
                continue
            if text.casefold() in cue_texts:
                continue
            original_text = str(case["cue_targets"][index]).casefold()
            if original_text in text.casefold() or text.casefold() in original_text:
                continue
            alternatives.append((index, rendered))
            break
    return alternatives


def counterfactual_cases(
    case: dict[str, Any], evidence_meta: dict[str, Any], *, seed: int
) -> list[dict[str, Any]]:
    cues = list(case.get("cue_targets") or [])
    if len(cues) < 3:
        return []
    editable = editable_alternatives(case, evidence_meta, seed=seed)
    if editable:
        ordered = stable_order(
            [str(index) for index, _ in editable],
            seed=seed,
            namespace=str(case["base_id"]),
        )
        chosen_index = int(ordered[0])
        replacement = next(rendered for index, rendered in editable if index == chosen_index)
    else:
        chosen_index = int(
            stable_order(
                [str(index) for index in range(len(cues))],
                seed=seed,
                namespace=str(case["base_id"]),
            )[0]
        )
        replacement = None

    age, sex = case.get("age"), case.get("sex")
    original_cue = str(cues[chosen_index])
    common = {
        "cf_target_index": chosen_index,
        "cf_original_cue": original_cue,
        "cf_original_evidence_id": case["cue_evidence_ids"][chosen_index],
        "cf_original_value_id": case["cue_value_ids"][chosen_index],
    }
    deleted_cues = [cue for index, cue in enumerate(cues) if index != chosen_index]
    cue_parallel_fields = (
        "cue_types",
        "cue_polarities",
        "cue_evidence_ids",
        "cue_evidence_entries",
        "cue_value_ids",
        "cue_value_labels",
        "cue_merged_value_counts",
    )
    deleted = {
        **case,
        **common,
        "id": f"{case['base_id']}__cue_deleted",
        "variant": "cue_deleted",
        "cue_targets": deleted_cues,
        "cue_count": len(deleted_cues),
        "prompt": make_prompt(deleted_cues, age=age, sex=sex),
        "prompt_cot": make_prompt(deleted_cues, condition="cot", age=age, sex=sex),
    }
    for field in cue_parallel_fields:
        if field in case:
            deleted[field] = [
                value for index, value in enumerate(case.get(field) or []) if index != chosen_index
            ]
    rows = [deleted]

    if replacement is not None:
        edited_cues = list(cues)
        edited_cues[chosen_index] = replacement["cue_text"]
        edited = {
            **case,
            **common,
            "id": f"{case['base_id']}__value_edited",
            "variant": "value_edited",
            "cue_targets": edited_cues,
            "prompt": make_prompt(edited_cues, age=age, sex=sex),
            "prompt_cot": make_prompt(edited_cues, condition="cot", age=age, sex=sex),
            "cf_replacement_cue": replacement["cue_text"],
            "cf_replacement_evidence_id": replacement["evidence_id"],
            "cf_replacement_value_id": replacement["value_id"],
            "cf_replacement_value_label": replacement["value_label"],
        }
        edited["cue_value_ids"] = list(case["cue_value_ids"])
        edited["cue_value_ids"][chosen_index] = replacement["value_id"]
        edited["cue_value_labels"] = list(case["cue_value_labels"])
        edited["cue_value_labels"][chosen_index] = replacement["value_label"]
        edited["cue_evidence_entries"] = list(case["cue_evidence_entries"])
        edited["cue_evidence_entries"][chosen_index] = replacement["evidence_entry"]
        edited["cue_polarities"] = list(case["cue_polarities"])
        edited["cue_polarities"][chosen_index] = replacement["cue_polarity"]
        if "cue_merged_value_counts" in case:
            edited["cue_merged_value_counts"] = list(case["cue_merged_value_counts"])
            edited["cue_merged_value_counts"][chosen_index] = 1
        rows.append(edited)
    return rows


def pair_hard_shuffles(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_diagnosis: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_diagnosis[str(case["diagnosis_id"])].append(case)

    pairs: list[dict[str, Any]] = []
    for diagnosis, group in sorted(by_diagnosis.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                int(row.get("cue_count") or 0),
                len(str(row.get("prompt") or "")),
                str(row["base_id"]),
            ),
        )
        if len(ordered) < 2:
            continue
        signatures = [
            tuple(zip(row["cue_evidence_ids"], row["cue_value_ids"])) for row in ordered
        ]
        # A cyclic assignment is a one-to-one derangement. Choose its offset by
        # first minimizing identical evidence/value signatures and then keeping
        # cue count and prompt length close. This is deterministic and avoids a
        # donor being reused for many easy comparisons.
        shift = min(
            range(1, len(ordered)),
            key=lambda offset: (
                sum(
                    signatures[index] == signatures[(index + offset) % len(ordered)]
                    for index in range(len(ordered))
                ),
                sum(
                    abs(
                        int(ordered[index]["cue_count"])
                        - int(ordered[(index + offset) % len(ordered)]["cue_count"])
                    )
                    for index in range(len(ordered))
                ),
                sum(
                    abs(
                        len(ordered[index]["prompt"])
                        - len(ordered[(index + offset) % len(ordered)]["prompt"])
                    )
                    for index in range(len(ordered))
                ),
                offset,
            ),
        )
        donors = [ordered[(index + shift) % len(ordered)] for index in range(len(ordered))]

        for own, donor in zip(ordered, donors):
            own_signature = tuple(zip(own["cue_evidence_ids"], own["cue_value_ids"]))
            donor_signature = tuple(zip(donor["cue_evidence_ids"], donor["cue_value_ids"]))
            pairs.append(
                {
                    "id": f"{own['base_id']}__hard_shuffle",
                    "official_split": own["official_split"],
                    "diagnosis_id": diagnosis,
                    "own_base_id": own["base_id"],
                    "donor_base_id": donor["base_id"],
                    "own_cue_count": own["cue_count"],
                    "donor_cue_count": donor["cue_count"],
                    "cue_count_difference": abs(int(own["cue_count"]) - int(donor["cue_count"])),
                    "prompt_length_difference": abs(len(own["prompt"]) - len(donor["prompt"])),
                    "different_evidence_value_signature": own_signature != donor_signature,
                    "source_answer_relation": "not_materialized_not_an_eligibility_gate",
                    "primary_pair_eligible": own_signature != donor_signature,
                }
            )
    return pairs


def write_summary(path: Path, protocol: dict[str, Any]) -> None:
    lines = [
        "# DDXPlus E5 Canonical Data",
        "",
        "Public synthetic data only. Official validation and test files remain disjoint.",
        "",
        f"- seed: **{protocol['seed']}**",
        f"- per-diagnosis cap: **{protocol['examples_per_diagnosis_cap']} per split**",
        f"- common eligible diagnoses: **{protocol['common_diagnosis_count']}**",
        "- primary training role: **none (DiReCT-only adaptation)**",
        "- validation role: threshold/control selection",
        "- test role: locked Table 3/Figure 3 evaluation",
        "- test mean activation is forbidden; the mean control comes from validation",
        "",
        "## Populations",
        "",
        (
            "| split | scanned | eligible | selected | gold-name excluded | deletion | "
            "native value edit | primary CoT-P0 | Direct-P0 control | hard pairs | "
            "pair eligible |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split, item in protocol["splits"].items():
        scan = item["scan"]
        lines.append(
            f"| {split} | {scan.get('rows_scanned', 0)} | {scan.get('eligible_rows', 0)} | "
            f"{item['cases']} | {scan.get('gold_named_in_prompt', 0)} | "
            f"{item['cue_deleted']} | {item['value_edited']} | "
            f"{item['activation_rows']} | {item['direct_p0_control_rows']} | "
            f"{item['hard_pairs']} | {item['primary_pair_eligible']} |"
        )
    lines.extend(
        [
            "",
            "## Frozen Rules",
            "",
            (
                "- The diagnosis set is the intersection of labels with at least one "
                "eligible case in every supplied official split; no top-k label selection is used."
            ),
            (
                "- Cases are reservoir-sampled independently inside each official split "
                "and diagnosis, up to the fixed per-diagnosis cap."
            ),
            (
                "- Eligibility requires at least three clean rendered cues and no literal "
                "gold diagnosis/alias in the prompt."
            ),
            "- Cue deletion and value edit stay within the same base case and split.",
            (
                "- Value edits use only another value declared for the same evidence ID; "
                "binary absence is tested by deletion, not an invented value."
            ),
            (
                "- Hard-shuffle donors have the same diagnosis; the one-to-one assignment "
                "minimizes signature collisions before size mismatch."
            ),
            (
                "- Source-answer agreement is reported later but does not define pair "
                "eligibility; the diagnosis is intentionally held fixed."
            ),
            (
                "- All primary E5 activations are CoT-P0/HS32 last-token states, "
                "matching the DiReCT Medical-NLA training condition."
            ),
            (
                "- Direct-P0 rows are a paired validation-only instruction control; "
                "they are not part of the locked primary test."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="append", required=True, type=parse_named_path)
    parser.add_argument("--evidences", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--examples-per-diagnosis", type=int, default=100)
    parser.add_argument("--expected-common-diagnoses", type=int)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    split_paths = dict(args.split)
    if len(split_paths) != len(args.split):
        raise ValueError("Duplicate --split name.")
    if set(split_paths) != {"validation", "test"}:
        raise ValueError(
            "E5 requires exactly --split validation=... and --split test=...."
        )
    if args.examples_per_diagnosis <= 0:
        raise ValueError("--examples-per-diagnosis must be positive.")
    evidence_meta = read_json(args.evidences)
    if not isinstance(evidence_meta, dict):
        raise ValueError("--evidences must be a JSON object keyed by evidence ID.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol: dict[str, Any] = {
        "schema_version": 3,
        "seed": args.seed,
        "examples_per_diagnosis_cap": args.examples_per_diagnosis,
        "diagnosis_support_rule": "eligible diagnosis intersection across supplied splits",
        "primary_adaptation_data": "DiReCT only",
        "mean_activation_control_split": "validation",
        "primary_hidden_state": "CoT-P0/HS32/last_token",
        "primary_instruction_condition": "cot",
        "instruction_sensitivity_control": "Direct-P0 on validation base cases only",
        "source_files": {
            "evidences": {"path": str(args.evidences), "sha256": sha256_file(args.evidences)}
        },
        "splits": {},
    }
    provided_patient_ids: dict[str, set[str]] = {}

    sampled_by_split: dict[str, list[dict[str, Any]]] = {}
    scans_by_split: dict[str, dict[str, Any]] = {}
    for split, source_path in split_paths.items():
        sampled, scan = sample_split(
            source_path,
            split=split,
            evidence_meta=evidence_meta,
            seed=args.seed,
            quota=args.examples_per_diagnosis,
        )
        sampled_by_split[split] = sampled
        scans_by_split[split] = scan

    common_diagnoses = common_diagnosis_support(sampled_by_split)
    if not common_diagnoses:
        raise ValueError("No eligible diagnosis is represented in every supplied split.")
    if (
        args.expected_common_diagnoses is not None
        and len(common_diagnoses) != args.expected_common_diagnoses
    ):
        raise ValueError(
            f"Found {len(common_diagnoses)} diagnoses on common eligible support; "
            f"expected {args.expected_common_diagnoses}."
        )
    protocol["common_diagnosis_count"] = len(common_diagnoses)
    protocol["common_diagnoses"] = common_diagnoses
    print(
        f"[population] common eligible diagnoses={len(common_diagnoses)} "
        f"cap={args.examples_per_diagnosis}",
        flush=True,
    )

    common_set = set(common_diagnoses)
    for split, source_path in split_paths.items():
        cases = [
            row
            for row in sampled_by_split[split]
            if str(row["diagnosis_id"]) in common_set
        ]
        scan = dict(scans_by_split[split])
        split_support = {
            str(row["diagnosis_id"]) for row in sampled_by_split[split]
        }
        scan.update(
            {
                "diagnoses_selected": len(common_diagnoses),
                "cases_selected": len(cases),
                "selected_diagnoses": common_diagnoses,
                "diagnoses_excluded_outside_common_support": sorted(
                    split_support - common_set
                ),
            }
        )
        counterfactuals = [
            derived
            for case in cases
            for derived in counterfactual_cases(case, evidence_meta, seed=args.seed)
        ]
        activation_rows = [make_activation_row(case) for case in [*cases, *counterfactuals]]
        direct_control_rows = (
            [make_activation_row(case, condition="direct") for case in cases]
            if split == "validation"
            else []
        )
        pairs = pair_hard_shuffles(cases)

        write_jsonl(args.out_dir / f"cases_{split}.jsonl", cases)
        write_jsonl(args.out_dir / f"counterfactual_cases_{split}.jsonl", counterfactuals)
        write_jsonl(args.out_dir / f"activation_rows_{split}.jsonl", activation_rows)
        if direct_control_rows:
            write_jsonl(
                args.out_dir / "activation_rows_validation_direct_control.jsonl",
                direct_control_rows,
            )
        write_jsonl(args.out_dir / f"hard_shuffle_pairs_{split}.jsonl", pairs)

        variant_counts = Counter(row["variant"] for row in counterfactuals)
        pair_eligible = sum(bool(row["primary_pair_eligible"]) for row in pairs)
        protocol["source_files"][split] = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
        protocol["splits"][split] = {
            "scan": scan,
            "cases": len(cases),
            "case_id_sha256": sha256_values(str(row["base_id"]) for row in cases),
            "activation_rows": len(activation_rows),
            "direct_p0_control_rows": len(direct_control_rows),
            "cue_deleted": variant_counts["cue_deleted"],
            "value_edited": variant_counts["value_edited"],
            "hard_pairs": len(pairs),
            "primary_pair_eligible": pair_eligible,
            "pair_sha256": sha256_values(
                f"{row['own_base_id']}->{row['donor_base_id']}" for row in pairs
            ),
            "provided_patient_ids": sum(
                row["patient_id_source"] == "provided" for row in cases
            ),
            "row_index_fallback_ids": sum(
                row["patient_id_source"] == "row_index_fallback" for row in cases
            ),
        }
        provided_patient_ids[split] = {
            str(row["source_patient_id"])
            for row in cases
            if row["patient_id_source"] == "provided"
        }
        print(
            f"[{split}] cases={len(cases)} deletion={variant_counts['cue_deleted']} "
            f"value_edit={variant_counts['value_edited']} cot_p0={len(activation_rows)} "
            f"direct_p0_control={len(direct_control_rows)} "
            f"pairs={pair_eligible}/{len(pairs)}",
            flush=True,
        )

    overlap = {}
    names = sorted(provided_patient_ids)
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            overlap[f"{left}__{right}"] = len(
                provided_patient_ids[left] & provided_patient_ids[right]
            )
    protocol["selected_provided_patient_id_overlap"] = overlap

    protocol_path = args.out_dir / "protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.out_dir / "summary.md", protocol)
    print(f"[protocol] {protocol_path}")
    print(f"[summary] {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
