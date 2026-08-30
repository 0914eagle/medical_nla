"""Prepare, generate, map, and score the frozen D22 Patchscope smoke."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_medical_nla_d22_geometry import (
    cluster_bootstrap_ci,
    mapped_path,
    parse_path_map,
    row_bootstrap_ci,
)
from scripts.run_ddxplus_structured_reader import micro_f1
from scripts.score_ddxplus_semantic_readouts import read_protocol, semantic_decisions
from src.config import load_config
from src.ddxplus_semantic_mapping import (
    make_batch_prompt,
    materialize_items,
    prepare_items,
    sha256_file,
)
from src.jsonl import append_jsonl, read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.reconstruction_scoring import load_activation


PROMPT = """A patient-state representation is provided at <STATE>.
List only the clinical findings represented in that state.
Return zero or more self-contained English clinical claims, one per bullet.
Each value-bearing claim must name both the finding and its value.
Do not output a diagnosis, rationale, question, yes/no answer, or bare value.
If no clinical finding is represented, output exactly: NONE"""
NAMESPACE = "d22_patchscope_v1"
VARIANTS = ("original", "cue_deleted", "value_edited")
CONDITIONS = ("real", "same_diagnosis_shuffled", "train_mean", "no_patch")


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def row_base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def stable_hash(*parts: str) -> str:
    return hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()


def normalize_variant(row: dict[str, Any]) -> str:
    value = str(row.get("variant") or "original")
    aliases = {"value_edit": "value_edited", "deletion": "cue_deleted"}
    return aliases.get(value, value)


def index_activation_rows(
    path: Path, mappings: list[tuple[str, str]]
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for raw in read_jsonl(path):
        row = dict(raw)
        variant = normalize_variant(row)
        if variant not in VARIANTS:
            continue
        if str(row.get("official_split") or "validation") == "test":
            raise ValueError("Locked-test row reached the validation Patchscope builder")
        identifier = row_base_id(row)
        if not identifier or variant in grouped[identifier]:
            raise ValueError(f"Missing/duplicate Patchscope row: {identifier}/{variant}")
        activation_path = mapped_path(row.get("activation_path"), mappings)
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        row["variant"] = variant
        row["activation_path"] = str(activation_path)
        grouped[identifier][variant] = row
    return grouped


def supported_value_edit(
    rows: dict[str, dict[str, Any]], protocol: dict[str, Any]
) -> bool:
    if set(rows) != set(VARIANTS):
        return False
    edit = rows["value_edited"]
    evidence = str(edit.get("cf_original_evidence_id") or "")
    old = str(edit.get("cf_original_value_id") or "")
    new = str(edit.get("cf_replacement_value_id") or "")
    supported = set(map(str, protocol["values_by_evidence"].get(evidence, [])))
    return bool(evidence and old and new and old != new and {old, new} <= supported)


def round_robin_sample(
    eligible: dict[str, dict[str, dict[str, Any]]], limit: int
) -> list[str]:
    by_diagnosis: dict[str, list[str]] = defaultdict(list)
    for identifier, rows in eligible.items():
        diagnosis = str(rows["original"].get("diagnosis_id") or "")
        if not diagnosis:
            raise ValueError(f"Missing diagnosis for {identifier}")
        by_diagnosis[diagnosis].append(identifier)
    for diagnosis, identifiers in by_diagnosis.items():
        identifiers.sort(key=lambda value: stable_hash(NAMESPACE, value))
    selected = []
    offsets = defaultdict(int)
    diagnoses = sorted(by_diagnosis)
    while len(selected) < min(limit, len(eligible)):
        progress = False
        for diagnosis in diagnoses:
            offset = offsets[diagnosis]
            if offset >= len(by_diagnosis[diagnosis]):
                continue
            selected.append(by_diagnosis[diagnosis][offset])
            offsets[diagnosis] += 1
            progress = True
            if len(selected) == min(limit, len(eligible)):
                break
        if not progress:
            break
    return selected


def deterministic_donors(
    eligible: dict[str, dict[str, dict[str, Any]]]
) -> dict[str, str]:
    by_diagnosis: dict[str, list[str]] = defaultdict(list)
    for identifier, rows in eligible.items():
        by_diagnosis[str(rows["original"]["diagnosis_id"])].append(identifier)
    result = {}
    for diagnosis, identifiers in sorted(by_diagnosis.items()):
        ordered = sorted(
            identifiers,
            key=lambda value: stable_hash(NAMESPACE, "donor", diagnosis, value),
        )
        if len(ordered) < 2:
            continue
        for index, identifier in enumerate(ordered):
            result[identifier] = ordered[(index + 1) % len(ordered)]
    return result


def compute_train_mean(
    manifest: Path, mappings: list[tuple[str, str]]
) -> tuple[torch.Tensor, int]:
    total = None
    seen = set()
    count = 0
    for row in read_jsonl(manifest):
        if normalize_variant(row) != "original":
            continue
        identifier = row_base_id(row)
        if not identifier or identifier in seen:
            continue
        seen.add(identifier)
        vector = load_activation(mapped_path(row.get("activation_path"), mappings)).double()
        if total is None:
            total = torch.zeros_like(vector)
        total += vector
        count += 1
    if total is None or not count:
        raise ValueError("No official-train original activations for Patchscope mean")
    return (total / count).float(), count


def generation_id(base_id: str, variant: str, condition: str) -> str:
    if condition in ("train_mean", "no_patch"):
        return f"patchscope::{base_id}::shared::{condition}"
    return f"patchscope::{base_id}::{variant}::{condition}"


def logical_id(base_id: str, variant: str, condition: str) -> str:
    return f"patchscope::{base_id}::{variant}::{condition}"


def prepare(args: argparse.Namespace) -> None:
    structured = json.loads(args.structured_protocol.read_text(encoding="utf-8"))
    if int(structured.get("layer", -1)) != 24:
        raise ValueError("Expected frozen HS24 structured scoring protocol")
    grouped = index_activation_rows(args.validation_manifest, args.path_map)
    preliminary = {
        identifier: rows
        for identifier, rows in grouped.items()
        if supported_value_edit(rows, structured)
    }
    donors = deterministic_donors(preliminary)
    eligible = {
        identifier: rows
        for identifier, rows in preliminary.items()
        if identifier in donors
    }
    selected = round_robin_sample(eligible, args.cases)
    if len(selected) != args.cases:
        raise ValueError(f"Expected {args.cases} eligible cases, got {len(selected)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    mean, train_count = compute_train_mean(args.train_manifest, args.path_map)
    mean_path = args.out_dir / "frozen_train_mean_hs32.pt"
    torch.save(mean, mean_path)

    generation_rows = []
    logical_rows = []
    seen_generation = set()
    for identifier in selected:
        donor = donors[identifier]
        for variant in VARIANTS:
            source = eligible[identifier][variant]
            donor_source = eligible[donor][variant]
            for condition in CONDITIONS:
                gen_id = generation_id(identifier, variant, condition)
                if gen_id not in seen_generation:
                    patch_path = None
                    donor_id = None
                    if condition == "real":
                        patch_path = source["activation_path"]
                    elif condition == "same_diagnosis_shuffled":
                        patch_path = donor_source["activation_path"]
                        donor_id = donor
                    elif condition == "train_mean":
                        patch_path = str(mean_path)
                    generation_rows.append(
                        {
                            "id": gen_id,
                            "base_id": identifier,
                            "variant": (
                                variant
                                if condition not in ("train_mean", "no_patch")
                                else "shared"
                            ),
                            "condition": condition,
                            "patch_activation_path": patch_path,
                            "donor_base_id": donor_id,
                        }
                    )
                    seen_generation.add(gen_id)
                logical = dict(source)
                logical["id"] = logical_id(identifier, variant, condition)
                logical["base_id"] = identifier
                logical["variant"] = variant
                logical["patch_condition"] = condition
                logical["generation_id"] = gen_id
                logical["donor_base_id"] = donor if condition == "same_diagnosis_shuffled" else None
                logical_rows.append(logical)

    if len(generation_rows) != args.cases * 8:
        raise AssertionError(f"Expected {args.cases * 8} unique generations")
    if len(logical_rows) != args.cases * len(VARIANTS) * len(CONDITIONS):
        raise AssertionError("Logical Patchscope population mismatch")
    generation_manifest = args.out_dir / "generation_manifest.jsonl"
    logical_manifest = args.out_dir / "logical_manifest.jsonl"
    write_jsonl(generation_manifest, generation_rows)
    write_jsonl(logical_manifest, logical_rows)
    protocol = {
        "schema_version": 1,
        "name": NAMESPACE,
        "validation_only": True,
        "locked_test_read": False,
        "prompt": PROMPT,
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "cases": len(selected),
        "selected_base_ids": selected,
        "eligible_before_donor": len(preliminary),
        "eligible_after_donor": len(eligible),
        "unique_generations": len(generation_rows),
        "logical_cells": len(logical_rows),
        "generation_manifest_sha256": sha256_file(generation_manifest),
        "logical_manifest_sha256": sha256_file(logical_manifest),
        "train_mean_rows": train_count,
        "train_mean_path": str(mean_path),
        "train_mean_sha256": sha256_file(mean_path),
        "validation_manifest": str(args.validation_manifest),
        "validation_manifest_sha256": sha256_file(args.validation_manifest),
        "train_manifest": str(args.train_manifest),
        "train_manifest_sha256": sha256_file(args.train_manifest),
        "structured_protocol": str(args.structured_protocol),
        "structured_protocol_sha256": sha256_file(args.structured_protocol),
        "selection": (
            "diagnosis round-robin, then "
            "SHA256(UTF8(d22_patchscope_v1 || NUL || base_id))"
        ),
        "generation": {
            "do_sample": False,
            "max_new_tokens": 128,
            "eos_termination": True,
        },
    }
    (args.out_dir / "population_protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"[prepare] eligible={len(eligible)} cases={len(selected)} "
        f"unique={len(generation_rows)} logical={len(logical_rows)}"
    )


def nested_attr(root: Any, path: str) -> Any:
    value = root
    for part in path.split("."):
        value = getattr(value, part)
    return value


def resolve_layer_stack(model: Any) -> tuple[str, Any]:
    candidates = (
        "model.language_model.layers",
        "language_model.model.layers",
        "model.model.layers",
        "model.layers",
    )
    for path in candidates:
        try:
            layers = nested_attr(model, path)
        except AttributeError:
            continue
        if len(layers) >= 33:
            return path, layers
    raise ValueError("Could not locate the Gemma transformer layer stack")


def find_subsequence(sequence: list[int], needle: list[int]) -> int:
    hits = [
        index
        for index in range(len(sequence) - len(needle) + 1)
        if sequence[index : index + len(needle)] == needle
    ]
    if len(hits) != 1:
        raise ValueError(f"Expected one <STATE> marker token sequence, got {len(hits)}")
    return hits[0] + len(needle) - 1


@contextlib.contextmanager
def patched_prefill(
    layer: Any,
    vectors: torch.Tensor | None,
    marker_position: int,
) -> Iterator[None]:
    if vectors is None:
        yield
        return
    state = {"applied": False}

    def hook(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
        hidden = inputs[0]
        if not state["applied"] and hidden.ndim == 3 and hidden.shape[1] > marker_position:
            if hidden.shape[0] != vectors.shape[0] or hidden.shape[2] != vectors.shape[1]:
                raise ValueError("Patch vector and target hidden-state shapes do not align")
            updated = hidden.clone()
            updated[:, marker_position, :] = vectors.to(hidden.device, hidden.dtype)
            state["applied"] = True
            return (updated, *inputs[1:])
        return inputs

    handle = layer.register_forward_pre_hook(hook)
    try:
        yield
    finally:
        handle.remove()
    if not state["applied"]:
        raise RuntimeError("Patch hook never reached the target prefill state")


def output_contract_valid(text: str) -> bool:
    stripped = text.strip()
    if stripped == "NONE":
        return True
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return bool(lines) and all(line.startswith("- ") and len(line) > 2 for line in lines)


def generate(args: argparse.Namespace) -> None:
    protocol = json.loads(args.population_protocol.read_text(encoding="utf-8"))
    if protocol.get("locked_test_read") is not False or protocol["prompt"] != PROMPT:
        raise ValueError("Patchscope population protocol is not the frozen validation protocol")
    if protocol.get("generation_manifest_sha256") != sha256_file(
        args.generation_manifest
    ):
        raise ValueError("Patchscope generation manifest changed after population freeze")
    mean_path = Path(protocol["train_mean_path"])
    if protocol.get("train_mean_sha256") != sha256_file(mean_path):
        raise ValueError("Patchscope train mean changed after population freeze")
    rows = list(read_jsonl(args.generation_manifest))
    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    chat_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": PROMPT}],
        tokenize=False,
        add_generation_prompt=True,
    )
    encoded = tokenizer(chat_text, add_special_tokens=False, return_tensors="pt")
    marker_ids = tokenizer("<STATE>", add_special_tokens=False)["input_ids"]
    input_ids = encoded["input_ids"][0].tolist()
    marker_position = find_subsequence(input_ids, list(marker_ids))
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    layer_path, layers = resolve_layer_stack(model)
    target_layer_index = 32
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = {str(row["id"]) for row in read_jsonl(output_path)} if output_path.exists() else set()

    receipt = {
        "schema_version": 1,
        "written_before_first_generation": True,
        "population_protocol_sha256": sha256_file(args.population_protocol),
        "generation_manifest_sha256": sha256_file(args.generation_manifest),
        "model_id": model_cfg["model_id"],
        "model_revision": clean(getattr(model.config, "_commit_hash", "")),
        "tokenizer_revision": clean(
            getattr(tokenizer, "_commit_hash", "")
            or getattr(tokenizer, "init_kwargs", {}).get("_commit_hash")
        ),
        "prompt_sha256": hashlib.sha256(PROMPT.encode("utf-8")).hexdigest(),
        "chat_text_sha256": hashlib.sha256(chat_text.encode("utf-8")).hexdigest(),
        "marker_token_ids": list(map(int, marker_ids)),
        "marker_final_subtoken_index": marker_position,
        "hidden_state_tuple_index": 32,
        "target_layer_module_path": layer_path,
        "target_layer_pre_hook_index": target_layer_index,
        "do_sample": False,
        "max_new_tokens": 128,
    }
    args.receipt.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt_sha256 = sha256_file(args.receipt)
    existing_rows = list(read_jsonl(output_path)) if output_path.exists() else []
    if any(
        str(row.get("generation_receipt_sha256") or "") != receipt_sha256
        for row in existing_rows
    ):
        raise ValueError("Existing Patchscope output was produced under another receipt")

    pending = [row for row in rows if str(row["id"]) not in done]
    groups = [
        [row for row in pending if row.get("patch_activation_path") is not None],
        [row for row in pending if row.get("patch_activation_path") is None],
    ]
    finished_total = 0
    for group in groups:
        for start in range(0, len(group), args.batch_size):
            batch = group[start : start + args.batch_size]
            batch_input_ids = encoded["input_ids"].repeat(len(batch), 1).to(model.device)
            attention_mask = encoded["attention_mask"].repeat(len(batch), 1).to(model.device)
            patch_paths = [row.get("patch_activation_path") for row in batch]
            vectors = (
                None
                if all(path is None for path in patch_paths)
                else torch.stack([load_activation(str(path)) for path in patch_paths])
            )
            with torch.inference_mode(), patched_prefill(
                layers[target_layer_index], vectors, marker_position
            ):
                generated = model.generate(
                    input_ids=batch_input_ids,
                    attention_mask=attention_mask,
                    do_sample=False,
                    max_new_tokens=128,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            input_length = batch_input_ids.shape[1]
            for offset, row in enumerate(batch):
                response = tokenizer.decode(
                    generated[offset, input_length:], skip_special_tokens=True
                ).strip()
                append_jsonl(
                    output_path,
                    {
                        **row,
                        "response": response,
                        "output_contract_valid": output_contract_valid(response),
                        "model_id": model_cfg["model_id"],
                        "generation_receipt_sha256": receipt_sha256,
                    },
                )
            finished_total += len(batch)
            if finished_total % (args.batch_size * 10) == 0 or finished_total == len(pending):
                print(f"[patchscope] {finished_total}/{len(pending)}", flush=True)
    output_rows = list(read_jsonl(output_path))
    if len(output_rows) != len(rows) or {str(row["id"]) for row in output_rows} != {
        str(row["id"]) for row in rows
    }:
        raise ValueError("Patchscope generation population mismatch")
    seal = {
        "schema_version": 1,
        "rows": len(output_rows),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
        "receipt_sha256": sha256_file(args.receipt),
    }
    args.seal.write_text(json.dumps(seal, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[generated] {len(output_rows)} -> {output_path}")


def materialize_logical(args: argparse.Namespace) -> None:
    generated = {str(row["id"]): row for row in read_jsonl(args.generated)}
    logical = list(read_jsonl(args.logical_manifest))
    outputs = []
    for row in logical:
        source = generated.get(str(row["generation_id"]))
        if source is None:
            raise ValueError(f"Missing unique generation {row['generation_id']}")
        response = str(source.get("response") or "").strip()
        outputs.append(
            {
                "id": row["id"],
                "base_id": row["base_id"],
                "variant": row["variant"],
                "patch_condition": row["patch_condition"],
                "response": "" if response == "NONE" else response,
                "raw_response": response,
                "output_contract_valid": bool(source.get("output_contract_valid")),
                "generation_id": row["generation_id"],
            }
        )
    write_jsonl(args.output, outputs)
    print(f"[logical] {len(outputs)} -> {args.output}")


def mapper_prepare(args: argparse.Namespace) -> None:
    protocol, alias_table, ontology, template = read_protocol(args.protocol)
    source = list(read_jsonl(args.readouts))
    items = [
        {
            "id": str(row["id"]),
            "base_id": str(row["base_id"]),
            "variant": str(row["variant"]),
            "text": str(row.get("response") or ""),
        }
        for row in source
    ]
    prepared, residual = prepare_items(items, alias_table)
    ordered = [{"claim_id": key, "claim": residual[key]} for key in sorted(residual)]
    requests = []
    batch_size = int(protocol["batch_size"])
    for start in range(0, len(ordered), batch_size):
        batch = ordered[start : start + batch_size]
        prompt = make_batch_prompt(batch, ontology, template)
        request_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]
        requests.append(
            {
                "id": f"patchscope_semantic_{start // batch_size:06d}_{request_hash}",
                "prompt": prompt,
                "claim_ids": [item["claim_id"] for item in batch],
            }
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "prepared_items.jsonl", prepared)
    write_jsonl(args.out_dir / "semantic_requests.jsonl", requests)
    report = {
        "schema_version": 1,
        "readout_rows": len(items),
        "claims": sum(len(row["claims"]) for row in prepared),
        "unique_residual_claims": len(residual),
        "requests": len(requests),
        "protocol_sha256": sha256_file(args.protocol),
        "locked_test_read": False,
    }
    (args.out_dir / "prepare_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[mapper-prepare] residual={len(residual)} requests={len(requests)}")


def selected_set(row: dict[str, Any]) -> set[str]:
    return {str(item["evidence_id"]) for item in row.get("selected_claims", [])}


def value_prediction(row: dict[str, Any], evidence: str) -> str | None:
    for item in row.get("selected_claims", []):
        if str(item["evidence_id"]) == evidence:
            return None if item.get("value_id") is None else str(item["value_id"])
    return None


def case_f1(prediction: set[str], target: set[str]) -> float:
    if not prediction and not target:
        return 1.0
    return 2 * len(prediction & target) / (len(prediction) + len(target))


def paired_summary(values: list[float], clusters: list[str]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "row_bootstrap_95_ci": row_bootstrap_ci(values),
        "diagnosis_cluster_bootstrap_95_ci": cluster_bootstrap_ci(values, clusters),
    }


def mapper_finalize(args: argparse.Namespace) -> None:
    protocol, _alias, ontology, _template = read_protocol(args.protocol)
    receipt = json.loads(args.mapper_receipt.read_text(encoding="utf-8"))
    if receipt.get("all_gates_passed") is not True:
        raise ValueError("Frozen mapper receipt did not pass G1-G4")
    if receipt.get("protocol_sha256") != sha256_file(args.protocol):
        raise ValueError("Frozen mapper receipt/protocol hash mismatch")
    prepared = list(read_jsonl(args.prepared_items))
    requests = list(read_jsonl(args.requests))
    judgements = list(read_jsonl(args.judgements)) if args.judgements.exists() else []
    if requests:
        decisions, model, semantic_audit = semantic_decisions(
            prepared, requests, judgements, protocol=protocol, ontology=ontology
        )
    else:
        decisions, semantic_audit = {}, []
        model = str(receipt.get("primary_model_id") or "")
    if model != str(receipt.get("primary_model_id") or ""):
        raise ValueError("Patchscope mapper model differs from frozen receipt")
    mapped = materialize_items(prepared, decisions)
    write_jsonl(args.out_dir / "mapped_readouts.jsonl", mapped)
    write_jsonl(args.out_dir / "semantic_decisions.jsonl", semantic_audit)

    manifest = {str(row["id"]): row for row in read_jsonl(args.logical_manifest)}
    raw = {str(row["id"]): row for row in read_jsonl(args.logical_readouts)}
    output = {str(row["id"]): row for row in mapped}
    if set(manifest) != set(raw) or set(manifest) != set(output):
        raise ValueError("Patchscope logical/mapped population mismatch")
    structured = json.loads(
        Path(protocol["structured_protocol"]["path"]).read_text(encoding="utf-8")
    )
    labels = set(map(str, structured["finding_labels"]))

    by_key = {}
    for identifier, row in manifest.items():
        key = (str(row["base_id"]), str(row["variant"]), str(row["patch_condition"]))
        by_key[key] = output[identifier]
    base_ids = sorted({key[0] for key in by_key})
    originals = {
        base: next(
            row
            for row in manifest.values()
            if str(row["base_id"]) == base and str(row["variant"]) == "original"
        )
        for base in base_ids
    }
    clusters = [str(originals[base].get("diagnosis_id") or "") for base in base_ids]

    condition_metrics = {}
    per_case_f1 = {}
    for condition in CONDITIONS:
        predictions = []
        targets = []
        case_values = []
        for base in base_ids:
            prediction = selected_set(by_key[(base, "original", condition)])
            target = set(map(str, originals[base].get("cue_evidence_ids") or [])) & labels
            predictions.append(prediction)
            targets.append(target)
            case_values.append(case_f1(prediction, target))
        per_case_f1[condition] = case_values
        condition_metrics[condition] = {
            "finding_micro_f1": micro_f1(predictions, targets),
            "mean_case_f1": statistics.fmean(case_values),
            "mean_claims": statistics.fmean(len(value) for value in predictions),
        }
    specificity = {}
    for control in CONDITIONS[1:]:
        deltas = [
            left - right
            for left, right in zip(per_case_f1["real"], per_case_f1[control], strict=True)
        ]
        specificity[control] = paired_summary(deltas, clusters)
    finding_gate = all(
        item["diagnosis_cluster_bootstrap_95_ci"][0] > 0
        for item in specificity.values()
    )

    deletion_drops = []
    retained_before = retained_after = 0
    for base in base_ids:
        original_row = originals[base]
        deleted_row = next(
            row
            for row in manifest.values()
            if str(row["base_id"]) == base and str(row["variant"]) == "cue_deleted"
        )
        before_set = selected_set(by_key[(base, "original", "real")])
        after_set = selected_set(by_key[(base, "cue_deleted", "real")])
        changed = str(deleted_row.get("cf_original_evidence_id") or "")
        deletion_drops.append(float(changed in before_set) - float(changed in after_set))
        original_cues = set(map(str, original_row.get("cue_evidence_ids") or [])) & labels
        deleted_cues = set(map(str, deleted_row.get("cue_evidence_ids") or [])) & labels
        for evidence in original_cues & deleted_cues:
            if evidence in before_set:
                retained_before += 1
                retained_after += evidence in after_set
    deletion = paired_summary(deletion_drops, clusters)
    retention = retained_after / retained_before if retained_before else None
    deletion_gate = (
        deletion["diagnosis_cluster_bootstrap_95_ci"][0] > 0
        and retention is not None
        and retention >= 0.90
    )

    replacement_deltas = []
    old_deltas = []
    clean_switch = []
    for base in base_ids:
        edit_row = next(
            row
            for row in manifest.values()
            if str(row["base_id"]) == base and str(row["variant"]) == "value_edited"
        )
        evidence = str(edit_row.get("cf_original_evidence_id") or "")
        old = str(edit_row.get("cf_original_value_id") or "")
        new = str(edit_row.get("cf_replacement_value_id") or "")
        before = value_prediction(by_key[(base, "original", "real")], evidence)
        after = value_prediction(by_key[(base, "value_edited", "real")], evidence)
        replacement_deltas.append(float(after == new) - float(before == new))
        old_deltas.append(float(after == old) - float(before == old))
        clean_switch.append(float(before == old and after == new))
    replacement = paired_summary(replacement_deltas, clusters)
    old_persistence = paired_summary(old_deltas, clusters)
    value_gate = (
        replacement["diagnosis_cluster_bootstrap_95_ci"][0] > 0
        and old_persistence["diagnosis_cluster_bootstrap_95_ci"][1] < 0
    )

    parse_rate = sum(bool(row.get("output_contract_valid")) for row in raw.values()) / len(raw)
    diagnosis_mentions = 0
    for identifier, row in manifest.items():
        diagnosis = (
            clean(row.get("diagnosis_name") or row.get("diagnosis_id"))
            .replace("_", " ")
            .lower()
        )
        text = clean(raw[identifier].get("raw_response")).lower()
        diagnosis_mentions += bool(diagnosis and diagnosis in text)
    lexical_claims = sum(
        bool(claim.get("lexical_mappings"))
        for row in prepared
        for claim in row.get("claims", [])
    )
    total_claims = sum(len(row.get("claims", [])) for row in prepared)
    result = {
        "schema_version": 1,
        "validation_only": True,
        "locked_test_read": False,
        "cases": len(base_ids),
        "logical_cells": len(manifest),
        "parse_rate": parse_rate,
        "diagnosis_mention_rate": diagnosis_mentions / len(manifest),
        "claims": total_claims,
        "lexical_claims": lexical_claims,
        "llm_residual_claims": total_claims - lexical_claims,
        "conditions": condition_metrics,
        "patient_specificity": specificity,
        "deletion": {
            **deletion,
            "untouched_retention": retention,
            "retained_original_hits": retained_before,
        },
        "value_edit": {
            "replacement_hit_delta": replacement,
            "old_persistence_delta": old_persistence,
            "clean_switch_rate": statistics.fmean(clean_switch),
        },
        "gates": {
            "output_contract_parse_rate_ge_0p95": parse_rate >= 0.95,
            "patient_specific_finding_readout": finding_gate,
            "selective_deletion_response": deletion_gate,
            "value_edit_response": value_gate,
        },
        "mapper_model_id": model,
        "mapper_protocol_sha256": sha256_file(args.protocol),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# D22 Frozen Patchscope Smoke",
        "",
        "Validation-only, self-contained-claim Patchscope scored by the frozen G1-G4 mapper.",
        "",
        f"- cases / logical cells: **{len(base_ids)} / {len(manifest)}**",
        f"- output-contract parse rate: **{parse_rate:.4f}**",
        f"- diagnosis mention rate: **{diagnosis_mentions / len(manifest):.4f}**",
        f"- mapped claims (lexical / residual): **{total_claims} "
        f"({lexical_claims} / {total_claims - lexical_claims})**",
        "- locked test read: **no**",
        "",
        "| original condition | finding micro F1 | mean case F1 | mean claims | "
        "real-minus-control [cluster CI] |",
        "|---|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        item = condition_metrics[condition]
        if condition == "real":
            gap = "-"
        else:
            comparison = specificity[condition]
            ci = comparison["diagnosis_cluster_bootstrap_95_ci"]
            gap = f"{comparison['mean']:+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]"
        lines.append(
            f"| {condition} | {item['finding_micro_f1']:.4f} | "
            f"{item['mean_case_f1']:.4f} | {item['mean_claims']:.2f} | {gap} |"
        )
    deletion_ci = deletion["diagnosis_cluster_bootstrap_95_ci"]
    replacement_ci = replacement["diagnosis_cluster_bootstrap_95_ci"]
    old_ci = old_persistence["diagnosis_cluster_bootstrap_95_ci"]
    lines.extend(
        [
            "",
            "## Counterfactual Response",
            "",
            f"- deletion target hit drop: **{deletion['mean']:+.4f}** "
            f"[{deletion_ci[0]:+.4f}, {deletion_ci[1]:+.4f}]",
            f"- untouched retention: **{retention if retention is not None else 'N/A'}**",
            f"- replacement-hit delta: **{replacement['mean']:+.4f}** "
            f"[{replacement_ci[0]:+.4f}, {replacement_ci[1]:+.4f}]",
            f"- old-persistence delta: **{old_persistence['mean']:+.4f}** "
            f"[{old_ci[0]:+.4f}, {old_ci[1]:+.4f}]",
            f"- clean-switch rate: **{statistics.fmean(clean_switch):.4f}**",
            "",
            "## Frozen Gates",
            "",
        ]
    )
    for key, value in result["gates"].items():
        lines.append(f"- {key}: **{value}**")
    lines.append("")
    (args.out_dir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--validation-manifest", required=True, type=Path)
    prep.add_argument("--train-manifest", required=True, type=Path)
    prep.add_argument("--structured-protocol", required=True, type=Path)
    prep.add_argument("--path-map", action="append", default=[], type=parse_path_map)
    prep.add_argument("--cases", type=int, default=50)
    prep.add_argument("--out-dir", required=True, type=Path)

    gen = sub.add_parser("generate")
    gen.add_argument("--config", default="configs/default.yaml")
    gen.add_argument("--generation-manifest", required=True, type=Path)
    gen.add_argument("--population-protocol", required=True, type=Path)
    gen.add_argument("--output", required=True, type=Path)
    gen.add_argument("--receipt", required=True, type=Path)
    gen.add_argument("--seal", required=True, type=Path)
    gen.add_argument("--batch-size", type=int, default=4)

    logical = sub.add_parser("materialize-logical")
    logical.add_argument("--generated", required=True, type=Path)
    logical.add_argument("--logical-manifest", required=True, type=Path)
    logical.add_argument("--output", required=True, type=Path)

    map_prep = sub.add_parser("mapper-prepare")
    map_prep.add_argument("--readouts", required=True, type=Path)
    map_prep.add_argument("--protocol", required=True, type=Path)
    map_prep.add_argument("--out-dir", required=True, type=Path)

    final = sub.add_parser("mapper-finalize")
    final.add_argument("--prepared-items", required=True, type=Path)
    final.add_argument("--requests", required=True, type=Path)
    final.add_argument("--judgements", required=True, type=Path)
    final.add_argument("--logical-manifest", required=True, type=Path)
    final.add_argument("--logical-readouts", required=True, type=Path)
    final.add_argument("--protocol", required=True, type=Path)
    final.add_argument("--mapper-receipt", required=True, type=Path)
    final.add_argument("--out-dir", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "generate":
        generate(args)
    elif args.command == "materialize-logical":
        materialize_logical(args)
    elif args.command == "mapper-prepare":
        mapper_prepare(args)
    else:
        mapper_finalize(args)


if __name__ == "__main__":
    main()
