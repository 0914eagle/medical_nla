#!/usr/bin/env python3
"""Select a same-layer Patchscope on controls, then apply that DDXPlus layer."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.calibrate_ddxplus_d22_patchscope import (
    distribution_metrics,
    patched_generate,
    patched_target_logits,
)
from scripts.calibrate_ddxplus_d22_patchscope_feature import (
    CLINICAL_PROMPTS,
    TARGET_LAYERS,
    choose_control_cell,
    run_control_family,
    summarize_control_rows,
)
from scripts.run_ddxplus_d22_patchscope import (
    compute_train_mean,
    index_activation_rows,
    resolve_layer_stack,
)
from src.config import load_config
from src.ddxplus_semantic_mapping import sha256_file
from src.jsonl import read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.reconstruction_scoring import load_activation


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def parse_layer_path(value: str) -> tuple[int, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("layer path must be LAYER=PATH")
    layer, path = value.split("=", 1)
    return int(layer), Path(path)


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path map must be OLD=NEW")
    old, new = value.split("=", 1)
    return old, new


def exact_layer_map(
    values: list[tuple[int, Path]], label: str
) -> dict[int, Path]:
    result = dict(values)
    if set(result) != set(TARGET_LAYERS) or len(result) != len(values):
        raise ValueError(
            f"{label} must provide exactly layers {TARGET_LAYERS}, got {sorted(result)}"
        )
    for path in result.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    return result


def fixed_donors(generation_manifest: Path) -> dict[str, str]:
    donors = {}
    for row in read_jsonl(generation_manifest):
        if (
            str(row.get("variant")) == "original"
            and str(row.get("condition")) == "same_diagnosis_shuffled"
        ):
            base_id = str(row["base_id"])
            donor = str(row.get("donor_base_id") or "")
            if not donor or base_id in donors:
                raise ValueError(f"Missing/duplicate donor for {base_id}")
            donors[base_id] = donor
    return donors


def run(args: argparse.Namespace) -> None:
    validation_manifests = exact_layer_map(
        args.validation_layer_manifest, "validation manifests"
    )
    train_manifests = exact_layer_map(
        args.train_layer_manifest, "train manifests"
    )
    cfg = load_config(args.config)
    model_cfg = cfg["source_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prior = json.loads(args.v1_protocol.read_text(encoding="utf-8"))
    base_ids = list(map(str, prior["selected_base_ids"]))[: args.cases]
    if len(base_ids) != args.cases or len(set(base_ids)) != args.cases:
        raise ValueError(
            f"Expected {args.cases} unique frozen clinical IDs, got {len(base_ids)}"
        )
    protocol_path = args.out_dir / "protocol.json"
    source_paths = [
        args.config,
        args.v1_protocol,
        args.v1_generation_manifest,
        *validation_manifests.values(),
        *train_manifests.values(),
    ]
    protocol = {
        "schema_version": 1,
        "written_before_model_inference": True,
        "validation_only": True,
        "locked_test_read": False,
        "selection_uses_clinical_outputs": False,
        "model_id": model_cfg["model_id"],
        "candidate_cells": [
            {
                "family": family,
                "source_layer": layer,
                "target_layer": layer,
            }
            for family in ("entity_description", "relation_specific")
            for layer in TARGET_LAYERS
        ],
        "selection_gate": {
            "keyword_hits_min_of_5": 3,
            "keyword_gain_must_be_positive": True,
            "outputs_differing_from_no_patch_min_of_5": 4,
        },
        "tie_break_order": [
            "keyword_hit_rate",
            "keyword_gain",
            "divergence_rate",
            "layer_distance_to_hs32",
            "relation_specific_over_entity_description",
        ],
        "clinical_prompts": CLINICAL_PROMPTS,
        "clinical_base_ids": base_ids,
        "fixed_donor_source": str(args.v1_generation_manifest),
        "validation_layer_manifests": {
            str(layer): str(path)
            for layer, path in sorted(validation_manifests.items())
        },
        "train_layer_manifests": {
            str(layer): str(path)
            for layer, path in sorted(train_manifests.items())
        },
        "sources": {str(path): sha256_file(path) for path in source_paths},
    }
    protocol_path.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[protocol] frozen before model inference: {protocol_path}", flush=True)

    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    layer_path, layers = resolve_layer_stack(model)

    control_rows = []
    for family in ("entity_description", "relation_specific"):
        control_rows.extend(
            run_control_family(
                model,
                tokenizer,
                layers,
                family,
                target_layers=TARGET_LAYERS,
                source_layers_by_target={layer: layer for layer in TARGET_LAYERS},
            )
        )
    control_summaries = summarize_control_rows(control_rows)
    selected = choose_control_cell(control_summaries)
    selection_path = args.out_dir / "selection.json"
    selection = {
        "schema_version": 1,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selection_used_only_general_domain_controls": True,
        "control_summaries": control_summaries,
        "selected_cell": selected,
        "control_gate_passed": selected is not None,
        "clinical_activation_content_used_for_selection": False,
    }
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_jsonl(args.out_dir / "control_rows.jsonl", control_rows)
    print(f"[selection] {selected}", flush=True)

    clinical_rows: list[dict[str, Any]] = []
    clinical_no_patch = ""
    mean_path: Path | None = None
    if selected is not None:
        source_layer = int(selected["source_layer"])
        target_layer = int(selected["target_layer"])
        if source_layer != target_layer:
            raise AssertionError("Same-layer calibration selected a cross-layer cell")
        family = str(selected["family"])
        target_prompt = CLINICAL_PROMPTS[family]
        donors = fixed_donors(args.v1_generation_manifest)
        activation_rows = index_activation_rows(
            validation_manifests[source_layer], args.path_map
        )
        train_mean, train_count = compute_train_mean(
            train_manifests[source_layer], args.path_map
        )
        mean_path = args.out_dir / f"frozen_train_mean_hs{source_layer}.pt"
        torch.save(train_mean, mean_path)
        print(
            f"[clinical population] layer={source_layer} train_mean_n={train_count}",
            flush=True,
        )

        no_patch_logits = patched_target_logits(
            model, tokenizer, layers, target_prompt, None, target_layer
        )
        clinical_no_patch = patched_generate(
            model,
            tokenizer,
            layers,
            target_prompt,
            None,
            layer_index=target_layer,
            max_new_tokens=80,
        )
        for index, base_id in enumerate(base_ids):
            donor_id = donors.get(base_id)
            if not donor_id:
                raise ValueError(f"No frozen donor for {base_id}")
            if base_id not in activation_rows or donor_id not in activation_rows:
                raise ValueError(
                    f"Selected/donor base missing at HS{source_layer}: "
                    f"{base_id}/{donor_id}"
                )
            paths = {
                "real": Path(activation_rows[base_id]["original"]["activation_path"]),
                "same_diagnosis_shuffled": Path(
                    activation_rows[donor_id]["original"]["activation_path"]
                ),
                "train_mean": mean_path,
            }
            for condition, activation_path in paths.items():
                vector = load_activation(activation_path)
                patched_logits = patched_target_logits(
                    model,
                    tokenizer,
                    layers,
                    target_prompt,
                    vector,
                    target_layer,
                )
                response = patched_generate(
                    model,
                    tokenizer,
                    layers,
                    target_prompt,
                    vector,
                    layer_index=target_layer,
                    max_new_tokens=80,
                )
                metrics = distribution_metrics(
                    patched_logits,
                    no_patch_logits,
                    int(patched_logits.argmax().item()),
                )
                clinical_rows.append(
                    {
                        "base_id": base_id,
                        "donor_base_id": (
                            donor_id
                            if condition == "same_diagnosis_shuffled"
                            else None
                        ),
                        "condition": condition,
                        "family": family,
                        "source_layer": source_layer,
                        "target_layer": target_layer,
                        "target_prompt": target_prompt,
                        "activation_path": str(activation_path),
                        "response": response,
                        "no_patch_response": clinical_no_patch,
                        "differs_from_no_patch": response != clinical_no_patch,
                        "first_token_kl_to_no_patch": metrics[
                            "kl_patched_to_no_patch"
                        ],
                        "max_abs_logit_delta": metrics["max_abs_logit_delta"],
                    }
                )
            print(f"[clinical] {index + 1}/{len(base_ids)}", flush=True)
        write_jsonl(args.out_dir / "clinical_rows.jsonl", clinical_rows)
    else:
        print("[stop] no same-layer general-domain cell passed", flush=True)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clinical_rows:
        grouped[str(row["condition"])].append(row)
    result = {
        "schema_version": 1,
        "validation_only": True,
        "locked_test_read": False,
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "selection": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "model_id": model_cfg["model_id"],
        "model_revision": str(getattr(model.config, "_commit_hash", "") or ""),
        "layer_module_path": layer_path,
        "control_summaries": control_summaries,
        "selected_cell": selected,
        "clinical_semantic_audit_authorized": selected is not None,
        "clinical_no_patch_response": clinical_no_patch,
        "clinical_rows": clinical_rows,
        "train_mean_path": str(mean_path) if mean_path else None,
        "train_mean_sha256": sha256_file(mean_path) if mean_path else None,
        "clinical_unique_responses": len(
            {row["response"] for row in clinical_rows}
        ),
        "clinical_response_counts": dict(
            Counter(row["response"] for row in clinical_rows)
        ),
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if selected:
        selected_label = (
            f"{selected['family']} / "
            f"HS{selected['source_layer']}->HS{selected['target_layer']}"
        )
    else:
        selected_label = "none"
    lines = [
        "# D22 Same-Layer Patchscope Source Sweep",
        "",
        "General-domain controls select a same-layer source/target cell before",
        "the corresponding DDXPlus activation tensors are loaded.",
        "",
        f"- control gate passed: **{selected is not None}**",
        f"- selected cell: **{selected_label}**",
        f"- clinical semantic audit authorized: **{selected is not None}**",
        f"- locked test read: **no**",
        "",
        "## General-Domain Selection",
        "",
        "| family | source HS | target HS | hits | no-patch hits | gain | "
        "changed | mean KL | eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in control_summaries:
        eligible = (
            row["keyword_hits"] >= 3
            and row["keyword_gain"] > 0
            and row["outputs_differing_from_no_patch"] >= 4
        )
        lines.append(
            f"| {row['family']} | {row['source_layer']} | {row['target_layer']} | "
            f"{row['keyword_hits']}/{row['n']} | {row['no_patch_keyword_hits']}/{row['n']} | "
            f"{row['keyword_gain']:+.4f} | {row['outputs_differing_from_no_patch']}/{row['n']} | "
            f"{row['mean_first_token_kl']:.6f} | {eligible} |"
        )
    lines.extend(
        [
            "",
            "## Clinical Application",
            "",
            "Raw continuations remain diagnostic outputs, not semantic scores.",
            "",
            "| condition | n | changed from no-patch | mean KL | unique outputs |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for condition, rows in sorted(grouped.items()):
        lines.append(
            f"| {condition} | {len(rows)} | "
            f"{mean([float(row['differs_from_no_patch']) for row in rows]):.4f} | "
            f"{mean([float(row['first_token_kl_to_no_patch']) for row in rows]):.6f} | "
            f"{len({row['response'] for row in rows})} |"
        )
    lines.append("")
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"))
    print(f"[done] {args.out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--validation-layer-manifest",
        action="append",
        default=[],
        type=parse_layer_path,
    )
    parser.add_argument(
        "--train-layer-manifest",
        action="append",
        default=[],
        type=parse_layer_path,
    )
    parser.add_argument("--v1-protocol", required=True, type=Path)
    parser.add_argument("--v1-generation-manifest", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], type=parse_mapping)
    parser.add_argument("--cases", type=int, default=5)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()
    if args.cases < 2:
        raise ValueError("At least two clinical cases are required")
    run(args)


if __name__ == "__main__":
    main()
