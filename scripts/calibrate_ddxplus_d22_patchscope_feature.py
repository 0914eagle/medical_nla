#!/usr/bin/env python3
"""Select a Patchscope interface on public controls, then apply it clinically."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch

from scripts.calibrate_ddxplus_d22_patchscope import (
    distribution_metrics,
    mapped_path,
    patched_generate,
    patched_target_logits,
    select_ddx_cases,
    source_states,
)
from scripts.run_ddxplus_d22_patchscope import resolve_layer_stack
from src.config import load_config
from src.ddxplus_semantic_mapping import sha256_file
from src.jsonl import write_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.reconstruction_scoring import load_activation


SOURCE_LAYER = 32
TARGET_LAYERS = (16, 24, 32)
MARKER = "foo"
ENTITY_PROMPT = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean electronics company, foo"
)
CLINICAL_PROMPTS = {
    "entity_description": (
        "Patient A: fever and productive cough, "
        "Patient B: substernal chest pain and exertional dyspnea, "
        "Patient C: itchy swollen rash, Patient foo"
    ),
    "relation_specific": "The clinical findings of patient foo",
}

ENTITY_CONTROLS = (
    {
        "source": "This text is about Paris",
        "keywords": ("france", "french", "capital", "city"),
    },
    {
        "source": "This text is about Saturn",
        "keywords": ("planet", "solar", "ring"),
    },
    {
        "source": "This text is about Mozart",
        "keywords": ("composer", "music", "austrian"),
    },
    {
        "source": "This text is about oxygen",
        "keywords": ("element", "gas", "chemical"),
    },
    {
        "source": "This text is about Mount Everest",
        "keywords": ("mountain", "highest", "tallest"),
    },
)

RELATION_CONTROLS = (
    {
        "source": "This text is about France",
        "target": "The capital city of foo",
        "keywords": ("paris",),
    },
    {
        "source": "This text is about Japan",
        "target": "The official currency of foo",
        "keywords": ("yen",),
    },
    {
        "source": "This text is about Mozart",
        "target": "The profession of foo",
        "keywords": ("composer", "musician"),
    },
    {
        "source": "This text is about Saturn",
        "target": "The astronomical object type of foo",
        "keywords": ("planet",),
    },
    {
        "source": "This text is about Mount Everest",
        "target": "The mountain range containing foo",
        "keywords": ("himalaya",),
    },
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def keyword_hit(text: str, keywords: tuple[str, ...] | list[str]) -> bool:
    lowered = text.lower()
    return any(
        re.search(
            rf"(?<!\w){re.escape(str(keyword).lower())}(?!\w)",
            lowered,
        )
        is not None
        for keyword in keywords
    )


def summarize_control_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["family"]), int(row["target_layer"]))].append(row)
    summaries = []
    for (family, target_layer), group in sorted(grouped.items()):
        summaries.append(
            {
                "family": family,
                "source_layer": SOURCE_LAYER,
                "target_layer": target_layer,
                "n": len(group),
                "keyword_hits": sum(bool(row["keyword_hit"]) for row in group),
                "keyword_hit_rate": mean(
                    [float(row["keyword_hit"]) for row in group]
                ),
                "no_patch_keyword_hits": sum(
                    bool(row["no_patch_keyword_hit"]) for row in group
                ),
                "no_patch_keyword_hit_rate": mean(
                    [float(row["no_patch_keyword_hit"]) for row in group]
                ),
                "keyword_gain": mean(
                    [
                        float(row["keyword_hit"])
                        - float(row["no_patch_keyword_hit"])
                        for row in group
                    ]
                ),
                "outputs_differing_from_no_patch": sum(
                    bool(row["differs_from_no_patch"]) for row in group
                ),
                "divergence_rate": mean(
                    [float(row["differs_from_no_patch"]) for row in group]
                ),
                "mean_first_token_kl": mean(
                    [float(row["first_token_kl_to_no_patch"]) for row in group]
                ),
            }
        )
    return summaries


def choose_control_cell(
    summaries: list[dict[str, Any]],
) -> dict[str, Any] | None:
    eligible = [
        row
        for row in summaries
        if int(row["keyword_hits"]) >= 3
        and float(row["keyword_gain"]) > 0
        and int(row["outputs_differing_from_no_patch"]) >= 4
    ]
    if not eligible:
        return None
    family_preference = {"relation_specific": 1, "entity_description": 0}
    return max(
        eligible,
        key=lambda row: (
            float(row["keyword_hit_rate"]),
            float(row["keyword_gain"]),
            float(row["divergence_rate"]),
            -abs(int(row["target_layer"]) - SOURCE_LAYER),
            family_preference[str(row["family"])],
        ),
    )


def run_control_family(
    model: Any,
    tokenizer: Any,
    layers: Any,
    family: str,
) -> list[dict[str, Any]]:
    controls = ENTITY_CONTROLS if family == "entity_description" else RELATION_CONTROLS
    rows = []
    for index, control in enumerate(controls):
        target = ENTITY_PROMPT if family == "entity_description" else str(control["target"])
        no_patch_logits = patched_target_logits(
            model, tokenizer, layers, target, None, SOURCE_LAYER
        )
        no_patch_response = patched_generate(
            model,
            tokenizer,
            layers,
            target,
            None,
            layer_index=SOURCE_LAYER,
            max_new_tokens=40,
        )
        states, _source_logits, _position = source_states(
            model,
            tokenizer,
            str(control["source"]),
            add_special_tokens=True,
        )
        for target_layer in TARGET_LAYERS:
            patched_logits = patched_target_logits(
                model,
                tokenizer,
                layers,
                target,
                states[SOURCE_LAYER],
                target_layer,
            )
            response = patched_generate(
                model,
                tokenizer,
                layers,
                target,
                states[SOURCE_LAYER],
                layer_index=target_layer,
                max_new_tokens=40,
            )
            metrics = distribution_metrics(
                patched_logits,
                no_patch_logits,
                int(patched_logits.argmax().item()),
            )
            keywords = tuple(map(str, control["keywords"]))
            rows.append(
                {
                    "family": family,
                    "control_index": index,
                    "source": str(control["source"]),
                    "target_prompt": target,
                    "keywords": list(keywords),
                    "source_layer": SOURCE_LAYER,
                    "target_layer": target_layer,
                    "response": response,
                    "no_patch_response": no_patch_response,
                    "keyword_hit": keyword_hit(response, keywords),
                    "no_patch_keyword_hit": keyword_hit(
                        no_patch_response, keywords
                    ),
                    "differs_from_no_patch": response != no_patch_response,
                    "first_token_kl_to_no_patch": metrics[
                        "kl_patched_to_no_patch"
                    ],
                    "max_abs_logit_delta": metrics["max_abs_logit_delta"],
                }
            )
        print(
            f"[control:{family}] {index + 1}/{len(controls)}",
            flush=True,
        )
    return rows


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    model_cfg = cfg["source_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    v1_protocol = json.loads(args.v1_protocol.read_text(encoding="utf-8"))
    selected_base_ids = list(map(str, v1_protocol["selected_base_ids"]))[: args.cases]
    protocol_path = args.out_dir / "protocol.json"
    protocol = {
        "schema_version": 1,
        "written_before_model_inference": True,
        "validation_only": True,
        "locked_test_read": False,
        "selection_uses_clinical_outputs": False,
        "model_id": model_cfg["model_id"],
        "source_layer": SOURCE_LAYER,
        "target_layers": list(TARGET_LAYERS),
        "families": ["entity_description", "relation_specific"],
        "entity_prompt": ENTITY_PROMPT,
        "clinical_prompts": CLINICAL_PROMPTS,
        "entity_controls": list(ENTITY_CONTROLS),
        "relation_controls": list(RELATION_CONTROLS),
        "selection_gate": {
            "keyword_hits_min_of_5": 3,
            "keyword_gain_must_be_positive": True,
            "outputs_differing_from_no_patch_min_of_5": 4,
        },
        "tie_break_order": [
            "keyword_hit_rate",
            "keyword_gain",
            "divergence_rate",
            "target_layer_distance_to_hs32",
            "relation_specific_over_entity_description",
        ],
        "clinical_base_ids": selected_base_ids,
        "sources": {
            str(path): sha256_file(path)
            for path in (
                args.config,
                args.validation_manifest,
                args.v1_protocol,
                args.v1_generation_manifest,
            )
        },
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
            run_control_family(model, tokenizer, layers, family)
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
    if selected is not None:
        ddx_rows = select_ddx_cases(
            args.validation_manifest,
            args.v1_protocol,
            args.v1_generation_manifest,
            args.path_map,
            args.cases,
        )
        family = str(selected["family"])
        target_layer = int(selected["target_layer"])
        target_prompt = CLINICAL_PROMPTS[family]
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
        for index, row in enumerate(ddx_rows):
            for condition in ("real", "same_diagnosis_shuffled", "train_mean"):
                activation_path = mapped_path(row[condition], args.path_map)
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
                        "base_id": row["base_id"],
                        "condition": condition,
                        "family": family,
                        "source_layer": SOURCE_LAYER,
                        "target_layer": target_layer,
                        "target_prompt": target_prompt,
                        "response": response,
                        "no_patch_response": clinical_no_patch,
                        "differs_from_no_patch": response != clinical_no_patch,
                        "first_token_kl_to_no_patch": metrics[
                            "kl_patched_to_no_patch"
                        ],
                        "max_abs_logit_delta": metrics["max_abs_logit_delta"],
                    }
                )
            print(f"[clinical] {index + 1}/{len(ddx_rows)}", flush=True)
        write_jsonl(args.out_dir / "clinical_rows.jsonl", clinical_rows)
    else:
        print("[stop] no general-domain control cell passed", flush=True)

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

    selected_label = (
        f"{selected['family']} / HS{selected['target_layer']}"
        if selected
        else "none"
    )
    lines = [
        "# D22 Patchscope Feature-Interface Calibration",
        "",
        "General-domain controls select the prompt family and target layer before",
        "any clinical activation is read.",
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
            "Raw clinical continuations are diagnostic outputs, not semantic scores.",
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


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path map must be OLD=NEW")
    old, new = value.split("=", 1)
    return old, new


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--validation-manifest", required=True, type=Path)
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
