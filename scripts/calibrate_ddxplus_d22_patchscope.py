#!/usr/bin/env python3
"""Paper-style Patchscopes calibration before another clinical readout run."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from scripts.run_ddxplus_d22_patchscope import (
    marker_token_span,
    patched_prefill,
    resolve_layer_stack,
)
from src.config import load_config
from src.ddxplus_semantic_mapping import sha256_file
from src.jsonl import read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.reconstruction_scoring import load_activation


IDENTITY_PROMPT = "apple -> apple; river -> river; seven -> seven; music -> music; foo"
ENTITY_PROMPT = (
    "Syria: Country in the Middle East, "
    "Leonardo DiCaprio: American actor, "
    "Samsung: South Korean electronics company, foo"
)
CLINICAL_PROMPT = (
    "state_a -> fever and productive cough; "
    "state_b -> substernal chest pain and exertional dyspnea; "
    "state_c -> itchy swollen rash; foo"
)
MARKER = "foo"
LAYERS = (16, 24, 32, 40)

IDENTITY_SOURCES = (
    "The capital of France is",
    "The largest planet in the Solar System is",
    "Two plus two equals",
    "Water freezes at zero degrees",
    "The opposite of hot is",
    "A baby cat is called a",
    "The chemical symbol for oxygen is",
    "The first month of the year is",
    "The color of grass is usually",
    "Birds can fly through the",
    "Humans breathe in oxygen and breathe out",
    "The square root of nine is",
    "The author of Hamlet was William",
    "The currency of Japan is the",
    "The tallest mountain on Earth is Mount",
    "An instrument with black and white keys is a",
    "The planet known as the Red Planet is",
    "One week contains seven",
    "A triangle has three",
    "The ocean west of California is the",
)

ENTITY_SOURCES = (
    ("This text is about Paris", ("france", "french", "capital", "city")),
    ("This text is about Saturn", ("planet", "solar", "ring")),
    ("This text is about Mozart", ("composer", "music", "austrian")),
    ("This text is about oxygen", ("element", "gas", "chemical")),
    ("This text is about Mount Everest", ("mountain", "highest", "tallest")),
)


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def mapped_path(value: str | Path, mappings: list[tuple[str, str]]) -> Path:
    text = str(value)
    for old, new in mappings:
        if text.startswith(old):
            text = new + text[len(old) :]
            break
    return Path(text)


def parse_mapping(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("path map must be OLD=NEW")
    old, new = value.split("=", 1)
    return old, new


def encode_raw(tokenizer: Any, text: str) -> tuple[dict[str, torch.Tensor], int]:
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = [tuple(map(int, pair)) for pair in encoded.pop("offset_mapping")[0].tolist()]
    start, end = marker_token_span(text, offsets, marker=MARKER)
    if end - start < 1:
        raise ValueError("Patchscope target marker has no token")
    return encoded, end - 1


def model_inputs(encoded: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in encoded.items()}


def top_rank(logits: torch.Tensor, token_id: int) -> int:
    target = logits[token_id]
    return int((logits > target).sum().item()) + 1


def distribution_metrics(
    patched_logits: torch.Tensor,
    no_patch_logits: torch.Tensor,
    target_id: int,
) -> dict[str, Any]:
    patched = patched_logits.float()
    baseline = no_patch_logits.float()
    patched_logp = F.log_softmax(patched, dim=-1)
    baseline_logp = F.log_softmax(baseline, dim=-1)
    patched_p = patched_logp.exp()
    return {
        "patched_top1_id": int(patched.argmax().item()),
        "no_patch_top1_id": int(baseline.argmax().item()),
        "target_rank_patched": top_rank(patched, target_id),
        "target_rank_no_patch": top_rank(baseline, target_id),
        "target_logprob_patched": float(patched_logp[target_id]),
        "target_logprob_no_patch": float(baseline_logp[target_id]),
        "target_logprob_lift": float(patched_logp[target_id] - baseline_logp[target_id]),
        "kl_patched_to_no_patch": float(
            (patched_p * (patched_logp - baseline_logp)).sum()
        ),
        "max_abs_logit_delta": float((patched - baseline).abs().max()),
    }


def raw_forward(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
    output_hidden_states: bool = False,
) -> Any:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens, return_tensors="pt")
    inputs = model_inputs(encoded, model.device)
    with torch.inference_mode():
        return model(
            **inputs,
            output_hidden_states=output_hidden_states,
            use_cache=False,
        )


def patched_target_logits(
    model: Any,
    tokenizer: Any,
    layers: Any,
    prompt: str,
    vector: torch.Tensor | None,
    layer_index: int,
) -> torch.Tensor:
    encoded, marker_position = encode_raw(tokenizer, prompt)
    inputs = model_inputs(encoded, model.device)
    vectors = None if vector is None else vector.float().reshape(1, -1)
    with torch.inference_mode(), patched_prefill(
        layers[layer_index], vectors, marker_position
    ):
        output = model(**inputs, use_cache=False)
    return output.logits[0, -1].detach().float().cpu()


def patched_generate(
    model: Any,
    tokenizer: Any,
    layers: Any,
    prompt: str,
    vector: torch.Tensor | None,
    *,
    layer_index: int,
    max_new_tokens: int,
) -> str:
    encoded, marker_position = encode_raw(tokenizer, prompt)
    inputs = model_inputs(encoded, model.device)
    vectors = None if vector is None else vector.float().reshape(1, -1)
    with torch.inference_mode(), patched_prefill(
        layers[layer_index], vectors, marker_position
    ):
        generated = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    return tokenizer.decode(
        generated[0, input_length:], skip_special_tokens=True
    ).strip()


def source_states(
    model: Any,
    tokenizer: Any,
    text: str,
    *,
    add_special_tokens: bool,
    position: int | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, int]:
    output = raw_forward(
        model,
        tokenizer,
        text,
        add_special_tokens=add_special_tokens,
        output_hidden_states=True,
    )
    index = output.logits.shape[1] - 1 if position is None else position
    if index < 0 or index >= output.logits.shape[1]:
        raise ValueError(
            f"Source position {index} outside sequence length {output.logits.shape[1]}"
        )
    states = {
        layer: output.hidden_states[layer][0, index].detach().float().cpu()
        for layer in LAYERS
    }
    logits = output.logits[0, index].detach().float().cpu()
    return states, logits, index


def select_ddx_cases(
    validation_manifest: Path,
    v1_protocol: Path,
    v1_generation_manifest: Path,
    mappings: list[tuple[str, str]],
    cases: int,
) -> list[dict[str, Any]]:
    protocol = json.loads(v1_protocol.read_text(encoding="utf-8"))
    selected = list(map(str, protocol["selected_base_ids"]))[:cases]
    originals = {
        str(row.get("base_id") or row["id"]): row
        for row in read_jsonl(validation_manifest)
        if str(row.get("variant") or "original") == "original"
    }
    generated = list(read_jsonl(v1_generation_manifest))
    by_key = {
        (str(row["base_id"]), str(row["variant"]), str(row["condition"])): row
        for row in generated
    }
    rows = []
    for base_id in selected:
        source = originals[base_id]
        real = by_key[(base_id, "original", "real")]
        shuffled = by_key[(base_id, "original", "same_diagnosis_shuffled")]
        mean = by_key[(base_id, "shared", "train_mean")]
        activation_path = mapped_path(source["activation_path"], mappings)
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        rows.append(
            {
                "base_id": base_id,
                "chat_text": str(source["chat_text"]),
                "position": int(source["position"]),
                "real": str(mapped_path(real["patch_activation_path"], mappings)),
                "same_diagnosis_shuffled": str(
                    mapped_path(shuffled["patch_activation_path"], mappings)
                ),
                "train_mean": str(mapped_path(mean["patch_activation_path"], mappings)),
            }
        )
    return rows


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else float("nan")


def run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    ddx_rows = select_ddx_cases(
        args.validation_manifest,
        args.v1_protocol,
        args.v1_generation_manifest,
        args.path_map,
        args.cases,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = args.out_dir / "protocol.json"
    protocol = {
        "schema_version": 2,
        "written_before_model_inference": True,
        "validation_only": True,
        "locked_test_read": False,
        "paper_reference": "Ghandeharioun et al. 2024 Patchscopes",
        "model_id": model_cfg["model_id"],
        "selected_base_ids": [row["base_id"] for row in ddx_rows],
        "layers": list(LAYERS),
        "prompts": {
            "identity": IDENTITY_PROMPT,
            "entity": ENTITY_PROMPT,
            "clinical": CLINICAL_PROMPT,
        },
        "gates": {
            "saved_activation_recomputed_cosine_min": 0.999,
            "identity_hs32_precision_at_1_min": 0.40,
            "identity_hs32_must_exceed_no_patch": True,
            "identity_hs32_target_logprob_lift_must_be_positive": True,
            "entity_keyword_hits_min": 3,
            "entity_outputs_differing_from_no_patch_min": 4,
        },
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
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
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

    extraction_rows = []
    for row in ddx_rows:
        states, _logits, position = source_states(
            model,
            tokenizer,
            row["chat_text"],
            add_special_tokens=False,
            position=row["position"],
        )
        saved = load_activation(row["real"])
        extraction_rows.append(
            {
                "base_id": row["base_id"],
                "position": position,
                "saved_vs_recomputed_hs32_cosine": float(
                    F.cosine_similarity(saved.float(), states[32], dim=0)
                ),
                "saved_vs_recomputed_hs32_max_abs": float(
                    (saved.float() - states[32]).abs().max()
                ),
            }
        )

    identity_no_patch = patched_target_logits(
        model, tokenizer, layers, IDENTITY_PROMPT, None, 32
    )
    identity_rows = []
    for source_index, text in enumerate(IDENTITY_SOURCES):
        states, source_logits, _position = source_states(
            model, tokenizer, text, add_special_tokens=True
        )
        target_id = int(source_logits.argmax().item())
        for layer in LAYERS:
            patched = patched_target_logits(
                model, tokenizer, layers, IDENTITY_PROMPT, states[layer], layer
            )
            metrics = distribution_metrics(patched, identity_no_patch, target_id)
            identity_rows.append(
                {
                    "source_index": source_index,
                    "source_prompt": text,
                    "layer": layer,
                    "source_top1_id": target_id,
                    "source_top1_token": tokenizer.decode([target_id]),
                    "patched_top1_token": tokenizer.decode([metrics["patched_top1_id"]]),
                    "no_patch_top1_token": tokenizer.decode([metrics["no_patch_top1_id"]]),
                    "precision_at_1": metrics["patched_top1_id"] == target_id,
                    "no_patch_precision_at_1": metrics["no_patch_top1_id"] == target_id,
                    **metrics,
                }
            )
        print(f"[identity] {source_index + 1}/{len(IDENTITY_SOURCES)}", flush=True)

    entity_no_patch = patched_generate(
        model,
        tokenizer,
        layers,
        ENTITY_PROMPT,
        None,
        layer_index=32,
        max_new_tokens=40,
    )
    entity_rows = []
    for index, (source_text, keywords) in enumerate(ENTITY_SOURCES):
        states, _source_logits, _position = source_states(
            model, tokenizer, source_text, add_special_tokens=True
        )
        response = patched_generate(
            model,
            tokenizer,
            layers,
            ENTITY_PROMPT,
            states[32],
            layer_index=32,
            max_new_tokens=40,
        )
        lowered = response.lower()
        entity_rows.append(
            {
                "source": source_text,
                "keywords": list(keywords),
                "response": response,
                "no_patch_response": entity_no_patch,
                "keyword_hit": any(keyword in lowered for keyword in keywords),
                "differs_from_no_patch": response != entity_no_patch,
            }
        )
        print(f"[entity] {index + 1}/{len(ENTITY_SOURCES)}", flush=True)

    clinical_no_patch_logits = patched_target_logits(
        model, tokenizer, layers, CLINICAL_PROMPT, None, 32
    )
    clinical_no_patch = patched_generate(
        model,
        tokenizer,
        layers,
        CLINICAL_PROMPT,
        None,
        layer_index=32,
        max_new_tokens=80,
    )
    clinical_rows = []
    for index, row in enumerate(ddx_rows):
        for condition in ("real", "same_diagnosis_shuffled", "train_mean"):
            vector = load_activation(row[condition])
            logits = patched_target_logits(
                model, tokenizer, layers, CLINICAL_PROMPT, vector, 32
            )
            response = patched_generate(
                model,
                tokenizer,
                layers,
                CLINICAL_PROMPT,
                vector,
                layer_index=32,
                max_new_tokens=80,
            )
            metrics = distribution_metrics(
                logits, clinical_no_patch_logits, int(logits.argmax().item())
            )
            clinical_rows.append(
                {
                    "base_id": row["base_id"],
                    "condition": condition,
                    "response": response,
                    "no_patch_response": clinical_no_patch,
                    "differs_from_no_patch": response != clinical_no_patch,
                    "kl_to_no_patch": metrics["kl_patched_to_no_patch"],
                    "max_abs_logit_delta": metrics["max_abs_logit_delta"],
                }
            )
        print(f"[clinical] {index + 1}/{len(ddx_rows)}", flush=True)

    identity_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in identity_rows:
        identity_by_layer[int(row["layer"])].append(row)
    identity_summary = {}
    for layer, rows in sorted(identity_by_layer.items()):
        identity_summary[str(layer)] = {
            "n": len(rows),
            "source_top1_unique": len({row["source_top1_id"] for row in rows}),
            "precision_at_1": mean([float(row["precision_at_1"]) for row in rows]),
            "no_patch_precision_at_1": mean(
                [float(row["no_patch_precision_at_1"]) for row in rows]
            ),
            "mean_target_logprob_lift": mean(
                [float(row["target_logprob_lift"]) for row in rows]
            ),
            "mean_kl_to_no_patch": mean(
                [float(row["kl_patched_to_no_patch"]) for row in rows]
            ),
            "mean_max_abs_logit_delta": mean(
                [float(row["max_abs_logit_delta"]) for row in rows]
            ),
        }

    extraction_gate = all(
        row["saved_vs_recomputed_hs32_cosine"] >= 0.999
        for row in extraction_rows
    )
    layer32 = identity_summary["32"]
    identity_gate = (
        layer32["precision_at_1"] >= 0.40
        and layer32["precision_at_1"] > layer32["no_patch_precision_at_1"]
        and layer32["mean_target_logprob_lift"] > 0
    )
    entity_keyword_rate = mean([float(row["keyword_hit"]) for row in entity_rows])
    entity_divergence_rate = mean(
        [float(row["differs_from_no_patch"]) for row in entity_rows]
    )
    entity_gate = entity_keyword_rate >= 0.60 and entity_divergence_rate >= 0.80
    clinical_interpretable = extraction_gate and identity_gate and entity_gate

    result = {
        "schema_version": 2,
        "validation_only": True,
        "locked_test_read": False,
        "paper_reference": "Ghandeharioun et al. 2024 Patchscopes",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "model_id": model_cfg["model_id"],
        "model_revision": clean(getattr(model.config, "_commit_hash", "")),
        "layer_module_path": layer_path,
        "layers": list(LAYERS),
        "prompts": {
            "identity": IDENTITY_PROMPT,
            "entity": ENTITY_PROMPT,
            "clinical": CLINICAL_PROMPT,
        },
        "extraction": extraction_rows,
        "identity": identity_rows,
        "identity_summary": identity_summary,
        "entity": entity_rows,
        "entity_no_patch_response": entity_no_patch,
        "clinical": clinical_rows,
        "clinical_no_patch_response": clinical_no_patch,
        "gates": {
            "saved_activation_recomputed_cosine_ge_0p999": extraction_gate,
            "paper_style_identity_hs32": identity_gate,
            "entity_description_hs32": entity_gate,
            "clinical_outputs_interpretable": clinical_interpretable,
        },
        "entity_keyword_rate": entity_keyword_rate,
        "entity_divergence_rate": entity_divergence_rate,
        "sources": {
            str(path): sha256_file(path)
            for path in (
                args.validation_manifest,
                args.v1_protocol,
                args.v1_generation_manifest,
            )
        },
    }
    (args.out_dir / "results.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.out_dir / "private_identity_rows.jsonl", identity_rows)
    write_jsonl(args.out_dir / "entity_rows.jsonl", entity_rows)
    write_jsonl(args.out_dir / "clinical_rows.jsonl", clinical_rows)

    lines = [
        "# D22 Paper-Style Patchscopes Calibration",
        "",
        "Validation-only calibration. Clinical outputs are interpretable only after both",
        "the few-shot token-identity and entity-description positive controls pass.",
        "",
        f"- extraction consistency passed: **{extraction_gate}**",
        f"- paper-style identity HS32 passed: **{identity_gate}**",
        f"- entity-description HS32 passed: **{entity_gate}**",
        f"- clinical outputs interpretable: **{clinical_interpretable}**",
        f"- locked test read: **no**",
        "",
        "## Few-Shot Token Identity",
        "",
        "| HS | n | source top-1 unique | precision@1 | no-patch precision@1 | "
        "target logprob lift | KL to no-patch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for layer, item in sorted(identity_summary.items(), key=lambda pair: int(pair[0])):
        lines.append(
            f"| {layer} | {item['n']} | {item['source_top1_unique']} | "
            f"{item['precision_at_1']:.4f} | {item['no_patch_precision_at_1']:.4f} | "
            f"{item['mean_target_logprob_lift']:+.4f} | {item['mean_kl_to_no_patch']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Entity Description",
            "",
            f"- keyword hit: **{entity_keyword_rate:.4f}**",
            f"- differs from no-patch: **{entity_divergence_rate:.4f}**",
            "",
            "| source | keyword hit | differs from no-patch | response |",
            "|---|---:|---:|---|",
        ]
    )
    for row in entity_rows:
        response = clean(row["response"]).replace("|", "\\|")[:180]
        lines.append(
            f"| {row['source']} | {row['keyword_hit']} | "
            f"{row['differs_from_no_patch']} | {response} |"
        )
    lines.extend(
        [
            "",
            "## Clinical Calibration",
            "",
            "These rows are report-only and must not be semantically interpreted when the",
            "positive-control gate is false.",
            "",
            "| condition | n | differs from no-patch | mean KL | mean max logit delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in clinical_rows:
        grouped[str(row["condition"])].append(row)
    for condition, rows in sorted(grouped.items()):
        lines.append(
            f"| {condition} | {len(rows)} | "
            f"{mean([float(row['differs_from_no_patch']) for row in rows]):.4f} | "
            f"{mean([float(row['kl_to_no_patch']) for row in rows]):.6f} | "
            f"{mean([float(row['max_abs_logit_delta']) for row in rows]):.4f} |"
        )
    lines.append("")
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"))
    print(f"[done] {args.out_dir}")


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
        raise ValueError("At least two DDXPlus cases are required")
    run(args)


if __name__ == "__main__":
    main()
