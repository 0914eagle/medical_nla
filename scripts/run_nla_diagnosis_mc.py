"""Run a constrained multiple-choice diagnosis prompt through injected NLA AV.

This mirrors scripts/run_source_model_mc.py, but the question is asked to the
Activation Verbalizer after injecting each saved activation vector.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_source_model_mc import parse_selection
from scripts.score_source_diagnosis_logprobs import read_candidates
from src.config import ensure_dir, load_config
from src.jsonl import append_jsonl, read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter
from src.nla import build_nla_inputs_embeds, cjk_fraction, extract_explanation, load_nla_sidecar


PASSTHROUGH_FIELDS = [
    "base_id",
    "prompt",
    "variant",
    "source",
    "patient_id",
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "target_role",
    "target_text",
    "position",
    "position_family",
    "position_mode",
    "layer",
    "cue_targets",
    "cue_types",
    "cue_evidence_ids",
    "notes",
]


def row_seed(base_seed: int, row_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def ordered_candidates(
    candidates: list[dict[str, str]],
    *,
    shuffle: bool,
    seed: int,
    row_id: str,
) -> list[dict[str, str]]:
    ordered = list(candidates)
    if shuffle:
        random.Random(row_seed(seed, row_id)).shuffle(ordered)
    return ordered


def build_actor_prompt_template(
    candidates: list[dict[str, str]],
    *,
    answer_style: str,
) -> str:
    options = "\n".join(
        f"{idx}. {candidate['diagnosis_name']}" for idx, candidate in enumerate(candidates, start=1)
    )
    if answer_style == "number":
        answer_instruction = "Return only the option number inside <explanation> tags."
    elif answer_style == "number_and_name":
        answer_instruction = (
            "Return only the option number and diagnosis name inside <explanation> tags."
        )
    else:
        raise ValueError(f"Unsupported answer_style: {answer_style}")

    return (
        "You are a meticulous AI researcher interpreting an activation vector from a "
        "medical diagnosis model.\n\n"
        "We will pass the vector enclosed in <concept> tags into your context. "
        "Your task is to choose the single diagnosis label best represented by "
        "that vector from the numbered list below.\n\n"
        "Here is the vector:\n\n"
        "<concept>{injection_char}</concept>\n\n"
        "Diagnosis options:\n"
        f"{options}\n\n"
        f"{answer_instruction}"
    )


def generation_kwargs(cfg: dict[str, Any], max_new_tokens: int | None) -> dict[str, Any]:
    gen = dict(cfg["generation"])
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return {key: value for key, value in gen.items() if value is not None}


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    parsed = [row for row in rows if row.get("selected_diagnosis_id")]
    hits = [bool(row["diagnosis_hit"]) for row in rows]
    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold[str(row["gold_diagnosis_id"])].append(row)

    with path.open("w", encoding="utf-8") as f:
        f.write("# NLA Multiple-Choice Diagnosis Baseline\n\n")
        f.write("The injected AV sees the closed diagnosis-label set and generates one option.\n\n")
        f.write(f"- n: {len(rows)}\n")
        if rows:
            f.write(f"- shuffled_options: `{rows[0].get('shuffled_options')}`\n")
            f.write(f"- answer_style: `{rows[0].get('answer_style')}`\n")
        f.write(f"- parsed: {len(parsed)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit: {sum(hits)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit_rate: {mean(hits):.4f}\n\n")

        f.write("## By Gold Diagnosis\n\n")
        f.write("| diagnosis_id | n | hit_rate | hits |\n")
        f.write("|---|---:|---:|---:|\n")
        rates = []
        for diagnosis_id, items in by_gold.items():
            count = sum(bool(row["diagnosis_hit"]) for row in items)
            rates.append((count / len(items), diagnosis_id, len(items), count))
        for rate, diagnosis_id, n_items, count in sorted(rates):
            f.write(f"| {diagnosis_id} | {n_items} | {rate:.4f} | {count} |\n")

        f.write("\n## Top Selected Diagnoses\n\n")
        f.write("| diagnosis_id | count |\n")
        f.write("|---|---:|\n")
        selected_counts = Counter(
            str(row["selected_diagnosis_id"]) for row in rows if row.get("selected_diagnosis_id")
        )
        for diagnosis_id, count in selected_counts.most_common(25):
            f.write(f"| {diagnosis_id} | {count} |\n")

        f.write("\n## Examples\n\n")
        f.write("| id | expected | selected | hit | answer |\n")
        f.write("|---|---|---|---:|---|\n")
        for row in rows[:200]:
            answer = str(row["nla_mc_output"]).replace("\n", " ")
            if len(answer) > 180:
                answer = answer[:177] + "..."
            f.write(
                f"| {row['id']} | {row['gold_diagnosis_name']} | "
                f"{row.get('selected_diagnosis_name') or '-'} | "
                f"{'Y' if row['diagnosis_hit'] else 'N'} | {answer} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--candidates-jsonl", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--answer-style", choices=["number", "number_and_name"], default="number_and_name")
    parser.add_argument(
        "--adapter-id",
        default=None,
        help="Optional PEFT/LoRA adapter path or HF id for evaluating Medical-NLA.",
    )
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    cfg = load_config(args.config)
    paths = cfg["paths"]
    all_rows = [
        row for row in read_jsonl(args.manifest) if row.get("diagnosis_id") and row.get("activation_path")
    ]
    if args.limit is not None:
        all_rows = all_rows[: args.limit]
    rows = [row for idx, row in enumerate(all_rows) if idx % args.num_shards == args.shard_index]
    if not rows:
        raise ValueError("No rows selected for this shard.")

    candidates = read_candidates(args.candidates_jsonl, all_rows)
    candidate_by_id = {row["diagnosis_id"]: row for row in candidates}
    missing = sorted({str(row["diagnosis_id"]) for row in rows} - set(candidate_by_id))
    if missing:
        raise ValueError(f"Gold diagnoses missing from candidate set: {missing[:10]}")

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(args.config, output_path.parent / f"{output_path.stem}.config.yaml")

    cache_dir = paths.get("cache_dir")
    nla_cfg = cfg["nla_model"]
    torch.manual_seed(int(cfg.get("seed", 17)))

    tokenizer = load_tokenizer(
        nla_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=nla_cfg.get("trust_remote_code", True),
    )
    sidecar = load_nla_sidecar(
        nla_cfg["model_id"],
        tokenizer=tokenizer,
        cache_dir=cache_dir,
        filename=nla_cfg.get("sidecar_filename", "nla_meta.yaml"),
        expected_d_model=nla_cfg.get("expected_d_model"),
        expected_injection_token_id=nla_cfg.get("expected_injection_token_id"),
    )
    model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    adapter_id = args.adapter_id or nla_cfg.get("adapter_id")
    model = maybe_load_peft_adapter(model, adapter_id, cache_dir=cache_dir)
    model.eval()
    embed_layer = model.get_input_embeddings()
    gen_kwargs = generation_kwargs(cfg, args.max_new_tokens)

    out_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        ordered = ordered_candidates(
            candidates,
            shuffle=args.shuffle_options,
            seed=args.seed,
            row_id=str(row.get("id") or row.get("base_id")),
        )
        actor_prompt_template = build_actor_prompt_template(
            ordered,
            answer_style=args.answer_style,
        )
        activation = torch.load(row["activation_path"], map_location="cpu", weights_only=True)
        injected = build_nla_inputs_embeds(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            activation=activation,
            device=model.device,
            actor_prompt_template=actor_prompt_template,
        )
        with torch.inference_mode():
            generated = model.generate(
                inputs_embeds=injected.inputs_embeds,
                attention_mask=injected.attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        raw_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        explanation, parsed_explanation = extract_explanation(raw_text)
        selection = parse_selection(explanation, ordered)
        gold_id = str(row["diagnosis_id"])
        hit = selection["selected_diagnosis_id"] == gold_id
        gold_candidate = candidate_by_id[gold_id]
        gold_index = next(
            index
            for index, candidate in enumerate(ordered, start=1)
            if candidate["diagnosis_id"] == gold_id
        )
        result_row = {
            "id": row["id"],
            "base_id": row.get("base_id", row["id"]),
            "variant": row.get("variant"),
            "gold_diagnosis_id": gold_id,
            "gold_diagnosis_name": gold_candidate["diagnosis_name"],
            "gold_option_index": gold_index,
            "selected_index": selection["selected_index"],
            "selected_diagnosis_id": selection["selected_diagnosis_id"],
            "selected_diagnosis_name": selection["selected_diagnosis_name"],
            "parse_method": selection["parse_method"],
            "diagnosis_hit": hit,
            "nla_mc_output": explanation,
            "raw_nla_output": raw_text,
            "parsed_explanation_tag": parsed_explanation,
            "cjk_fraction": cjk_fraction(raw_text),
            "query": injected.prompt_text,
            "num_candidates": len(candidates),
            "shuffled_options": args.shuffle_options,
            "answer_style": args.answer_style,
            "seed": args.seed,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "activation_path": row["activation_path"],
            "activation_norm": injected.activation_norm,
            "scaled_activation_norm": injected.scaled_activation_norm,
            "injection_position": injected.injection_position,
            "injection_scale": sidecar.injection_scale,
            "injection_token_id": sidecar.injection_token_id,
            "sidecar_path": sidecar.path,
            "adapter_id": adapter_id,
            "gen_config": gen_kwargs,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        for field in PASSTHROUGH_FIELDS:
            if field in row:
                result_row[field] = row.get(field)
        append_jsonl(output_path, result_row)
        out_rows.append(result_row)
        print(
            f"[nla-mc] {idx}/{len(rows)} {row['id']} "
            f"gold={gold_id} selected={selection['selected_diagnosis_id']} hit={hit}",
            flush=True,
        )

    write_summary(summary_path, out_rows)
    print(f"[done] wrote {len(out_rows)} rows to {output_path}", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
