from __future__ import annotations

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import ensure_dir, load_config
from .injection import replace_placeholder_embeddings
from .jsonl import append_jsonl, read_jsonl
from .modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter


def generation_kwargs(cfg: dict) -> dict:
    gen = dict(cfg["generation"])
    return {k: v for k, v in gen.items() if v is not None}


def build_query(template: str, prompt: str, activation_text: str) -> str:
    if "{activation}" in template:
        return template.format(prompt=prompt, activation=activation_text)
    return template.format(prompt=prompt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(args.config, output_path.parent / f"{output_path.stem}.config.yaml")

    torch.manual_seed(int(cfg.get("seed", 17)))
    cache_dir = paths.get("cache_dir")
    nla_cfg = cfg["nla_model"]
    if nla_cfg["model_id"].startswith("TODO"):
        raise ValueError("Set nla_model.model_id in configs/default.yaml before running NLA inference.")

    tokenizer = load_tokenizer(
        nla_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=nla_cfg.get("trust_remote_code", True),
    )
    placeholder = nla_cfg["placeholder_token"]
    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    if placeholder_id == tokenizer.unk_token_id:
        added = tokenizer.add_special_tokens({"additional_special_tokens": [placeholder]})
    else:
        added = 0

    model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    if added:
        model.resize_token_embeddings(len(tokenizer))
    model = maybe_load_peft_adapter(model, nla_cfg.get("adapter_id"), cache_dir=cache_dir)
    model.eval()

    placeholder_id = tokenizer.convert_tokens_to_ids(placeholder)
    if placeholder_id is None or placeholder_id < 0:
        raise ValueError(f"Could not resolve placeholder token id for {placeholder!r}.")

    embed_layer = model.get_input_embeddings()
    gen_kwargs = generation_kwargs(cfg)
    for row in read_jsonl(args.manifest):
        activation = torch.load(row["activation_path"], map_location="cpu")
        span_len = 1 if activation.ndim == 1 else int(activation.shape[0])
        activation_text = " ".join([placeholder] * span_len)
        query = build_query(nla_cfg["query_template"], row["prompt"], activation_text)
        prompt_text = query if "{activation}" in nla_cfg["query_template"] else f"{activation_text}\n{query}"
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)
        input_ids = encoded["input_ids"].to(model.device)
        attention_mask = encoded["attention_mask"].to(model.device)
        base_embeds = embed_layer(input_ids)
        result = replace_placeholder_embeddings(
            input_ids=input_ids,
            base_embeds=base_embeds,
            placeholder_token_id=placeholder_id,
            activation=activation,
            normalization=nla_cfg.get("normalization", "none"),
        )
        generated = model.generate(
            inputs_embeds=result.inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
        text = tokenizer.decode(generated[0], skip_special_tokens=True)
        append_jsonl(
            output_path,
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "query": query,
                "nla_output": text,
                "layer": row["layer"],
                "position": row["position"],
                "activation_path": row["activation_path"],
                "placeholder_positions": result.placeholder_positions,
                "gen_config": gen_kwargs,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
