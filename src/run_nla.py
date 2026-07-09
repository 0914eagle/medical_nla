from __future__ import annotations

import argparse
import shutil
from datetime import UTC, datetime
from pathlib import Path

import torch

from .config import ensure_dir, load_config
from .jsonl import append_jsonl, read_jsonl
from .modeling import load_causal_lm, load_tokenizer
from .nla import build_nla_inputs_embeds, cjk_fraction, extract_explanation, load_nla_sidecar


def generation_kwargs(cfg: dict) -> dict:
    gen = dict(cfg["generation"])
    return {k: v for k, v in gen.items() if v is not None}


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
    model.eval()

    embed_layer = model.get_input_embeddings()
    gen_kwargs = generation_kwargs(cfg)
    for row in read_jsonl(args.manifest):
        activation = torch.load(row["activation_path"], map_location="cpu")
        result = build_nla_inputs_embeds(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            activation=activation,
            device=model.device,
        )
        generated = model.generate(
            inputs_embeds=result.inputs_embeds,
            attention_mask=result.attention_mask,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **gen_kwargs,
        )
        raw_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        explanation, parsed_explanation = extract_explanation(raw_text)
        append_jsonl(
            output_path,
            {
                "id": row["id"],
                "prompt": row["prompt"],
                "query": result.prompt_text,
                "nla_output": explanation,
                "raw_nla_output": raw_text,
                "parsed_explanation_tag": parsed_explanation,
                "cjk_fraction": cjk_fraction(raw_text),
                "layer": row["layer"],
                "position": row["position"],
                "activation_path": row["activation_path"],
                "activation_norm": result.activation_norm,
                "scaled_activation_norm": result.scaled_activation_norm,
                "injection_position": result.injection_position,
                "injection_scale": sidecar.injection_scale,
                "injection_token_id": sidecar.injection_token_id,
                "sidecar_path": sidecar.path,
                "gen_config": gen_kwargs,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
