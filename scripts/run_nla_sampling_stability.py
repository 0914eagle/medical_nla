"""Sample Medical-NLA readouts multiple times per activation.

This probes whether readout instability alone can predict source-model errors.
Unlike source/NLA disagreement, the features derived from this output do not
require looking at the source model's generated answer.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config
from src.jsonl import append_jsonl, read_jsonl
from src.modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter
from src.nla import build_nla_inputs_embeds, cjk_fraction, extract_explanation, load_nla_sidecar


def stable_seed(base_seed: int, row_id: str, sample_index: int) -> int:
    digest = hashlib.sha256(f"{base_seed}:{row_id}:{sample_index}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def generation_kwargs(
    cfg: dict[str, Any],
    *,
    max_new_tokens: int | None,
    temperature: float,
    top_p: float,
) -> dict[str, Any]:
    gen = dict(cfg["generation"])
    gen["do_sample"] = True
    gen["temperature"] = temperature
    gen["top_p"] = top_p
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return {key: value for key, value in gen.items() if value is not None}


def read_actor_prompt_template(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--adapter-id", required=True)
    parser.add_argument("--actor-prompt-template-file", default=None)
    parser.add_argument("--samples-per-row", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    if args.samples_per_row < 1:
        raise ValueError("--samples-per-row must be >= 1")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    cfg = load_config(args.config)
    paths = cfg["paths"]
    output_path = Path(args.output_jsonl)
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(args.config, output_path.parent / f"{output_path.stem}.config.yaml")

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
    actor_prompt_template = read_actor_prompt_template(args.actor_prompt_template_file)
    model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    model = maybe_load_peft_adapter(model, args.adapter_id, cache_dir=cache_dir)
    model.eval()
    embed_layer = model.get_input_embeddings()
    gen_kwargs = generation_kwargs(
        cfg,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    rows = [
        row
        for index, row in enumerate(read_jsonl(args.manifest))
        if index % args.num_shards == args.shard_index
    ]
    if args.limit is not None:
        rows = rows[: args.limit]
    for row_index, row in enumerate(rows, start=1):
        activation = torch.load(row["activation_path"], map_location="cpu", weights_only=True)
        injected = build_nla_inputs_embeds(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            activation=activation,
            device=model.device,
            actor_prompt_template=actor_prompt_template,
        )
        for sample_index in range(args.samples_per_row):
            seed = stable_seed(args.seed, str(row["id"]), sample_index)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
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
            out = {
                **row,
                "sample_id": f"{row['id']}__sample_{sample_index:02d}",
                "sample_index": sample_index,
                "samples_per_row": args.samples_per_row,
                "sample_seed": seed,
                "query": injected.prompt_text,
                "actor_prompt_template_file": args.actor_prompt_template_file,
                "adapter_id": args.adapter_id,
                "nla_output": explanation,
                "raw_nla_output": raw_text,
                "parsed_explanation_tag": parsed_explanation,
                "cjk_fraction": cjk_fraction(raw_text),
                "activation_norm": injected.activation_norm,
                "scaled_activation_norm": injected.scaled_activation_norm,
                "injection_position": injected.injection_position,
                "injection_scale": sidecar.injection_scale,
                "injection_token_id": sidecar.injection_token_id,
                "sidecar_path": sidecar.path,
                "gen_config": gen_kwargs,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            append_jsonl(output_path, out)
            print(
                f"[sample] {row_index}/{len(rows)} {row['id']} "
                f"sample={sample_index + 1}/{args.samples_per_row}",
                flush=True,
            )

    del model
    torch.cuda.empty_cache()
    print(f"[done] wrote samples to {output_path}", flush=True)


if __name__ == "__main__":
    main()
