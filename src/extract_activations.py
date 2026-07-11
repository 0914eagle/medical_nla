from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import ensure_dir, load_config
from .jsonl import append_jsonl, read_jsonl
from .modeling import load_causal_lm, load_tokenizer


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def chat_tokens(tokenizer, prompt: str) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
    else:
        input_ids = encoded
    return {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}


def select_activation(
    hidden_state: torch.Tensor,
    attention_mask: torch.Tensor,
    row: dict[str, Any],
    activation_cfg: dict[str, Any],
) -> tuple[torch.Tensor, str]:
    mode = row.get("position_mode") or activation_cfg.get("position_mode", "last_token")
    seq_hidden = hidden_state[0]
    mask = attention_mask[0].bool()

    if mode == "last_token":
        pos = int(mask.nonzero(as_tuple=False)[-1].item())
        return seq_hidden[pos].detach().cpu(), str(pos)

    if mode == "token_index":
        pos = row.get("target_token_position", activation_cfg.get("default_token_index"))
        if pos is None:
            raise ValueError(f"Row {row.get('id')} requires target_token_position.")
        pos = int(pos)
        if not bool(mask[pos]):
            raise ValueError(f"Row {row.get('id')} selected padding position {pos}.")
        return seq_hidden[pos].detach().cpu(), str(pos)

    if mode == "token_span":
        span = row.get("target_token_span", activation_cfg.get("default_token_span"))
        if not span or len(span) != 2:
            raise ValueError(f"Row {row.get('id')} requires target_token_span [start, end).")
        start, end = int(span[0]), int(span[1])
        if start >= end:
            raise ValueError(f"Invalid span for row {row.get('id')}: {span}")
        if not bool(mask[start:end].all()):
            raise ValueError(f"Row {row.get('id')} selected span containing padding: {span}")
        return seq_hidden[start:end].detach().cpu(), f"{start}:{end}"

    raise ValueError(f"Unsupported position_mode: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_name = args.run_name or cfg.get("run_name", "pilot")
    paths = cfg["paths"]
    out_dir = ensure_dir(Path(paths["activation_dir"]) / run_name)
    manifest_path = out_dir / "manifest.jsonl"
    if manifest_path.exists():
        manifest_path.unlink()
    shutil.copy2(args.config, out_dir / "config.yaml")

    torch.manual_seed(int(cfg.get("seed", 17)))
    cache_dir = paths.get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    tokenizer.padding_side = "left"
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()

    layer = int(model_cfg["layer"])
    offset = int(model_cfg.get("hidden_state_index_offset", 0))
    hidden_state_index = layer + offset

    for row in read_jsonl(args.input):
        row_id = str(row["id"])
        prompt = row["prompt"]
        text = chat_text(tokenizer, prompt)
        encoded = chat_tokens(tokenizer, prompt)
        encoded = {k: v.to(model.device) for k, v in encoded.items()}
        with torch.inference_mode():
            outputs = model(**encoded, output_hidden_states=True, use_cache=False)

        hidden_states = outputs.hidden_states
        if hidden_state_index >= len(hidden_states):
            raise IndexError(
                f"hidden_state_index {hidden_state_index} out of range for {len(hidden_states)} states"
            )
        activation, position = select_activation(
            hidden_states[hidden_state_index],
            encoded["attention_mask"],
            row,
            cfg["activation"],
        )
        activation_path = out_dir / f"{row_id}.pt"
        torch.save(activation, activation_path)
        append_jsonl(
            manifest_path,
            {
                "id": row_id,
                "prompt": prompt,
                "chat_text": text,
                "activation_path": str(activation_path),
                "model_id": model_cfg["model_id"],
                "layer": layer,
                "hidden_state_index": hidden_state_index,
                "position": position,
                "dtype": str(activation.dtype),
                "shape": list(activation.shape),
            },
        )

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
