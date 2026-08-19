from __future__ import annotations

from typing import Any

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import torch_dtype


def load_tokenizer(model_id: str, *, cache_dir: str | None, trust_remote_code: bool):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def describe_placement(model) -> str:
    """Summarize which devices hold the weights.

    `device_map="auto"` silently falls back to CPU when no GPU is visible,
    which looks like a successful load but runs orders of magnitude slower,
    so the placement is reported rather than assumed.
    """
    counts: dict[str, int] = {}
    for param in model.parameters():
        key = str(param.device)
        counts[key] = counts.get(key, 0) + param.numel()
    total = sum(counts.values()) or 1
    parts = [
        f"{device}: {numel / 1e9:.1f}B params ({100 * numel / total:.0f}%)"
        for device, numel in sorted(counts.items())
    ]
    return " | ".join(parts)


def load_causal_lm(model_cfg: dict[str, Any], *, cache_dir: str | None):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "cuda"),
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    placement = describe_placement(model)
    print(f"[model] {model_cfg['model_id']} placement -> {placement}", flush=True)
    if "cpu" in placement:
        print(
            "[model] WARNING: part of the model is on CPU. Check "
            "CUDA_VISIBLE_DEVICES and free GPU memory; inference will be very slow.",
            flush=True,
        )
    return model


def maybe_load_peft_adapter(model, adapter_id: str | None, *, cache_dir: str | None):
    if not adapter_id:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_id, cache_dir=cache_dir)
