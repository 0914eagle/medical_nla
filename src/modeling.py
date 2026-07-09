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


def load_causal_lm(model_cfg: dict[str, Any], *, cache_dir: str | None):
    return AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "cuda"),
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )


def maybe_load_peft_adapter(model, adapter_id: str | None, *, cache_dir: str | None):
    if not adapter_id:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_id, cache_dir=cache_dir)
