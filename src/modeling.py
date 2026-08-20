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


def free_memory_report() -> str:
    """Free memory per visible device, for explaining a placement decision."""
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device visible"
    parts = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        parts.append(f"cuda:{index} {free / 1e9:.1f}/{total / 1e9:.1f} GB free")
    return " | ".join(parts)


def load_causal_lm(
    model_cfg: dict[str, Any], *, cache_dir: str | None, allow_offload: bool = False
):
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        torch_dtype=torch_dtype(model_cfg.get("dtype", "bfloat16")),
        device_map=model_cfg.get("device_map", "cuda"),
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    placement = describe_placement(model)
    print(f"[model] {model_cfg['model_id']} placement -> {placement}", flush=True)

    # Offloaded weights do not merely run slowly: computing against a meta-device
    # tensor fails deep inside cuBLAS, minutes later, with an error that names
    # neither the offload nor the memory that caused it. Refuse at load time,
    # where the cause is still visible.
    offloaded = [part for part in placement.split(" | ") if part.startswith(("cpu", "meta", "disk"))]
    if offloaded and not allow_offload:
        raise RuntimeError(
            f"{model_cfg['model_id']} was partly offloaded ({', '.join(offloaded)}). "
            f"Free memory: {free_memory_report()}. Free the GPUs, reduce the visible "
            "set, or pass allow_offload=True to accept a very slow run."
        )
    return model


def maybe_load_peft_adapter(model, adapter_id: str | None, *, cache_dir: str | None):
    if not adapter_id:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_id, cache_dir=cache_dir)
