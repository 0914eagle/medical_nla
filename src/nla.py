from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml
from huggingface_hub import hf_hub_download


EXPLANATION_RE = re.compile(r"<explanation>\s*(.*?)\s*</explanation>", re.DOTALL)


@dataclass(frozen=True)
class NlaSidecar:
    d_model: int
    injection_char: str
    injection_token_id: int
    injection_left_neighbor_id: int
    injection_right_neighbor_id: int
    actor_prompt_template: str
    injection_scale: float
    path: str


@dataclass(frozen=True)
class NlaInjectionResult:
    inputs_embeds: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    injection_position: int
    activation_norm: float
    scaled_activation_norm: float
    prompt_text: str


def load_nla_sidecar(
    model_id: str,
    *,
    tokenizer: Any,
    cache_dir: str | None,
    filename: str = "nla_meta.yaml",
    expected_d_model: int | None = None,
    expected_injection_token_id: int | None = None,
) -> NlaSidecar:
    path = hf_hub_download(model_id, filename=filename, cache_dir=cache_dir)
    meta = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if meta["kind"] not in ("nla_model", "nla_dataset"):
        raise ValueError(f"Unsupported NLA sidecar kind: {meta['kind']!r}")

    d_model = meta["d_model"] if meta["kind"] == "nla_model" else meta["extraction"]["d_model"]
    injection_scale = meta.get("extraction", {}).get("injection_scale")
    if injection_scale is None:
        raise ValueError(f"{path} does not contain extraction.injection_scale.")

    tokens = meta["tokens"]
    prompt_templates = meta["prompt_templates"]
    actor_prompt_template = prompt_templates.get("av") or prompt_templates.get("actor")
    if actor_prompt_template is None:
        raise ValueError(f"{path} has no AV prompt template under prompt_templates.av/actor.")

    sidecar = NlaSidecar(
        d_model=int(d_model),
        injection_char=tokens["injection_char"],
        injection_token_id=int(tokens["injection_token_id"]),
        injection_left_neighbor_id=int(tokens["injection_left_neighbor_id"]),
        injection_right_neighbor_id=int(tokens["injection_right_neighbor_id"]),
        actor_prompt_template=actor_prompt_template,
        injection_scale=float(injection_scale),
        path=path,
    )

    if expected_d_model is not None and sidecar.d_model != int(expected_d_model):
        raise ValueError(f"Sidecar d_model={sidecar.d_model}, expected {expected_d_model}.")
    if (
        expected_injection_token_id is not None
        and sidecar.injection_token_id != int(expected_injection_token_id)
    ):
        raise ValueError(
            "Sidecar injection_token_id="
            f"{sidecar.injection_token_id}, expected {expected_injection_token_id}."
        )

    live_ids = tokenizer.encode(sidecar.injection_char, add_special_tokens=False)
    if live_ids != [sidecar.injection_token_id]:
        raise ValueError(
            "Tokenizer drift for injection char "
            f"{sidecar.injection_char!r}: live={live_ids}, sidecar={[sidecar.injection_token_id]}"
        )

    prompt_text, input_ids = build_nla_prompt(tokenizer, sidecar)
    positions = find_verified_injection_positions(input_ids, sidecar)
    if len(positions) != 1:
        raise ValueError(
            f"Canonical NLA prompt should contain exactly one verified injection position; got {positions}."
        )
    return sidecar


def build_nla_prompt(tokenizer: Any, sidecar: NlaSidecar) -> tuple[str, list[int]]:
    content = sidecar.actor_prompt_template.format(injection_char=sidecar.injection_char)
    input_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=True,
        add_generation_prompt=True,
    )
    return content, input_ids


def find_verified_injection_positions(input_ids: list[int] | torch.Tensor, sidecar: NlaSidecar) -> list[int]:
    if isinstance(input_ids, torch.Tensor):
        ids = input_ids.detach().cpu().flatten().tolist()
    else:
        ids = input_ids
    positions: list[int] = []
    for idx, token_id in enumerate(ids):
        if token_id != sidecar.injection_token_id:
            continue
        if idx == 0 or idx == len(ids) - 1:
            continue
        if ids[idx - 1] != sidecar.injection_left_neighbor_id:
            continue
        if ids[idx + 1] != sidecar.injection_right_neighbor_id:
            continue
        positions.append(idx)
    return positions


def scale_activation_for_nla(activation: torch.Tensor, injection_scale: float) -> torch.Tensor:
    if activation.ndim != 1:
        raise ValueError(
            "Released NLA AV checkpoints expect one activation vector, "
            f"got shape {tuple(activation.shape)}."
        )
    norm = activation.float().norm().clamp_min(1e-12)
    return activation.float() * (float(injection_scale) / norm)


def build_nla_inputs_embeds(
    *,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: NlaSidecar,
    activation: torch.Tensor,
    device: torch.device | str,
) -> NlaInjectionResult:
    prompt_text, input_ids_list = build_nla_prompt(tokenizer, sidecar)
    positions = find_verified_injection_positions(input_ids_list, sidecar)
    if len(positions) != 1:
        raise ValueError(f"Expected one verified injection position, got {positions}.")

    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    with torch.no_grad():
        inputs_embeds = embed_layer(input_ids).clone()

    if inputs_embeds.shape[-1] != sidecar.d_model:
        raise ValueError(f"Embedding dim={inputs_embeds.shape[-1]}, sidecar d_model={sidecar.d_model}.")
    if activation.numel() != sidecar.d_model:
        raise ValueError(f"Activation dim={activation.numel()}, sidecar d_model={sidecar.d_model}.")

    scaled = scale_activation_for_nla(activation, sidecar.injection_scale).to(
        device=inputs_embeds.device,
        dtype=inputs_embeds.dtype,
    )
    pos = positions[0]
    inputs_embeds[0, pos] = scaled
    return NlaInjectionResult(
        inputs_embeds=inputs_embeds,
        input_ids=input_ids,
        attention_mask=attention_mask,
        injection_position=pos,
        activation_norm=float(activation.float().norm().item()),
        scaled_activation_norm=float(scaled.float().norm().item()),
        prompt_text=prompt_text,
    )


def extract_explanation(text: str) -> tuple[str, bool]:
    match = EXPLANATION_RE.search(text)
    if match is None:
        return text.strip(), False
    return match.group(1).strip(), True


def cjk_fraction(text: str) -> float:
    if not text:
        return 0.0
    cjk = 0
    for ch in text:
        code = ord(ch)
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
    return cjk / len(text)
