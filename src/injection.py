from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class InjectionResult:
    inputs_embeds: torch.Tensor
    placeholder_positions: list[int]


def find_token_positions(input_ids: torch.Tensor, token_id: int) -> list[int]:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1-D, got shape {tuple(input_ids.shape)}")
    return (input_ids == token_id).nonzero(as_tuple=False).flatten().tolist()


def normalize_activation(activation: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return activation
    if mode == "l2":
        denom = activation.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return activation / denom
    raise ValueError(f"Unsupported activation normalization mode: {mode}")


def replace_placeholder_embeddings(
    *,
    input_ids: torch.Tensor,
    base_embeds: torch.Tensor,
    placeholder_token_id: int,
    activation: torch.Tensor,
    normalization: str = "none",
) -> InjectionResult:
    """Replace placeholder token embeddings with saved activation vectors.

    Shapes:
      input_ids: (seq_len,) or (1, seq_len)
      base_embeds: (seq_len, embed_dim) or (1, seq_len, embed_dim)
      activation: (embed_dim,) or (span_len, embed_dim)
    """
    squeeze_batch = False
    if input_ids.ndim == 2:
        if input_ids.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for injection.")
        input_ids_1d = input_ids[0]
    else:
        input_ids_1d = input_ids

    if base_embeds.ndim == 3:
        if base_embeds.shape[0] != 1:
            raise ValueError("Only batch size 1 is supported for injection.")
        embeds_2d = base_embeds[0].clone()
        squeeze_batch = True
    elif base_embeds.ndim == 2:
        embeds_2d = base_embeds.clone()
    else:
        raise ValueError(f"base_embeds must be 2-D or 3-D, got shape {tuple(base_embeds.shape)}")

    positions = find_token_positions(input_ids_1d, placeholder_token_id)
    if not positions:
        raise ValueError(f"Placeholder token id {placeholder_token_id} was not found.")

    activation = normalize_activation(activation, normalization)
    if activation.ndim == 1:
        activation_2d = activation.unsqueeze(0)
    elif activation.ndim == 2:
        activation_2d = activation
    else:
        raise ValueError(f"activation must be 1-D or 2-D, got shape {tuple(activation.shape)}")

    if len(positions) != activation_2d.shape[0]:
        raise ValueError(
            "Placeholder count must match activation span length: "
            f"{len(positions)} placeholders vs {activation_2d.shape[0]} activation rows"
        )
    if embeds_2d.shape[-1] != activation_2d.shape[-1]:
        raise ValueError(
            "Embedding dim must match activation dim: "
            f"{embeds_2d.shape[-1]} vs {activation_2d.shape[-1]}"
        )

    activation_2d = activation_2d.to(device=embeds_2d.device, dtype=embeds_2d.dtype)
    embeds_2d[positions, :] = activation_2d
    if squeeze_batch:
        return InjectionResult(inputs_embeds=embeds_2d.unsqueeze(0), placeholder_positions=positions)
    return InjectionResult(inputs_embeds=embeds_2d, placeholder_positions=positions)
