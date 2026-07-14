from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import torch

from .config import ensure_dir, load_config
from .jsonl import append_jsonl, read_jsonl
from .modeling import load_causal_lm, load_tokenizer


PASSTHROUGH_FIELDS = [
    "variant",
    "source_id",
    "primary_target",
    "distractor_target",
    "correct_dx",
    "distractor_dx",
    "distractor_position",
    "distractor_strength",
    "condition",
    "condition_order",
    "insertion_type",
    "target_role",
    "cue_index",
    "category",
    "nonspecific_target",
    "specific_target",
    "specific_targets",
    "nonspecific_expected",
    "specific_expected",
    "diagnostic_shift",
    "specific_aliases",
    "nonspecific_aliases",
    "diagnosis_aliases",
    "source",
    "patient_id",
    "diagnosis_id",
    "diagnosis_name",
    "cue_targets",
    "cue_types",
    "cue_evidence_ids",
    "cue_evidence_entries",
    "notes",
]


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_chat_encoded(encoded: Any) -> dict[str, Any]:
    if hasattr(encoded, "input_ids"):
        input_ids = encoded.input_ids
        attention_mask = getattr(encoded, "attention_mask", None)
        offset_mapping = getattr(encoded, "offset_mapping", None)
    elif isinstance(encoded, dict):
        input_ids = encoded["input_ids"]
        attention_mask = encoded.get("attention_mask")
        offset_mapping = encoded.get("offset_mapping")
    else:
        input_ids = encoded
        attention_mask = None
        offset_mapping = None
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "offset_mapping": offset_mapping,
    }


def chat_tokens(tokenizer, prompt: str, *, return_offsets: bool = False) -> dict[str, Any]:
    messages = [{"role": "user", "content": prompt}]
    if return_offsets:
        text = chat_text(tokenizer, prompt)
        encoded = tokenizer(
            text,
            return_tensors="pt",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
    else:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    return normalize_chat_encoded(encoded)


def substring_char_span(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    if not needle:
        raise ValueError("target_text must be non-empty.")
    text_l = text.lower()
    needle_l = needle.lower()
    start = -1
    search_from = 0
    for _ in range(int(occurrence) + 1):
        start = text_l.find(needle_l, search_from)
        if start < 0:
            raise ValueError(f"target_text {needle!r} not found in chat text.")
        search_from = start + len(needle_l)
    return start, start + len(needle)


def token_span_for_char_span(offset_mapping: torch.Tensor, start: int, end: int) -> tuple[int, int]:
    offsets = offset_mapping[0].detach().cpu().tolist()
    token_positions: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(offsets):
        if tok_start == tok_end:
            continue
        if tok_start < end and tok_end > start:
            token_positions.append(idx)
    if not token_positions:
        raise ValueError(f"No tokens overlap char span {start}:{end}.")
    return token_positions[0], token_positions[-1] + 1


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

    if mode == "target_text":
        span = row.get("target_token_span")
        if not span or len(span) != 2:
            raise ValueError(f"Row {row.get('id')} target_text was not resolved to target_token_span.")
        start, end = int(span[0]), int(span[1])
        strategy = row.get("target_text_strategy") or activation_cfg.get(
            "target_text_strategy", "last_subtoken"
        )
        if not bool(mask[start:end].all()):
            raise ValueError(f"Row {row.get('id')} selected span containing padding: {span}")
        if strategy == "first_subtoken":
            pos = start
            return seq_hidden[pos].detach().cpu(), str(pos)
        if strategy == "last_subtoken":
            pos = end - 1
            return seq_hidden[pos].detach().cpu(), str(pos)
        if strategy == "span":
            return seq_hidden[start:end].detach().cpu(), f"{start}:{end}"
        if strategy == "span_mean":
            return seq_hidden[start:end].mean(dim=0).detach().cpu(), f"{start}:{end}:mean"
        raise ValueError(f"Unsupported target_text_strategy: {strategy}")

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
        needs_offsets = (row.get("position_mode") or cfg["activation"].get("position_mode")) == "target_text"
        encoded = chat_tokens(tokenizer, prompt, return_offsets=needs_offsets)
        offset_mapping = encoded.pop("offset_mapping")
        if needs_offsets:
            target_text = row.get("target_text")
            if target_text is None:
                raise ValueError(f"Row {row_id} uses target_text mode but has no target_text.")
            occurrence = row.get(
                "target_text_occurrence",
                cfg["activation"].get("target_text_occurrence", 0),
            )
            char_start, char_end = substring_char_span(text, str(target_text), int(occurrence))
            tok_start, tok_end = token_span_for_char_span(offset_mapping, char_start, char_end)
            row = dict(row)
            row["target_token_span"] = [tok_start, tok_end]
            row["target_char_span"] = [char_start, char_end]
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
        manifest_row = {
                "id": row_id,
                "base_id": row.get("base_id", row_id),
                "prompt": prompt,
                "chat_text": text,
                "activation_path": str(activation_path),
                "model_id": model_cfg["model_id"],
                "layer": layer,
                "hidden_state_index": hidden_state_index,
                "position": position,
                "position_family": row.get("position_family"),
                "position_mode": row.get("position_mode") or cfg["activation"].get("position_mode"),
                "target_text": row.get("target_text"),
                "target_text_strategy": row.get("target_text_strategy")
                or cfg["activation"].get("target_text_strategy"),
                "target_token_span": row.get("target_token_span"),
                "target_char_span": row.get("target_char_span"),
                "dtype": str(activation.dtype),
                "shape": list(activation.shape),
        }
        for field in PASSTHROUGH_FIELDS:
            if field in row:
                manifest_row[field] = row.get(field)
        append_jsonl(manifest_path, manifest_row)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
