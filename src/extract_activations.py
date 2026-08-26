"""Read hidden states out of the backbone at chosen layers and token positions.

Three things about this file drive its shape.

**One forward pass per prompt, not per row.** A case with four cues produces
four extraction rows that all describe the *same* forward pass; the pilot ran
it four times. Rows are grouped by prompt text and the group is served by a
single pass, which is where most of the wall-clock saving lives (DDXPlus:
18,646 rows over 4,900 prompts).

**One forward pass per prompt, not per layer.** `output_hidden_states=True`
already returns every layer. The layer sweep therefore costs nothing beyond
the writes, and re-running the model once per layer -- as the pilot did -- was
pure waste.

**Hidden-state indices are not layer numbers.** The tuple has 48+1 entries:
index 0 is the embedding output (before any block), indices 1..47 are block
outputs *before* the final norm, and index 48 is the post-final-RMSNorm state.
Index 48 lives on a different scale entirely (norm ~158 against ~213,000 at
index 47) and must not be plotted on the same trajectory as the rest. Index 0
is the lexical-vs-contextual control: whatever a readout recovers there was
available from the token identity alone.

Activations are stored as float32. The forward runs in bfloat16, but bfloat16
has 8 mantissa bits, and everything downstream -- cosine similarities between
near-identical counterfactual pairs, per-layer norm curves -- measures small
differences between large vectors.

Layout, one directory per (layer, selection), each with its own manifest, so a
downstream script is pointed at a directory and needs to know nothing else:

    {activation_dir}/{run_name}/
        config.yaml
        run.json
        layer24/
            last_subtoken/manifest.jsonl, shard_037/{id}.pt
            span_mean/manifest.jsonl,     shard_037/{id}.pt
"""

from __future__ import annotations

import argparse
import json
import shutil
import zlib
from collections import Counter, OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is imported where it is used, so the position and
    import torch  # span logic below stays testable without a GPU stack.

from .config import ensure_dir, load_config
from .jsonl import append_jsonl, read_jsonl


PASSTHROUGH_FIELDS = [
    "variant",
    "cue_text",
    "case_id",
    "cue_pool",
    "cf_variant",
    "cf_role",
    "cf_slot",
    "cf_original_cue",
    "cf_replacement_cue",
    "cf_removed_cue",
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
    "cue_count",
    "cue_count_condition",
    "cue_targets",
    "cue_types",
    "cue_evidence_ids",
    "cue_evidence_entries",
    "notes",
    # Which case and which arm of an experiment this row is. Without them a
    # manifest of several arms is one undifferentiated pile, and the analysis
    # that needs the pairing has nothing to pair on.
    "base_id",
    "hint_variant",
    "hint_diagnosis_name",
    "gold_in_prompt",
    "split",
    "disease_category",
    "canonical_pdd",
    "patient_group",
    "position_label",
    "source_correct",
    "answer_forced",
    "diagnosis_alias_in_reasoning",
    "gold_alias_in_reasoning",
]

# Every fourth block, plus the last one before the final norm. 48 (post-norm)
# is available but is deliberately not in the default set: it is a different
# scale and belongs in its own figure.
DEFAULT_LAYERS = [0, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 47]

SPAN_SELECTION = "span"


def chat_text(tokenizer, prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def encode_chat(tokenizer, prompt: str) -> dict[str, Any]:
    """Tokenize one prompt's chat text, keeping character offsets.

    Always this path, never `apply_chat_template(tokenize=True)`: rows sharing
    a prompt must share one tokenization, or a token index resolved for one row
    would not name the same token in another. The template emits its own BOS,
    so `add_special_tokens=False` avoids a second one.
    """
    text = chat_text(tokenizer, prompt)
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return {
        "text": text,
        "input_ids": list(encoded["input_ids"]),
        "offset_mapping": [tuple(pair) for pair in encoded["offset_mapping"]],
        "n_tokens": len(encoded["input_ids"]),
    }


def encode_messages(tokenizer, messages: list[dict[str, str]]) -> dict[str, Any]:
    """Tokenize a complete user/assistant transcript for teacher forcing."""
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return {
        "text": text,
        "input_ids": list(encoded["input_ids"]),
        "offset_mapping": [tuple(pair) for pair in encoded["offset_mapping"]],
        "n_tokens": len(encoded["input_ids"]),
    }


def encode_row(tokenizer, row: dict[str, Any]) -> dict[str, Any]:
    messages = row.get("chat_messages")
    if messages is not None:
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"Row {row.get('id')} has invalid chat_messages.")
        return encode_messages(tokenizer, messages)
    prompt = row.get("prompt")
    if not prompt:
        raise ValueError(f"Row {row.get('id')} has no prompt.")
    return encode_chat(tokenizer, str(prompt))


def substring_char_span(text: str, needle: str, occurrence: int = 0) -> tuple[int, int]:
    if not needle:
        raise ValueError("target_text must be non-empty.")
    text_l = text.lower()
    needle_l = needle.lower()
    occurrence = int(occurrence)
    if occurrence == -1:
        start = text_l.rfind(needle_l)
        if start < 0:
            raise ValueError(f"target_text {needle!r} not found in chat text.")
    elif occurrence >= 0:
        start = -1
        search_from = 0
        for _ in range(occurrence + 1):
            start = text_l.find(needle_l, search_from)
            if start < 0:
                raise ValueError(f"target_text {needle!r} not found in chat text.")
            search_from = start + len(needle_l)
    else:
        raise ValueError("target_text_occurrence must be -1 or non-negative.")
    return start, start + len(needle)


def token_span_for_char_span(
    offset_mapping: list[tuple[int, int]], start: int, end: int
) -> tuple[int, int]:
    token_positions: list[int] = []
    for idx, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:
            continue
        if tok_start < end and tok_end > start:
            token_positions.append(idx)
    if not token_positions:
        raise ValueError(f"No tokens overlap char span {start}:{end}.")
    return token_positions[0], token_positions[-1] + 1


def resolve_positions(row: dict[str, Any], encoded: dict[str, Any], activation_cfg: dict[str, Any]):
    """Work out which token positions this row names, in the unpadded sequence.

    Returns (span, selections) where span is [start, end) and selections is the
    ordered list of selection names this row wants at every layer.
    """
    mode = row.get("position_mode") or activation_cfg.get("position_mode", "last_token")
    n_tokens = int(encoded["n_tokens"])

    if mode == "last_token":
        return (n_tokens - 1, n_tokens), ["last_token"]

    if mode == "token_index":
        pos = row.get("target_token_position", activation_cfg.get("default_token_index"))
        if pos is None:
            raise ValueError(f"Row {row.get('id')} requires target_token_position.")
        pos = int(pos)
        if not 0 <= pos < n_tokens:
            raise ValueError(f"Row {row.get('id')} token index {pos} outside 0..{n_tokens - 1}.")
        return (pos, pos + 1), ["token_index"]

    if mode == "token_span":
        span = row.get("target_token_span", activation_cfg.get("default_token_span"))
        if not span or len(span) != 2:
            raise ValueError(f"Row {row.get('id')} requires target_token_span [start, end).")
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= n_tokens:
            raise ValueError(f"Row {row.get('id')} span {span} outside 0..{n_tokens}.")
        return (start, end), ["token_span"]

    if mode == "target_text":
        target_text = row.get("target_text")
        if target_text is None:
            raise ValueError(f"Row {row.get('id')} uses target_text mode but has no target_text.")
        occurrence = int(
            row.get(
                "target_text_occurrence",
                activation_cfg.get("target_text_occurrence", 0),
            )
        )
        char_start, char_end = substring_char_span(encoded["text"], str(target_text), occurrence)
        start, end = token_span_for_char_span(encoded["offset_mapping"], char_start, char_end)
        row["target_char_span"] = [char_start, char_end]
        return (start, end), []

    raise ValueError(f"Unsupported position_mode: {mode}")


def select_from_span(seq_hidden: "torch.Tensor", span: tuple[int, int], selection: str):
    """Reduce a token span to the tensor a selection names, plus a position label."""
    start, end = span
    if selection in {"last_token", "token_index"}:
        return seq_hidden[end - 1], str(end - 1)
    if selection == "first_subtoken":
        return seq_hidden[start], str(start)
    if selection == "last_subtoken":
        return seq_hidden[end - 1], str(end - 1)
    if selection == "span_mean":
        return seq_hidden[start:end].mean(dim=0), f"{start}:{end}:mean"
    if selection in {"span", "token_span"}:
        return seq_hidden[start:end], f"{start}:{end}"
    raise ValueError(f"Unsupported selection: {selection}")


def shard_dir_name(row_id: str, n_shards: int) -> str:
    """Stable across runs, so a resumed run rewrites the same paths."""
    return f"shard_{zlib.crc32(row_id.encode('utf-8')) % n_shards:03d}"


class SelectionWriter:
    """One (layer, selection) output directory and its manifest."""

    def __init__(self, root: Path, layer: int, selection: str, *, n_shards: int, resume: bool):
        self.dir = root / f"layer{layer:02d}" / selection
        self.manifest = self.dir / "manifest.jsonl"
        self.layer = layer
        self.selection = selection
        self.n_shards = n_shards
        self.seen: set[str] = set()
        if resume and self.manifest.exists():
            self.seen = {str(row["id"]) for row in read_jsonl(self.manifest)}
        elif self.dir.exists():
            shutil.rmtree(self.dir)
        ensure_dir(self.dir)

    def has(self, row_id: str) -> bool:
        return row_id in self.seen

    def write(self, row_id: str, tensor: "torch.Tensor", manifest_row: dict[str, Any]) -> None:
        import torch

        shard = ensure_dir(self.dir / shard_dir_name(row_id, self.n_shards))
        path = shard / f"{row_id}.pt"
        torch.save(tensor, path)
        manifest_row = dict(manifest_row)
        manifest_row["activation_path"] = str(path)
        manifest_row["layer"] = self.layer
        manifest_row["hidden_state_index"] = self.layer
        manifest_row["selection"] = self.selection
        manifest_row["dtype"] = str(tensor.dtype)
        manifest_row["shape"] = list(tensor.shape)
        append_jsonl(self.manifest, manifest_row)
        self.seen.add(row_id)


def group_by_prompt(rows: list[dict[str, Any]]) -> "OrderedDict[str, list[dict[str, Any]]]":
    groups: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for row in rows:
        messages = row.get("chat_messages")
        if messages is not None:
            key = "messages:" + json.dumps(messages, ensure_ascii=False, sort_keys=True)
        else:
            prompt = row.get("prompt")
            if not prompt:
                raise ValueError(f"Row {row.get('id')} has no prompt.")
            key = str(prompt)
        groups.setdefault(key, []).append(row)
    return groups


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, nargs="+", help="One or more row JSONL files.")
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Exact output directory. Use this for restricted inputs whose manifests "
            "must not be written under the ordinary artifact tree."
        ),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Hidden-state indices, NOT layer numbers: 0 is the embedding output, "
            "1..47 are block outputs before the final norm, 48 is post-final-norm "
            f"and is on a different scale. Default: {DEFAULT_LAYERS}."
        ),
    )
    parser.add_argument(
        "--span-layers",
        nargs="+",
        type=int,
        default=[],
        help=(
            "Layers at which target_text rows additionally keep the whole token "
            "span, not just a reduction of it. Costs one file of "
            "(span_length x d_model) floats per row, so it is asked for by layer."
        ),
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=None,
        choices=["first_subtoken", "last_subtoken", "span_mean"],
        help=(
            "Reductions to store for every target_text row. Defaults to the "
            "strategy each row names, so an existing row file behaves as before."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--shards", type=int, default=256, help="Subdirectories per selection.")
    parser.add_argument("--limit-prompts", type=int, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep existing outputs and write only the rows missing from them.",
    )
    args = parser.parse_args()

    import torch

    from .modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    activation_cfg = cfg.get("activation") or {}
    run_name = args.run_name or cfg.get("run_name", "pilot")
    layers = sorted(set(args.layers if args.layers is not None else DEFAULT_LAYERS))
    span_layers = sorted(set(args.span_layers) & set(layers))
    if set(args.span_layers) - set(layers):
        raise SystemExit(
            f"--span-layers {sorted(set(args.span_layers) - set(layers))} are not in --layers."
        )
    batch_size = int(args.batch_size or activation_cfg.get("batch_size") or 8)

    out_dir = ensure_dir(
        args.output_dir or (Path(cfg["paths"]["activation_dir"]) / run_name)
    )
    shutil.copy2(args.config, out_dir / "config.yaml")

    rows: list[dict[str, Any]] = []
    for path in args.input:
        rows.extend(read_jsonl(path))
    if not rows:
        raise SystemExit(f"no rows in {args.input}")
    # Ids address files, so a repeat would have one row overwrite another's
    # tensor while both appear in the manifest.
    counts = Counter(str(row.get("id")) for row in rows)
    repeated = [row_id for row_id, n in counts.most_common(5) if n > 1]
    if repeated:
        raise SystemExit(
            f"duplicate row ids in input, e.g. {repeated}. To store several "
            "reductions of one span, pass --strategies rather than one row file "
            "per strategy."
        )
    groups = group_by_prompt(rows)
    if args.limit_prompts:
        groups = OrderedDict(list(groups.items())[: args.limit_prompts])
        rows = [row for group in groups.values() for row in group]
    print(
        f"[input] {len(rows):,} rows over {len(groups):,} distinct prompts "
        f"({len(rows) / max(len(groups), 1):.1f} rows per forward pass)",
        flush=True,
    )

    torch.manual_seed(int(cfg.get("seed", 17)))
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    # Right padding, unlike generation: it leaves every real token at the index
    # the unpadded tokenization gave it, so a resolved span needs no shifting.
    # Padding is masked out of attention either way.
    tokenizer.padding_side = "right"

    # Resolve every span before loading the model, so a bad row fails in seconds
    # rather than after a several-minute load.
    encodings: dict[str, dict[str, Any]] = {}
    plans: list[tuple[str, list[dict[str, Any]]]] = []
    for input_key, group in groups.items():
        encoded = encode_row(tokenizer, group[0])
        encodings[input_key] = encoded
        planned = []
        for row in group:
            row = dict(row)
            span, selections = resolve_positions(row, encoded, activation_cfg)
            if not selections:
                selections = list(
                    args.strategies
                    or [
                        row.get("target_text_strategy")
                        or activation_cfg.get("target_text_strategy", "last_subtoken")
                    ]
                )
            row["target_token_span"] = [int(span[0]), int(span[1])]
            planned.append({"row": row, "span": span, "selections": selections})
        plans.append((input_key, planned))

    token_counts = [int(e["n_tokens"]) for e in encodings.values()]
    print(
        f"[tokens] prompt length min {min(token_counts)} / mean "
        f"{sum(token_counts) / len(token_counts):.0f} / max {max(token_counts)}",
        flush=True,
    )

    selection_names = sorted({s for _, planned in plans for p in planned for s in p["selections"]})
    writers: dict[tuple[int, str], SelectionWriter] = {}
    for layer in layers:
        wanted = list(selection_names)
        if layer in span_layers:
            wanted.append(SPAN_SELECTION)
        for selection in wanted:
            writers[(layer, selection)] = SelectionWriter(
                out_dir, layer, selection, n_shards=args.shards, resume=args.resume
            )
    print(
        f"[layers] {layers} x selections {selection_names}"
        + (f" (+ full span at {span_layers})" if span_layers else ""),
        flush=True,
    )

    def outputs_for(planned: dict[str, Any]) -> list[tuple[int, str]]:
        keys = [(layer, sel) for layer in layers for sel in planned["selections"]]
        if planned["row"].get("position_mode") == "target_text":
            keys += [(layer, SPAN_SELECTION) for layer in span_layers]
        return keys

    if args.resume:
        pending = [
            (prompt, planned)
            for prompt, planned in plans
            if any(
                not writers[key].has(str(p["row"]["id"]))
                for p in planned
                for key in outputs_for(p)
            )
        ]
        if len(pending) != len(plans):
            print(
                f"[resume] {len(plans) - len(pending):,} prompts already complete, "
                f"{len(pending):,} to run",
                flush=True,
            )
        plans = pending
    if not plans:
        print("[done] nothing to do; every requested output already exists")
        return

    # Batch prompts of similar length together. Padding is computed per batch,
    # so mixing a 200-token prompt with a 900-token one makes the model attend
    # over 700 tokens of nothing; sorting removes most of that waste and changes
    # no output, since each row is keyed by its own id.
    plans.sort(key=lambda item: encodings[item[0]]["n_tokens"])

    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    n_written = 0

    for batch_start in range(0, len(plans), batch_size):
        batch = plans[batch_start : batch_start + batch_size]
        padded = tokenizer.pad(
            [{"input_ids": encodings[prompt]["input_ids"]} for prompt, _ in batch],
            padding=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(model.device) for k, v in padded.items()}
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden_states = outputs.hidden_states
        if max(layers) >= len(hidden_states):
            raise IndexError(
                f"layer {max(layers)} out of range for {len(hidden_states)} hidden states"
            )

        for layer in layers:
            # One host transfer per (batch, layer); float32 here so nothing
            # downstream sees bfloat16's 8 mantissa bits.
            layer_hidden = hidden_states[layer].to("cpu", torch.float32)
            for index, (input_key, planned_rows) in enumerate(batch):
                seq_hidden = layer_hidden[index]
                for planned in planned_rows:
                    row, span = planned["row"], planned["span"]
                    row_id = str(row["id"])
                    selections = list(planned["selections"])
                    if layer in span_layers and row.get("position_mode") == "target_text":
                        selections.append(SPAN_SELECTION)
                    for selection in selections:
                        writer = writers[(layer, selection)]
                        if writer.has(row_id):
                            continue
                        tensor, position = select_from_span(seq_hidden, span, selection)
                        manifest_row = {
                            "id": row_id,
                            "base_id": row.get("base_id", row_id),
                            "prompt": row.get("prompt"),
                            "chat_text": encodings[input_key]["text"],
                            "model_id": model_cfg["model_id"],
                            "position": position,
                            "position_family": row.get("position_family"),
                            "position_mode": row.get("position_mode")
                            or activation_cfg.get("position_mode"),
                            "target_text": row.get("target_text"),
                            "target_text_strategy": selection,
                            "target_token_span": row.get("target_token_span"),
                            "target_char_span": row.get("target_char_span"),
                            "prompt_token_count": int(encodings[input_key]["n_tokens"]),
                        }
                        for field in PASSTHROUGH_FIELDS:
                            if field in row:
                                manifest_row[field] = row.get(field)
                        writer.write(row_id, tensor.clone(), manifest_row)
                        n_written += 1
            del layer_hidden
        del outputs, hidden_states

        done = min(batch_start + batch_size, len(plans))
        if done % (batch_size * 25) == 0 or done == len(plans):
            print(
                f"[extract] {done:,}/{len(plans):,} prompts | {n_written:,} tensors written",
                flush=True,
            )

    run_record = {
        "run_name": run_name,
        "inputs": list(args.input),
        "model_id": model_cfg["model_id"],
        "layers": layers,
        "span_layers": span_layers,
        "selections": selection_names,
        "n_rows": len(rows),
        "n_prompts": len(groups),
        "batch_size": batch_size,
        "shards": args.shards,
        "store_dtype": "float32",
        "prompt_token_count_max": max(token_counts),
        "tensors_written": n_written,
        "manifests": {
            f"layer{layer:02d}/{selection}": str(writer.manifest)
            for (layer, selection), writer in writers.items()
        },
    }
    (out_dir / "run.json").write_text(json.dumps(run_record, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in run_record.items() if k != "manifests"}, indent=2))
    print(f"[done] {n_written:,} tensors under {out_dir}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
