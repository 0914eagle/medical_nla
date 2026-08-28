"""Continue a Medical-NLA adapter with symmetric activation-target contrast.

Each pair contributes two matched and two crossed sequences:

    matched = (h_i, y_i), (h_j, y_j)
    crossed = (h_i, y_j), (h_j, y_i)

The ordinary SFT loss is retained on the matched sequences. A softplus ranking
term increases content-NLL(crossed) - content-NLL(matched). Pairing stays
within source dataset and disease stratum so the model cannot win by detecting
dataset format or broad diagnosis alone.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_medical_nla_lora import build_training_example, collate_examples
from src.config import load_config
from src.jsonl import read_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.nla import AV_PROMPT_FILENAME, adapter_av_prompt, load_nla_sidecar


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def source(row: dict[str, Any]) -> str:
    return clean(row.get("source_dataset")) or "<missing>"


def stratum(row: dict[str, Any]) -> str:
    fields = (
        ("diagnosis_id", "disease_category", "canonical_pdd")
        if source(row) == "ddxplus"
        else ("disease_category", "canonical_pdd", "diagnosis_id")
    )
    for field in fields:
        value = clean(row.get(field)).casefold()
        if value:
            return value
    return "<missing>"


def row_id(row: dict[str, Any]) -> str:
    return clean(row.get("base_id") or row.get("id"))


def build_disjoint_pairs(
    rows: list[dict[str, Any]], *, seed: int, epoch: int
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Build source/stratum-matched pairs; each row appears at most once."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("activation_path") and row.get("target_text"):
            grouped[(source(row), stratum(row))].append(row)

    output: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for (source_name, group_name), group in sorted(grouped.items()):
        ordered = sorted(group, key=row_id)
        random.Random(f"{seed}:{epoch}:{source_name}:{group_name}").shuffle(ordered)
        for index in range(0, len(ordered) - 1, 2):
            first, second = ordered[index], ordered[index + 1]
            if clean(first.get("target_text")) == clean(second.get("target_text")):
                continue
            output[source_name].append((first, second))
    return dict(output)


def balanced_pairs(
    rows: list[dict[str, Any]], *, seed: int, epoch: int, max_pairs_per_source: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    grouped = build_disjoint_pairs(rows, seed=seed, epoch=epoch)
    if len(grouped) < 2:
        raise ValueError("Contrastive training requires at least two source datasets")
    cap = min(min(len(items) for items in grouped.values()), max_pairs_per_source)
    if cap <= 0:
        raise ValueError("No eligible contrastive pairs")
    selected = []
    for source_name, items in sorted(grouped.items()):
        items = list(items)
        random.Random(f"{seed}:{epoch}:{source_name}:cap").shuffle(items)
        selected.extend(items[:cap])
    random.Random(f"{seed}:{epoch}:mixed-pairs").shuffle(selected)
    return selected


def crossed_rows(
    first: dict[str, Any], second: dict[str, Any]
) -> list[dict[str, Any]]:
    first_cross = dict(first)
    first_cross["target_text"] = second["target_text"]
    second_cross = dict(second)
    second_cross["target_text"] = first["target_text"]
    return [first, second, first_cross, second_cross]


def symmetric_pair_objective(
    content_nll: torch.Tensor, *, n_pairs: int, temperature: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return softplus ranking loss and per-pair cross-minus-matched gaps."""
    if content_nll.ndim != 1 or content_nll.numel() != 4 * n_pairs:
        raise ValueError(
            f"Expected {4 * n_pairs} scalar NLLs for {n_pairs} pairs; "
            f"got shape {tuple(content_nll.shape)}"
        )
    blocks = content_nll.reshape(n_pairs, 4)
    gaps = 0.5 * (blocks[:, 2] + blocks[:, 3] - blocks[:, 0] - blocks[:, 1])
    loss = (F.softplus(-gaps / temperature) * temperature).mean()
    return loss, gaps


def low_memory_token_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy without materializing a full float32 logits copy.

    The 12B model emits bfloat16 logits. Calling ``logits.float()`` on the
    four matched/crossed sequences adds roughly 1.1 GiB at the point of peak
    training memory on a 24 GiB card. Log-sum-exp and target gathering are
    equivalent to cross-entropy while keeping the vocabulary tensor in its
    model dtype. The two reduced tensors are cast before subtraction.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError(
            f"Logit/label shape mismatch: {tuple(logits.shape)} vs {tuple(labels.shape)}"
        )
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    target_logits = logits.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    log_normalizer = torch.logsumexp(logits, dim=-1)
    losses = log_normalizer.float() - target_logits.float()
    return losses.masked_fill(~valid, 0.0)


def losses_for_pair_batch(
    *,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows = [row for first, second in pairs for row in crossed_rows(first, second)]
    examples = [
        build_training_example(
            row=row,
            tokenizer=tokenizer,
            model=model,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
            eos_token_id=tokenizer.eos_token_id,
        )
        for row in rows
    ]
    inputs_embeds, attention_mask, labels, content = collate_examples(examples)
    logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
    per_token = low_memory_token_nll(logits[:, :-1, :], labels[:, 1:])
    supervised = labels[:, 1:] != -100
    content_mask = supervised & content[:, 1:]

    # Rows are laid out [own_i, own_j, cross_ij, cross_ji] per pair.
    own_indices = [4 * index + offset for index in range(len(pairs)) for offset in (0, 1)]
    own_mask = supervised[own_indices]
    sft_loss = per_token[own_indices][own_mask].mean()

    content_nll = []
    for losses, mask in zip(per_token, content_mask, strict=True):
        if not bool(mask.any()):
            raise ValueError("Contrastive target has no content tokens")
        content_nll.append(losses[mask].mean())
    pair_loss, gaps = symmetric_pair_objective(
        torch.stack(content_nll), n_pairs=len(pairs), temperature=temperature
    )
    return sft_loss, pair_loss, gaps.mean()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--init-adapter", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--pairs-per-batch", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-pairs-per-source", type=int, default=124)
    parser.add_argument("--pair-loss-weight", type=float, required=True)
    parser.add_argument("--pair-temperature", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")
    if args.pairs_per_batch <= 0:
        raise ValueError("--pairs-per-batch must be positive")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad-accum-steps must be positive")
    if args.max_pairs_per_source <= 0:
        raise ValueError("--max-pairs-per-source must be positive")
    if args.pair_temperature <= 0:
        raise ValueError("--pair-temperature must be positive")
    if args.pair_loss_weight < 0:
        raise ValueError("--pair-loss-weight must be nonnegative")

    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.out_dir}")

    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    nla_cfg = cfg["nla_model"]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    rows = list(read_jsonl(args.train_jsonl))
    initial_pairs = build_disjoint_pairs(rows, seed=args.seed, epoch=1)
    pair_counts = dict(sorted((key, len(value)) for key, value in initial_pairs.items()))
    print(
        f"[pairs] available={pair_counts}",
        flush=True,
    )
    if len(initial_pairs) < 2 or any(not items for items in initial_pairs.values()):
        raise ValueError("Need nonempty contrastive pairs from both source datasets")

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
    actor_prompt_template = adapter_av_prompt(str(args.init_adapter))
    if actor_prompt_template is None:
        raise FileNotFoundError(f"Adapter has no recorded AV prompt: {args.init_adapter}")
    base = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    from peft import PeftModel

    model = PeftModel.from_pretrained(
        base, str(args.init_adapter), cache_dir=cache_dir, is_trainable=True
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("Loaded adapter has no trainable parameters")
    for parameter in trainable:
        parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    # Gradients work in eval mode. Disabling dropout is important here because
    # the pre-training alignment gap is only about 0.005 NLL; independent LoRA
    # dropout masks across matched and crossed rows would be larger than the
    # signal this objective is intended to amplify.
    model.eval()
    embed_layer = model.get_input_embeddings()
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, weight_decay=args.weight_decay
    )

    args.out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.out_dir / "train.config.yaml")
    metrics_path = args.out_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    epoch = 0
    recent = []
    while optimizer_step < args.max_steps:
        epoch += 1
        pairs = balanced_pairs(
            rows,
            seed=args.seed,
            epoch=epoch,
            max_pairs_per_source=args.max_pairs_per_source,
        )
        print(f"[epoch] {epoch} balanced_pairs={len(pairs)}", flush=True)
        for start in range(0, len(pairs), args.pairs_per_batch):
            batch = pairs[start : start + args.pairs_per_batch]
            sft_loss, pair_loss, gap = losses_for_pair_batch(
                pairs=batch,
                model=model,
                tokenizer=tokenizer,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt_template,
                temperature=args.pair_temperature,
            )
            total = sft_loss + args.pair_loss_weight * pair_loss
            (total / args.grad_accum_steps).backward()
            micro_step += 1
            recent.append((float(sft_loss.item()), float(pair_loss.item()), float(gap.item())))
            if micro_step % args.grad_accum_steps != 0:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window = recent[-args.grad_accum_steps :]
            metrics = {
                "step": optimizer_step,
                "epoch": epoch,
                "sft_loss": mean(item[0] for item in window),
                "pair_loss": mean(item[1] for item in window),
                "cross_minus_matched": mean(item[2] for item in window),
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"[train] step={optimizer_step}/{args.max_steps} "
                f"sft={metrics['sft_loss']:.4f} "
                f"pair={metrics['pair_loss']:.4f} "
                f"gap={metrics['cross_minus_matched']:+.4f}",
                flush=True,
            )
            if optimizer_step >= args.max_steps:
                break

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    (args.out_dir / AV_PROMPT_FILENAME).write_text(
        actor_prompt_template, encoding="utf-8"
    )
    metadata = {
        "objective": "matched_sft_plus_symmetric_content_nll_contrast",
        "init_adapter": str(args.init_adapter),
        "train_jsonl": str(args.train_jsonl),
        "optimizer_steps": optimizer_step,
        "epochs_touched": epoch,
        "args": vars(args),
    }
    # Path values are serialized explicitly for reproducibility.
    metadata["args"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in metadata["args"].items()
    }
    (args.out_dir / "best.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
