"""LoRA SFT for a diagnosis-preserving Medical-NLA AV adapter.

This trains only a PEFT adapter on top of the released AV checkpoint. The input
is a JSONL file from `scripts/make_medical_nla_sft_dataset.py` with activation
paths and target `<explanation>...</explanation>` strings.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import read_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.nla import build_nla_inputs_embeds, load_nla_sidecar


@dataclass
class TrainMetrics:
    step: int
    epoch: int
    loss: float


def split_rows(rows: list[dict[str, Any]], *, val_frac: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    by_base: dict[str, dict[str, Any]] = {}
    for row in rows:
        by_base[str(row.get("base_id", row["id"]))] = row
    base_ids = sorted(by_base)
    rng.shuffle(base_ids)
    n_val = max(1, int(round(len(base_ids) * val_frac))) if len(base_ids) > 1 else 0
    val_ids = set(base_ids[:n_val])
    train = [row for row in rows if str(row.get("base_id", row["id"])) not in val_ids]
    val = [row for row in rows if str(row.get("base_id", row["id"])) in val_ids]
    return train, val


def build_training_example(
    *,
    row: dict[str, Any],
    tokenizer: Any,
    model: torch.nn.Module,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str | None,
    eos_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    activation = torch.load(row["activation_path"], map_location="cpu")
    injected = build_nla_inputs_embeds(
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        sidecar=sidecar,
        activation=activation,
        device=model.device,
        actor_prompt_template=actor_prompt_template,
    )
    target_ids = tokenizer.encode(str(row["target_text"]), add_special_tokens=False)
    if eos_token_id is not None:
        target_ids = target_ids + [int(eos_token_id)]
    target = torch.tensor(target_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    with torch.no_grad():
        target_embeds = embed_layer(target)
    inputs_embeds = torch.cat([injected.inputs_embeds, target_embeds], dim=1).squeeze(0)
    attention_mask = torch.ones(inputs_embeds.shape[0], dtype=torch.long, device=model.device)
    labels = torch.full((inputs_embeds.shape[0],), -100, dtype=torch.long, device=model.device)
    prefix_len = injected.inputs_embeds.shape[1]
    labels[prefix_len:] = target.squeeze(0)
    return inputs_embeds, attention_mask, labels


def collate_examples(
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(example[0].shape[0] for example in examples)
    hidden = examples[0][0].shape[-1]
    device = examples[0][0].device
    dtype = examples[0][0].dtype
    batch_embeds = torch.zeros((len(examples), max_len, hidden), dtype=dtype, device=device)
    batch_attention = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    batch_labels = torch.full((len(examples), max_len), -100, dtype=torch.long, device=device)
    for idx, (embeds, attention, labels) in enumerate(examples):
        length = embeds.shape[0]
        batch_embeds[idx, :length] = embeds
        batch_attention[idx, :length] = attention
        batch_labels[idx, :length] = labels
    return batch_embeds, batch_attention, batch_labels


@torch.inference_mode()
def evaluate(
    *,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str | None,
    batch_size: int,
    max_eval_rows: int | None,
) -> float:
    model.eval()
    if max_eval_rows is not None:
        rows = rows[:max_eval_rows]
    losses = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
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
            for row in batch_rows
        ]
        inputs_embeds, attention_mask, labels = collate_examples(examples)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        losses.append(float(out.loss.item()))
    return sum(losses) / max(len(losses), 1)


def read_actor_prompt_template(path: str | None) -> str | None:
    if path is None:
        return None
    return Path(path).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--actor-prompt-template-file", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument("--max-eval-rows", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model

    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    nla_cfg = cfg["nla_model"]
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    rows = list(read_jsonl(args.train_jsonl))
    rows = [row for row in rows if row.get("activation_path") and row.get("target_text")]
    if not rows:
        raise ValueError("No rows with activation_path and target_text found.")
    train_rows, val_rows = split_rows(rows, val_frac=args.val_frac, seed=args.seed)

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
    model.config.use_cache = False
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=args.target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.train()
    embed_layer = model.get_input_embeddings()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "train.config.yaml")
    metadata = {
        "train_jsonl": str(Path(args.train_jsonl)),
        "n_rows": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    global_step = 0
    optimizer_step = 0
    metrics_path = out_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    for epoch in range(1, args.epochs + 1):
        random.shuffle(train_rows)
        for start in range(0, len(train_rows), args.batch_size):
            batch_rows = train_rows[start : start + args.batch_size]
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
                for row in batch_rows
            ]
            inputs_embeds, attention_mask, labels = collate_examples(examples)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            global_step += 1
            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                metric = TrainMetrics(
                    step=optimizer_step,
                    epoch=epoch,
                    loss=float(loss.item() * args.grad_accum_steps),
                )
                with metrics_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(metric), sort_keys=True) + "\n")
                print(
                    f"[train] step={optimizer_step} epoch={epoch} loss={metric.loss:.4f}",
                    flush=True,
                )
                if args.max_steps is not None and optimizer_step >= args.max_steps:
                    break
        val_loss = evaluate(
            rows=val_rows,
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
            batch_size=args.batch_size,
            max_eval_rows=args.max_eval_rows,
        )
        print(f"[eval] epoch={epoch} val_loss={val_loss:.4f}", flush=True)
        if args.max_steps is not None and optimizer_step >= args.max_steps:
            break

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print(f"[done] saved LoRA adapter to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
