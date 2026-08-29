"""Train the approved D10 one-claim-by-two-activations objective.

The original-only arm and paired-ranking arm use the same rows and order. The
only difference is the frozen ranking weight: zero for control, one for D10.
No target is assigned to the deleted activation.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_medical_nla_contrastive import losses_for_rows
from scripts.train_medical_nla_lora import read_actor_prompt_template
from src.config import load_config
from src.jsonl import read_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.nla import AV_PROMPT_FILENAME, adapter_av_prompt, load_nla_sidecar


def paired_variants(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    required = (
        "base_id",
        "target_text",
        "original_activation_path",
        "deleted_activation_path",
    )
    missing = [field for field in required if not row.get(field)]
    if missing:
        raise ValueError(f"D10 row lacks {missing}: {row.get('id')}")
    original = dict(row)
    original["activation_path"] = row["original_activation_path"]
    original["d10_state"] = "original"
    deleted = dict(row)
    deleted["activation_path"] = row["deleted_activation_path"]
    deleted["d10_state"] = "cue_deleted"
    return original, deleted


def one_by_two_objective(
    original_content_nll: torch.Tensor,
    deleted_content_nll: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if original_content_nll.shape != deleted_content_nll.shape:
        raise ValueError("Original/deleted NLL shapes differ")
    gap = deleted_content_nll - original_content_nll
    loss = (temperature * F.softplus(-gap / temperature)).mean()
    return loss, gap


def backward_pair(
    *,
    row: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str,
    ranking_weight: float,
    temperature: float,
    grad_accum_steps: int,
) -> tuple[float, float, float]:
    original, deleted = paired_variants(row)
    with torch.no_grad():
        original_sum, original_counts, original_content = losses_for_rows(
            rows=[original],
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
        )
        _deleted_sum, _deleted_counts, deleted_content = losses_for_rows(
            rows=[deleted],
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
        )
        ranking_loss, gap = one_by_two_objective(
            original_content, deleted_content, temperature=temperature
        )
        sft_loss = original_sum[0] / original_counts[0]
        strength = torch.sigmoid(-gap[0] / temperature)

    original_sum_grad, original_counts_grad, original_content_grad = losses_for_rows(
        rows=[original],
        model=model,
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        sidecar=sidecar,
        actor_prompt_template=actor_prompt_template,
    )
    scalar = original_sum_grad[0] / original_counts_grad[0]
    if ranking_weight:
        scalar = scalar + ranking_weight * strength * original_content_grad[0]
    (scalar / grad_accum_steps).backward()

    if ranking_weight:
        _deleted_sum_grad, _deleted_counts_grad, deleted_content_grad = losses_for_rows(
            rows=[deleted],
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
        )
        scalar_deleted = -ranking_weight * strength * deleted_content_grad[0]
        (scalar_deleted / grad_accum_steps).backward()

    return float(sft_loss.item()), float(ranking_loss.item()), float(gap.item())


def epoch_rows(rows: list[dict[str, Any]], *, seed: int, epoch: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row.get("base_id") or row.get("id")))
    random.Random(f"{seed}:{epoch}:d10-pair-order").shuffle(ordered)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--init-adapter", type=Path)
    parser.add_argument(
        "--actor-prompt-template-file",
        default=str(REPO_ROOT / "prompt_templates" / "cue_position_readout.txt"),
    )
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--ranking-weight", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    args = parser.parse_args()
    if args.max_steps <= 0 or args.grad_accum_steps <= 0:
        raise ValueError("Step counts must be positive")
    if args.temperature != 1.0:
        raise ValueError("D10 freezes temperature=1.0")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.out_dir}")

    rows = list(read_jsonl(args.train_jsonl))
    if not rows:
        raise ValueError("No D10 training rows")
    identifiers = [str(row.get("base_id") or "") for row in rows]
    if "" in identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("D10 rows have missing or duplicate base_id")
    for row in rows:
        original, deleted = paired_variants(row)
        for variant in (original, deleted):
            if not Path(str(variant["activation_path"])).is_file():
                raise FileNotFoundError(variant["activation_path"])

    cfg = load_config(args.config)
    nla_cfg = cfg["nla_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

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
    base = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    if args.init_adapter:
        from peft import PeftModel

        actor_prompt_template = adapter_av_prompt(str(args.init_adapter))
        if actor_prompt_template is None:
            raise FileNotFoundError(
                f"Adapter has no recorded AV prompt: {args.init_adapter}"
            )
        model = PeftModel.from_pretrained(
            base, str(args.init_adapter), cache_dir=cache_dir, is_trainable=True
        )
        initialization = str(args.init_adapter)
    else:
        from peft import LoraConfig, get_peft_model

        actor_prompt_template = read_actor_prompt_template(
            args.actor_prompt_template_file
        )
        if actor_prompt_template is None:
            actor_prompt_template = sidecar.actor_prompt_template
        model = get_peft_model(
            base,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=args.target_modules,
            ),
        )
        initialization = "fresh LoRA on released AV checkpoint"

    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    model.print_trainable_parameters()
    # Deterministic paired comparison: gradients are enabled, dropout is not.
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
    recent: list[tuple[float, float, float]] = []
    while optimizer_step < args.max_steps:
        epoch += 1
        for row in epoch_rows(rows, seed=args.seed, epoch=epoch):
            sft_loss, ranking_loss, gap = backward_pair(
                row=row,
                model=model,
                tokenizer=tokenizer,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt_template,
                ranking_weight=args.ranking_weight,
                temperature=args.temperature,
                grad_accum_steps=args.grad_accum_steps,
            )
            recent.append((sft_loss, ranking_loss, gap))
            micro_step += 1
            if micro_step % args.grad_accum_steps:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window = recent[-args.grad_accum_steps :]
            metrics = {
                "step": optimizer_step,
                "epoch": epoch,
                "sft_loss": mean(value[0] for value in window),
                "ranking_loss": mean(value[1] for value in window),
                "deleted_minus_original_content_nll": mean(
                    value[2] for value in window
                ),
                "ranking_weight": args.ranking_weight,
                "temperature": args.temperature,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"[train] step={optimizer_step}/{args.max_steps} "
                f"sft={metrics['sft_loss']:.4f} "
                f"rank={metrics['ranking_loss']:.4f} "
                f"gap={metrics['deleted_minus_original_content_nll']:+.4f}",
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
        "objective": (
            "original_only_sft"
            if args.ranking_weight == 0.0
            else "d10_one_claim_two_activation_ranking"
        ),
        "initialization": initialization,
        "train_jsonl": str(args.train_jsonl),
        "n_rows": len(rows),
        "optimizer_steps": optimizer_step,
        "epochs_touched": epoch,
        "locked_hyperparameters": {
            "ranking_weight": args.ranking_weight,
            "temperature": args.temperature,
        },
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    (args.out_dir / "best.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
