"""Train the approved D10/D20 changed-cue objectives.

The original-only arm and paired-ranking arm use the same rows and order. The
only difference is the frozen ranking weight: zero for control, one for D10.
For D20 only, the retained claim is additionally anchored on both the original
and deleted activations. No deleted-state abstention target is ever invented.
"""

from __future__ import annotations

import argparse
import hashlib
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


def retained_variants(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original, deleted = paired_variants(row)
    retained_target = row.get("retained_target_text")
    if not retained_target:
        raise ValueError(f"D20 row lacks retained_target_text: {row.get('id')}")
    original["target_text"] = retained_target
    original["d20_target"] = "retained"
    deleted["target_text"] = retained_target
    deleted["d20_target"] = "retained"
    return original, deleted


def one_by_two_objective(
    original_content_nll: torch.Tensor,
    deleted_content_nll: torch.Tensor,
    *,
    temperature: float,
    margin: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if original_content_nll.shape != deleted_content_nll.shape:
        raise ValueError("Original/deleted NLL shapes differ")
    gap = deleted_content_nll - original_content_nll
    loss = (temperature * F.softplus((margin - gap) / temperature)).mean()
    return loss, gap


def specificity_anchored_objective(
    changed_original_content_nll: torch.Tensor,
    changed_deleted_content_nll: torch.Tensor,
    retained_original_content_nll: torch.Tensor,
    retained_deleted_content_nll: torch.Tensor,
    *,
    anchor_weight: float,
    ranking_weight: float,
    temperature: float,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the D20 content objective and changed deleted-minus-original gap."""
    ranking_loss, changed_gap = one_by_two_objective(
        changed_original_content_nll,
        changed_deleted_content_nll,
        temperature=temperature,
        margin=margin,
    )
    loss = (
        changed_original_content_nll.mean()
        + anchor_weight
        * (
            retained_original_content_nll.mean()
            + retained_deleted_content_nll.mean()
        )
        + ranking_weight * ranking_loss
    )
    return loss, changed_gap


def backward_pair(
    *,
    row: dict[str, Any],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str,
    ranking_weight: float,
    retained_anchor_weight: float,
    temperature: float,
    margin: float,
    grad_accum_steps: int,
) -> dict[str, float]:
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
            original_content,
            deleted_content,
            temperature=temperature,
            margin=margin,
        )
        sft_loss = original_sum[0] / original_counts[0]
        strength = torch.sigmoid((margin - gap[0]) / temperature)

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

    retained_original_nll = float("nan")
    retained_deleted_nll = float("nan")
    if retained_anchor_weight:
        retained_original, retained_deleted = retained_variants(row)
        (
            _retained_original_sum,
            _retained_original_counts,
            retained_original_content,
        ) = losses_for_rows(
            rows=[retained_original],
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
        )
        (
            retained_anchor_weight
            * retained_original_content[0]
            / grad_accum_steps
        ).backward()
        retained_original_nll = float(retained_original_content[0].detach().item())

        (
            _retained_deleted_sum,
            _retained_deleted_counts,
            retained_deleted_content,
        ) = losses_for_rows(
            rows=[retained_deleted],
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
        )
        (
            retained_anchor_weight
            * retained_deleted_content[0]
            / grad_accum_steps
        ).backward()
        retained_deleted_nll = float(retained_deleted_content[0].detach().item())

    return {
        "sft_loss": float(sft_loss.item()),
        "ranking_loss": float(ranking_loss.item()),
        "changed_gap": float(gap.item()),
        "retained_original_content_nll": retained_original_nll,
        "retained_deleted_content_nll": retained_deleted_nll,
    }


def epoch_rows(rows: list[dict[str, Any]], *, seed: int, epoch: int) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: str(row.get("base_id") or row.get("id")))
    random.Random(f"{seed}:{epoch}:d10-pair-order").shuffle(ordered)
    return ordered


def normalize_checkpoint_steps(values: list[int], *, max_steps: int) -> list[int]:
    steps = sorted(set(values))
    if not steps or any(step <= 0 or step > max_steps for step in steps):
        raise ValueError("Checkpoint steps must be unique values in [1, max_steps]")
    if max_steps not in steps:
        raise ValueError("The final optimizer step must be a checkpoint")
    return steps


def advance_cursor(*, epoch: int, row_index: int, n_rows: int) -> tuple[int, int]:
    row_index += 1
    if row_index == n_rows:
        return epoch + 1, 0
    if row_index > n_rows:
        raise ValueError("Training cursor moved past the epoch boundary")
    return epoch, row_index


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def training_contract(
    *, args: argparse.Namespace, train_sha256: str, n_rows: int, checkpoints: list[int]
) -> dict[str, Any]:
    actor_prompt_path = Path(args.actor_prompt_template_file)
    contract = {
        "train_sha256": train_sha256,
        "config_sha256": sha256_file(Path(args.config)),
        "actor_prompt_sha256": (
            sha256_file(actor_prompt_path) if actor_prompt_path.is_file() else None
        ),
        "n_rows": n_rows,
        "max_steps": args.max_steps,
        "grad_accum_steps": args.grad_accum_steps,
        "ranking_weight": args.ranking_weight,
        "temperature": args.temperature,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "init_adapter": str(args.init_adapter) if args.init_adapter else None,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "target_modules": list(args.target_modules),
        "checkpoint_steps": checkpoints,
    }
    # Preserve byte-for-byte D10 resume contracts at the legacy defaults.
    if args.retained_anchor_weight or args.margin:
        contract.update(
            {
                "objective_schema": "d20_specificity_anchor_v1",
                "retained_anchor_weight": args.retained_anchor_weight,
                "margin": args.margin,
            }
        )
    return contract


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def prepare_metrics_for_resume(path: Path, *, optimizer_step: int) -> None:
    if not path.is_file():
        if optimizer_step:
            raise FileNotFoundError(f"Missing metrics for resume: {path}")
        return
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    rows = [json.loads(line) for line in lines]
    expected = list(range(1, len(rows) + 1))
    actual = [int(row["step"]) for row in rows]
    if actual != expected or len(rows) < optimizer_step:
        raise ValueError("Training metrics are not contiguous through the resume step")
    if len(rows) > optimizer_step:
        kept = rows[:optimizer_step]
        temporary = path.with_suffix(".jsonl.resume-tmp")
        temporary.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in kept),
            encoding="utf-8",
        )
        temporary.replace(path)
        print(
            f"[resume] truncated metrics from {len(rows)} to {optimizer_step} steps",
            flush=True,
        )


def save_training_checkpoint(
    *,
    out_dir: Path,
    model: torch.nn.Module,
    actor_prompt_template: str,
    optimizer: torch.optim.Optimizer,
    optimizer_step: int,
    micro_step: int,
    epoch: int,
    next_row_index: int,
    contract: dict[str, Any],
    initialization: str,
) -> Path:
    checkpoint = out_dir / f"checkpoint-step{optimizer_step:06d}"
    state_path = checkpoint / "trainer_state.pt"
    if checkpoint.exists():
        if state_path.is_file():
            raise FileExistsError(f"Checkpoint already complete: {checkpoint}")
        shutil.rmtree(checkpoint)
    checkpoint.mkdir(parents=True)
    model.save_pretrained(checkpoint)
    (checkpoint / AV_PROMPT_FILENAME).write_text(
        actor_prompt_template, encoding="utf-8"
    )
    state = {
        "schema_version": 1,
        "contract": contract,
        "initialization": initialization,
        "cursor": {
            "optimizer_step": optimizer_step,
            "micro_step": micro_step,
            "epoch": epoch,
            "next_row_index": next_row_index,
        },
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        },
    }
    temporary = checkpoint / "trainer_state.pt.tmp"
    torch.save(state, temporary)
    temporary.replace(state_path)

    # Adapter snapshots remain available for dose-response evaluation. Only the
    # newest optimizer state is retained because Adam states dominate disk use.
    for old_state in sorted(out_dir.glob("checkpoint-step*/trainer_state.pt")):
        if old_state != state_path:
            old_state.unlink()
    print(f"[checkpoint] step={optimizer_step} path={checkpoint}", flush=True)
    return checkpoint


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
    parser.add_argument("--checkpoint-steps", nargs="+", type=int)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--ranking-weight", type=float, choices=(0.0, 1.0), required=True)
    parser.add_argument(
        "--retained-anchor-weight", type=float, choices=(0.0, 1.0), default=0.0
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--margin", type=float, default=0.0)
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
    if args.margin != 0.0:
        raise ValueError("D10/D20 freeze the ranking margin at 0.0")
    if args.retained_anchor_weight and args.ranking_weight != 1.0:
        raise ValueError("D20 retained anchoring requires ranking_weight=1.0")
    checkpoint_steps = normalize_checkpoint_steps(
        args.checkpoint_steps or [args.max_steps], max_steps=args.max_steps
    )
    if args.out_dir.exists() and not args.resume_from_checkpoint:
        raise FileExistsError(f"Refusing to overwrite output: {args.out_dir}")
    if args.resume_from_checkpoint and not args.out_dir.is_dir():
        raise FileNotFoundError(f"Resume output directory does not exist: {args.out_dir}")

    rows = list(read_jsonl(args.train_jsonl))
    if not rows:
        raise ValueError("No D10 training rows")
    identifiers = [str(row.get("base_id") or "") for row in rows]
    if "" in identifiers or len(set(identifiers)) != len(identifiers):
        raise ValueError("D10 rows have missing or duplicate base_id")
    for row in rows:
        original, deleted = paired_variants(row)
        if args.retained_anchor_weight:
            retained_variants(row)
        for variant in (original, deleted):
            if not Path(str(variant["activation_path"])).is_file():
                raise FileNotFoundError(variant["activation_path"])
    contract = training_contract(
        args=args,
        train_sha256=sha256_file(args.train_jsonl),
        n_rows=len(rows),
        checkpoints=checkpoint_steps,
    )

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
    resume_state = None
    if args.resume_from_checkpoint:
        state_path = args.resume_from_checkpoint / "trainer_state.pt"
        if not state_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint has no trainer state: {state_path}")
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        if resume_state.get("contract") != contract:
            raise ValueError("Resume checkpoint training contract does not match arguments")
        from peft import PeftModel

        actor_prompt_template = adapter_av_prompt(str(args.resume_from_checkpoint))
        if actor_prompt_template is None:
            raise FileNotFoundError(
                f"Checkpoint has no recorded AV prompt: {args.resume_from_checkpoint}"
            )
        model = PeftModel.from_pretrained(
            base,
            str(args.resume_from_checkpoint),
            cache_dir=cache_dir,
            is_trainable=True,
        )
        initialization = str(resume_state["initialization"])
    elif args.init_adapter:
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

    if not args.resume_from_checkpoint:
        args.out_dir.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.config, args.out_dir / "train.config.yaml")
    metrics_path = args.out_dir / "metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    micro_step = 0
    epoch = 1
    next_row_index = 0
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        cursor = resume_state["cursor"]
        optimizer_step = int(cursor["optimizer_step"])
        micro_step = int(cursor["micro_step"])
        epoch = int(cursor["epoch"])
        next_row_index = int(cursor["next_row_index"])
        if micro_step % args.grad_accum_steps:
            raise ValueError("Resume checkpoint is not on an optimizer boundary")
        prepare_metrics_for_resume(metrics_path, optimizer_step=optimizer_step)
        restore_rng_state(resume_state["rng"])
        print(
            f"[resume] step={optimizer_step} epoch={epoch} row={next_row_index}",
            flush=True,
        )
    recent: list[dict[str, float]] = []
    while optimizer_step < args.max_steps:
        ordered_rows = epoch_rows(rows, seed=args.seed, epoch=epoch)
        for row in ordered_rows[next_row_index:]:
            active_epoch = epoch
            row_metrics = backward_pair(
                row=row,
                model=model,
                tokenizer=tokenizer,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt_template,
                ranking_weight=args.ranking_weight,
                retained_anchor_weight=args.retained_anchor_weight,
                temperature=args.temperature,
                margin=args.margin,
                grad_accum_steps=args.grad_accum_steps,
            )
            recent.append(row_metrics)
            micro_step += 1
            epoch, next_row_index = advance_cursor(
                epoch=epoch, row_index=next_row_index, n_rows=len(ordered_rows)
            )
            if micro_step % args.grad_accum_steps:
                continue
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1
            window = recent[-args.grad_accum_steps :]
            metrics = {
                "step": optimizer_step,
                "epoch": active_epoch,
                "sft_loss": mean(value["sft_loss"] for value in window),
                "ranking_loss": mean(value["ranking_loss"] for value in window),
                "deleted_minus_original_content_nll": mean(
                    value["changed_gap"] for value in window
                ),
                "ranking_weight": args.ranking_weight,
                "retained_anchor_weight": args.retained_anchor_weight,
                "temperature": args.temperature,
                "margin": args.margin,
            }
            if args.retained_anchor_weight:
                metrics.update(
                    {
                        "retained_original_content_nll": mean(
                            value["retained_original_content_nll"] for value in window
                        ),
                        "retained_deleted_content_nll": mean(
                            value["retained_deleted_content_nll"] for value in window
                        ),
                    }
                )
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"[train] step={optimizer_step}/{args.max_steps} "
                f"sft={metrics['sft_loss']:.4f} "
                f"rank={metrics['ranking_loss']:.4f} "
                f"gap={metrics['deleted_minus_original_content_nll']:+.4f}"
                + (
                    " ret_orig="
                    f"{metrics['retained_original_content_nll']:.4f}"
                    " ret_del="
                    f"{metrics['retained_deleted_content_nll']:.4f}"
                    if args.retained_anchor_weight
                    else ""
                ),
                flush=True,
            )
            if optimizer_step >= args.max_steps:
                break
            if optimizer_step in checkpoint_steps:
                save_training_checkpoint(
                    out_dir=args.out_dir,
                    model=model,
                    actor_prompt_template=actor_prompt_template,
                    optimizer=optimizer,
                    optimizer_step=optimizer_step,
                    micro_step=micro_step,
                    epoch=epoch,
                    next_row_index=next_row_index,
                    contract=contract,
                    initialization=initialization,
                )

    final_checkpoint = args.out_dir / f"checkpoint-step{optimizer_step:06d}"
    if (
        optimizer_step in checkpoint_steps
        and not (final_checkpoint / "trainer_state.pt").is_file()
    ):
        save_training_checkpoint(
            out_dir=args.out_dir,
            model=model,
            actor_prompt_template=actor_prompt_template,
            optimizer=optimizer,
            optimizer_step=optimizer_step,
            micro_step=micro_step,
            epoch=epoch,
            next_row_index=next_row_index,
            contract=contract,
            initialization=initialization,
        )

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    (args.out_dir / AV_PROMPT_FILENAME).write_text(
        actor_prompt_template, encoding="utf-8"
    )
    metadata = {
        "objective": (
            "d20_specificity_anchored_two_target_two_activation"
            if args.retained_anchor_weight
            else (
                "original_only_sft"
                if args.ranking_weight == 0.0
                else "d10_one_claim_two_activation_ranking"
            )
        ),
        "initialization": initialization,
        "train_jsonl": str(args.train_jsonl),
        "n_rows": len(rows),
        "optimizer_steps": optimizer_step,
        "epochs_touched": epoch - 1 if next_row_index == 0 else epoch,
        "next_row_index": next_row_index,
        "checkpoint_steps": checkpoint_steps,
        "training_contract": contract,
        "locked_hyperparameters": {
            "ranking_weight": args.ranking_weight,
            "retained_anchor_weight": args.retained_anchor_weight,
            "temperature": args.temperature,
            "margin": args.margin,
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
