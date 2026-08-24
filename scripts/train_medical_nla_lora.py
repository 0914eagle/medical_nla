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
from scripts.make_medical_nla_v3_cue_first_targets import content_char_spans
from src.nla import AV_PROMPT_FILENAME, build_nla_inputs_embeds, load_nla_sidecar


@dataclass
class TrainMetrics:
    step: int
    epoch: int
    loss: float


@dataclass
class EvalMetrics:
    """Validation loss split by what the token is.

    A target is seven lines of which six are the same XML in every row, so
    roughly thirty of a thirty-six-token target are a constant. The adapter
    learns that constant in the first hundred steps and the mean loss collapses
    toward zero whether or not the vector was read at all -- DDXPlus reached
    1e-4 a quarter of the way through its first epoch. Selecting a best epoch
    on that number, or reading an L16-vs-L24-vs-L32 trajectory off it, is
    reading the brackets.

    `content_loss` is the same cross-entropy restricted to the tokens of the
    clinical finding itself, which is the only part of the target the vector
    can supply.
    """

    loss: float
    content_loss: float
    scaffold_loss: float
    content_tokens: int
    scaffold_tokens: int




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
    activation = torch.load(row["activation_path"], map_location="cpu", weights_only=True)
    injected = build_nla_inputs_embeds(
        tokenizer=tokenizer,
        embed_layer=embed_layer,
        sidecar=sidecar,
        activation=activation,
        device=model.device,
        actor_prompt_template=actor_prompt_template,
    )
    target_text = str(row["target_text"])
    encoded = tokenizer(target_text, add_special_tokens=False, return_offsets_mapping=True)
    target_ids = list(encoded["input_ids"])
    spans = content_char_spans(target_text)
    is_content = [
        any(start < span_end and end > span_start for span_start, span_end in spans)
        for start, end in encoded["offset_mapping"]
    ]
    if eos_token_id is not None:
        target_ids = target_ids + [int(eos_token_id)]
        is_content = is_content + [False]
    target = torch.tensor(target_ids, dtype=torch.long, device=model.device).unsqueeze(0)
    with torch.no_grad():
        target_embeds = embed_layer(target)
    inputs_embeds = torch.cat([injected.inputs_embeds, target_embeds], dim=1).squeeze(0)
    attention_mask = torch.ones(inputs_embeds.shape[0], dtype=torch.long, device=model.device)
    labels = torch.full((inputs_embeds.shape[0],), -100, dtype=torch.long, device=model.device)
    content = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=model.device)
    prefix_len = injected.inputs_embeds.shape[1]
    labels[prefix_len:] = target.squeeze(0)
    content[prefix_len:] = torch.tensor(is_content, dtype=torch.bool, device=model.device)
    return inputs_embeds, attention_mask, labels, content


def collate_examples(
    examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(example[0].shape[0] for example in examples)
    hidden = examples[0][0].shape[-1]
    device = examples[0][0].device
    dtype = examples[0][0].dtype
    batch_embeds = torch.zeros((len(examples), max_len, hidden), dtype=dtype, device=device)
    batch_attention = torch.zeros((len(examples), max_len), dtype=torch.long, device=device)
    batch_labels = torch.full((len(examples), max_len), -100, dtype=torch.long, device=device)
    batch_content = torch.zeros((len(examples), max_len), dtype=torch.bool, device=device)
    for idx, (embeds, attention, labels, content) in enumerate(examples):
        length = embeds.shape[0]
        batch_embeds[idx, :length] = embeds
        batch_attention[idx, :length] = attention
        batch_labels[idx, :length] = labels
        batch_content[idx, :length] = content
    return batch_embeds, batch_attention, batch_labels, batch_content


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
) -> EvalMetrics:
    """Loss over `rows`, split scaffold from content, leaving the model's mode as found.

    Restoring it is not tidiness. Without it the model stayed in eval mode
    after the first epoch's evaluation, so epoch 1 trained with dropout and
    every later epoch trained without it -- and since the layer sweep also ran
    different epoch counts per layer, the resulting L16/L24/L32 trajectory mixed
    a layer effect with a dropout-coverage effect.

    The caller chooses `rows`; this no longer truncates. Truncation here took
    the file's first n rows, which on a corpus grouped by diagnosis is one
    corner of the label space rather than a sample of it.
    """
    was_training = model.training
    model.eval()
    # Summed rather than averaged per batch: batches hold unequal numbers of
    # supervised tokens, so a mean of batch means weights a short target the
    # same as a long one.
    totals = {"all": 0.0, "content": 0.0, "scaffold": 0.0}
    counts = {"all": 0, "content": 0, "scaffold": 0}
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
        inputs_embeds, attention_mask, labels, content = collate_examples(examples)
        out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # The shift the model applies internally, applied here so the same
        # per-token losses can be grouped: logits at t predict the label at t+1.
        flat_logits = out.logits[:, :-1, :].float().reshape(-1, out.logits.shape[-1])
        flat_labels = labels[:, 1:].reshape(-1)
        flat_content = content[:, 1:].reshape(-1)
        per_token = torch.nn.functional.cross_entropy(
            flat_logits, flat_labels, reduction="none", ignore_index=-100
        )
        supervised = flat_labels != -100
        is_content = supervised & flat_content
        is_scaffold = supervised & ~flat_content
        for key, mask in (("all", supervised), ("content", is_content), ("scaffold", is_scaffold)):
            totals[key] += float(per_token[mask].sum().item())
            counts[key] += int(mask.sum().item())
    if was_training:
        model.train()
    mean = lambda key: totals[key] / counts[key] if counts[key] else float("nan")
    return EvalMetrics(
        loss=mean("all"),
        content_loss=mean("content"),
        scaffold_loss=mean("scaffold"),
        content_tokens=counts["content"],
        scaffold_tokens=counts["scaffold"],
    )


def read_actor_prompt_template(path: str | None) -> str | None:
    """The AV prompt text, or None to fall back to the checkpoint's own.

    The literal "sidecar" is accepted so the fallback stays reachable without
    anyone having to know that an absent flag means it.
    """
    if path is None or path == "sidecar":
        return None
    text = Path(path).read_text(encoding="utf-8")
    if "{injection_char}" not in text:
        raise ValueError(f"{path} has no {{injection_char}} placeholder; the vector would not be injected.")
    return text


def save_av_prompt(out_dir: Path, template: str | None, sidecar: Any) -> None:
    """Record the AV prompt beside the adapter it was trained under.

    Generation reads it back (`src.nla.adapter_av_prompt`), so the pair cannot
    drift apart the way it did when both ends took a flag that could be omitted.
    """
    text = template if template is not None else sidecar.actor_prompt_template
    (out_dir / AV_PROMPT_FILENAME).write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument(
        "--val-jsonl",
        default=None,
        help=(
            "Optional fixed validation JSONL. If omitted, validation rows are split "
            "from --train-jsonl with --val-frac."
        ),
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--actor-prompt-template-file",
        default=str(REPO_ROOT / "prompt_templates" / "cue_position_readout.txt"),
        help=(
            "The AV prompt. Defaults to the cue-position template because "
            "leaving it unset fell back to the checkpoint sidecar's own prompt, "
            "which asks for a diagnosis while the supervised target contains "
            "only <observed> findings -- the adapter then learns a format the "
            "prompt never asked for. Pass a path to override; pass 'sidecar' to "
            "restore the old fallback."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help=(
            "Must be the same for every layer and both corpora. The pilot ran 3 "
            "at L32 and 2 at L16/L24 while the write-up said one recipe, which "
            "left the layer trajectory mixing a layer effect with an epoch one."
        ),
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Cap the training set, sampled at random. Used to give both corpora "
            "the same budget: MedCaseReasoning carries 32,724 training rows "
            "against DDXPlus's 10,195, so without a cap a difference between "
            "them could be the prose or could be the 3.2x more data."
        ),
    )
    parser.add_argument(
        "--train-subsample-seed",
        type=int,
        default=17,
        help=(
            "Seed for --max-train-rows, held separate from --seed so that "
            "several seeds see the same subset and vary only in initialization "
            "and ordering. Sharing the seed would confound the two."
        ),
    )
    parser.add_argument("--val-frac", type=float, default=0.05)
    parser.add_argument(
        "--max-eval-rows",
        type=int,
        default=128,
        help=(
            "Validation rows to score per epoch, sampled at random once and "
            "reused across epochs so the per-epoch losses are comparable."
        ),
    )
    parser.add_argument(
        "--select-on",
        choices=("content", "total"),
        default="content",
        help=(
            "Which validation loss picks the best epoch. 'content' uses only "
            "the tokens of the clinical finding; 'total' also counts the fixed "
            "XML scaffold, which is most of every target and is learned in the "
            "first hundred steps."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=17,
        help=(
            "One run is a point estimate. Train at least three seeds per "
            "configuration and report mean +/- sd; the seed is recorded in "
            "metadata.json and best.json so runs can be pooled afterwards."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Recompute block activations in the backward pass instead of "
            "storing them. About 30% slower and the difference between fitting "
            "and not fitting a 12B adapter run on two 24GB cards."
        ),
    )
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
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)

    train_input_rows = list(read_jsonl(args.train_jsonl))
    train_input_rows = [
        row for row in train_input_rows if row.get("activation_path") and row.get("target_text")
    ]
    if not train_input_rows:
        raise ValueError("No rows with activation_path and target_text found.")
    if args.val_jsonl:
        train_rows = train_input_rows
        val_rows = [
            row
            for row in read_jsonl(args.val_jsonl)
            if row.get("activation_path") and row.get("target_text")
        ]
        if not val_rows:
            raise ValueError("--val-jsonl contained no rows with activation_path and target_text.")
    else:
        train_rows, val_rows = split_rows(
            train_input_rows,
            val_frac=args.val_frac,
            seed=args.seed,
        )

    if args.max_train_rows is not None and len(train_rows) > args.max_train_rows:
        before = len(train_rows)
        train_rows = random.Random(args.train_subsample_seed).sample(
            train_rows, args.max_train_rows
        )
        print(
            f"[data] training on {len(train_rows):,} of {before:,} rows "
            f"(random, seed {args.train_subsample_seed})",
            flush=True,
        )

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
    if args.gradient_checkpointing:
        # Storing every block's activations for the backward pass costs about
        # 12GB at batch 8 here, which is what pushed a card holding 11.8GB of
        # weights over its 24GB. Recomputing them instead trades roughly 30%
        # throughput for most of that memory.
        #
        # enable_input_require_grads is not optional with checkpointing: this
        # trainer feeds `inputs_embeds` it builds itself, and a checkpointed
        # first block whose input does not require grad produces no gradient
        # for the adapter at all -- a silent no-op, not an error.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
        print("[train] gradient checkpointing on", flush=True)
    # The base is loaded in bfloat16, so the adapter is created in bfloat16 and
    # the AdamW moments are kept in it too -- 8 mantissa bits for a quantity
    # that accumulates over thousands of steps. Standard practice is to train
    # the adapter in float32 while the frozen base stays bfloat16, and it
    # affects stability rather than only precision.
    trainable = [p for p in model.parameters() if p.requires_grad]
    for param in trainable:
        param.data = param.data.float()
    model.print_trainable_parameters()
    model.train()
    embed_layer = model.get_input_embeddings()
    # Only the adapter: handing AdamW the frozen base as well invites optimizer
    # state for parameters that never receive a gradient.
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, out_dir / "train.config.yaml")
    metadata = {
        "train_jsonl": str(Path(args.train_jsonl)),
        "val_jsonl": str(Path(args.val_jsonl)) if args.val_jsonl else None,
        "n_rows": len(train_rows) + len(val_rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    # Drawn once, so the per-epoch losses are comparable, and drawn at random,
    # because the first n rows of a file grouped by diagnosis are one corner of
    # the label space.
    eval_rows = val_rows
    if args.max_eval_rows is not None and len(val_rows) > args.max_eval_rows:
        eval_rows = random.Random(args.seed).sample(val_rows, args.max_eval_rows)
    print(
        f"[eval] validating on {len(eval_rows):,} of {len(val_rows):,} rows "
        f"(random, seed {args.seed})",
        flush=True,
    )

    # Checked before the first forward pass, because the failure it catches is
    # silent and expensive: a target shape whose content spans do not parse
    # gives zero content tokens, a content loss of NaN, and `NaN < best` False
    # at every epoch -- so the loop trains to the end and saves no adapter at
    # all. That cost one full MCR conclusion run. Two rows are enough: the
    # spans come from the target's shape, which is constant within a split.
    if args.select_on == "content":
        probe_rows = (train_rows + eval_rows)[:2]
        if probe_rows and not any(
            content_char_spans(str(row.get("target_text") or "")) for row in probe_rows
        ):
            raise SystemExit(
                "no content spans in the first training targets, so --select-on "
                "content would rank every epoch by NaN and save nothing.\n"
                "  Either the target shape is new (teach content_char_spans "
                "about it) or pass --select-on total.\n"
                f"  first target: {str(probe_rows[0].get('target_text'))[:200]!r}"
            )

    global_step = 0
    optimizer_step = 0
    # The adapter kept was the last epoch's, whatever the validation loss did.
    # out_dir now always holds the best epoch, and best.json says which.
    best_val_loss = float("inf")
    best_val: EvalMetrics | None = None
    best_epoch: int | None = None
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
            inputs_embeds, attention_mask, labels, _content = collate_examples(examples)
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
            loss = out.loss / args.grad_accum_steps
            loss.backward()
            global_step += 1
            if global_step % args.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
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
        val = evaluate(
            rows=eval_rows,
            model=model,
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            actor_prompt_template=actor_prompt_template,
            batch_size=args.batch_size,
        )
        # Selected on the finding's tokens, not on the whole target. Six of
        # every seven target lines are the same XML in every row, so the mean
        # is a constant the adapter has already learned; ranking epochs by it
        # ranks them by rounding error.
        selector = val.content_loss if args.select_on == "content" else val.loss
        if selector != selector:  # NaN, and NaN < anything is False
            # The startup check should have caught this; if something else
            # produced a NaN mid-run, fall back rather than silently declining
            # to save for the rest of training.
            print("[eval] content loss is NaN -- selecting on total loss instead", flush=True)
            selector = val.loss
        improved = selector < best_val_loss
        print(
            f"[eval] epoch={epoch} val_loss={val.loss:.4f} "
            f"content={val.content_loss:.4f} scaffold={val.scaffold_loss:.4f} "
            f"({val.content_tokens:,} content / {val.scaffold_tokens:,} scaffold tokens)"
            + (" (best so far, saving)" if improved else f" (best {best_val_loss:.4f})"),
            flush=True,
        )
        with metrics_path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {"epoch": epoch, "step": optimizer_step, **asdict(val)},
                    sort_keys=True,
                )
                + "\n"
            )
        if improved:
            best_val_loss, best_epoch = selector, epoch
            best_val = val
            model.save_pretrained(out_dir)
            tokenizer.save_pretrained(out_dir)
            save_av_prompt(out_dir, actor_prompt_template, sidecar)
        if args.max_steps is not None and optimizer_step >= args.max_steps:
            break

    if best_epoch is None:
        # No epoch improved on infinity only if evaluation never ran.
        model.save_pretrained(out_dir)
        tokenizer.save_pretrained(out_dir)
        save_av_prompt(out_dir, actor_prompt_template, sidecar)
        print(f"[done] saved the final LoRA adapter to {out_dir}", flush=True)
    else:
        (out_dir / "best.json").write_text(
            json.dumps(
                {
                    "best_epoch": best_epoch,
                    "best_val_loss": round(best_val.loss, 6),
                    "best_val_content_loss": round(best_val.content_loss, 6),
                    "best_val_scaffold_loss": round(best_val.scaffold_loss, 6),
                    "selected_on": args.select_on,
                    "epochs_run": epoch,
                    "n_val_rows_used": len(eval_rows),
                    "n_val_rows_available": len(val_rows),
                    "seed": args.seed,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            f"[done] kept epoch {best_epoch} (content {best_val.content_loss:.4f}, "
            f"val_loss {best_val.loss:.4f}) of {epoch} in {out_dir}",
            flush=True,
        )


if __name__ == "__main__":
    main()
