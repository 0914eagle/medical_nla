"""Train one frozen D16 control or soft-auxiliary bottleneck smoke arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

from scripts.train_medical_nla_lora import (
    build_training_example,
    collate_examples,
    read_actor_prompt_template,
    row_target_nlls,
)
from src.config import load_config
from src.jsonl import read_jsonl
from src.modeling import load_causal_lm, load_tokenizer
from src.nla import AV_PROMPT_FILENAME, load_nla_sidecar
from src.nla_bottleneck import (
    BOTTLENECK_FILENAME,
    NlaBottleneckProjector,
    load_bottleneck,
    save_bottleneck,
    sha256_file,
    sha256_state_dict,
)


def parse_path_maps(values: list[str]) -> list[tuple[str, str]]:
    mappings = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected OLD=NEW path map, got {value!r}")
        old, new = value.split("=", 1)
        if not old:
            raise ValueError("Path-map OLD prefix cannot be empty")
        mappings.append((old, new))
    return mappings


def remap_rows(
    rows: list[dict[str, Any]],
    path_maps: list[tuple[str, str]],
    fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    output = []
    for source in rows:
        row = dict(source)
        for field in fields:
            value = str(row.get(field) or "")
            for old, new in path_maps:
                if value.startswith(old):
                    value = new + value[len(old) :]
                    break
            row[field] = value
        output.append(row)
    return output


def hash_order(rows: list[dict[str, Any]], *, seed: int) -> list[dict[str, Any]]:
    def key(row: dict[str, Any]) -> tuple[str, str]:
        identifier = str(row.get("base_id") or row.get("id") or "")
        digest = hashlib.sha256(f"{seed}:{identifier}".encode()).hexdigest()
        return digest, identifier

    return sorted(rows, key=key)


def ordered_id_sha256(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row.get("base_id") or row.get("id")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def initialize_auxiliary_head(d_z: int, labels: int) -> torch.nn.Linear:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        head = torch.nn.Linear(d_z, labels)
        torch.nn.init.kaiming_uniform_(head.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(d_z)
        torch.nn.init.uniform_(head.bias, -bound, bound)
    return head


def load_teacher(
    scores_path: Path, report_path: Path
) -> tuple[list[str], dict[str, list[float]], dict[str, Any]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    labels = [str(value) for value in report.get("finding_labels") or []]
    if len(labels) != 91 or len(set(labels)) != 91:
        raise ValueError("D16 requires the frozen 91-label K=5 teacher")
    if int(report.get("num_folds") or -1) != 5:
        raise ValueError("D16 requires the one-shot K=5 OOF teacher")
    original = {}
    for row in read_jsonl(scores_path):
        if str(row.get("variant")) != "original":
            continue
        identifier = str(row.get("base_id") or "")
        probabilities = list(map(float, row.get("finding_probabilities") or []))
        if not identifier or identifier in original or len(probabilities) != len(labels):
            raise ValueError(f"Invalid original teacher row: {identifier!r}")
        original[identifier] = probabilities
    if len(original) != 4655:
        raise ValueError(f"Teacher has {len(original)} originals, expected 4655")
    return labels, original, report


def validate_rows(
    direct_rows: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    teacher: dict[str, list[float]],
    labels: list[str],
) -> None:
    if len(direct_rows) != 248:
        raise ValueError(f"Direct train has {len(direct_rows)} rows, expected 248")
    if len(pairs) != 3104:
        raise ValueError(f"D9a has {len(pairs)} pairs, expected 3104")
    direct_ids = [str(row.get("base_id") or row.get("id") or "") for row in direct_rows]
    pair_ids = [str(row.get("base_id") or row.get("id") or "") for row in pairs]
    if len(set(direct_ids)) != len(direct_ids):
        raise ValueError("Direct train contains duplicate base IDs")
    if len(set(pair_ids)) != len(pair_ids):
        raise ValueError("D9a contains duplicate base IDs")
    label_set = set(labels)
    for row in direct_rows:
        if not row.get("target_text") or not Path(str(row.get("activation_path"))).is_file():
            raise ValueError(f"Invalid Direct row: {row.get('base_id')}")
    for row in pairs:
        identifier = str(row.get("base_id") or "")
        if identifier not in teacher:
            raise ValueError(f"D9a pair absent from K=5 teacher: {identifier}")
        if str(row.get("changed_evidence_id") or "") not in label_set:
            raise ValueError(f"D9a changed label outside teacher ontology: {identifier}")
        for field in ("original_activation_path", "deleted_activation_path"):
            if not Path(str(row.get(field) or "")).is_file():
                raise FileNotFoundError(row.get(field))


def load_activation_batch(rows: list[dict[str, Any]], field: str, device: Any) -> torch.Tensor:
    return torch.stack(
        [
            torch.load(row[field], map_location="cpu", weights_only=True).float().flatten()
            for row in rows
        ]
    ).to(device)


def auxiliary_losses(
    *,
    rows: list[dict[str, Any]],
    projector: NlaBottleneckProjector,
    head: torch.nn.Linear,
    teacher: dict[str, list[float]],
    label_index: dict[str, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = projector.mean.device
    original = load_activation_batch(rows, "original_activation_path", device)
    deleted = load_activation_batch(rows, "deleted_activation_path", device)
    _reconstructed_original, z_original = projector(original)
    _reconstructed_deleted, z_deleted = projector(deleted)
    original_logits = head(z_original)
    deleted_logits = head(z_deleted)
    targets = torch.tensor(
        [teacher[str(row["base_id"])] for row in rows],
        dtype=original_logits.dtype,
        device=original_logits.device,
    )
    original_bce = F.binary_cross_entropy_with_logits(
        original_logits, targets, reduction="mean"
    )
    columns = torch.tensor(
        [label_index[str(row["changed_evidence_id"])] for row in rows],
        dtype=torch.long,
        device=original_logits.device,
    )
    batch = torch.arange(len(rows), device=original_logits.device)
    changed_delta = original_logits[batch, columns] - deleted_logits[batch, columns]
    deleted_margin = F.softplus(-changed_delta).mean()
    return original_bce, deleted_margin, changed_delta.detach()


def load_training_stack(config: str, *, seed: int, actor_prompt_file: str):
    from peft import LoraConfig, get_peft_model

    cfg = load_config(config)
    nla_cfg = cfg["nla_model"]
    cache_dir = cfg["paths"].get("cache_dir")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
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
    actor_prompt = read_actor_prompt_template(actor_prompt_file)
    if actor_prompt is None:
        raise ValueError("D16 requires an explicit common clinical-state AV prompt")
    base = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    model = get_peft_model(
        base,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.enable_input_require_grads()
    return model, tokenizer, sidecar, actor_prompt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--direct-train", required=True, type=Path)
    parser.add_argument("--d9a-pairs", required=True, type=Path)
    parser.add_argument("--teacher-scores", required=True, type=Path)
    parser.add_argument("--teacher-report", required=True, type=Path)
    parser.add_argument("--pca-artifact", required=True, type=Path)
    parser.add_argument("--lambda-protocol", type=Path)
    parser.add_argument("--floor-protocol", type=Path)
    parser.add_argument("--matched-control", type=Path)
    parser.add_argument("--aux-head-audit-dir", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--arm", choices=("control", "auxiliary"), required=True)
    parser.add_argument("--seed", type=int, choices=(17, 29, 43), required=True)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--microbatch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument(
        "--actor-prompt-template-file",
        default=str(REPO_ROOT / "prompt_templates" / "common_p0_clinical_state_readout.txt"),
    )
    args = parser.parse_args()
    if args.max_steps != 20 or args.batch_size != 8 or args.microbatch_size != 4:
        raise ValueError("D16 freezes 20 steps, batch 8, microbatch 4")
    if args.out_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out_dir}")

    path_maps = parse_path_maps(args.path_map)
    direct_rows = remap_rows(
        list(read_jsonl(args.direct_train)), path_maps, ("activation_path",)
    )
    pairs = remap_rows(
        list(read_jsonl(args.d9a_pairs)),
        path_maps,
        ("original_activation_path", "deleted_activation_path"),
    )
    labels, teacher, _teacher_report = load_teacher(args.teacher_scores, args.teacher_report)
    validate_rows(direct_rows, pairs, teacher, labels)
    direct_rows = hash_order(direct_rows, seed=args.seed)[: args.max_steps * args.batch_size]
    pairs = hash_order(pairs, seed=args.seed)[: args.max_steps * args.batch_size]
    if len(direct_rows) != 160 or len(pairs) != 160:
        raise AssertionError("D16 smoke must use 160 rows from each source")
    selected_direct_sha256 = ordered_id_sha256(direct_rows)
    selected_d9a_sha256 = ordered_id_sha256(pairs)

    if args.arm == "auxiliary":
        if args.lambda_protocol is None:
            raise ValueError("Auxiliary arm requires --lambda-protocol")
        lambda_protocol = json.loads(args.lambda_protocol.read_text(encoding="utf-8"))
        if not lambda_protocol.get("frozen_before_control"):
            raise ValueError("Lambda protocol is not frozen")
        if str(lambda_protocol.get("pca_artifact_sha256")) != sha256_file(
            args.pca_artifact
        ):
            raise ValueError("Lambda protocol was calibrated for another PCA artifact")
        auxiliary_weight = float(lambda_protocol["lambda"])
        if not math.isfinite(auxiliary_weight) or auxiliary_weight <= 0:
            raise ValueError("Invalid frozen lambda")
        if args.floor_protocol is None or args.matched_control is None:
            raise ValueError("Auxiliary arm requires frozen floor and matched control")
        if args.aux_head_audit_dir is None:
            raise ValueError("Auxiliary arm requires an external aux-head audit directory")
        floor_protocol = json.loads(args.floor_protocol.read_text(encoding="utf-8"))
        if not floor_protocol.get("frozen_before_proposed"):
            raise ValueError("Effect-floor protocol is not frozen before proposed")
        control_metadata = json.loads(
            (args.matched_control / "best.json").read_text(encoding="utf-8")
        )
        if control_metadata.get("arm") != "control" or int(
            control_metadata.get("seed") or -1
        ) != args.seed:
            raise ValueError("Matched control arm/seed mismatch")
        frozen_control = floor_protocol.get("control_gaps", {}).get(str(args.seed), {})
        if frozen_control.get("best_sha256") != sha256_file(
            args.matched_control / "best.json"
        ):
            raise ValueError("Matched control is not the checkpoint frozen in the floor")
        if selected_direct_sha256 != control_metadata.get(
            "selected_direct_id_sha256"
        ) or selected_d9a_sha256 != control_metadata.get("selected_d9a_id_sha256"):
            raise ValueError("Proposed row order differs from matched control")
    else:
        lambda_protocol = None
        auxiliary_weight = 0.0
        floor_protocol = None
        control_metadata = None

    model, tokenizer, sidecar, actor_prompt = load_training_stack(
        args.config, seed=args.seed, actor_prompt_file=args.actor_prompt_template_file
    )
    projector, pca_metadata = load_bottleneck(
        args.pca_artifact, device=model.device, require_gate_passed=True
    )
    if projector.d_z != 256 or projector.d_model != 3840:
        raise ValueError("D16 projector dimensions changed")
    head = None
    if args.arm == "auxiliary":
        head = initialize_auxiliary_head(projector.d_z, len(labels)).to(model.device)
        head.float()
        initial_head_sha256 = sha256_state_dict(head.state_dict())
        if initial_head_sha256 != str(lambda_protocol.get("aux_head_initial_sha256")):
            raise ValueError("Auxiliary-head initialization differs from lambda calibration")
    else:
        initial_head_sha256 = None

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    trainable.extend(projector.parameters())
    if head is not None:
        trainable.extend(head.parameters())
    for parameter in trainable:
        parameter.data = parameter.data.float()
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    model.train()
    projector.train()
    if head is not None:
        head.train()
    embed_layer = model.get_input_embeddings()
    label_index = {label: index for index, label in enumerate(labels)}

    args.out_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, args.out_dir / "train.config.yaml")
    metrics_path = args.out_dir / "metrics.jsonl"
    for step in range(args.max_steps):
        direct_batch = direct_rows[step * args.batch_size : (step + 1) * args.batch_size]
        pair_batch = pairs[step * args.batch_size : (step + 1) * args.batch_size]
        optimizer.zero_grad(set_to_none=True)
        language_losses = []
        for start in range(0, args.batch_size, args.microbatch_size):
            micro_rows = direct_batch[start : start + args.microbatch_size]
            examples = [
                build_training_example(
                    row=row,
                    tokenizer=tokenizer,
                    model=model,
                    embed_layer=embed_layer,
                    sidecar=sidecar,
                    actor_prompt_template=actor_prompt,
                    eos_token_id=tokenizer.eos_token_id,
                    activation_transform=lambda value: projector(
                        value.to(model.device)
                    )[0],
                )
                for row in micro_rows
            ]
            inputs_embeds, attention_mask, labels_tensor, _content = collate_examples(examples)
            language_loss = row_target_nlls(
                model=model,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels_tensor,
            ).mean()
            (language_loss / (args.batch_size / args.microbatch_size)).backward()
            language_losses.append(float(language_loss.detach()))

        if head is not None:
            original_bce, deleted_margin, changed_delta = auxiliary_losses(
                rows=pair_batch,
                projector=projector,
                head=head,
                teacher=teacher,
                label_index=label_index,
            )
            auxiliary_loss = original_bce + deleted_margin
            (auxiliary_weight * auxiliary_loss).backward()
            bce_value = float(original_bce.detach())
            margin_value = float(deleted_margin.detach())
            delta_value = float(changed_delta.mean())
        else:
            with torch.no_grad():
                original = load_activation_batch(
                    pair_batch, "original_activation_path", projector.mean.device
                )
                deleted = load_activation_batch(
                    pair_batch, "deleted_activation_path", projector.mean.device
                )
                projector(original)
                projector(deleted)
            bce_value = margin_value = delta_value = 0.0

        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        metrics = {
            "step": step + 1,
            "arm": args.arm,
            "seed": args.seed,
            "language_loss": mean(language_losses),
            "original_soft_bce": bce_value,
            "deleted_pair_margin": margin_value,
            "changed_logit_delta": delta_value,
            "auxiliary_weight": auxiliary_weight,
            "direct_ids": [str(row["base_id"]) for row in direct_batch],
            "d9a_ids": [str(row["base_id"]) for row in pair_batch],
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics, sort_keys=True) + "\n")
        print(
            f"[train] {args.arm} seed={args.seed} step={step + 1}/20 "
            f"lang={metrics['language_loss']:.4f} bce={bce_value:.4f} "
            f"rank={margin_value:.4f} delta={delta_value:+.4f}",
            flush=True,
        )

    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)
    (args.out_dir / AV_PROMPT_FILENAME).write_text(actor_prompt, encoding="utf-8")
    bottleneck_path = args.out_dir / BOTTLENECK_FILENAME
    save_bottleneck(
        bottleneck_path,
        projector,
        metadata={
            **pca_metadata,
            "training_arm": args.arm,
            "training_seed": args.seed,
            "auxiliary_head_included": False,
            "auxiliary_head_registered_with_inference_model": False,
            "pca_artifact": str(args.pca_artifact),
            "pca_artifact_sha256": sha256_file(args.pca_artifact),
        },
    )
    final_head_sha256 = sha256_state_dict(head.state_dict()) if head is not None else None
    aux_head_audit_path = None
    if head is not None:
        args.aux_head_audit_dir.mkdir(parents=True, exist_ok=True)
        aux_head_audit_path = args.aux_head_audit_dir / f"{args.out_dir.name}.pt"
        if aux_head_audit_path.exists():
            raise FileExistsError(aux_head_audit_path)
        torch.save(
            {
                "training_only": True,
                "inference_connected": False,
                "finding_labels": labels,
                "state_dict": {
                    key: value.detach().cpu() for key, value in head.state_dict().items()
                },
            },
            aux_head_audit_path,
        )
    metadata = {
        "decision": "D16",
        "arm": args.arm,
        "seed": args.seed,
        "optimizer_steps": args.max_steps,
        "direct_rows": len(direct_rows),
        "d9a_pairs": len(pairs),
        "selected_direct_id_sha256": selected_direct_sha256,
        "selected_d9a_id_sha256": selected_d9a_sha256,
        "auxiliary_weight": auxiliary_weight,
        "initial_aux_head_sha256": initial_head_sha256,
        "final_training_only_aux_head_sha256": final_head_sha256,
        "external_aux_head_audit_artifact": (
            str(aux_head_audit_path) if aux_head_audit_path else None
        ),
        "external_aux_head_audit_artifact_sha256": (
            sha256_file(aux_head_audit_path) if aux_head_audit_path else None
        ),
        "bottleneck": str(bottleneck_path),
        "bottleneck_sha256": sha256_file(bottleneck_path),
        "auxiliary_head_removed_from_inference_checkpoint": True,
        "auxiliary_head_artifact_saved_outside_inference_checkpoint": head is not None,
        "teacher_scores_sha256": sha256_file(args.teacher_scores),
        "teacher_report_sha256": sha256_file(args.teacher_report),
        "d9a_pairs_sha256": sha256_file(args.d9a_pairs),
        "direct_train_sha256": sha256_file(args.direct_train),
        "lambda_protocol": str(args.lambda_protocol) if args.lambda_protocol else None,
        "lambda_protocol_sha256": (
            sha256_file(args.lambda_protocol) if args.lambda_protocol else None
        ),
        "floor_protocol": str(args.floor_protocol) if args.floor_protocol else None,
        "floor_protocol_sha256": (
            sha256_file(args.floor_protocol) if args.floor_protocol else None
        ),
        "matched_control": str(args.matched_control) if args.matched_control else None,
        "locked_test_read": False,
    }
    (args.out_dir / "best.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
