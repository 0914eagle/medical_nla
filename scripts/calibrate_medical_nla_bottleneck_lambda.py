"""Freeze the one-shot D16 auxiliary weight by row-gradient RMS parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    build_training_example_from_activation,
    collate_examples,
    row_target_nlls,
)
from scripts.train_medical_nla_soft_bottleneck import (
    initialize_auxiliary_head,
    load_activation_batch,
    load_teacher,
    load_training_stack,
    parse_path_maps,
    remap_rows,
    validate_rows,
)
from src.jsonl import read_jsonl
from src.nla_bottleneck import load_bottleneck, sha256_file, sha256_state_dict


def calibration_order(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            hashlib.sha256(str(row.get("base_id") or row.get("id") or "").encode()).hexdigest(),
            str(row.get("base_id") or row.get("id") or ""),
        ),
    )


def hash_ids(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(str(row.get("base_id") or row.get("id")) for row in rows)
    return hashlib.sha256(payload.encode()).hexdigest()


def round_significant(value: float, digits: int = 2) -> float:
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Cannot round nonpositive gradient ratio: {value}")
    return round(value, digits - int(math.floor(math.log10(abs(value)))) - 1)


def language_gradient_norms(
    *,
    rows: list[dict[str, Any]],
    projector: Any,
    model: Any,
    tokenizer: Any,
    sidecar: Any,
    actor_prompt: str,
    batch_size: int,
) -> list[float]:
    embed_layer = model.get_input_embeddings()
    norms: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        activations = load_activation_batch(batch_rows, "activation_path", model.device)
        with torch.no_grad():
            latent = projector.encode(activations)
        latent = latent.detach().requires_grad_(True)
        reconstructed = projector.reconstruct_from_latent(latent)
        examples = [
            build_training_example_from_activation(
                row=row,
                activation=reconstructed[index],
                tokenizer=tokenizer,
                model=model,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt,
                eos_token_id=tokenizer.eos_token_id,
            )
            for index, row in enumerate(batch_rows)
        ]
        inputs_embeds, attention_mask, labels, _content = collate_examples(examples)
        losses = row_target_nlls(
            model=model,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        # Rows do not attend across the batch dimension, so d(sum_i L_i)/dz_i
        # is exactly each row's own gradient and needs only one backward pass.
        gradients = torch.autograd.grad(losses.sum(), latent)[0]
        norms.extend(gradients.float().norm(dim=1).detach().cpu().tolist())
        print(f"[lambda] language {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return norms


def auxiliary_gradient_norms(
    *,
    rows: list[dict[str, Any]],
    projector: Any,
    head: torch.nn.Linear,
    teacher: dict[str, list[float]],
    label_index: dict[str, int],
    batch_size: int,
) -> list[float]:
    norms: list[float] = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        original = load_activation_batch(
            batch_rows, "original_activation_path", projector.mean.device
        )
        deleted = load_activation_batch(
            batch_rows, "deleted_activation_path", projector.mean.device
        )
        with torch.no_grad():
            z_original = projector.encode(original)
            z_deleted = projector.encode(deleted)
        z_original = z_original.detach().requires_grad_(True)
        z_deleted = z_deleted.detach().requires_grad_(True)
        original_logits = head(z_original)
        deleted_logits = head(z_deleted)
        targets = torch.tensor(
            [teacher[str(row["base_id"])] for row in batch_rows],
            dtype=original_logits.dtype,
            device=original_logits.device,
        )
        per_row_bce = F.binary_cross_entropy_with_logits(
            original_logits, targets, reduction="none"
        ).mean(dim=1)
        columns = torch.tensor(
            [label_index[str(row["changed_evidence_id"])] for row in batch_rows],
            dtype=torch.long,
            device=original_logits.device,
        )
        indices = torch.arange(len(batch_rows), device=original_logits.device)
        per_row_rank = F.softplus(
            -(original_logits[indices, columns] - deleted_logits[indices, columns])
        )
        losses = per_row_bce + per_row_rank
        grad_original, grad_deleted = torch.autograd.grad(
            losses.sum(), (z_original, z_deleted)
        )
        squared = grad_original.float().square().sum(dim=1)
        squared += grad_deleted.float().square().sum(dim=1)
        norms.extend(squared.sqrt().detach().cpu().tolist())
        print(f"[lambda] auxiliary {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return norms


def rms(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot compute RMS of an empty population")
    return math.sqrt(mean(value * value for value in values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--direct-train", required=True, type=Path)
    parser.add_argument("--d9a-pairs", required=True, type=Path)
    parser.add_argument("--teacher-scores", required=True, type=Path)
    parser.add_argument("--teacher-report", required=True, type=Path)
    parser.add_argument("--pca-artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--path-map", action="append", default=[])
    parser.add_argument(
        "--actor-prompt-template-file",
        default=str(REPO_ROOT / "prompt_templates" / "common_p0_clinical_state_readout.txt"),
    )
    args = parser.parse_args()
    if args.batch_size != 4:
        raise ValueError("D16 freezes lambda-calibration microbatch size at 4")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")

    mappings = parse_path_maps(args.path_map)
    direct = remap_rows(
        list(read_jsonl(args.direct_train)), mappings, ("activation_path",)
    )
    pairs = remap_rows(
        list(read_jsonl(args.d9a_pairs)),
        mappings,
        ("original_activation_path", "deleted_activation_path"),
    )
    labels, teacher, _teacher_report = load_teacher(args.teacher_scores, args.teacher_report)
    validate_rows(direct, pairs, teacher, labels)
    direct = calibration_order(direct)[:248]
    pairs = calibration_order(pairs)[:248]

    model, tokenizer, sidecar, actor_prompt = load_training_stack(
        args.config, seed=17, actor_prompt_file=args.actor_prompt_template_file
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.train()
    projector, _metadata = load_bottleneck(
        args.pca_artifact, device=model.device, require_gate_passed=True
    )
    for parameter in projector.parameters():
        parameter.requires_grad_(False)
    head = initialize_auxiliary_head(projector.d_z, len(labels)).to(model.device).float()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    head_sha256 = sha256_state_dict(head.state_dict())

    language_norms = language_gradient_norms(
        rows=direct,
        projector=projector,
        model=model,
        tokenizer=tokenizer,
        sidecar=sidecar,
        actor_prompt=actor_prompt,
        batch_size=args.batch_size,
    )
    auxiliary_norms = auxiliary_gradient_norms(
        rows=pairs,
        projector=projector,
        head=head,
        teacher=teacher,
        label_index={label: index for index, label in enumerate(labels)},
        batch_size=args.batch_size,
    )
    language_rms = rms(language_norms)
    auxiliary_rms = rms(auxiliary_norms)
    weight_unrounded = language_rms / auxiliary_rms
    weight = round_significant(weight_unrounded, 2)
    report = {
        "decision": "D16",
        "frozen_before_control": True,
        "calibration_seed": 17,
        "direct_rows": len(direct),
        "d9a_pairs": len(pairs),
        "direct_order": "SHA256(base_id)",
        "d9a_order": "SHA256(base_id)",
        "direct_id_sha256": hash_ids(direct),
        "d9a_id_sha256": hash_ids(pairs),
        "gradient_tensor": "z",
        "gradient_norm": "row_L2_then_RMS",
        "aux_pair_norm": "L2_concatenation_of_original_and_deleted_z_gradients",
        "language_reduction": "row_mean_target_token_NLL",
        "auxiliary_reduction": "row_mean_91_label_BCE_plus_selected_cue_softplus_T1",
        "language_gradient_rms": language_rms,
        "auxiliary_gradient_rms": auxiliary_rms,
        "lambda_unrounded": weight_unrounded,
        "lambda": weight,
        "significant_digits": 2,
        "aux_head_initial_sha256": head_sha256,
        "pca_artifact": str(args.pca_artifact),
        "pca_artifact_sha256": sha256_file(args.pca_artifact),
        "direct_train_sha256": sha256_file(args.direct_train),
        "d9a_pairs_sha256": sha256_file(args.d9a_pairs),
        "teacher_scores_sha256": sha256_file(args.teacher_scores),
        "teacher_report_sha256": sha256_file(args.teacher_report),
        "validation_read": False,
        "locked_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = args.output.with_name("lambda_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# D16 Gradient-Parity Lambda",
                "",
                "- calibration seed: **17**",
                "- Direct / D9a rows: **248 / 248**",
                f"- language dL/dz RMS: **{language_rms:.8f}**",
                f"- auxiliary dL/dz RMS: **{auxiliary_rms:.8f}**",
                f"- unrounded ratio: **{weight_unrounded:.8f}**",
                f"- frozen lambda (2 significant digits): **{weight:g}**",
                f"- initial auxiliary-head SHA256: `{head_sha256}`",
                "- validation / locked test read: **no / no**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
