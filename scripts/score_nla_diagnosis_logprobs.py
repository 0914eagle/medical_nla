"""Score diagnosis candidate logprobs under the injected NLA AV model.

This probes whether the AV assigns high probability to the gold diagnosis even
when free generation collapses to generic format descriptions.
"""

from __future__ import annotations

import argparse
import math
import shutil
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import ensure_dir, load_config
from src.jsonl import append_jsonl, read_jsonl
from src.modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter
from src.nla import build_nla_inputs_embeds, build_nla_prompt, load_nla_sidecar
from src.run_nla import actor_prompt_template_with_suffix, read_actor_prompt_template


PASSTHROUGH_FIELDS = [
    "base_id",
    "prompt",
    "variant",
    "source",
    "patient_id",
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "target_role",
    "target_text",
    "position",
    "position_family",
    "position_mode",
    "layer",
    "cue_targets",
    "cue_types",
    "cue_evidence_ids",
    "notes",
]


def normalize_candidate_id(text: str) -> str:
    return "_".join(text.strip().lower().replace("/", " ").split())


def collect_candidates(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_id: dict[str, str] = {}
    for row in rows:
        diagnosis_id = str(row.get("diagnosis_id") or "").strip()
        diagnosis_name = str(row.get("diagnosis_name") or "").strip()
        if not diagnosis_name:
            aliases = row.get("diagnosis_aliases") or []
            if aliases:
                diagnosis_name = str(aliases[0]).strip()
        if not diagnosis_name:
            continue
        if not diagnosis_id:
            diagnosis_id = normalize_candidate_id(diagnosis_name)
        by_id.setdefault(diagnosis_id, diagnosis_name)
    return [
        {"diagnosis_id": diagnosis_id, "diagnosis_name": diagnosis_name}
        for diagnosis_id, diagnosis_name in sorted(by_id.items())
    ]


def read_candidates(path: str | None, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    if path is None:
        candidates = collect_candidates(rows)
    else:
        candidates = []
        for row in read_jsonl(path):
            diagnosis_name = str(row.get("diagnosis_name") or row.get("name") or "").strip()
            if not diagnosis_name:
                raise ValueError(f"Candidate row lacks diagnosis_name/name: {row}")
            diagnosis_id = str(row.get("diagnosis_id") or normalize_candidate_id(diagnosis_name))
            candidates.append({"diagnosis_id": diagnosis_id, "diagnosis_name": diagnosis_name})
    if not candidates:
        raise ValueError("No diagnosis candidates found.")
    return candidates


def candidate_surface(prefix: str, diagnosis_name: str) -> str:
    if not prefix or prefix[-1].isspace():
        return diagnosis_name
    return " " + diagnosis_name


def tokenize_candidate(
    tokenizer: Any,
    *,
    completion_prefix: str,
    diagnosis_name: str,
) -> tuple[list[int], list[int]]:
    prefix_ids = tokenizer.encode(completion_prefix, add_special_tokens=False)
    candidate_ids = tokenizer.encode(
        candidate_surface(completion_prefix, diagnosis_name),
        add_special_tokens=False,
    )
    if not candidate_ids:
        raise ValueError(f"Candidate tokenized to no tokens: {diagnosis_name!r}")
    return prefix_ids, candidate_ids


def pad_text_ids(
    rows: list[tuple[list[int], list[int]]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int]]:
    text_ids = [prefix_ids + candidate_ids for prefix_ids, candidate_ids in rows]
    prefix_lens = [len(prefix_ids) for prefix_ids, _ in rows]
    candidate_lens = [len(candidate_ids) for _, candidate_ids in rows]
    max_len = max(len(ids) for ids in text_ids)
    ids = torch.full((len(text_ids), max_len), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros((len(text_ids), max_len), dtype=torch.long, device=device)
    for idx, token_ids in enumerate(text_ids):
        ids[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)
        attention[idx, : len(token_ids)] = 1
    return ids, attention, prefix_lens, candidate_lens


@torch.inference_mode()
def score_candidate_batch(
    *,
    model: torch.nn.Module,
    embed_layer: torch.nn.Module,
    prefix_embeds: torch.Tensor,
    prefix_attention_mask: torch.Tensor,
    tokenizer: Any,
    tokenized: list[tuple[list[int], list[int]]],
) -> list[dict[str, float | int]]:
    text_ids, text_attention, prefix_lens, candidate_lens = pad_text_ids(
        tokenized,
        pad_token_id=int(tokenizer.pad_token_id),
        device=prefix_embeds.device,
    )
    text_embeds = embed_layer(text_ids)
    batch_size = text_ids.shape[0]
    repeated_prefix = prefix_embeds.expand(batch_size, -1, -1)
    repeated_attention = prefix_attention_mask.expand(batch_size, -1)
    inputs_embeds = torch.cat([repeated_prefix, text_embeds], dim=1)
    attention_mask = torch.cat([repeated_attention, text_attention], dim=1)

    logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    prefix_len = prefix_embeds.shape[1]

    scores: list[dict[str, float | int]] = []
    for row_idx, (prefix_ids, candidate_ids) in enumerate(tokenized):
        del prefix_ids
        candidate_start = prefix_lens[row_idx]
        total = 0.0
        token_logprobs: list[float] = []
        for offset, token_id in enumerate(candidate_ids):
            text_index = candidate_start + offset
            logits_index = prefix_len + text_index - 1
            token_logprob = float(log_probs[row_idx, logits_index, int(token_id)].item())
            token_logprobs.append(token_logprob)
            total += token_logprob
        length = candidate_lens[row_idx]
        scores.append(
            {
                "logprob_sum": total,
                "logprob_mean": total / max(length, 1),
                "num_tokens": length,
                "first_token_logprob": token_logprobs[0],
            }
        )
    return scores


@torch.inference_mode()
def build_token_baseline_embeds(
    *,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    device: torch.device | str,
    actor_prompt_template: str | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the NLA prompt with the literal injection token embedding.

    This captures candidate-name language priors under the same AV prompt but
    without replacing the placeholder with an activation vector.
    """

    _, input_ids_list = build_nla_prompt(
        tokenizer,
        sidecar,
        actor_prompt_template=actor_prompt_template,
    )
    input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=device).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, device=device)
    return embed_layer(input_ids).clone(), attention_mask


def rank_candidates(
    candidates: list[dict[str, str]],
    scores: list[dict[str, float | int]],
    *,
    rank_field: str,
) -> list[dict[str, Any]]:
    ranked = []
    for candidate, score in zip(candidates, scores, strict=True):
        ranked.append({**candidate, **score})
    ranked.sort(key=lambda row: float(row[rank_field]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def summarize(rows: list[dict[str, Any]], output_path: Path, *, rank_field: str) -> None:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant") or "all")].append(row)

    def rate(items: list[dict[str, Any]], field: str) -> float:
        if not items:
            return math.nan
        return sum(bool(row[field]) for row in items) / len(items)

    def mrr(items: list[dict[str, Any]]) -> float:
        if not items:
            return math.nan
        return sum(1.0 / int(row["gold_rank"]) for row in items) / len(items)

    with output_path.open("w", encoding="utf-8") as f:
        f.write("# NLA Diagnosis Logprob Probe\n\n")
        f.write(f"- n: {len(rows)}\n")
        f.write(f"- rank_field: `{rank_field}`\n")
        f.write(f"- top1: {sum(row['gold_rank'] == 1 for row in rows)}/{len(rows)}\n")
        f.write(f"- top5: {sum(row['gold_rank'] <= 5 for row in rows)}/{len(rows)}\n")
        f.write(f"- mrr: {mrr(rows):.4f}\n\n")
        f.write("| variant | n | top1 | top5 | mrr | mean_gold_rank |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for variant, items in sorted(by_variant.items()):
            mean_rank = sum(int(row["gold_rank"]) for row in items) / len(items)
            f.write(
                f"| {variant} | {len(items)} | "
                f"{rate(items, 'gold_top1'):.4f} | {rate(items, 'gold_top5'):.4f} | "
                f"{mrr(items):.4f} | {mean_rank:.2f} |\n"
            )
        f.write("\n## Top Predicted Diagnoses\n\n")
        top_counts = Counter(str(row["top1_diagnosis_id"]) for row in rows)
        f.write("| diagnosis_id | count |\n")
        f.write("|---|---:|\n")
        for diagnosis_id, count in top_counts.most_common(20):
            f.write(f"| {diagnosis_id} | {count} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument(
        "--candidates-jsonl",
        default=None,
        help="Optional JSONL with diagnosis_id and diagnosis_name. Defaults to manifest diagnoses.",
    )
    parser.add_argument(
        "--completion-prefix",
        default="<explanation>The most likely diagnosis is",
        help="Fixed AV completion prefix before each candidate diagnosis.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=int,
        default=8,
        help="Number of candidate diagnoses scored per forward pass.",
    )
    parser.add_argument(
        "--rank-field",
        choices=[
            "logprob_mean",
            "logprob_sum",
            "first_token_logprob",
            "calibrated_logprob_mean",
            "calibrated_logprob_sum",
            "calibrated_first_token_logprob",
        ],
        default="logprob_mean",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k-output", type=int, default=10)
    parser.add_argument("--actor-prompt-template-file", default=None)
    parser.add_argument("--actor-prompt-suffix-file", default=None)
    parser.add_argument(
        "--adapter-id",
        default=None,
        help="Optional PEFT/LoRA adapter path or HF id for evaluating Medical-NLA.",
    )
    parser.add_argument(
        "--calibrate-with-token-baseline",
        action="store_true",
        help=(
            "Subtract candidate scores under the same NLA prompt with the literal "
            "injection token embedding. This removes candidate-name priors such as "
            "easy multi-token continuations."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg["paths"]
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    ensure_dir(output_path.parent)
    ensure_dir(summary_path.parent)
    if output_path.exists():
        output_path.unlink()
    shutil.copy2(args.config, output_path.parent / f"{output_path.stem}.config.yaml")

    rows = list(read_jsonl(args.manifest))
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = [row for row in rows if row.get("diagnosis_id") and row.get("activation_path")]
    if not rows:
        raise ValueError("No manifest rows with diagnosis_id and activation_path found.")
    candidates = read_candidates(args.candidates_jsonl, rows)
    candidate_by_id = {row["diagnosis_id"]: row for row in candidates}

    cache_dir = paths.get("cache_dir")
    nla_cfg = cfg["nla_model"]
    torch.manual_seed(int(cfg.get("seed", 17)))

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
    if actor_prompt_template is not None and args.actor_prompt_suffix_file is not None:
        raise ValueError("Use either --actor-prompt-template-file or --actor-prompt-suffix-file, not both.")
    if actor_prompt_template is None:
        actor_prompt_template = actor_prompt_template_with_suffix(
            sidecar.actor_prompt_template,
            args.actor_prompt_suffix_file,
        )

    model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
    adapter_id = args.adapter_id or nla_cfg.get("adapter_id")
    model = maybe_load_peft_adapter(model, adapter_id, cache_dir=cache_dir)
    model.eval()
    embed_layer = model.get_input_embeddings()
    tokenized_candidates = [
        tokenize_candidate(
            tokenizer,
            completion_prefix=args.completion_prefix,
            diagnosis_name=candidate["diagnosis_name"],
        )
        for candidate in candidates
    ]
    baseline_scores: list[dict[str, float | int]] | None = None
    if args.calibrate_with_token_baseline:
        print("[calibration] scoring literal-token baseline candidate priors", flush=True)
        baseline_embeds, baseline_attention = build_token_baseline_embeds(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            device=model.device,
            actor_prompt_template=actor_prompt_template,
        )
        baseline_scores = []
        for start in range(0, len(tokenized_candidates), args.candidate_batch_size):
            batch = tokenized_candidates[start : start + args.candidate_batch_size]
            baseline_scores.extend(
                score_candidate_batch(
                    model=model,
                    embed_layer=embed_layer,
                    prefix_embeds=baseline_embeds,
                    prefix_attention_mask=baseline_attention,
                    tokenizer=tokenizer,
                    tokenized=batch,
                )
            )

    result_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        gold_id = str(row["diagnosis_id"])
        if gold_id not in candidate_by_id:
            raise ValueError(f"Gold diagnosis_id {gold_id!r} is not in candidate set.")
        activation = torch.load(row["activation_path"], map_location="cpu")
        injected = build_nla_inputs_embeds(
            tokenizer=tokenizer,
            embed_layer=embed_layer,
            sidecar=sidecar,
            activation=activation,
            device=model.device,
            actor_prompt_template=actor_prompt_template,
        )

        scores: list[dict[str, float | int]] = []
        for start in range(0, len(tokenized_candidates), args.candidate_batch_size):
            batch = tokenized_candidates[start : start + args.candidate_batch_size]
            scores.extend(
                score_candidate_batch(
                    model=model,
                    embed_layer=embed_layer,
                    prefix_embeds=injected.inputs_embeds,
                    prefix_attention_mask=injected.attention_mask,
                    tokenizer=tokenizer,
                    tokenized=batch,
                )
            )
        if baseline_scores is not None:
            for score, baseline in zip(scores, baseline_scores, strict=True):
                score["baseline_logprob_sum"] = baseline["logprob_sum"]
                score["baseline_logprob_mean"] = baseline["logprob_mean"]
                score["baseline_first_token_logprob"] = baseline["first_token_logprob"]
                score["calibrated_logprob_sum"] = float(score["logprob_sum"]) - float(
                    baseline["logprob_sum"]
                )
                score["calibrated_logprob_mean"] = float(score["logprob_mean"]) - float(
                    baseline["logprob_mean"]
                )
                score["calibrated_first_token_logprob"] = float(
                    score["first_token_logprob"]
                ) - float(baseline["first_token_logprob"])
        ranked = rank_candidates(candidates, scores, rank_field=args.rank_field)
        gold = next(item for item in ranked if item["diagnosis_id"] == gold_id)
        top = ranked[0]
        out = {
            "id": row["id"],
            "gold_diagnosis_id": gold_id,
            "gold_diagnosis_name": row.get("diagnosis_name") or candidate_by_id[gold_id]["diagnosis_name"],
            "completion_prefix": args.completion_prefix,
            "rank_field": args.rank_field,
            "num_candidates": len(candidates),
            "gold_rank": gold["rank"],
            "gold_top1": gold["rank"] == 1,
            "gold_top5": gold["rank"] <= 5,
            "gold_logprob_sum": gold["logprob_sum"],
            "gold_logprob_mean": gold["logprob_mean"],
            "gold_first_token_logprob": gold["first_token_logprob"],
            "gold_num_tokens": gold["num_tokens"],
            "top1_diagnosis_id": top["diagnosis_id"],
            "top1_diagnosis_name": top["diagnosis_name"],
            "top1_logprob_mean": top["logprob_mean"],
            "top1_logprob_sum": top["logprob_sum"],
            "top_candidates": ranked[: args.top_k_output],
            "activation_norm": injected.activation_norm,
            "scaled_activation_norm": injected.scaled_activation_norm,
            "injection_position": injected.injection_position,
            "injection_scale": sidecar.injection_scale,
            "sidecar_path": sidecar.path,
            "adapter_id": adapter_id,
            "timestamp": datetime.now(UTC).isoformat(),
        }
        for field in PASSTHROUGH_FIELDS:
            if field in row:
                out[field] = row.get(field)
        append_jsonl(output_path, out)
        result_rows.append(out)
        print(
            f"[logprob] {idx}/{len(rows)} {row['id']} "
            f"gold_rank={gold['rank']} top1={top['diagnosis_id']}",
            flush=True,
        )

    summarize(result_rows, summary_path, rank_field=args.rank_field)
    print(f"[done] wrote {len(result_rows)} rows to {output_path}", flush=True)
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
