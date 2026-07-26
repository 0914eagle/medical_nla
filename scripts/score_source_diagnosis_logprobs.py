"""Score diagnosis candidate logprobs under the source model.

This is the no-NLA confidence baseline for later error-prediction experiments.
It asks whether Gemma/Qwen itself assigns high probability to the gold diagnosis
when prompted with the clinical case.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_config
from src.jsonl import append_jsonl, read_jsonl
from src.modeling import load_causal_lm, load_tokenizer


def normalize_candidate_id(text: str) -> str:
    return "_".join(text.strip().lower().replace("/", " ").split())


def collect_candidates(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    by_id: dict[str, str] = {}
    for row in rows:
        diagnosis_name = str(row.get("diagnosis_name") or "").strip()
        if not diagnosis_name:
            aliases = row.get("diagnosis_aliases") or []
            if aliases:
                diagnosis_name = str(aliases[0]).strip()
        if not diagnosis_name:
            continue
        diagnosis_id = str(row.get("diagnosis_id") or normalize_candidate_id(diagnosis_name))
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
        raise ValueError("No candidates found.")
    return candidates


def source_question(prompt: str) -> str:
    return (
        "Answer the clinical question with the most likely diagnosis or syndrome first.\n\n"
        f"Question: {prompt}"
    )


def chat_prefix_ids(tokenizer: Any, *, prompt: str, completion_prefix: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": source_question(prompt)}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        ids = encoded.detach().cpu().flatten().tolist()
    elif encoded and isinstance(encoded[0], list):
        ids = encoded[0]
    else:
        ids = list(encoded)
    prefix_ids = tokenizer.encode(completion_prefix, add_special_tokens=False)
    return ids + prefix_ids


def candidate_surface(prefix: str, diagnosis_name: str) -> str:
    if not prefix or prefix[-1].isspace():
        return diagnosis_name
    return " " + diagnosis_name


def tokenize_candidates(
    tokenizer: Any,
    *,
    completion_prefix: str,
    candidates: list[dict[str, str]],
) -> list[list[int]]:
    tokenized = []
    for candidate in candidates:
        ids = tokenizer.encode(
            candidate_surface(completion_prefix, candidate["diagnosis_name"]),
            add_special_tokens=False,
        )
        if not ids:
            raise ValueError(f"Candidate tokenized to no tokens: {candidate}")
        tokenized.append(ids)
    return tokenized


def score_candidates(
    *,
    model: torch.nn.Module,
    prefix_ids: list[int],
    candidate_ids: list[list[int]],
    pad_token_id: int,
    batch_size: int,
) -> list[dict[str, float | int]]:
    out_scores: list[dict[str, float | int]] = []
    device = model.device
    for start in range(0, len(candidate_ids), batch_size):
        batch = candidate_ids[start : start + batch_size]
        text_ids = [prefix_ids + ids for ids in batch]
        max_len = max(len(ids) for ids in text_ids)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for idx, ids in enumerate(text_ids):
            input_ids[idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention[idx, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        prefix_len = len(prefix_ids)
        for row_idx, ids in enumerate(batch):
            total = 0.0
            token_logprobs = []
            for offset, token_id in enumerate(ids):
                token_index = prefix_len + offset
                logits_index = token_index - 1
                lp = float(log_probs[row_idx, logits_index, int(token_id)].item())
                total += lp
                token_logprobs.append(lp)
            out_scores.append(
                {
                    "logprob_sum": total,
                    "logprob_mean": total / max(len(ids), 1),
                    "num_tokens": len(ids),
                    "first_token_logprob": token_logprobs[0],
                }
            )
    return out_scores


def add_calibrated_scores(
    scores: list[dict[str, float | int]],
    baseline_scores: list[dict[str, float | int]],
) -> None:
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
        score["calibrated_first_token_logprob"] = float(score["first_token_logprob"]) - float(
            baseline["first_token_logprob"]
        )


def rank_candidates(
    candidates: list[dict[str, str]],
    scores: list[dict[str, float | int]],
    *,
    rank_field: str,
) -> list[dict[str, Any]]:
    ranked = [{**candidate, **score} for candidate, score in zip(candidates, scores, strict=True)]
    ranked.sort(key=lambda row: float(row[rank_field]), reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    return ranked


def summarize(rows: list[dict[str, Any]], path: Path, *, rank_field: str) -> None:
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant") or "all")].append(row)

    def rate(items: list[dict[str, Any]], predicate) -> float:
        return sum(1 for row in items if predicate(row)) / max(len(items), 1)

    def mrr(items: list[dict[str, Any]]) -> float:
        return sum(1.0 / int(row["gold_rank"]) for row in items) / max(len(items), 1)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Diagnosis Logprob Baseline\n\n")
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
                f"{rate(items, lambda row: row['gold_rank'] == 1):.4f} | "
                f"{rate(items, lambda row: row['gold_rank'] <= 5):.4f} | "
                f"{mrr(items):.4f} | {mean_rank:.2f} |\n"
            )
        f.write("\n## Top Predicted Diagnoses\n\n")
        f.write("| diagnosis_id | count |\n")
        f.write("|---|---:|\n")
        for diagnosis_id, count in Counter(str(row["top1_diagnosis_id"]) for row in rows).most_common(20):
            f.write(f"| {diagnosis_id} | {count} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True, help="JSONL rows with prompt and diagnosis_id/name.")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--candidates-jsonl", default=None)
    parser.add_argument("--completion-prefix", default="The most likely diagnosis is")
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
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k-output", type=int, default=10)
    parser.add_argument(
        "--calibration-prompt",
        default=None,
        help=(
            "Optional neutral prompt for subtracting candidate-name and generic "
            "diagnosis priors. Required when rank-field starts with calibrated_."
        ),
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    rows = [row for row in rows if row.get("prompt") and row.get("diagnosis_id")]
    if not rows:
        raise ValueError("No input rows with prompt and diagnosis_id found.")
    candidates = read_candidates(args.candidates_jsonl, rows)
    candidate_by_id = {row["diagnosis_id"]: row for row in candidates}

    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    tokenized_candidates = tokenize_candidates(
        tokenizer,
        completion_prefix=args.completion_prefix,
        candidates=candidates,
    )
    baseline_scores = None
    if args.rank_field.startswith("calibrated_"):
        if not args.calibration_prompt:
            raise ValueError("--calibration-prompt is required for calibrated rank fields.")
        print("[calibration] scoring neutral source-model candidate priors", flush=True)
        baseline_prefix_ids = chat_prefix_ids(
            tokenizer,
            prompt=args.calibration_prompt,
            completion_prefix=args.completion_prefix,
        )
        baseline_scores = score_candidates(
            model=model,
            prefix_ids=baseline_prefix_ids,
            candidate_ids=tokenized_candidates,
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=args.candidate_batch_size,
        )

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        gold_id = str(row["diagnosis_id"])
        if gold_id not in candidate_by_id:
            raise ValueError(f"Gold diagnosis_id {gold_id!r} is not in candidate set.")
        prefix_ids = chat_prefix_ids(
            tokenizer,
            prompt=str(row["prompt"]),
            completion_prefix=args.completion_prefix,
        )
        scores = score_candidates(
            model=model,
            prefix_ids=prefix_ids,
            candidate_ids=tokenized_candidates,
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=args.candidate_batch_size,
        )
        if baseline_scores is not None:
            add_calibrated_scores(scores, baseline_scores)
        ranked = rank_candidates(candidates, scores, rank_field=args.rank_field)
        gold = next(item for item in ranked if item["diagnosis_id"] == gold_id)
        top = ranked[0]
        out = {
            "id": row["id"],
            "base_id": row.get("base_id", row["id"]),
            "prompt": row["prompt"],
            "variant": row.get("variant"),
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
            "top1_diagnosis_id": top["diagnosis_id"],
            "top1_diagnosis_name": top["diagnosis_name"],
            "top1_logprob_mean": top["logprob_mean"],
            "top1_logprob_sum": top["logprob_sum"],
            "top_candidates": ranked[: args.top_k_output],
        }
        append_jsonl(output_path, out)
        out_rows.append(out)
        print(
            f"[source-logprob] {idx}/{len(rows)} {row['id']} "
            f"gold_rank={gold['rank']} top1={top['diagnosis_id']}",
            flush=True,
        )
    summarize(out_rows, summary_path, rank_field=args.rank_field)
    print(f"[done] wrote {len(out_rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
