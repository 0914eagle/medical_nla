"""Score option-completion logprobs for source-model MC prompts.

This is a stricter source-confidence baseline than diagnosis-name logprobs. It
uses the exact multiple-choice prompt already shown to the source model and
scores completions such as `17. Pneumonia` for every listed option.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
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


OPTION_RE = re.compile(r"^(\d{1,3})\.\s*(.+?)\s*$")


def normalize_candidate_id(text: str) -> str:
    return "_".join(text.strip().lower().replace("/", " ").split())


def normalize_name(text: Any) -> str:
    text = str(text or "").lower().replace("/", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def candidate_id_by_name(rows: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in rows:
        diagnosis_id = str(row.get("gold_diagnosis_id") or row.get("diagnosis_id") or "")
        diagnosis_name = str(row.get("gold_diagnosis_name") or row.get("diagnosis_name") or "")
        if diagnosis_id and diagnosis_name:
            out.setdefault(normalize_name(diagnosis_name), diagnosis_id)
    return out


def parse_options(mc_prompt: str) -> list[dict[str, Any]]:
    marker = "Diagnosis options:"
    if marker not in mc_prompt:
        raise ValueError("MC prompt lacks 'Diagnosis options:' marker.")
    options_text = mc_prompt.split(marker, 1)[1]
    options = []
    for line in options_text.splitlines():
        match = OPTION_RE.match(line.strip())
        if not match:
            continue
        index = int(match.group(1))
        name = match.group(2).strip()
        options.append({"index": index, "diagnosis_name": name})
    if not options:
        raise ValueError("No numbered options parsed from MC prompt.")
    return options


def chat_prefix_ids(tokenizer: Any, prompt: str) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(encoded, "input_ids"):
        encoded = encoded.input_ids
    elif isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    if isinstance(encoded, torch.Tensor):
        return encoded.detach().cpu().flatten().tolist()
    if encoded and isinstance(encoded[0], list):
        return list(encoded[0])
    return list(encoded)


def option_surface(option: dict[str, Any], answer_style: str) -> str:
    if answer_style == "number":
        return str(option["index"])
    if answer_style == "number_and_name":
        return f"{option['index']}. {option['diagnosis_name']}"
    raise ValueError(f"Unsupported answer_style: {answer_style}")


def tokenize_options(
    tokenizer: Any,
    *,
    options: list[dict[str, Any]],
    answer_style: str,
) -> list[list[int]]:
    tokenized = []
    for option in options:
        surface = option_surface(option, answer_style)
        ids = tokenizer.encode(surface, add_special_tokens=False)
        if not ids:
            raise ValueError(f"Option tokenized to no tokens: {surface}")
        tokenized.append(ids)
    return tokenized


def score_option_ids(
    *,
    model: torch.nn.Module,
    prefix_ids: list[int],
    option_ids: list[list[int]],
    pad_token_id: int,
    batch_size: int,
) -> list[dict[str, float | int]]:
    scores = []
    device = model.device
    for start in range(0, len(option_ids), batch_size):
        batch = option_ids[start : start + batch_size]
        text_ids = [prefix_ids + ids for ids in batch]
        max_len = max(len(ids) for ids in text_ids)
        input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long, device=device)
        attention = torch.zeros((len(batch), max_len), dtype=torch.long, device=device)
        for row_idx, ids in enumerate(text_ids):
            input_ids[row_idx, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
            attention[row_idx, : len(ids)] = 1
        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
        log_probs = F.log_softmax(logits, dim=-1)
        prefix_len = len(prefix_ids)
        for row_idx, ids in enumerate(batch):
            token_logprobs = []
            total = 0.0
            for offset, token_id in enumerate(ids):
                token_index = prefix_len + offset
                logits_index = token_index - 1
                lp = float(log_probs[row_idx, logits_index, int(token_id)].item())
                token_logprobs.append(lp)
                total += lp
            scores.append(
                {
                    "logprob_sum": total,
                    "logprob_mean": total / max(len(ids), 1),
                    "first_token_logprob": token_logprobs[0],
                    "num_tokens": len(ids),
                }
            )
    return scores


def rank_options(
    options: list[dict[str, Any]],
    scores: list[dict[str, float | int]],
    *,
    rank_field: str,
    id_by_name: dict[str, str],
) -> list[dict[str, Any]]:
    ranked = []
    for option, score in zip(options, scores):
        name = str(option["diagnosis_name"])
        ranked.append(
            {
                **option,
                "diagnosis_id": id_by_name.get(normalize_name(name), normalize_candidate_id(name)),
                **score,
            }
        )
    ranked.sort(key=lambda row: float(row[rank_field]), reverse=True)
    values = torch.tensor([float(row[rank_field]) for row in ranked], dtype=torch.float32)
    probs = torch.softmax(values, dim=0)
    for rank, (row, prob) in enumerate(zip(ranked, probs), start=1):
        row["rank"] = rank
        row["prob"] = float(prob.item())
    return ranked


def add_distribution_features(out: dict[str, Any], ranked: list[dict[str, Any]], rank_field: str) -> None:
    entropy = -sum(float(row["prob"]) * math.log(max(float(row["prob"]), 1e-30)) for row in ranked)
    max_entropy = math.log(max(len(ranked), 1))
    out["top1_diagnosis_id"] = ranked[0]["diagnosis_id"]
    out["top1_diagnosis_name"] = ranked[0]["diagnosis_name"]
    out["top1_score"] = float(ranked[0][rank_field])
    out["top1_prob"] = float(ranked[0]["prob"])
    out["top2_diagnosis_id"] = ranked[1]["diagnosis_id"] if len(ranked) > 1 else None
    out["top2_diagnosis_name"] = ranked[1]["diagnosis_name"] if len(ranked) > 1 else None
    out["top2_score"] = float(ranked[1][rank_field]) if len(ranked) > 1 else None
    out["top2_prob"] = float(ranked[1]["prob"]) if len(ranked) > 1 else None
    out["top1_top2_margin"] = (
        float(ranked[0][rank_field]) - float(ranked[1][rank_field]) if len(ranked) > 1 else None
    )
    out["top1_top2_prob_margin"] = (
        float(ranked[0]["prob"]) - float(ranked[1]["prob"]) if len(ranked) > 1 else None
    )
    out["candidate_entropy"] = entropy
    out["candidate_entropy_norm"] = entropy / max_entropy if max_entropy else 0.0


def summarize(rows: list[dict[str, Any]], path: Path, *, rank_field: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source MC Option Logprob Confidence\n\n")
        f.write(f"- n: {len(rows)}\n")
        f.write(f"- rank_field: `{rank_field}`\n")
        f.write(f"- top1_matches_generated_selection: {sum(row['top1_matches_generated_selection'] for row in rows)}/{len(rows)}\n")
        f.write(f"- top1_matches_gold: {sum(row['top1_matches_gold'] for row in rows)}/{len(rows)}\n")
        f.write(f"- generated_selection_matches_gold: {sum(row['generated_selection_matches_gold'] for row in rows)}/{len(rows)}\n")
        f.write(f"- mean_generated_selection_rank: {sum(int(row['generated_selection_rank'] or 0) for row in rows) / max(len(rows), 1):.2f}\n\n")
        f.write("## Top Logprob Option Diagnoses\n\n")
        f.write("| diagnosis_id | count |\n")
        f.write("|---|---:|\n")
        for diagnosis_id, count in Counter(str(row["top1_diagnosis_id"]) for row in rows).most_common(25):
            f.write(f"| {diagnosis_id} | {count} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source-mc", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--rank-field", choices=["logprob_mean", "logprob_sum", "first_token_logprob"], default="logprob_mean")
    parser.add_argument("--answer-style", choices=["number", "number_and_name"], default=None)
    parser.add_argument("--candidate-batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--top-k-output", type=int, default=10)
    args = parser.parse_args()

    rows = list(read_jsonl(args.source_mc))
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError("No source MC rows found.")

    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()

    id_by_name = candidate_id_by_name(rows)
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    out_rows = []
    for idx, row in enumerate(rows, start=1):
        mc_prompt = str(row.get("mc_prompt") or "")
        options = parse_options(mc_prompt)
        answer_style = args.answer_style or str(row.get("answer_style") or "number_and_name")
        prefix_ids = chat_prefix_ids(tokenizer, mc_prompt)
        option_ids = tokenize_options(tokenizer, options=options, answer_style=answer_style)
        scores = score_option_ids(
            model=model,
            prefix_ids=prefix_ids,
            option_ids=option_ids,
            pad_token_id=int(tokenizer.pad_token_id),
            batch_size=args.candidate_batch_size,
        )
        ranked = rank_options(options, scores, rank_field=args.rank_field, id_by_name=id_by_name)
        gold_id = str(row.get("gold_diagnosis_id") or "")
        selected_id = row.get("selected_diagnosis_id")
        generated_selection = next(
            (item for item in ranked if item["diagnosis_id"] == selected_id),
            None,
        )
        gold = next((item for item in ranked if item["diagnosis_id"] == gold_id), None)
        out = {
            "id": row["id"],
            "base_id": row.get("base_id", row["id"]),
            "prompt": row.get("prompt"),
            "rank_field": args.rank_field,
            "answer_style": answer_style,
            "gold_diagnosis_id": gold_id,
            "gold_diagnosis_name": row.get("gold_diagnosis_name"),
            "selected_diagnosis_id": selected_id,
            "selected_diagnosis_name": row.get("selected_diagnosis_name"),
            "generated_selection_rank": generated_selection["rank"] if generated_selection else None,
            "generated_selection_prob": generated_selection["prob"] if generated_selection else None,
            "generated_selection_score": generated_selection[args.rank_field] if generated_selection else None,
            "gold_rank": gold["rank"] if gold else None,
            "gold_prob": gold["prob"] if gold else None,
            "gold_score": gold[args.rank_field] if gold else None,
            "top_candidates": ranked[: args.top_k_output],
            "top1_matches_generated_selection": ranked[0]["diagnosis_id"] == selected_id,
            "top1_matches_gold": ranked[0]["diagnosis_id"] == gold_id,
            "generated_selection_matches_gold": bool(row.get("diagnosis_hit")),
            "num_candidates": len(ranked),
        }
        add_distribution_features(out, ranked, args.rank_field)
        append_jsonl(output_path, out)
        out_rows.append(out)
        print(
            f"[source-mc-logprob] {idx}/{len(rows)} {row['id']} "
            f"generated_rank={out['generated_selection_rank']} top1={out['top1_diagnosis_id']}",
            flush=True,
        )
    summarize(out_rows, summary_path, rank_field=args.rank_field)
    print(f"[done] wrote {len(out_rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
