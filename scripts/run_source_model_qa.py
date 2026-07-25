"""Generate and score source-model answers for generic diagnosis QA rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_specificity_outputs import contains_term
from src.config import load_config
from src.jsonl import append_jsonl, read_jsonl
from src.modeling import load_causal_lm, load_tokenizer


def diagnosis_name(row: dict[str, Any]) -> str:
    if row.get("diagnosis_name"):
        return str(row["diagnosis_name"])
    aliases = row.get("diagnosis_aliases") or []
    if aliases:
        return str(aliases[0])
    return str(row.get("diagnosis_id", ""))


def aliases(row: dict[str, Any]) -> list[str]:
    values = row.get("diagnosis_aliases") or []
    if isinstance(values, str):
        values = [values]
    if not values and diagnosis_name(row):
        values = [diagnosis_name(row)]
    return [str(value) for value in values if str(value).strip()]


def answer_prompt(prompt: str) -> str:
    return (
        "Answer the clinical question with the most likely diagnosis or syndrome "
        "first, in one short sentence. Then give one brief reason.\n\n"
        f"Question: {prompt}"
    )


def chat_inputs(tokenizer: Any, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
    encoded = tokenizer.apply_chat_template(
        [{"role": "user", "content": answer_prompt(prompt)}],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {key: value.to(device) for key, value in encoded.items()}


def generation_kwargs(cfg: dict[str, Any], max_new_tokens: int | None) -> dict[str, Any]:
    gen = dict(cfg["generation"])
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return {key: value for key, value in gen.items() if value is not None}


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    hits = [bool(row["diagnosis_hit"]) for row in rows]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Model QA Answers\n\n")
        f.write("Lexical diagnosis-alias scoring only.\n\n")
        f.write(f"- n: {len(rows)}\n")
        f.write(f"- diagnosis_hit: {sum(hits)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit_rate: {mean(hits):.4f}\n\n")
        f.write("| id | expected | hit | hits | answer |\n")
        f.write("|---|---|---:|---|---|\n")
        for row in rows[:200]:
            answer = str(row["answer"]).replace("\n", " ")
            if len(answer) > 220:
                answer = answer[:217] + "..."
            f.write(
                f"| {row['id']} | {row['diagnosis_name']} | "
                f"{'Y' if row['diagnosis_hit'] else 'N'} | "
                f"{', '.join(row['diagnosis_hits']) or '-'} | {answer} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

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
    gen_kwargs = generation_kwargs(cfg, args.max_new_tokens)

    rows = list(read_jsonl(args.input))
    if args.limit is not None:
        rows = rows[: args.limit]
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()
    out_rows = []
    for idx, row in enumerate(rows, start=1):
        prompt = str(row.get(args.prompt_field) or "")
        if not prompt:
            continue
        encoded = chat_inputs(tokenizer, prompt, model.device)
        input_len = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        answer_ids = generated[0, input_len:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        row_aliases = aliases(row)
        hits = [alias for alias in row_aliases if contains_term(answer, alias)]
        out = {
            "id": row["id"],
            "base_id": row.get("base_id", row["id"]),
            "prompt": prompt,
            "diagnosis_id": row.get("diagnosis_id"),
            "diagnosis_name": diagnosis_name(row),
            "diagnosis_aliases": row_aliases,
            "answer": answer,
            "diagnosis_hits": hits,
            "diagnosis_hit": bool(hits),
            "gen_config": gen_kwargs,
            "variant": row.get("variant"),
            "source": row.get("source"),
            "patient_id": row.get("patient_id"),
        }
        append_jsonl(output_path, out)
        out_rows.append(out)
        print(f"[source-answer] {idx}/{len(rows)} {row['id']} hit={bool(hits)}", flush=True)
    write_summary(summary_path, out_rows)
    print(f"[done] wrote {len(out_rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
