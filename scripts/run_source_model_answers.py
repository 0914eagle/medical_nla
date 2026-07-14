"""Generate source-model answers for specificity full prompts.

This baseline separates source-model diagnostic ability from NLA readout
ability. It asks Gemma directly for the answer to each `specific_prompt` and
scores the answer with the same lexical probes used for NLA specificity output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from scripts.score_specificity_outputs import SPECIFIC_ALIASES, contains_term
from src.config import ensure_dir, load_config
from src.modeling import load_causal_lm, load_tokenizer


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def generation_kwargs(cfg: dict, max_new_tokens: int | None) -> dict:
    gen = dict(cfg["generation"])
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return {k: v for k, v in gen.items() if v is not None}


def chat_inputs(tokenizer, prompt: str, device) -> dict[str, torch.Tensor]:
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    return {k: v.to(device) for k, v in encoded.items()}


def alias_hits(text: str, aliases: list[str]) -> list[str]:
    return [alias for alias in aliases if contains_term(text, alias)]


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hits = [bool(row["answer_hit"]) for row in rows]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Model Answer Baseline\n\n")
        f.write("Lexical screening only; not claim-level clinical grading.\n\n")
        f.write(f"- n: {len(rows)}\n")
        f.write(f"- answer_hit: {sum(hits)}/{len(rows)}\n")
        f.write(f"- answer_hit_rate: {mean(hits):.3f}\n\n")
        f.write("| case | expected | hit | hits | answer |\n")
        f.write("|---|---|---:|---|---|\n")
        for row in rows:
            answer = row["answer"].replace("\n", " ")
            if len(answer) > 220:
                answer = answer[:217] + "..."
            f.write(
                f"| {row['id']} | {row['specific_expected']} | "
                f"{'Y' if row['answer_hit'] else 'N'} | "
                f"{', '.join(row['answer_hits']) or '-'} | "
                f"{answer} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--input", default="data/prompts_medical_specificity_cases_v1.jsonl"
    )
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
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

    out_rows = []
    for case in read_jsonl(Path(args.input)):
        prompt = case["specific_prompt"]
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
        aliases = SPECIFIC_ALIASES.get(case["id"], [])
        hits = alias_hits(answer, aliases)
        out_rows.append(
            {
                "id": case["id"],
                "prompt": prompt,
                "specific_expected": case["specific_expected"],
                "diagnostic_shift": case["diagnostic_shift"],
                "answer": answer,
                "answer_aliases": aliases,
                "answer_hits": hits,
                "answer_hit": bool(hits),
                "gen_config": gen_kwargs,
            }
        )

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    ensure_dir(output_path.parent)
    write_jsonl(output_path, out_rows)
    write_summary(summary_path, out_rows)
    print(f"wrote {len(out_rows)} source answers to {output_path}")
    print(f"wrote summary to {summary_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
