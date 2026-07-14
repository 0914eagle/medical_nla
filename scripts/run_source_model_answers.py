"""Generate source-model answers for specificity full prompts.

This baseline separates source-model diagnostic ability from NLA readout
ability. It asks Gemma directly for the answer to each `specific_prompt` and
scores the answer with the same lexical probes used for NLA specificity output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_specificity_outputs import SPECIFIC_ALIASES, contains_term
from src.config import ensure_dir, load_config
from src.modeling import load_causal_lm, load_tokenizer

SOURCE_DIAGNOSIS_ALIASES = {
    "shift_001": ["meningitis", "meningeal"],
    "shift_002": ["wernicke", "thiamine", "vitamin b1", "b1 deficiency"],
    "shift_003": ["aortic dissection", "dissection"],
    "shift_004": ["pheochromocytoma"],
    "shift_005": ["deep vein thrombosis", "dvt"],
    "shift_006": ["infective endocarditis", "endocarditis"],
    "shift_007": ["pulmonary embolism", "pe"],
    "shift_008": ["appendicitis"],
    "shift_009": ["subarachnoid hemorrhage", "subarachnoid", "sah"],
    "shift_010": ["stroke", "cerebrovascular accident", "cva"],
    "shift_011": ["pneumonia"],
    "shift_012": ["hemolysis", "hemolytic", "prehepatic"],
}


def case_id(case: dict) -> str:
    return case.get("id") or case["base_id"]


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


def source_answer_prompt(prompt: str) -> str:
    return (
        "Answer the clinical question with the most likely diagnosis or syndrome "
        "first, in one short sentence. Then give one brief reason.\n\n"
        f"Question: {prompt}"
    )


def write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    hits = [bool(row["diagnosis_hit"]) for row in rows]
    broad_hits = [bool(row["broad_hit"]) for row in rows]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Model Answer Baseline\n\n")
        f.write("Lexical screening only; not claim-level clinical grading.\n")
        f.write(
            "`diagnosis_hit` uses diagnosis-level aliases only; `broad_hit` also "
            "includes cue/symptom aliases used for NLA screening.\n\n"
        )
        f.write(f"- n: {len(rows)}\n")
        f.write(f"- diagnosis_hit: {sum(hits)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit_rate: {mean(hits):.3f}\n")
        f.write(f"- broad_hit: {sum(broad_hits)}/{len(rows)}\n")
        f.write(f"- broad_hit_rate: {mean(broad_hits):.3f}\n\n")
        f.write("| case | expected | diagnosis_hit | diagnosis_hits | broad_hits | answer |\n")
        f.write("|---|---|---:|---|---|---|\n")
        for row in rows:
            answer = row["answer"].replace("\n", " ")
            if len(answer) > 220:
                answer = answer[:217] + "..."
            f.write(
                f"| {row['id']} | {row['specific_expected']} | "
                f"{'Y' if row['diagnosis_hit'] else 'N'} | "
                f"{', '.join(row['diagnosis_hits']) or '-'} | "
                f"{', '.join(row['broad_hits']) or '-'} | "
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
        cid = case_id(case)
        prompt = source_answer_prompt(case["specific_prompt"])
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
        broad_aliases = case.get("specific_aliases") or SPECIFIC_ALIASES.get(cid, [])
        broad_hits = alias_hits(answer, broad_aliases)
        diagnosis_aliases = case.get("diagnosis_aliases") or SOURCE_DIAGNOSIS_ALIASES.get(cid, broad_aliases)
        diagnosis_hits = alias_hits(answer, diagnosis_aliases)
        out_rows.append(
            {
                "id": cid,
                "category": case.get("category"),
                "prompt": case["specific_prompt"],
                "source_answer_prompt": prompt,
                "specific_expected": case["specific_expected"],
                "diagnostic_shift": case.get("diagnostic_shift"),
                "answer": answer,
                "answer_aliases": diagnosis_aliases,
                "answer_hits": diagnosis_hits,
                "answer_hit": bool(diagnosis_hits),
                "diagnosis_aliases": diagnosis_aliases,
                "diagnosis_hits": diagnosis_hits,
                "diagnosis_hit": bool(diagnosis_hits),
                "broad_aliases": broad_aliases,
                "broad_hits": broad_hits,
                "broad_hit": bool(broad_hits),
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
