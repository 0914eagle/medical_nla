"""Run a constrained multiple-choice diagnosis baseline with the source model.

The free-generation source baseline can miss DDXPlus canonical labels even when
the answer is clinically nearby. This script gives the model the closed set of
DDXPlus labels and asks it to select exactly one option.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_specificity_outputs import contains_term
from src.config import load_config
from src.sampling import sample_rows
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


def row_seed(base_seed: int, row_id: str) -> int:
    digest = hashlib.sha256(f"{base_seed}:{row_id}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def ordered_candidates(
    candidates: list[dict[str, str]],
    *,
    shuffle: bool,
    seed: int,
    row_id: str,
) -> list[dict[str, str]]:
    ordered = list(candidates)
    if shuffle:
        random.Random(row_seed(seed, row_id)).shuffle(ordered)
    return ordered


def build_mc_prompt(
    clinical_prompt: str,
    candidates: list[dict[str, str]],
    *,
    answer_style: str,
) -> str:
    options = "\n".join(
        f"{idx}. {candidate['diagnosis_name']}" for idx, candidate in enumerate(candidates, start=1)
    )
    if answer_style == "number":
        answer_instruction = "Return only the option number."
    elif answer_style == "number_and_name":
        answer_instruction = "Return only the option number and diagnosis name."
    else:
        raise ValueError(f"Unsupported answer_style: {answer_style}")
    return (
        "Choose the single best diagnosis for the clinical case from the numbered list.\n"
        f"{answer_instruction}\n\n"
        f"Clinical case:\n{clinical_prompt}\n\n"
        f"Diagnosis options:\n{options}"
    )


def chat_inputs(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    chat_texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    encoded = tokenizer(chat_texts, padding=True, return_tensors="pt")
    tokenizer.padding_side = original_padding_side
    return {key: value.to(device) for key, value in encoded.items()}


def generation_kwargs(cfg: dict[str, Any], max_new_tokens: int | None) -> dict[str, Any]:
    gen = dict(cfg["generation"])
    if max_new_tokens is not None:
        gen["max_new_tokens"] = max_new_tokens
    return {key: value for key, value in gen.items() if value is not None}


def parse_selection(answer: str, candidates: list[dict[str, str]]) -> dict[str, Any]:
    text = answer.strip()
    for match in re.finditer(r"(?<!\d)(\d{1,3})(?!\d)", text[:120]):
        number = int(match.group(1))
        if 1 <= number <= len(candidates):
            candidate = candidates[number - 1]
            return {
                "selected_index": number,
                "selected_diagnosis_id": candidate["diagnosis_id"],
                "selected_diagnosis_name": candidate["diagnosis_name"],
                "parse_method": "number",
            }

    hits = []
    for idx, candidate in enumerate(candidates, start=1):
        if contains_term(text, candidate["diagnosis_name"]):
            hits.append((idx, candidate))
    if len(hits) == 1:
        idx, candidate = hits[0]
        return {
            "selected_index": idx,
            "selected_diagnosis_id": candidate["diagnosis_id"],
            "selected_diagnosis_name": candidate["diagnosis_name"],
            "parse_method": "name",
        }
    return {
        "selected_index": None,
        "selected_diagnosis_id": None,
        "selected_diagnosis_name": None,
        "parse_method": "unparsed",
    }


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    parsed = [row for row in rows if row.get("selected_diagnosis_id")]
    hits = [bool(row["diagnosis_hit"]) for row in rows]
    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold[str(row["gold_diagnosis_id"])].append(row)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Source Model Multiple-Choice Baseline\n\n")
        f.write("The source model sees the closed diagnosis-label set and generates one option.\n\n")
        f.write(f"- n: {len(rows)}\n")
        if rows:
            f.write(f"- shuffled_options: `{rows[0].get('shuffled_options')}`\n")
            f.write(f"- answer_style: `{rows[0].get('answer_style')}`\n")
        f.write(f"- parsed: {len(parsed)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit: {sum(hits)}/{len(rows)}\n")
        f.write(f"- diagnosis_hit_rate: {mean(hits):.4f}\n\n")

        f.write("## By Gold Diagnosis\n\n")
        f.write("| diagnosis_id | n | hit_rate | hits |\n")
        f.write("|---|---:|---:|---:|\n")
        rates = []
        for diagnosis_id, items in by_gold.items():
            count = sum(bool(row["diagnosis_hit"]) for row in items)
            rates.append((count / len(items), diagnosis_id, len(items), count))
        for rate, diagnosis_id, n_items, count in sorted(rates):
            f.write(f"| {diagnosis_id} | {n_items} | {rate:.4f} | {count} |\n")

        f.write("\n## Top Selected Diagnoses\n\n")
        f.write("| diagnosis_id | count |\n")
        f.write("|---|---:|\n")
        selected_counts = Counter(
            str(row["selected_diagnosis_id"]) for row in rows if row.get("selected_diagnosis_id")
        )
        for diagnosis_id, count in selected_counts.most_common(25):
            f.write(f"| {diagnosis_id} | {count} |\n")

        f.write("\n## Examples\n\n")
        f.write("| id | expected | selected | hit | answer |\n")
        f.write("|---|---|---|---:|---|\n")
        for row in rows[:200]:
            answer = str(row["answer"]).replace("\n", " ")
            if len(answer) > 180:
                answer = answer[:177] + "..."
            f.write(
                f"| {row['id']} | {row['gold_diagnosis_name']} | "
                f"{row.get('selected_diagnosis_name') or '-'} | "
                f"{'Y' if row['diagnosis_hit'] else 'N'} | {answer} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--candidates-jsonl", default=None)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=17,
        help="Seed for --limit, which samples rather than taking the front.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--answer-style", choices=["number", "number_and_name"], default="number_and_name")
    args = parser.parse_args()

    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must satisfy 0 <= shard_index < num_shards")

    cfg = load_config(args.config)
    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    all_rows = [row for row in read_jsonl(args.input) if row.get(args.prompt_field) and row.get("diagnosis_id")]
    # Candidates from every row, before the limit -- see the note in
    # score_source_diagnosis_logprobs: a limited row set carries only a few
    # labels, and the option list must not shrink with it.
    candidates = read_candidates(args.candidates_jsonl, all_rows)
    all_rows = sample_rows(all_rows, args.limit, seed=args.sample_seed)
    rows = [row for idx, row in enumerate(all_rows) if idx % args.num_shards == args.shard_index]
    if not rows:
        raise ValueError("No rows selected for this shard.")

    candidate_by_id = {row["diagnosis_id"]: row for row in candidates}
    missing = sorted({str(row["diagnosis_id"]) for row in rows} - set(candidate_by_id))
    if missing:
        raise ValueError(f"Gold diagnoses missing from candidate set: {missing[:10]}")

    tokenizer = load_tokenizer(
        model_cfg["model_id"],
        cache_dir=cache_dir,
        trust_remote_code=model_cfg.get("trust_remote_code", True),
    )
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()
    gen_kwargs = generation_kwargs(cfg, args.max_new_tokens)

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    out_rows: list[dict[str, Any]] = []
    total = len(rows)
    for start in range(0, total, args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        batch_prompts = []
        batch_options = []
        for row in batch_rows:
            ordered = ordered_candidates(
                candidates,
                shuffle=args.shuffle_options,
                seed=args.seed,
                row_id=str(row.get("id") or row.get("base_id")),
            )
            batch_options.append(ordered)
            batch_prompts.append(
                build_mc_prompt(
                    str(row[args.prompt_field]),
                    ordered,
                    answer_style=args.answer_style,
                )
            )
        encoded = chat_inputs(tokenizer, batch_prompts, model.device)
        input_len = int(encoded["input_ids"].shape[-1])
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                **gen_kwargs,
            )
        for offset, row in enumerate(batch_rows):
            answer_ids = generated[offset, input_len:]
            answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            selection = parse_selection(answer, batch_options[offset])
            gold_id = str(row["diagnosis_id"])
            hit = selection["selected_diagnosis_id"] == gold_id
            gold_candidate = candidate_by_id[gold_id]
            gold_index = next(
                idx
                for idx, candidate in enumerate(batch_options[offset], start=1)
                if candidate["diagnosis_id"] == gold_id
            )
            out = {
                "id": row["id"],
                "base_id": row.get("base_id", row["id"]),
                "prompt": row[args.prompt_field],
                "variant": row.get("variant"),
                "gold_diagnosis_id": gold_id,
                "gold_diagnosis_name": gold_candidate["diagnosis_name"],
                "gold_option_index": gold_index,
                "selected_index": selection["selected_index"],
                "selected_diagnosis_id": selection["selected_diagnosis_id"],
                "selected_diagnosis_name": selection["selected_diagnosis_name"],
                "parse_method": selection["parse_method"],
                "diagnosis_hit": hit,
                "answer": answer,
                "mc_prompt": batch_prompts[offset],
                "num_candidates": len(candidates),
                "shuffled_options": args.shuffle_options,
                "answer_style": args.answer_style,
                "seed": args.seed,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
                "gen_config": gen_kwargs,
                "batch_size": args.batch_size,
            }
            append_jsonl(output_path, out)
            out_rows.append(out)
            print(
                f"[source-mc] {start + offset + 1}/{total} {row['id']} "
                f"gold={gold_id} selected={selection['selected_diagnosis_id']} hit={hit}",
                flush=True,
            )
    write_summary(summary_path, out_rows)
    print(f"[done] wrote {len(out_rows)} rows to {output_path}", flush=True)


if __name__ == "__main__":
    main()
