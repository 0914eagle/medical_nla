"""Ask the backbone for a diagnosis, on the prompt the case actually carries.

This produces the correctness labels every error-related claim rests on, and it
is the one GPU job that does not wait for activation extraction: the prompt is
frozen, so the answers can be generated now.

The prompt is used **as written**. The pilot's version wrapped it in a further
instruction while extraction used the bare case prompt, so the labels described
a different forward pass than the activations they were joined to. The
instruction now lives inside the case prompt, and nothing here may add to it.

Correctness is decided on the parsed answer rather than on the whole response.
The prompt asks for a fixed closing string, so the diagnosis is recoverable
exactly; searching the full text instead would count a diagnosis that appeared
only while being ruled out. Where the source provides a ranked differential, the
answer's rank in it is recorded too: picking the second-ranked condition is a
different failure from picking an unrelated one, and averaging them into one
accuracy hides that.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.case_prompts import parse_answer
from src.jsonl import append_jsonl, read_jsonl

PROMPT_FIELDS = {"direct": "prompt", "cot": "prompt_cot"}


def normalize(text: str) -> str:
    return " ".join(str(text or "").split()).lower().strip(" .")


def is_correct(answer: str | None, gold: str, aliases: list[str]) -> bool:
    """Match the parsed answer against the gold name or one of its aliases.

    Containment in either direction, since a model may answer "acute otitis
    media" for "Otitis media" or the reverse; both name the same condition.
    """
    if not answer:
        return False
    got = normalize(answer)
    for candidate in [gold, *aliases]:
        want = normalize(candidate)
        if want and (want in got or got in want):
            return True
    return False


def differential_rank(answer: str | None, differential: list[dict[str, Any]]) -> int | None:
    """1-based position of the answer in a ranked differential, or None."""
    if not answer or not differential:
        return None
    got = normalize(answer)
    for index, entry in enumerate(differential, start=1):
        name = normalize(entry.get("diagnosis"))
        if name and (name in got or got in name):
            return index
    return None


def batched(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--condition", default="direct", choices=sorted(PROMPT_FIELDS))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Defaults to 48 for direct answers and 768 for chain-of-thought.",
    )
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    # Imported here so the scoring helpers stay testable without a GPU stack.
    import torch

    from src.config import ensure_dir, load_config
    from src.modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    field = PROMPT_FIELDS[args.condition]
    max_new = args.max_new_tokens or (48 if args.condition == "direct" else 768)

    rows = [row for row in read_jsonl(args.cases) if row.get(field)]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows in {args.cases} carrying {field!r}")

    output_path = Path(args.output_jsonl)
    ensure_dir(output_path.parent)
    if output_path.exists():
        output_path.unlink()

    cache_dir = cfg["paths"].get("cache_dir")
    model_cfg = cfg["source_model"]
    tokenizer = load_tokenizer(
        model_cfg["model_id"], cache_dir=cache_dir, trust_remote_code=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_causal_lm(model_cfg, cache_dir=cache_dir)
    model.eval()

    torch.manual_seed(int(cfg.get("seed", 17)))
    n_correct = 0
    n_parsed = 0
    ranks: list[int] = []

    for index, batch in enumerate(batched(rows, args.batch_size)):
        texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row[field]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        completions = tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for row, response in zip(batch, completions):
            answer = parse_answer(response)
            gold = str(row.get("diagnosis_name") or "")
            aliases = [str(a) for a in (row.get("diagnosis_aliases") or [])]
            correct = is_correct(answer, gold, aliases)
            rank = differential_rank(answer, row.get("differential_diagnosis") or [])
            n_parsed += answer is not None
            n_correct += correct
            if rank is not None:
                ranks.append(rank)
            append_jsonl(
                output_path,
                {
                    "id": row.get("id"),
                    "base_id": row.get("base_id", row.get("id")),
                    "condition": args.condition,
                    "prompt": row[field],
                    "response": response,
                    "answer": answer,
                    "answer_parsed": answer is not None,
                    "diagnosis_name": gold,
                    "source_correct": correct,
                    "differential_rank": rank,
                    "model_id": model_cfg["model_id"],
                },
            )
        if index and index % 25 == 0:
            done = (index + 1) * args.batch_size
            print(
                f"[gen] {min(done, len(rows)):,}/{len(rows):,} | "
                f"correct {n_correct / max(done, 1):.3f}",
                flush=True,
            )

    summary = {
        "cases": args.cases,
        "condition": args.condition,
        "n": len(rows),
        "answer_parse_rate": round(n_parsed / len(rows), 4),
        "accuracy": round(n_correct / len(rows), 4),
        "differential_rank_available": len(ranks),
        "differential_rank_mean": round(sum(ranks) / len(ranks), 2) if ranks else None,
        "answer_in_top3_of_differential": (
            round(sum(1 for r in ranks if r <= 3) / len(ranks), 4) if ranks else None
        ),
    }
    print(json.dumps(summary, indent=2))
    if args.summary_json:
        path = Path(args.summary_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(rows):,} answers to {output_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
