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

**The direct condition is prefilled.** Asked for "the single most likely
diagnosis", gemma-3-12b-it answers "Okay, let's break down this case" and
reasons; on 832 MCR cases at 128 new tokens, 0 of them reached the closing
string and 87% were still mid-sentence when the budget ran out. Raising the
budget would have produced parseable answers and a worthless experiment, since
the direct arm would then be a chain-of-thought arm and the contrast the paper
rests on would not exist. The assistant turn is instead started at the answer
cue, which leaves no room to reason. Causal masking makes this free for the
activations: appended tokens cannot alter any earlier token's hidden state, so
the cue positions and the format position are unchanged.

**The chain-of-thought condition may run out of budget honestly**, and a
reasoner cut off mid-sentence has no answer to score. Those cases get a second
pass in which the model's own full chain is given back to it and only the
answer is requested -- a completion, not an intervention, and marked
`answer_forced` so the analysis can check whether forced rows differ. The
truncated-chain version of the same mechanism is the early-answering test and
lives in `src.case_prompts.early_answer_prompt`; it must not be used for labels.
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

from src.answer_matching import is_correct, normalize, token_f1
from src.case_prompts import ANSWER_CUE, parse_answer, prefilled_assistant_turn
from src.jsonl import append_jsonl, read_jsonl
from src.sampling import sample_rows

PROMPT_FIELDS = {"direct": "prompt", "cot": "prompt_cot"}

# Chain-of-thought needs room to finish reasoning AND emit the closing string;
# the source format this follows allowed 1500-2048. The direct figure is what a
# free-running direct arm would need, and is only reachable with --no-prefill.
DEFAULT_MAX_NEW_TOKENS = {"direct": 512, "cot": 2048}

# A prefilled turn has already been started at the answer, so all that remains
# is a diagnosis name and a full stop. 32 cut the long end of MedCaseReasoning's
# labels mid-word ("...atypical peripheral retinal degeneration (PR"), which
# costs a match rather than a few tokens.
PREFILLED_MAX_NEW_TOKENS = 64

# Enough to state a diagnosis after a chain of thought is handed back.
FORCED_ANSWER_MAX_NEW_TOKENS = 32

# Case annotations copied onto the answer row. These say which arm of an
# experiment a row belongs to, and nothing downstream can reconstruct them from
# a prompt and an answer.
CARRIED_FIELDS = (
    "base_id",
    "variant",
    "hint_wording",
    "ladder_rung",
    "first_answer",
    "first_correct",
    "moved",
    "readout_conclusion",
    "correction_flag",
    "probe_argmax",
    "channel_map",
    "group",
    "readout_channel",
    "label_moved",
    "hint_variant",
    "hint_diagnosis_name",
    "gold_in_prompt",
    "suggestion_source",
    "suggestion_score",
    "target_role",
    "target_text",
    "cue_text",
    "cue_targets",
    "position_mode",
    "age",
    "sex",
    "source",
    "patient_id",
    "split",
    "disease_category",
    "canonical_pdd",
    "patient_group",
)


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


def reasoning_before_final_answer(response: str) -> str:
    """Return the generated reasoning before the final answer boundary.

    Forced-answer rows can contain an earlier, unfinished chain followed by a
    second answer cue. The last cue is therefore the only boundary that
    consistently separates the state at P1 from the diagnosis emitted at P2.
    """
    text = str(response or "")
    boundary = text.casefold().rfind(ANSWER_CUE.casefold())
    return text[:boundary].rstrip() if boundary >= 0 else text


def parse_where(specs: list[str]) -> dict[str, set[str]]:
    """`hint_variant=none,wrong` -> {"hint_variant": {"none", "wrong"}}.

    Compared as lowercased strings, so `gold_in_prompt=false` selects the rows
    whose flag is the JSON boolean without the caller having to know that.
    """
    filters: dict[str, set[str]] = {}
    for spec in specs:
        key, sep, values = str(spec).partition("=")
        if not sep or not key.strip():
            raise SystemExit(f"--where wants KEY=VALUE, got {spec!r}")
        filters.setdefault(key.strip(), set()).update(
            value.strip().lower() for value in values.split(",") if value.strip()
        )
    return filters


def matches_where(row: dict[str, Any], filters: dict[str, set[str]]) -> bool:
    return all(str(row.get(key)).lower() in wanted for key, wanted in filters.items())


def batched(rows: list[dict[str, Any]], size: int):
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def answer_row_key(row: dict[str, Any]) -> tuple[str, str]:
    """Stable key for interrupted-run resumption.

    Some experiment files stack several variants of one base case, so base_id
    alone is not unique. The emitted row preserves the case id and variant.
    """
    identifier = row.get("id")
    if identifier is None:
        raise ValueError("A source-answer row lacks id")
    return str(identifier), str(row.get("variant") or "")


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
        help=(
            f"Defaults to {PREFILLED_MAX_NEW_TOKENS} when the answer is prefilled, "
            f"otherwise {DEFAULT_MAX_NEW_TOKENS['direct']} for direct answers and "
            f"{DEFAULT_MAX_NEW_TOKENS['cot']} for chain-of-thought."
        ),
    )
    parser.add_argument(
        "--prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Start the model's turn at the answer cue so nothing can precede the "
            "answer. On by default for --condition direct, where the model "
            "otherwise reasons at length and the arm stops being a baseline; off "
            "for cot, where reasoning is the point."
        ),
    )
    parser.add_argument(
        "--force-answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For chains that ran out of budget, hand the model its own full "
            "reasoning back and ask only for the answer. Marked answer_forced."
        ),
    )
    parser.add_argument(
        "--where",
        action="append",
        default=[],
        metavar="KEY=VALUE[,VALUE]",
        help=(
            "Keep rows whose KEY is one of the VALUEs. Repeat to require several "
            "keys at once. For case files that stack the arms of an experiment: "
            "a chain-of-thought run over an arm nothing will read is a third of "
            "the run spent on 2048-token generations for nobody."
        ),
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=17,
        help="Seed for --limit, which samples rather than taking the front.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Keep valid rows already present in --output-jsonl and generate only "
            "missing case IDs. Existing rows outside the selected case set or "
            "duplicate keys stop the run."
        ),
    )
    args = parser.parse_args()

    # Imported here so the scoring helpers stay testable without a GPU stack.
    import torch

    from src.config import ensure_dir, load_config
    from src.modeling import load_causal_lm, load_tokenizer

    cfg = load_config(args.config)
    field = PROMPT_FIELDS[args.condition]

    # Generation settings come from the config so the copy stored beside the
    # results describes the run; the condition default fills in only what the
    # config leaves unset, and the flag overrides both.
    generation = {k: v for k, v in (cfg.get("generation") or {}).items() if v is not None}
    generation.pop("max_new_tokens", None)
    generation.setdefault("do_sample", False)
    prefill = args.prefill if args.prefill is not None else args.condition == "direct"
    if args.max_new_tokens:
        max_new = args.max_new_tokens
    elif prefill:
        max_new = PREFILLED_MAX_NEW_TOKENS
    else:
        max_new = DEFAULT_MAX_NEW_TOKENS[args.condition]
    force_answer = args.force_answer and not prefill

    rows = [row for row in read_jsonl(args.cases) if row.get(field)]
    if args.where:
        filters = parse_where(args.where)
        kept = [row for row in rows if matches_where(row, filters)]
        if not kept:
            raise SystemExit(
                f"--where kept none of {len(rows):,} rows. Filters: {filters}\n"
                "A silent empty run would look like a finished one."
            )
        print(f"[where] {len(kept):,} of {len(rows):,} rows kept by {filters}", flush=True)
        rows = kept
    if args.limit:
        rows = sample_rows(rows, args.limit, seed=args.sample_seed, label="cases")
    if not rows:
        raise SystemExit(f"no rows in {args.cases} carrying {field!r}")

    output_path = Path(args.output_jsonl)
    ensure_dir(output_path.parent)
    existing_rows: list[dict[str, Any]] = []
    if output_path.exists() and args.resume:
        existing_rows = list(read_jsonl(output_path))
    elif output_path.exists():
        output_path.unlink()

    target_keys = [answer_row_key(row) for row in rows]
    if len(target_keys) != len(set(target_keys)):
        raise SystemExit("selected case rows contain duplicate (id, variant) keys")
    existing_keys = [answer_row_key(row) for row in existing_rows]
    if len(existing_keys) != len(set(existing_keys)):
        raise SystemExit("existing output contains duplicate (id, variant) keys")
    extra_keys = set(existing_keys) - set(target_keys)
    if extra_keys:
        raise SystemExit(
            f"existing output contains {len(extra_keys)} rows outside the selected cases"
        )
    wrong_conditions = {
        str(row.get("condition"))
        for row in existing_rows
        if str(row.get("condition")) != args.condition
    }
    if wrong_conditions:
        raise SystemExit(
            "existing output was produced for another condition: "
            f"{sorted(wrong_conditions)}"
        )
    existing_key_set = set(existing_keys)
    generation_rows = [row for row in rows if answer_row_key(row) not in existing_key_set]
    if existing_rows:
        print(
            f"[resume] valid existing rows {len(existing_rows):,}/{len(rows):,}; "
            f"generating {len(generation_rows):,}",
            flush=True,
        )

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
    n_correct = sum(bool(row.get("source_correct")) for row in existing_rows)
    n_parsed = sum(bool(row.get("answer_parsed")) for row in existing_rows)
    n_forced = sum(bool(row.get("answer_forced")) for row in existing_rows)
    n_answer_in_reasoning = sum(
        bool(row.get("diagnosis_alias_in_reasoning")) for row in existing_rows
    )
    n_gold_in_reasoning = sum(
        bool(row.get("gold_alias_in_reasoning")) for row in existing_rows
    )
    ranks: list[int] = [
        int(row["differential_rank"])
        for row in existing_rows
        if row.get("differential_rank") is not None
    ]

    def generate(texts: list[str], budget: int) -> list[str]:
        encoded = tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=budget,
                pad_token_id=tokenizer.pad_token_id,
                **generation,
            )
        return tokenizer.batch_decode(
            generated[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def carried(row: dict[str, Any]) -> dict[str, Any]:
        """The case's own annotations, copied through to the answer row.

        A case file that stacks several arms of one experiment says which arm
        each row is; this file is what gets analyzed, and without those keys the
        arms are indistinguishable. The referring-note run wrote 1,143 answers
        with no `hint_variant` on any of them, so every case looked incomplete
        and the analysis reported nothing -- the same shape of loss as scoring
        with the alias table and not recording it.

        Listed rather than copied wholesale: the case row also carries the
        other conditions' prompts and the full differential, which would treble
        the file for no reader.
        """
        return {key: row[key] for key in CARRIED_FIELDS if key in row}

    def emit(row: dict[str, Any], response: str, *, forced: bool) -> None:
        nonlocal n_parsed, n_correct, n_forced
        nonlocal n_answer_in_reasoning, n_gold_in_reasoning
        answer = parse_answer(response)
        gold = str(row.get("diagnosis_name") or "")
        aliases = [str(a) for a in (row.get("diagnosis_aliases") or [])]
        correct = is_correct(answer, gold, aliases)
        reasoning = reasoning_before_final_answer(response)
        answer_in_reasoning = bool(answer) and is_correct(reasoning, answer, [])
        gold_in_reasoning = bool(gold) and is_correct(reasoning, gold, aliases)
        # Recorded, never used as the metric: it is how the strict rule's
        # undercount is measured on free-text labels.
        overlap = token_f1(answer, gold, aliases)
        rank = differential_rank(answer, row.get("differential_diagnosis") or [])
        n_parsed += answer is not None
        n_correct += correct
        n_forced += forced
        n_answer_in_reasoning += answer_in_reasoning
        n_gold_in_reasoning += gold_in_reasoning
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
                "answer_prefilled": prefill,
                "answer_forced": forced,
                # P1 is only a clean pre-answer read when the diagnosis has not
                # already appeared in the chain. Keep model-answer and expert-
                # gold leakage separate because they differ on source-wrong rows.
                "diagnosis_alias_in_reasoning": answer_in_reasoning,
                "gold_alias_in_reasoning": gold_in_reasoning,
                "diagnosis_name": gold,
                # Carried, not just consulted. Scoring here and omitting the
                # alias list left every later re-scoring of this file silently
                # stricter than the run that produced it: the audit recomputed
                # DDXPlus at 0.2920 against the recorded 0.3724, the whole
                # difference being the 49-name alias table it could not see.
                "diagnosis_aliases": aliases,
                "source_correct": correct,
                "answer_token_f1": round(overlap, 4),
                "differential_rank": rank,
                "model_id": model_cfg["model_id"],
                **carried(row),
            },
        )

    print(
        f"[setup] condition={args.condition} prefill={prefill} "
        f"max_new_tokens={max_new} force_answer={force_answer}",
        flush=True,
    )

    # Rows whose chain ran out of budget are held back rather than scored as
    # failures, and finished in one batched second pass at the end.
    pending: list[tuple[dict[str, Any], str, str]] = []

    for index, batch in enumerate(batched(generation_rows, args.batch_size)):
        chat_texts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": row[field]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for row in batch
        ]
        texts = [
            prefilled_assistant_turn(chat_text) if prefill else chat_text
            for chat_text in chat_texts
        ]
        completions = generate(texts, max_new)

        for row, chat_text, completion in zip(batch, chat_texts, completions):
            response = f"{ANSWER_CUE}{completion}" if prefill else completion
            if force_answer and parse_answer(response) is None:
                pending.append((row, chat_text, response))
                continue
            emit(row, response, forced=False)
        if index and index % 25 == 0:
            generated = min((index + 1) * args.batch_size, len(generation_rows))
            done = len(existing_rows) + generated
            print(
                f"[gen] {done:,}/{len(rows):,} | "
                f"parsed {n_parsed / max(done, 1):.3f} | "
                f"correct {n_correct / max(done, 1):.3f}"
                + (f" | {len(pending):,} awaiting a forced answer" if pending else ""),
                flush=True,
            )

    if pending:
        print(
            f"[force] {len(pending):,}/{len(generation_rows):,} new chains reached "
            "the token budget "
            "without an answer; completing them from their own reasoning",
            flush=True,
        )
        for index, batch in enumerate(batched(pending, args.batch_size)):
            texts = [
                prefilled_assistant_turn(chat_text, chain) for _, chat_text, chain in batch
            ]
            completions = generate(texts, FORCED_ANSWER_MAX_NEW_TOKENS)
            for (row, _, chain), completion in zip(batch, completions):
                emit(row, f"{chain}\n\n{ANSWER_CUE}{completion}", forced=True)
            if index and index % 25 == 0:
                done = min((index + 1) * args.batch_size, len(pending))
                print(f"[force] {done:,}/{len(pending):,}", flush=True)

    summary = {
        "cases": args.cases,
        "condition": args.condition,
        "model_id": model_cfg["model_id"],
        "prefill": prefill,
        "max_new_tokens": max_new,
        "force_answer": force_answer,
        "generation": generation,
        "batch_size": args.batch_size,
        "n": len(rows),
        "answer_parse_rate": round(n_parsed / len(rows), 4),
        "answer_forced_rate": round(n_forced / len(rows), 4),
        "diagnosis_alias_in_reasoning_rate": round(
            n_answer_in_reasoning / max(n_parsed, 1), 4
        ),
        "gold_alias_in_reasoning_rate": round(n_gold_in_reasoning / len(rows), 4),
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
    final_rows = list(read_jsonl(output_path))
    final_keys = [answer_row_key(row) for row in final_rows]
    if len(final_rows) != len(rows) or set(final_keys) != set(target_keys):
        raise RuntimeError(
            "source-answer completeness check failed: "
            f"expected={len(rows)}, found={len(final_rows)}, "
            f"unique={len(set(final_keys))}"
        )
    print(f"[done] wrote {len(final_rows):,} answers to {output_path}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
