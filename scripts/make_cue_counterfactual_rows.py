"""Build cue-swap / cue-removal counterfactual extraction rows.

Faithfulness test for the cue-position reader (v4/v5). For each sampled
test-pool case:

- `orig`:    original prompt; probe the swap slot and up to two retained
             slots (golds = the original cues).
- `swap`:    the swap-slot cue is replaced by a different cue drawn from
             the corpus vocabulary; probe the new cue's span (gold = the
             replacement) plus the retained slots. Sensitivity: the
             readout must track the new content; still emitting the old
             cue means the reader keys on case context, not the span.
- `removed`: the swap-slot cue is deleted; probe the retained slots.
             Specificity: retained readouts must stay correct, and the
             removed cue must not appear as a phantom.

Prompts are rebuilt from the cue list with the original template and
verified against the stored prompt; cases that do not round-trip are
skipped (counted) so every counterfactual prompt is construction-exact.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_ddxplus_cue_position_rows import cue_spans_in_prompt
from scripts.make_ddxplus_probe_dataset import make_prompt as probe_make_prompt
from src.jsonl import read_jsonl, write_jsonl

CARRY_FIELDS = (
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "source",
    "patient_id",
    "age",
    "sex",
)


def make_prompt(
    cues: list[str], *, condition: str = "direct", age: Any = None, sex: Any = None
) -> str:
    """The frame the cases were built with, not a second copy of it.

    This held its own inline "A patient presents with X, Y and Z" sentence
    while the corpus moved to the findings-list frame, so the round-trip check
    below rejected every case and the script produced no rows at all. The check
    was right; the builder was stale. Importing the one the cases use makes the
    two impossible to drift apart again.
    """
    return probe_make_prompt(cues, condition=condition, age=age, sex=sex)


def extraction_row(
    *,
    base_id: str,
    case: dict[str, Any],
    variant: str,
    role: str,
    prompt: str,
    cot_prompt: str,
    cues_in_prompt: list[str],
    slot: int,
    gold_cue: str,
    strategy: str,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    spans = cue_spans_in_prompt(prompt, cues_in_prompt)
    span = spans[slot]
    if span is None:
        return None
    occurrence = prompt[: span[0]].count(gold_cue)
    row = {field: case.get(field) for field in CARRY_FIELDS}
    row.update(
        {
            "id": f"{base_id}__cf_{variant}__slot{slot:02d}",
            "base_id": base_id,
            "variant": f"cue_counterfactual_{variant}",
            "cf_variant": variant,
            "cf_role": role,
            "cf_slot": slot,
            "prompt": prompt,
            # The chain-of-thought form of the same presentation. Without it the
            # CoT arm has nothing to run on: run_source_answers reads
            # prompt_cot in the cot condition, and hypothesis 1 is a comparison
            # against that arm, so its absence left the comparison with no
            # other side.
            "prompt_cot": cot_prompt,
            "target_role": "cue",
            "cue_text": gold_cue,
            "cue_targets": [gold_cue],
            "position_mode": "target_text",
            "target_text": gold_cue,
            "target_text_strategy": strategy,
            "target_text_occurrence": occurrence,
            **extra,
        }
    )
    return row


def counterfactual_rows_for_case(
    case: dict[str, Any],
    *,
    vocab: list[str],
    rng: random.Random,
    strategy: str,
) -> list[dict[str, Any]] | None:
    cues = [str(cue) for cue in (case.get("cue_targets") or []) if str(cue).strip()]
    if len(cues) < 3:
        return None
    age, sex = case.get("age"), case.get("sex")

    def prompt_of(cue_list: list[str]) -> str:
        return make_prompt(cue_list, age=age, sex=sex)

    # Construction-exact: the prompt this script would build for the unchanged
    # cues has to be the prompt the case carries, or the counterfactual differs
    # from the original in ways beyond the one cue that was meant to change.
    if prompt_of(cues) != str(case.get("prompt") or ""):
        return None

    base_id = str(case.get("base_id") or case["id"])
    slot = rng.randrange(len(cues))
    retained_slots = rng.sample([i for i in range(len(cues)) if i != slot], k=2)
    case_cues_lower = {cue.lower() for cue in cues}
    candidates = [cue for cue in vocab if cue.lower() not in case_cues_lower]
    if not candidates:
        return None
    replacement = rng.choice(candidates)

    swap_cues = list(cues)
    swap_cues[slot] = replacement
    removed_cues = [cue for i, cue in enumerate(cues) if i != slot]
    # Slot indices shift after removal.
    removed_slot_of = {i: (i if i < slot else i - 1) for i in retained_slots}

    shared = {
        "cf_original_cue": cues[slot],
        "cf_replacement_cue": replacement,
        "cf_removed_cue": cues[slot],
    }
    rows = []
    specs = [
        ("orig", "swapped_slot", prompt_of(cues), cues, slot, cues[slot]),
        ("swap", "swapped_slot", prompt_of(swap_cues), swap_cues, slot, replacement),
    ]
    for r in retained_slots:
        specs.append(("orig", "retained", prompt_of(cues), cues, r, cues[r]))
        specs.append(("swap", "retained", prompt_of(swap_cues), swap_cues, r, cues[r]))
        specs.append(
            ("removed", "retained", prompt_of(removed_cues), removed_cues,
             removed_slot_of[r], cues[r])
        )
    for variant, role, prompt, cues_in_prompt, s, gold in specs:
        row = extraction_row(
            base_id=base_id,
            case=case,
            variant=variant,
            role=role,
            prompt=prompt,
            # Built from the same cue list as the direct prompt, so the two
            # conditions differ only in the instruction that follows the
            # presentation -- which is what makes the arms comparable.
            cot_prompt=make_prompt(cues_in_prompt, condition="cot", age=age, sex=sex),
            cues_in_prompt=cues_in_prompt,
            slot=s,
            gold_cue=gold,
            strategy=strategy,
            extra=shared,
        )
        if row is None:
            return None
        rows.append(row)
    # Distinct ids: orig/swap retained rows share (variant, slot) namespaces safely,
    # but orig swapped-slot and orig retained could collide only if slots equal, which
    # they cannot. Verify anyway.
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        return None
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="All-cue case manifest (prompt + cue_targets).")
    parser.add_argument(
        "--split-dir",
        required=True,
        help="Cue-position split dir; its test manifests define the eligible case pool.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=["cue_count_all"])
    parser.add_argument("--num-cases", type=int, default=150)
    parser.add_argument(
        "--target-text-strategy", default="last_subtoken", choices=["last_subtoken", "span_mean"]
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    test_case_ids: set[str] = set()
    for pool in ("test_seen_cue", "test_heldout_cue"):
        for row in read_jsonl(split_dir / f"manifest_{pool}.jsonl"):
            test_case_ids.add(str(row.get("base_id") or row["id"]))

    variants = set(args.variants)
    cases = []
    vocab_set: dict[str, str] = {}
    for row in read_jsonl(args.cases):
        if variants and row.get("variant") not in variants:
            continue
        for cue in row.get("cue_targets") or []:
            text = " ".join(str(cue).split())
            if text:
                vocab_set.setdefault(text.lower(), text)
        if str(row.get("base_id") or row["id"]) in test_case_ids:
            cases.append(row)
    if not cases:
        raise ValueError("No test-pool cases found. Check --cases and --split-dir.")
    vocab = sorted(vocab_set.values())

    rng = random.Random(args.seed)
    rng.shuffle(cases)
    out_rows: list[dict[str, Any]] = []
    used_cases = 0
    skipped = 0
    for case in cases:
        if used_cases >= args.num_cases:
            break
        rows = counterfactual_rows_for_case(
            case, vocab=vocab, rng=rng, strategy=args.target_text_strategy
        )
        if rows is None:
            skipped += 1
            continue
        out_rows.extend(rows)
        used_cases += 1
    if not out_rows:
        raise ValueError("No counterfactual rows produced (all cases skipped).")

    seen_ids = set()
    for row in out_rows:
        if row["id"] in seen_ids:
            raise ValueError(f"Duplicate counterfactual row id: {row['id']}")
        seen_ids.add(row["id"])
    write_jsonl(Path(args.output), out_rows)
    print(
        f"[done] wrote {len(out_rows)} counterfactual rows from {used_cases} cases "
        f"to {args.output} (skipped_cases={skipped}, cue_vocab={len(vocab)})"
    )


if __name__ == "__main__":
    main()
