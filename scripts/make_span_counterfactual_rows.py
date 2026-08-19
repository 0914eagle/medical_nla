"""Span-substitution counterfactuals for natural-text corpora.

`make_cue_counterfactual_rows` rebuilds the prompt from its cue list and
requires the rebuild to match the stored prompt byte-for-byte. That works for
DDXPlus, whose prompts are assembled from cues, but not for corpora written
as prose (MedCaseReasoning case reports, PHEE sentences), where no template
exists to rebuild from.

This generator gets the same guarantee a different way: it edits the prompt
in place, replacing exactly one cue's character span with a donor cue and
leaving every other byte untouched. The intervention is therefore still
minimal and exact — the swapped span is the only difference between the
original and counterfactual prompt — which is what the faithfulness claim
needs.

Emits the same row schema as the DDXPlus counterfactual generator, so
`src.extract_activations` and `scripts.evaluate_cue_counterfactuals` consume
it unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl

CARRY_FIELDS = (
    "diagnosis_id",
    "diagnosis_name",
    "diagnosis_aliases",
    "source",
    "case_id",
    "patient_id",
)


def cue_spans(prompt: str, cues: list[str]) -> list[tuple[int, int] | None]:
    """Char span per cue, scanning left to right so repeats resolve in order."""
    spans: list[tuple[int, int] | None] = []
    search_from = 0
    for cue in cues:
        pos = prompt.find(cue, search_from)
        if pos == -1:
            pos = prompt.find(cue)
        if pos == -1:
            spans.append(None)
            continue
        spans.append((pos, pos + len(cue)))
        search_from = pos + len(cue)
    return spans


def spans_overlap(spans: list[tuple[int, int]]) -> bool:
    ordered = sorted(spans)
    return any(a[1] > b[0] for a, b in zip(ordered, ordered[1:]))


def substitute(prompt: str, span: tuple[int, int], replacement: str) -> str:
    return prompt[: span[0]] + replacement + prompt[span[1] :]


def remove_span(prompt: str, span: tuple[int, int]) -> str:
    """Delete a span, then tidy the double space / stranded comma it leaves."""
    out = prompt[: span[0]] + prompt[span[1] :]
    out = out.replace("  ", " ").replace(" ,", ",").replace(" .", ".")
    return out.replace(", ,", ",").strip()


def extraction_row(
    *,
    base_id: str,
    case: dict[str, Any],
    variant: str,
    role: str,
    prompt: str,
    slot: int,
    gold_cue: str,
    strategy: str,
    extra: dict[str, Any],
) -> dict[str, Any] | None:
    if gold_cue not in prompt:
        return None
    occurrence = prompt[: prompt.find(gold_cue)].count(gold_cue)
    row = {field: case.get(field) for field in CARRY_FIELDS if case.get(field) is not None}
    row.update(
        {
            "id": f"{base_id}__cf_{variant}__slot{slot:02d}",
            "base_id": base_id,
            "variant": f"cue_counterfactual_{variant}",
            "cf_variant": variant,
            "cf_role": role,
            "cf_slot": slot,
            "cf_method": "span_substitution",
            "prompt": prompt,
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
    swap_slots: int,
) -> list[dict[str, Any]] | None:
    prompt = str(case.get("prompt") or "")
    cues = [str(cue) for cue in (case.get("cue_targets") or []) if str(cue).strip()]
    if not prompt or len(cues) < 3:
        return None

    spans = cue_spans(prompt, cues)
    if any(span is None for span in spans):
        return None
    resolved: list[tuple[int, int]] = [span for span in spans if span is not None]
    # Overlapping cue spans would make an edit to one corrupt another.
    if spans_overlap(resolved):
        return None

    base_id = str(case.get("base_id") or case["id"])
    case_cues_lower = {cue.lower() for cue in cues}
    candidates = [cue for cue in vocab if cue.lower() not in case_cues_lower]
    if not candidates:
        return None

    n_swaps = min(swap_slots, len(cues) - 2)
    swap_indices = rng.sample(range(len(cues)), n_swaps)
    rows: list[dict[str, Any]] = []

    for slot in sorted(swap_indices):
        replacement = rng.choice(candidates)
        retained_pool = [i for i in range(len(cues)) if i != slot]
        retained = rng.sample(retained_pool, min(2, len(retained_pool)))
        shared = {
            "cf_original_cue": cues[slot],
            "cf_replacement_cue": replacement,
            "cf_removed_cue": cues[slot],
            "cf_swap_group": f"{base_id}__g{slot:02d}",
        }
        swap_prompt = substitute(prompt, resolved[slot], replacement)
        removed_prompt = remove_span(prompt, resolved[slot])

        specs = [
            ("orig", "swapped_slot", prompt, slot, cues[slot]),
            ("swap", "swapped_slot", swap_prompt, slot, replacement),
        ]
        for r in retained:
            specs.append(("orig", "retained", prompt, r, cues[r]))
            specs.append(("swap", "retained", swap_prompt, r, cues[r]))
            specs.append(("removed", "retained", removed_prompt, r, cues[r]))

        group_rows = []
        for variant, role, variant_prompt, s, gold in specs:
            row = extraction_row(
                base_id=base_id,
                case=case,
                variant=variant,
                role=role,
                prompt=variant_prompt,
                slot=s,
                gold_cue=gold,
                strategy=strategy,
                extra=shared,
            )
            if row is None:
                group_rows = []
                break
            row["id"] = f"{row['id']}__g{slot:02d}"
            group_rows.append(row)
        rows.extend(group_rows)

    return rows or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Case manifest with prompt + cue_targets.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--variants", nargs="+", default=["cue_count_all"])
    parser.add_argument("--num-cases", type=int, default=500)
    parser.add_argument(
        "--swap-slots",
        type=int,
        default=2,
        help="Swapped slots per case; each yields its own orig/swap/removed group.",
    )
    parser.add_argument(
        "--target-text-strategy", default="last_subtoken", choices=["last_subtoken", "span_mean"]
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    variants = set(args.variants)
    keep_all = "any" in variants
    cases: list[dict[str, Any]] = []
    vocab_set: dict[str, str] = {}
    for row in read_jsonl(args.cases):
        if not keep_all and row.get("variant") not in variants:
            continue
        cases.append(row)
        for cue in row.get("cue_targets") or []:
            text = str(cue).strip()
            if text:
                vocab_set.setdefault(text.lower(), text)
    vocab = sorted(vocab_set.values())
    if not cases or not vocab:
        raise ValueError("No eligible cases or empty cue vocabulary.")

    rng = random.Random(args.seed)
    rng.shuffle(cases)

    rows: list[dict[str, Any]] = []
    used_cases = 0
    skipped = 0
    for case in cases:
        if used_cases >= args.num_cases:
            break
        case_rows = counterfactual_rows_for_case(
            case,
            vocab=vocab,
            rng=rng,
            strategy=args.target_text_strategy,
            swap_slots=args.swap_slots,
        )
        if case_rows is None:
            skipped += 1
            continue
        rows.extend(case_rows)
        used_cases += 1

    if not rows:
        raise ValueError("No counterfactual rows produced.")

    write_jsonl(Path(args.output), rows)
    n_swap_pairs = sum(1 for row in rows if row["cf_role"] == "swapped_slot") // 2
    summary = {
        "cases_used": used_cases,
        "cases_skipped": skipped,
        "rows": len(rows),
        "swap_pairs": n_swap_pairs,
        "cue_vocab": len(vocab),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(rows)} span-counterfactual rows to {args.output}")


if __name__ == "__main__":
    main()
