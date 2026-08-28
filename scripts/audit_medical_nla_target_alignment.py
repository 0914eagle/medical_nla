"""Audit whether a Medical-NLA assigns patient targets to patient activations.

The gate compares each Direct validation pair against deterministic donors from
the same disease category:

* matched: own activation, own physician-observation target;
* target shuffled: own activation, donor target;
* activation shuffled: donor activation, own target.

Content-token NLL is used so XML scaffold and target length do not decide the
result. This is a development audit and must not read locked test manifests.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_medical_nla_lora import build_training_example, collate_examples
from src.config import load_config
from src.jsonl import read_jsonl, write_jsonl
from src.modeling import load_causal_lm, load_tokenizer, maybe_load_peft_adapter
from src.nla import adapter_av_prompt, load_nla_sidecar


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def category(row: dict[str, Any]) -> str:
    return clean(row.get("disease_category")).casefold()


def row_id(row: dict[str, Any]) -> str:
    return clean(row.get("base_id") or row.get("id"))


def same_category_pairs(
    rows: list[dict[str, Any]], *, seed: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return a one-to-one within-category derangement with distinct targets."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = category(row)
        if key:
            grouped[key].append(row)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        ordered = sorted(group, key=row_id)
        random.Random(f"{seed}:{key}").shuffle(ordered)
        # Select the cyclic derangement that retains the most distinct target
        # pairs. It remains one-to-one and deterministic; identical-target
        # pairs are omitted because they are not a negative control.
        candidates = []
        for offset in range(1, len(ordered)):
            donors = ordered[offset:] + ordered[:offset]
            distinct = sum(
                clean(own.get("target_text")) != clean(donor.get("target_text"))
                for own, donor in zip(ordered, donors, strict=True)
            )
            candidates.append((distinct, -offset, donors))
        _distinct, _neg_offset, donors = max(candidates, key=lambda item: item[:2])
        for own, donor in zip(ordered, donors, strict=True):
            if clean(own.get("target_text")) == clean(donor.get("target_text")):
                continue
            pairs.append((own, donor))
    return sorted(pairs, key=lambda pair: row_id(pair[0]))


def condition_rows(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]], condition: str
) -> list[dict[str, Any]]:
    output = []
    for own, donor in pairs:
        row = dict(own)
        row["alignment_condition"] = condition
        row["donor_base_id"] = row_id(donor)
        if condition == "target_shuffled":
            row["target_text"] = donor["target_text"]
            row["cue_targets"] = donor.get("cue_targets")
        elif condition == "activation_shuffled":
            row["activation_path"] = donor["activation_path"]
        elif condition != "matched":
            raise ValueError(f"Unknown alignment condition: {condition}")
        output.append(row)
    return output


@torch.inference_mode()
def score_content_nll(
    *,
    rows: list[dict[str, Any]],
    model: torch.nn.Module,
    tokenizer: Any,
    embed_layer: torch.nn.Module,
    sidecar: Any,
    actor_prompt_template: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    model.eval()
    results = []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        examples = [
            build_training_example(
                row=row,
                tokenizer=tokenizer,
                model=model,
                embed_layer=embed_layer,
                sidecar=sidecar,
                actor_prompt_template=actor_prompt_template,
                eos_token_id=tokenizer.eos_token_id,
            )
            for row in batch_rows
        ]
        inputs_embeds, attention_mask, labels, content = collate_examples(examples)
        logits = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
        shifted_logits = logits[:, :-1, :].float().transpose(1, 2)
        shifted_labels = labels[:, 1:]
        per_token = torch.nn.functional.cross_entropy(
            shifted_logits,
            shifted_labels,
            reduction="none",
            ignore_index=-100,
        )
        masks = (shifted_labels != -100) & content[:, 1:]
        for row, losses, mask in zip(batch_rows, per_token, masks, strict=True):
            count = int(mask.sum().item())
            if not count:
                raise ValueError(f"No content tokens for {row_id(row)}")
            results.append(
                {
                    "base_id": row_id(row),
                    "donor_base_id": row["donor_base_id"],
                    "disease_category": row.get("disease_category"),
                    "condition": row["alignment_condition"],
                    "content_nll": float(losses[mask].mean().item()),
                    "content_tokens": count,
                }
            )
        print(f"[score] {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    return results


def bootstrap_ci(values: list[float], *, seed: int, draws: int = 5000) -> list[float]:
    if not values:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    estimates = sorted(
        mean(rng.choice(values) for _ in values) for _ in range(draws)
    )
    return [
        estimates[int(0.025 * (draws - 1))],
        estimates[int(0.975 * (draws - 1))],
    ]


def summarize(
    scores: list[dict[str, Any]], *, eligible_rows: int, seed: int
) -> dict[str, Any]:
    by_condition = {
        condition: {
            row["base_id"]: row
            for row in scores
            if row["condition"] == condition
        }
        for condition in ("matched", "target_shuffled", "activation_shuffled")
    }
    ids = sorted(set.intersection(*(set(rows) for rows in by_condition.values())))
    result: dict[str, Any] = {
        "eligible_direct_rows": eligible_rows,
        "paired_rows": len(ids),
        "pair_coverage": len(ids) / eligible_rows if eligible_rows else 0.0,
        "conditions": {},
        "gaps": {},
    }
    for condition, rows in by_condition.items():
        result["conditions"][condition] = {
            "mean_content_nll": mean(rows[identifier]["content_nll"] for identifier in ids),
        }
    matched = by_condition["matched"]
    for condition in ("target_shuffled", "activation_shuffled"):
        control = by_condition[condition]
        # Positive means the matched pair receives lower NLL.
        gaps = [
            control[identifier]["content_nll"] - matched[identifier]["content_nll"]
            for identifier in ids
        ]
        result["gaps"][condition] = {
            "control_minus_matched": mean(gaps),
            "bootstrap_95_ci": bootstrap_ci(gaps, seed=seed),
            "matched_win_rate": sum(value > 0 for value in gaps) / len(gaps),
        }
    # A one-sided target shuffle is noisy because some target texts are
    # intrinsically easier than others. The 2x2 cross difference cancels that
    # nuisance by scoring both targets under both activations:
    #   [NLL(y_j|h_i) + NLL(y_i|h_j) - NLL(y_i|h_i) - NLL(y_j|h_j)] / 2.
    symmetric = []
    for identifier in ids:
        donor_id = by_condition["target_shuffled"][identifier]["donor_base_id"]
        if donor_id not in matched:
            continue
        symmetric.append(
            0.5
            * (
                by_condition["target_shuffled"][identifier]["content_nll"]
                + by_condition["activation_shuffled"][identifier]["content_nll"]
                - matched[identifier]["content_nll"]
                - matched[donor_id]["content_nll"]
            )
        )
    result["symmetric_cross"] = {
        "n": len(symmetric),
        "cross_minus_matched": mean(symmetric) if symmetric else float("nan"),
        "bootstrap_95_ci": bootstrap_ci(symmetric, seed=seed),
        "matched_win_rate": (
            sum(value > 0 for value in symmetric) / len(symmetric)
            if symmetric
            else float("nan")
        ),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--source-dataset", default="direct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--resume-scores",
        action="store_true",
        help="Reuse a complete --output-jsonl and only recompute aggregate summaries.",
    )
    args = parser.parse_args()

    rows = [
        row
        for row in read_jsonl(args.manifest)
        if str(row.get("source_dataset")) == args.source_dataset
    ]
    if not rows:
        raise ValueError(f"No {args.source_dataset} rows in {args.manifest}")
    pairs = same_category_pairs(rows, seed=args.seed)
    if not pairs:
        raise ValueError("No within-category target derangements available")
    print(f"[pairs] eligible={len(rows)} paired={len(pairs)}", flush=True)

    expected_scores = 3 * len(pairs)
    if args.resume_scores:
        scores = list(read_jsonl(args.output_jsonl))
        if len(scores) != expected_scores:
            raise ValueError(
                f"Existing scores contain {len(scores)} rows; expected {expected_scores}"
            )
        print(f"[resume] reusing {len(scores)} score rows", flush=True)
    else:
        cfg = load_config(args.config)
        nla_cfg = cfg["nla_model"]
        cache_dir = cfg["paths"].get("cache_dir")
        tokenizer = load_tokenizer(
            nla_cfg["model_id"],
            cache_dir=cache_dir,
            trust_remote_code=nla_cfg.get("trust_remote_code", True),
        )
        sidecar = load_nla_sidecar(
            nla_cfg["model_id"],
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            filename=nla_cfg.get("sidecar_filename", "nla_meta.yaml"),
            expected_d_model=nla_cfg.get("expected_d_model"),
            expected_injection_token_id=nla_cfg.get("expected_injection_token_id"),
        )
        actor_prompt_template = adapter_av_prompt(str(args.adapter))
        if actor_prompt_template is None:
            raise FileNotFoundError(f"Adapter has no recorded AV prompt: {args.adapter}")
        model = load_causal_lm(nla_cfg, cache_dir=cache_dir)
        model = maybe_load_peft_adapter(model, str(args.adapter), cache_dir=cache_dir)
        embed_layer = model.get_input_embeddings()

        scores = []
        for condition in ("matched", "target_shuffled", "activation_shuffled"):
            print(f"[condition] {condition}", flush=True)
            scores.extend(
                score_content_nll(
                    rows=condition_rows(pairs, condition),
                    model=model,
                    tokenizer=tokenizer,
                    embed_layer=embed_layer,
                    sidecar=sidecar,
                    actor_prompt_template=actor_prompt_template,
                    batch_size=args.batch_size,
                )
            )
    result = summarize(scores, eligible_rows=len(rows), seed=args.seed)
    result.update(
        {
            "manifest": str(args.manifest),
            "adapter": str(args.adapter),
            "source_dataset": args.source_dataset,
            "seed": args.seed,
            "selection_rule": "symmetric cross-minus-matched bootstrap interval above zero",
        }
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if not args.resume_scores:
        write_jsonl(args.output_jsonl, scores)
    args.output_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# Medical-NLA Activation-Target Alignment Gate",
        "",
        "Validation-only; donors are deterministic derangements within disease category.",
        "Positive gaps mean the matched activation/target pair has lower content NLL.",
        "",
        f"- eligible Direct rows: **{result['eligible_direct_rows']}**",
        f"- paired rows: **{result['paired_rows']}** ({result['pair_coverage']:.4f})",
        "",
        "| condition | mean content NLL | control-minus-matched | bootstrap 95% CI | matched win rate |",
        "|---|---:|---:|---:|---:|",
    ]
    matched_nll = result["conditions"]["matched"]["mean_content_nll"]
    lines.append(f"| matched | {matched_nll:.4f} | - | - | - |")
    for condition in ("target_shuffled", "activation_shuffled"):
        values = result["gaps"][condition]
        ci = values["bootstrap_95_ci"]
        lines.append(
            f"| {condition} | {result['conditions'][condition]['mean_content_nll']:.4f} | "
            f"{values['control_minus_matched']:+.4f} | "
            f"[{ci[0]:+.4f}, {ci[1]:+.4f}] | {values['matched_win_rate']:.4f} |"
        )
    symmetric = result["symmetric_cross"]
    symmetric_ci = symmetric["bootstrap_95_ci"]
    lines.extend(
        [
            "",
            "| primary symmetric 2x2 gate | n | cross-minus-matched | bootstrap 95% CI | matched win rate |",
            "|---|---:|---:|---:|---:|",
            f"| target/activation cross | {symmetric['n']} | "
            f"{symmetric['cross_minus_matched']:+.4f} | "
            f"[{symmetric_ci[0]:+.4f}, {symmetric_ci[1]:+.4f}] | "
            f"{symmetric['matched_win_rate']:.4f} |",
        ]
    )
    lines.extend(
        [
            "",
            "The primary gate passes only when the symmetric bootstrap interval is strictly above zero.",
            "One-sided shuffle gaps are retained as diagnostics because they include target-difficulty noise.",
        ]
    )
    args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.summary_md.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
