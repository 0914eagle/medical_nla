"""Build train/validation SFT rows for the DiReCT P0 Medical-NLA pilot.

Observation targets come only from physician annotations that are exact
substrings of the note. The answer target is the backbone's own answer from the
same case, not the physician gold diagnosis. This keeps source-wrong cases from
being silently converted into gold-correction supervision.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from src.jsonl import read_jsonl, write_jsonl


SPLITS = ("train", "val_seen")


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def xml_text(value: Any) -> str:
    return (
        clean_text(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identifier = base_id(row)
        if not identifier or identifier in result:
            raise ValueError(f"Missing or duplicate {label} ID: {identifier!r}")
        result[identifier] = row
    return result


def grounded_observations(row: dict[str, Any], *, max_observations: int, seed: int) -> list[str]:
    observations: list[str] = []
    seen: set[str] = set()
    for deduction in row.get("gold_deductions") or []:
        if deduction.get("observation_exact_in_note") is not True:
            continue
        observation = clean_text(deduction.get("observation"))
        key = observation.casefold()
        if observation and key not in seen:
            observations.append(observation)
            seen.add(key)
    random.Random(f"{seed}:{row['id']}").shuffle(observations)
    return observations[:max_observations]


def target_text(observations: list[str], source_answer: str) -> str:
    observed = "\n".join(f"- {xml_text(item)}" for item in observations)
    return "\n".join(
        [
            "<explanation>",
            "<readout>",
            "<observed>",
            observed,
            "</observed>",
            f"<answer>{xml_text(source_answer)}</answer>",
            "</readout>",
            "</explanation>",
        ]
    )


def hash_ids(rows: list[dict[str, Any]]) -> str:
    payload = "\n".join(sorted(base_id(row) for row in rows)).encode()
    return hashlib.sha256(payload).hexdigest()


def build_split(
    *,
    split: str,
    split_rows: list[dict[str, Any]],
    activation_rows: list[dict[str, Any]],
    source_answers: dict[str, dict[str, Any]],
    max_observations: int,
    seed: int,
    include_gold_label_in_note: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    clinical = index_unique(split_rows, f"{split} clinical")
    activations = index_unique(activation_rows, f"{split} activation")
    if set(clinical) != set(activations):
        raise ValueError(
            f"{split} clinical/activation ID mismatch: "
            f"clinical_only={len(set(clinical) - set(activations))}, "
            f"activation_only={len(set(activations) - set(clinical))}"
        )

    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    for identifier in sorted(clinical):
        if (
            not include_gold_label_in_note
            and clinical[identifier].get("gold_label_exact_in_note") is True
        ):
            counts["gold_label_exact_in_note"] += 1
            continue
        source = source_answers.get(identifier)
        if source is None:
            counts["missing_source_answer"] += 1
            continue
        answer = clean_text(source.get("answer"))
        if not answer:
            counts["empty_source_answer"] += 1
            continue
        observations = grounded_observations(
            clinical[identifier], max_observations=max_observations, seed=seed
        )
        if not observations:
            counts["no_exact_grounded_observation"] += 1
            continue
        activation = activations[identifier]
        if str(activation.get("position_family")) != "P0":
            raise ValueError(f"Non-P0 activation for {identifier}")
        if int(activation.get("layer")) != 32:
            raise ValueError(f"Non-HS32 activation for {identifier}")
        activation_path = Path(str(activation.get("activation_path") or ""))
        if not activation_path.is_file():
            raise FileNotFoundError(activation_path)
        output.append(
            {
                "id": f"{identifier}__direct_e3_p0",
                "base_id": identifier,
                "source_dataset": "direct",
                "split": split,
                "activation_path": str(activation_path),
                "position_family": "P0",
                "layer": 32,
                "patient_group": clinical[identifier].get("patient_group"),
                "disease_category": clinical[identifier].get("disease_category"),
                "canonical_pdd": clinical[identifier].get("canonical_pdd"),
                "source_answer": answer,
                "source_correct": bool(source.get("source_correct")),
                "cue_targets": observations,
                "target_style": "direct_p0_grounded_observations_source_decision_v1",
                "target_text": target_text(observations, answer),
            }
        )
    counts["written"] = len(output)
    counts["source_correct"] = sum(row["source_correct"] for row in output)
    counts["source_wrong"] = len(output) - counts["source_correct"]
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--activation-root", required=True, type=Path)
    parser.add_argument("--source-answers", required=True, nargs="+", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--max-observations", type=int, default=12)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--include-gold-label-in-note",
        action="store_true",
        help="Sensitivity only: retain rows whose note states the normalized gold label.",
    )
    args = parser.parse_args()

    if args.max_observations <= 0:
        raise ValueError("--max-observations must be positive")
    source_rows = [
        row for path in args.source_answers for row in read_jsonl(path)
    ]
    source_answers = index_unique(source_rows, "source answer")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, tuple[list[dict[str, Any]], Counter[str]]] = {}
    for split in SPLITS:
        rows, counts = build_split(
            split=split,
            split_rows=list(read_jsonl(args.split_dir / f"{split}.jsonl")),
            activation_rows=list(
                read_jsonl(
                    args.activation_root
                    / "layer32"
                    / "last_token"
                    / f"manifest_{split}.jsonl"
                )
            ),
            source_answers=source_answers,
            max_observations=args.max_observations,
            seed=args.seed,
            include_gold_label_in_note=args.include_gold_label_in_note,
        )
        output_name = "sft_train.jsonl" if split == "train" else "sft_val.jsonl"
        write_jsonl(args.out_dir / output_name, rows)
        summaries[split] = (rows, counts)

    lines = [
        "# DiReCT E3 SFT Dataset",
        "",
        "Private train/validation targets. No test manifest was read.",
        "",
        "Targets contain exact-note-grounded physician observations and the source model's own diagnosis.",
        "Gold diagnosis is metadata only and is never substituted for a source-wrong answer.",
        f"Gold-label-in-note rows retained: **{args.include_gold_label_in_note}**.",
        "",
        "| split | rows | source correct | source wrong | ID hash |",
        "|---|---:|---:|---:|---|",
    ]
    for split in SPLITS:
        rows, counts = summaries[split]
        lines.append(
            f"| {split} | {len(rows)} | {counts['source_correct']} | "
            f"{counts['source_wrong']} | `{hash_ids(rows)}` |"
        )
        omitted = {
            key: value
            for key, value in counts.items()
            if key not in {"written", "source_correct", "source_wrong"} and value
        }
        if omitted:
            lines.append(f"\n- {split} omissions: `{dict(omitted)}`")
    (args.out_dir / "summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"[dataset] {args.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
