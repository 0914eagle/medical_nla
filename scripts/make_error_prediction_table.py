"""Merge source-model and NLA outputs into an error-prediction table.

The supervised label is whether the source model answer was correct. The table
also keeps post-hoc gold-aware fields, such as `nla_answer_hit`, for analysis.
Features that can be used at inference time should avoid gold labels; the main
gold-free feature here is agreement between the source selected diagnosis and
the Medical-NLA structured answer.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def key(row: dict[str, Any], field: str = "id") -> str:
    return str(row.get(field) or row.get("id") or row.get("base_id"))


def read_by_key(path: str | None, field: str = "id") -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = {}
    for row in read_jsonl(path):
        rows[key(row, field)] = row
    return rows


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"true", "t", "1", "yes", "y"}:
        return True
    if text in {"false", "f", "0", "no", "n"}:
        return False
    return None


def add_rank_features(out: dict[str, Any], prefix: str, row: dict[str, Any] | None) -> None:
    if not row:
        return
    for field in (
        "gold_rank",
        "gold_top1",
        "gold_top5",
        "gold_logprob_mean",
        "gold_logprob_sum",
        "gold_first_token_logprob",
        "top1_diagnosis_id",
        "top1_diagnosis_name",
        "top1_logprob_mean",
        "top1_logprob_sum",
        "top1_prob",
    ):
        if field in row:
            out[f"{prefix}_{field}"] = row[field]


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower()
    text = text.replace("/", " ")
    text = text.replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def answer_agree(source_name: Any, nla_answer: Any) -> bool | None:
    source = normalize_answer(source_name)
    nla = normalize_answer(nla_answer)
    if not source or not nla:
        return None
    return source == nla or source in nla or nla in source


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["is_error"] for row in rows if row.get("is_error") is not None]
    overlap = Counter(
        (bool(row.get("source_correct")), bool(row.get("nla_answer_hit")))
        for row in rows
        if row.get("source_correct") is not None and row.get("nla_answer_hit") is not None
    )
    agreement_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        agree = row.get("source_nla_answer_agree")
        label = "unknown" if agree is None else ("agree" if agree else "disagree")
        agreement_groups[label].append(row)

    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_gold[str(row.get("gold_diagnosis_id") or "-")].append(row)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Error Prediction Feature Table\n\n")
        f.write(f"- n: {len(rows)}\n")
        if labels:
            f.write(f"- errors: {sum(labels)}/{len(labels)}\n")
            f.write(f"- error_rate: {mean(labels):.4f}\n")

        if overlap:
            f.write("\n## Source vs Medical-NLA Gold Hits\n\n")
            f.write("| source_correct | nla_answer_hit | n | rate |\n")
            f.write("|---:|---:|---:|---:|\n")
            total = sum(overlap.values())
            for source_correct, nla_hit in ((False, False), (False, True), (True, False), (True, True)):
                count = overlap[(source_correct, nla_hit)]
                f.write(f"| {source_correct} | {nla_hit} | {count} | {count / total if total else 0:.4f} |\n")

        if agreement_groups:
            f.write("\n## Source Accuracy by Source/NLA Agreement\n\n")
            f.write("| group | n | source_correct | source_accuracy |\n")
            f.write("|---|---:|---:|---:|\n")
            for label in ("agree", "disagree", "unknown"):
                items = agreement_groups.get(label, [])
                if not items:
                    continue
                correct = sum(bool(row.get("source_correct")) for row in items)
                f.write(f"| {label} | {len(items)} | {correct} | {correct / len(items):.4f} |\n")

        if by_gold:
            f.write("\n## Largest Medical-NLA Gains by Gold Diagnosis\n\n")
            f.write("| diagnosis_id | n | source_hits | nla_hits | diff |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            stats = []
            for diagnosis_id, items in by_gold.items():
                source_hits = sum(bool(row.get("source_correct")) for row in items)
                nla_hits = sum(bool(row.get("nla_answer_hit")) for row in items)
                stats.append((nla_hits - source_hits, diagnosis_id, len(items), source_hits, nla_hits))
            for diff, diagnosis_id, count, source_hits, nla_hits in sorted(stats, reverse=True)[:20]:
                f.write(f"| {diagnosis_id} | {count} | {source_hits} | {nla_hits} | {diff:+d} |\n")

            f.write("\n## Largest Source Gains by Gold Diagnosis\n\n")
            f.write("| diagnosis_id | n | source_hits | nla_hits | diff |\n")
            f.write("|---|---:|---:|---:|---:|\n")
            for diff, diagnosis_id, count, source_hits, nla_hits in sorted(stats)[:20]:
                f.write(f"| {diagnosis_id} | {count} | {source_hits} | {nla_hits} | {diff:+d} |\n")

        f.write("\n| feature | present |\n")
        f.write("|---|---:|\n")
        feature_names = sorted({field for row in rows for field in row if field not in {"id", "base_id", "prompt"}})
        for field in feature_names:
            f.write(f"| {field} | {sum(field in row for row in rows)} |\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-answers", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--key-field", default="id")
    parser.add_argument("--source-logprobs", default=None)
    parser.add_argument("--nla-scored", default=None)
    parser.add_argument("--nla-logprobs", default=None)
    parser.add_argument("--probe-predictions", default=None)
    args = parser.parse_args()

    source_answers = list(read_jsonl(args.source_answers))
    source_logprobs = read_by_key(args.source_logprobs, args.key_field)
    nla_scored = read_by_key(args.nla_scored, args.key_field)
    nla_logprobs = read_by_key(args.nla_logprobs, args.key_field)
    probe_predictions = read_by_key(args.probe_predictions, args.key_field)

    rows = []
    for source in source_answers:
        k = key(source, args.key_field)
        correct = as_bool(
            source.get("diagnosis_hit", source.get("answer_hit", source.get("is_correct")))
        )
        source_selected_id = source.get("selected_diagnosis_id")
        source_selected_name = source.get("selected_diagnosis_name")
        out = {
            "id": source.get("id", k),
            "base_id": source.get("base_id", k),
            "prompt": source.get("prompt"),
            "gold_diagnosis_id": source.get("diagnosis_id") or source.get("gold_diagnosis_id"),
            "gold_diagnosis_name": source.get("diagnosis_name")
            or source.get("specific_expected")
            or source.get("gold_diagnosis_name"),
            "source_answer": source.get("answer"),
            "source_selected_diagnosis_id": source_selected_id,
            "source_selected_diagnosis_name": source_selected_name,
            "source_correct": correct,
            "is_error": None if correct is None else not correct,
        }
        add_rank_features(out, "source", source_logprobs.get(k))
        nla_row = nla_scored.get(k)
        if nla_row:
            nla_answer = nla_row.get("answer_readout")
            out["nla_answer"] = nla_answer
            out["nla_answer_hit"] = nla_row.get("answer_hit")
            out["nla_output_answer_hit"] = nla_row.get("output_answer_hit")
            out["nla_parsed_readout"] = nla_row.get("parsed_readout")
            out["nla_parsed_answer"] = nla_row.get("parsed_answer")
            out["nla_cue_recall"] = nla_row.get("cue_recall")
            out["nla_cue_hit_count"] = nla_row.get("cue_hit_count")
            out["nla_cue_count"] = nla_row.get("cue_count")
            out["nla_cjk_fraction"] = nla_row.get("cjk_fraction")
            out["source_nla_answer_agree"] = answer_agree(source_selected_name, nla_answer)
            out["nla_output"] = nla_row.get("nla_output")
        add_rank_features(out, "nla", nla_logprobs.get(k))
        add_rank_features(out, "probe", probe_predictions.get(k))
        rows.append(out)

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, rows)
    write_summary(summary_path, rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
