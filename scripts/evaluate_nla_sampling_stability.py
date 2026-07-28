"""Evaluate NLA-only sampling-stability features for source-error prediction."""

from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_error_prediction import auroc, average_precision
from scripts.score_medical_nla_v2_readouts import score_row
from src.jsonl import read_jsonl, write_jsonl


def normalize_answer(text: Any) -> str:
    text = str(text or "").lower().replace("/", " ").replace("_", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return " ".join(text.split())


def source_rows_by_id(path: str) -> dict[str, dict[str, Any]]:
    out = {}
    for row in read_jsonl(path):
        out[str(row["id"])] = row
    return out


def entropy_norm(counts: Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    entropy = -sum((count / total) * math.log(max(count / total, 1e-30)) for count in counts.values())
    return entropy / math.log(total)


def aggregate_group(row_id: str, samples: list[dict[str, Any]], source: dict[str, Any] | None) -> dict[str, Any]:
    scored = [score_row(row) for row in samples]
    answer_counts = Counter(normalize_answer(row["answer_readout"]) for row in scored if row["answer_readout"])
    top_answer, top_count = ("", 0)
    if answer_counts:
        top_answer, top_count = answer_counts.most_common(1)[0]
    n = len(scored)
    parsed_answer = sum(bool(row["parsed_answer"]) for row in scored)
    parsed_readout = sum(bool(row["parsed_readout"]) for row in scored)
    answer_hits = sum(bool(row["answer_hit"]) for row in scored)
    output_answer_hits = sum(bool(row["output_answer_hit"]) for row in scored)
    cue_recalls = [float(row["cue_recall"]) for row in scored if row["cue_recall"] is not None]
    cjk_values = [float(row.get("cjk_fraction") or 0.0) for row in scored]

    source_correct = None
    source_selected_id = None
    source_selected_name = None
    source_answer = None
    if source is not None:
        source_correct = bool(source.get("diagnosis_hit"))
        source_selected_id = source.get("selected_diagnosis_id")
        source_selected_name = source.get("selected_diagnosis_name")
        source_answer = source.get("answer")

    first = scored[0]
    return {
        "id": row_id,
        "base_id": first.get("base_id", row_id),
        "prompt": first.get("prompt"),
        "diagnosis_id": first.get("diagnosis_id") or first.get("gold_diagnosis_id"),
        "diagnosis_name": first.get("diagnosis_name") or first.get("gold_diagnosis_name"),
        "source_correct": source_correct,
        "is_error": None if source_correct is None else not source_correct,
        "source_selected_diagnosis_id": source_selected_id,
        "source_selected_diagnosis_name": source_selected_name,
        "source_answer": source_answer,
        "n_samples": n,
        "parsed_answer_rate": parsed_answer / n if n else 0.0,
        "parsed_readout_rate": parsed_readout / n if n else 0.0,
        "top_answer_norm": top_answer,
        "top_answer_count": top_count,
        "top_answer_fraction": top_count / n if n else 0.0,
        "unique_answer_count": len(answer_counts),
        "unique_answer_fraction": len(answer_counts) / n if n else 0.0,
        "answer_entropy_norm": entropy_norm(answer_counts),
        "sample_answer_hit_rate": answer_hits / n if n else 0.0,
        "sample_output_answer_hit_rate": output_answer_hits / n if n else 0.0,
        "mean_cue_recall": mean(cue_recalls) if cue_recalls else None,
        "mean_cjk_fraction": mean(cjk_values) if cjk_values else 0.0,
        "answer_counts": dict(answer_counts),
        "sample_answers": [row["answer_readout"] for row in scored],
    }


def score_specs() -> list[tuple[str, str, int]]:
    return [
        ("nla_answer_entropy", "answer_entropy_norm", 1),
        ("nla_low_top_answer_fraction", "top_answer_fraction", -1),
        ("nla_unique_answer_fraction", "unique_answer_fraction", 1),
        ("nla_low_parsed_answer_rate", "parsed_answer_rate", -1),
        ("nla_low_mean_cue_recall", "mean_cue_recall", -1),
    ]


def metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for name, field, direction in score_specs():
        labels = []
        scores = []
        for row in rows:
            label = row.get("is_error")
            value = row.get(field)
            if label is None or value is None:
                continue
            labels.append(bool(label))
            scores.append(direction * float(value))
        out.append(
            {
                "signal": name,
                "field": field,
                "n": len(labels),
                "auroc": auroc(labels, scores) if labels else None,
                "average_precision": average_precision(labels, scores) if labels else None,
                "notes": "NLA-only sampling-stability feature",
            }
        )
    return out


def write_summary(path: Path, rows: list[dict[str, Any]], metrics: list[dict[str, Any]]) -> None:
    labels = [bool(row["is_error"]) for row in rows if row.get("is_error") is not None]
    with path.open("w", encoding="utf-8") as f:
        f.write("# NLA Sampling Stability Error Prediction\n\n")
        f.write("NLA-only features; no source answer comparison is used as a feature.\n\n")
        f.write(f"- n: {len(rows)}\n")
        if labels:
            f.write(f"- errors: {sum(labels)}/{len(labels)}\n")
            f.write(f"- error_rate: {mean(labels):.4f}\n")
        f.write(f"- mean_top_answer_fraction: {mean(float(row['top_answer_fraction']) for row in rows):.4f}\n")
        f.write(f"- mean_answer_entropy_norm: {mean(float(row['answer_entropy_norm']) for row in rows):.4f}\n")
        f.write(f"- mean_unique_answer_count: {mean(int(row['unique_answer_count']) for row in rows):.2f}\n\n")

        f.write("| signal | n | auroc | average_precision | notes |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for row in metrics:
            auroc_text = "-" if row["auroc"] is None else f"{row['auroc']:.4f}"
            ap_text = "-" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
            f.write(f"| {row['signal']} | {row['n']} | {auroc_text} | {ap_text} | {row['notes']} |\n")

        f.write("\n## Most Unstable Examples\n\n")
        f.write("| id | source_correct | expected | top_fraction | entropy | unique | sample_answers |\n")
        f.write("|---|---:|---|---:|---:|---:|---|\n")
        unstable = sorted(
            rows,
            key=lambda row: (float(row["answer_entropy_norm"]), int(row["unique_answer_count"])),
            reverse=True,
        )[:50]
        for row in unstable:
            answers = "; ".join(str(answer) for answer in row["sample_answers"][:12])
            answers = answers.replace("\n", " ")
            if len(answers) > 180:
                answers = answers[:177] + "..."
            f.write(
                f"| {row['id']} | {row.get('source_correct')} | {row.get('diagnosis_name')} | "
                f"{row['top_answer_fraction']:.3f} | {row['answer_entropy_norm']:.3f} | "
                f"{row['unique_answer_count']} | {answers} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", required=True)
    parser.add_argument("--source-answers", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--metrics-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    source_by_id = source_rows_by_id(args.source_answers)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(args.samples):
        groups[str(row["id"])].append(row)
    aggregated = [
        aggregate_group(row_id, sorted(samples, key=lambda row: int(row.get("sample_index", 0))), source_by_id.get(row_id))
        for row_id, samples in sorted(groups.items())
    ]
    metrics = metric_rows(aggregated)
    write_jsonl(Path(args.output_jsonl), aggregated)
    write_jsonl(Path(args.metrics_jsonl), metrics)
    write_summary(Path(args.summary_md), aggregated, metrics)
    print(f"[done] wrote {len(aggregated)} aggregate rows to {args.output_jsonl}")
    print(f"[done] wrote metrics to {args.metrics_jsonl}")
    print(f"[done] wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
