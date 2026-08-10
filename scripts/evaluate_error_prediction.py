"""Evaluate error-prediction signals from a merged feature table."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def as_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def auroc(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    pairs = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum_pos = 0.0
    rank = 1
    idx = 0
    while idx < len(pairs):
        end = idx + 1
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + rank + (end - idx) - 1) / 2.0
        rank_sum_pos += avg_rank * sum(1 for _, label in pairs[idx:end] if label)
        rank += end - idx
        idx = end
    return (rank_sum_pos - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(labels: list[bool], scores: list[float]) -> float | None:
    positives = sum(labels)
    if positives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    hits = 0
    precision_sum = 0.0
    for idx, (_, label) in enumerate(ranked, start=1):
        if label:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / positives


def binary_metrics(labels: list[bool], predictions: list[bool]) -> dict[str, float | int]:
    tp = sum(pred and label for pred, label in zip(predictions, labels))
    fp = sum(pred and not label for pred, label in zip(predictions, labels))
    tn = sum((not pred) and (not label) for pred, label in zip(predictions, labels))
    fn = sum((not pred) and label for pred, label in zip(predictions, labels))
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / len(labels) if labels else 0.0,
    }


DISAGREE_SIGNALS = ("source_nla_disagree", "source_probe_disagree")


def score_specs() -> list[tuple[str, str, int]]:
    return [
        ("source_nla_disagree", "source_nla_answer_agree", -1),
        ("source_probe_disagree", "source_probe_answer_agree", -1),
        ("probe_low_top1_prob", "probe_top1_prob", -1),
        ("source_low_top1_prob", "source_top1_prob", -1),
        ("source_low_top1_top2_prob_margin", "source_top1_top2_prob_margin", -1),
        ("source_low_top1_top2_margin", "source_top1_top2_margin", -1),
        ("source_entropy", "source_candidate_entropy_norm", 1),
    ]


def collect_scores(
    rows: list[dict[str, Any]],
    *,
    name: str,
    field: str,
    direction: int,
) -> tuple[list[bool], list[float]]:
    labels = []
    scores = []
    for row in rows:
        label = as_bool(row.get("is_error"))
        if label is None:
            continue
        if name in DISAGREE_SIGNALS:
            agree = as_bool(row.get(field))
            if agree is None:
                continue
            score = 0.0 if agree else 1.0
        else:
            value = as_float(row.get(field))
            if value is None:
                continue
            score = direction * value
        labels.append(label)
        scores.append(score)
    return labels, scores


def paired_disagree_auroc(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """AUROC for both disagree signals on the rows where both are defined."""
    paired_rows = [
        row
        for row in rows
        if as_bool(row.get("source_nla_answer_agree")) is not None
        and as_bool(row.get("source_probe_answer_agree")) is not None
        and as_bool(row.get("is_error")) is not None
    ]
    if not paired_rows:
        return None
    nla_labels, nla_scores = collect_scores(
        paired_rows, name="source_nla_disagree", field="source_nla_answer_agree", direction=-1
    )
    probe_labels, probe_scores = collect_scores(
        paired_rows, name="source_probe_disagree", field="source_probe_answer_agree", direction=-1
    )
    return {
        "n": len(nla_labels),
        "nla_auroc": auroc(nla_labels, nla_scores),
        "probe_auroc": auroc(probe_labels, probe_scores),
    }


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    result_rows: list[dict[str, Any]],
    paired: dict[str, Any] | None,
) -> None:
    labels = [as_bool(row.get("is_error")) for row in rows]
    labels = [label for label in labels if label is not None]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Error Prediction Evaluation\n\n")
        f.write(f"- n: {len(labels)}\n")
        f.write(f"- errors: {sum(labels)}/{len(labels)}\n")
        f.write(f"- error_rate: {mean(labels):.4f}\n\n")
        f.write("| signal | n | auroc | average_precision | notes |\n")
        f.write("|---|---:|---:|---:|---|\n")
        for row in result_rows:
            auroc_text = "-" if row["auroc"] is None else f"{row['auroc']:.4f}"
            ap_text = "-" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
            f.write(
                f"| {row['signal']} | {row['n']} | {auroc_text} | {ap_text} | "
                f"{row.get('notes', '')} |\n"
            )
        titles = {
            "source_nla_disagree": "Source/NLA Disagree Predicts Source Error",
            "source_probe_disagree": "Source/Probe Disagree Predicts Source Error (control)",
        }
        for signal, title in titles.items():
            rule = next(
                (row for row in result_rows if row["signal"] == signal and "precision" in row),
                None,
            )
            if not rule:
                continue
            f.write(f"\n## Binary Rule: {title}\n\n")
            f.write(f"- precision: {rule['precision']:.4f}\n")
            f.write(f"- recall: {rule['recall']:.4f}\n")
            f.write(f"- specificity: {rule['specificity']:.4f}\n")
            f.write(f"- accuracy: {rule['accuracy']:.4f}\n")
            f.write(f"- tp/fp/tn/fn: {rule['tp']}/{rule['fp']}/{rule['tn']}/{rule['fn']}\n")

        if paired and paired["nla_auroc"] is not None and paired["probe_auroc"] is not None:
            f.write("\n## NLA vs Probe Disagreement Control (paired rows)\n\n")
            f.write("Computed only on rows where both agreement fields are present.\n\n")
            f.write(f"- paired_n: {paired['n']}\n")
            f.write(f"- nla_disagree_auroc: {paired['nla_auroc']:.4f}\n")
            f.write(f"- probe_disagree_auroc: {paired['probe_auroc']:.4f}\n")
            f.write(
                f"- nla_minus_probe_auroc: {paired['nla_auroc'] - paired['probe_auroc']:+.4f}\n"
            )
            f.write(
                "\nIf the probe control matches the NLA signal, the natural-language "
                "readout adds no error-detection value over a linear probe on the "
                "same activation; the case for NLA must then rest on explanation "
                "quality, not detection.\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    results = []
    for name, field, direction in score_specs():
        labels, scores = collect_scores(rows, name=name, field=field, direction=direction)
        result = {
            "signal": name,
            "field": field,
            "n": len(labels),
            "auroc": auroc(labels, scores) if labels else None,
            "average_precision": average_precision(labels, scores) if labels else None,
        }
        if name in DISAGREE_SIGNALS and labels:
            predictions = [score > 0.5 for score in scores]
            result.update(binary_metrics(labels, predictions))
            result["notes"] = (
                "gold-free rule (probe control)"
                if name == "source_probe_disagree"
                else "gold-free rule"
            )
        elif not labels:
            result["notes"] = "missing feature"
        else:
            result["notes"] = "higher score predicts source error"
        results.append(result)

    paired = paired_disagree_auroc(rows)
    if paired is not None:
        results.append(
            {
                "signal": "paired_disagree_control",
                "field": "source_nla_answer_agree+source_probe_answer_agree",
                "n": paired["n"],
                "auroc": None,
                "average_precision": None,
                "nla_auroc": paired["nla_auroc"],
                "probe_auroc": paired["probe_auroc"],
                "notes": "NLA vs probe disagreement on paired rows",
            }
        )

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, results)
    write_summary(summary_path, rows, results, paired)
    print(f"[done] wrote {output_path}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
