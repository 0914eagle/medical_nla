"""Merge experiment outputs into an error-prediction feature table.

The label is whether the source model answer was correct. Other inputs are
features/baselines: source diagnosis logprobs, NLA free-generation hits, NLA
diagnosis logprobs, and optional linear-probe predictions.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import read_jsonl, write_jsonl


def key(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id"))


def read_by_key(path: str | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    rows = {}
    for row in read_jsonl(path):
        rows[key(row)] = row
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


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["is_error"] for row in rows if row.get("is_error") is not None]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Error Prediction Feature Table\n\n")
        f.write(f"- n: {len(rows)}\n")
        if labels:
            f.write(f"- errors: {sum(labels)}/{len(labels)}\n")
            f.write(f"- error_rate: {mean(labels):.4f}\n")
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
    parser.add_argument("--source-logprobs", default=None)
    parser.add_argument("--nla-scored", default=None)
    parser.add_argument("--nla-logprobs", default=None)
    parser.add_argument("--probe-predictions", default=None)
    args = parser.parse_args()

    source_answers = list(read_jsonl(args.source_answers))
    source_logprobs = read_by_key(args.source_logprobs)
    nla_scored = read_by_key(args.nla_scored)
    nla_logprobs = read_by_key(args.nla_logprobs)
    probe_predictions = read_by_key(args.probe_predictions)

    rows = []
    for source in source_answers:
        k = key(source)
        correct = as_bool(
            source.get("diagnosis_hit", source.get("answer_hit", source.get("is_correct")))
        )
        out = {
            "id": source.get("id", k),
            "base_id": k,
            "prompt": source.get("prompt"),
            "gold_diagnosis_id": source.get("diagnosis_id") or source.get("gold_diagnosis_id"),
            "gold_diagnosis_name": source.get("diagnosis_name")
            or source.get("specific_expected")
            or source.get("gold_diagnosis_name"),
            "source_answer": source.get("answer"),
            "source_correct": correct,
            "is_error": None if correct is None else not correct,
        }
        add_rank_features(out, "source", source_logprobs.get(k))
        nla_row = nla_scored.get(k)
        if nla_row:
            out["nla_diagnosis_hit"] = nla_row.get("diagnosis_hit")
            out["nla_specific_hit"] = nla_row.get("specific_hit")
            out["nla_cjk_fraction"] = nla_row.get("cjk_fraction")
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
