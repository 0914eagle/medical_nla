"""Aggregate DiReCT E1 source-answer metrics without emitting restricted text."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, token_f1
from src.jsonl import read_jsonl


def labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label, Path(raw_path)


def exact_mcnemar(gain: int, loss: int) -> float:
    n = gain + loss
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(min(gain, loss) + 1))
    return min(1.0, 2 * tail / (2**n))


def annotate(row: dict[str, Any]) -> dict[str, Any]:
    answer = row.get("answer")
    gold = str(row.get("canonical_pdd") or row.get("diagnosis_name") or "")
    aliases = [str(value) for value in row.get("diagnosis_aliases") or []]
    category = str(row.get("disease_category") or "")
    return {
        "id": str(row.get("base_id") or row.get("id")),
        "split": str(row.get("split") or "unknown"),
        "parsed": bool(row.get("answer_parsed", answer)),
        "strict_correct": is_correct(answer, gold, aliases),
        "category_correct": bool(category) and is_correct(answer, category, []),
        "token_f1": float(token_f1(answer, gold, aliases)),
        "answer_in_reasoning": bool(row.get("diagnosis_alias_in_reasoning")),
        "gold_in_reasoning": bool(row.get("gold_alias_in_reasoning")),
        "forced": bool(row.get("answer_forced")),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    if not n:
        return {"n": 0}

    def rate(field: str) -> float:
        return sum(bool(row[field]) for row in rows) / n

    return {
        "n": n,
        "parse_rate": rate("parsed"),
        "strict_pdd_accuracy": rate("strict_correct"),
        "category_accuracy": rate("category_correct"),
        "mean_token_f1": sum(row["token_f1"] for row in rows) / n,
        "answer_in_reasoning_rate": rate("answer_in_reasoning"),
        "gold_in_reasoning_rate": rate("gold_in_reasoning"),
        "forced_rate": rate("forced"),
    }


def paired(
    left: list[dict[str, Any]], right: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    left_by_id = {row["id"]: row for row in left}
    right_by_id = {row["id"]: row for row in right}
    shared = sorted(left_by_id.keys() & right_by_id.keys())
    both = left_only = right_only = neither = 0
    for row_id in shared:
        lval = bool(left_by_id[row_id][field])
        rval = bool(right_by_id[row_id][field])
        if lval and rval:
            both += 1
        elif lval:
            left_only += 1
        elif rval:
            right_only += 1
        else:
            neither += 1
    n = len(shared)
    return {
        "n": n,
        "both_correct": both,
        "left_only_correct": left_only,
        "right_only_correct": right_only,
        "neither_correct": neither,
        "left_accuracy": (both + left_only) / n if n else None,
        "right_accuracy": (both + right_only) / n if n else None,
        "right_minus_left": (right_only - left_only) / n if n else None,
        "mcnemar_exact_p": exact_mcnemar(right_only, left_only),
    }


def fmt(value: Any) -> str:
    return "-" if value is None else f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--answers",
        action="append",
        required=True,
        type=labeled_path,
        metavar="LABEL=PATH",
    )
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    labeled_rows: dict[str, list[dict[str, Any]]] = {}
    for label, path in args.answers:
        if label in labeled_rows:
            raise SystemExit(f"duplicate label: {label}")
        rows = [annotate(row) for row in read_jsonl(path)]
        ids = [row["id"] for row in rows]
        if len(ids) != len(set(ids)):
            raise SystemExit(f"duplicate base_id in {path}")
        labeled_rows[label] = rows

    result: dict[str, Any] = {"arms": {}, "paired": {}}
    for label, rows in labeled_rows.items():
        by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_split[row["split"]].append(row)
        result["arms"][label] = {
            "overall": summarize(rows),
            "by_split": {
                split: summarize(split_rows)
                for split, split_rows in sorted(by_split.items())
            },
        }

    if "direct" in labeled_rows and "cot" in labeled_rows:
        for field, name in (
            ("strict_correct", "strict_pdd"),
            ("category_correct", "disease_category"),
        ):
            result["paired"][name] = paired(
                labeled_rows["direct"], labeled_rows["cot"], field
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# DiReCT E1 Source Answer Summary",
        "",
        "Aggregate-only summary; no restricted note or generated text is emitted.",
        "",
        "## Arms",
        "",
        "| arm | pool | n | parse | strict PDD | category | token F1 | answer in CoT | gold in CoT |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, arm in result["arms"].items():
        pools = [("overall", arm["overall"]), *arm["by_split"].items()]
        for pool, summary in pools:
            lines.append(
                f"| {label} | {pool} | {summary['n']} | {fmt(summary.get('parse_rate'))} "
                f"| {fmt(summary.get('strict_pdd_accuracy'))} "
                f"| {fmt(summary.get('category_accuracy'))} "
                f"| {fmt(summary.get('mean_token_f1'))} "
                f"| {fmt(summary.get('answer_in_reasoning_rate'))} "
                f"| {fmt(summary.get('gold_in_reasoning_rate'))} |"
            )

    if result["paired"]:
        lines.extend(["", "## Paired direct versus CoT", ""])
        for name, comparison in result["paired"].items():
            lines.extend(
                [
                    f"### {name}",
                    "",
                    f"- shared: **{comparison['n']}**",
                    f"- both correct: **{comparison['both_correct']}**",
                    f"- Direct only correct: **{comparison['left_only_correct']}**",
                    f"- CoT only correct: **{comparison['right_only_correct']}**",
                    f"- neither correct: **{comparison['neither_correct']}**",
                    f"- CoT minus Direct: **{fmt(comparison['right_minus_left'])}**",
                    f"- McNemar exact p: **{fmt(comparison['mcnemar_exact_p'])}**",
                    "",
                ]
            )

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(f"[json] {args.output_json}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
