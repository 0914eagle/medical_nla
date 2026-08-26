"""Interactively review the primary DiReCT E2 semantic-audit rows."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_FIELDS = ("human_source", "human_gold", "human_category")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl_atomic(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_labels(value: str) -> tuple[bool, bool, bool] | None:
    normalized = value.strip().lower().replace(",", " ")
    if normalized in {"q", "quit", "exit"}:
        return None
    tokens = normalized.split()
    if len(tokens) == 1 and len(tokens[0]) == 3:
        tokens = list(tokens[0])
    if len(tokens) != 3 or any(token not in {"y", "n"} for token in tokens):
        raise ValueError("Enter three labels, for example: y n n (or ynn).")
    return tuple(token == "y" for token in tokens)  # type: ignore[return-value]


def validate_resume(
    source_rows: list[dict[str, Any]], reviewed_rows: list[dict[str, Any]]
) -> None:
    source_ids = [str(row["id"]) for row in source_rows]
    reviewed_ids = [str(row["id"]) for row in reviewed_rows]
    if source_ids != reviewed_ids:
        raise ValueError("Reviewed output IDs/order do not match the source audit file")


def is_complete(row: dict[str, Any]) -> bool:
    return all(row.get(field) is not None for field in REVIEW_FIELDS)


def automatic_match(row: dict[str, Any], role: str) -> bool:
    return bool((row.get("verdicts") or {}).get(role, {}).get("semantic_match"))


def summary_markdown(rows: list[dict[str, Any]]) -> str:
    roles = (
        ("source_answer", "human_source", "Source answer"),
        ("gold_pdd", "human_gold", "Gold PDD"),
        ("category", "human_category", "Disease category"),
    )
    lines = [
        "# DiReCT E2 Primary Manual Semantic Audit",
        "",
        f"- rows: **{len(rows)}**",
        f"- fully reviewed: **{sum(is_complete(row) for row in rows)}/{len(rows)}**",
        "",
        "| target | reviewed | human positive | automatic positive | agreement |",
        "|---|---:|---:|---:|---:|",
    ]
    for role, field, label in roles:
        reviewed = [row for row in rows if row.get(field) is not None]
        human_positive = sum(row.get(field) is True for row in reviewed)
        automatic_positive = sum(automatic_match(row, role) for row in reviewed)
        agreement = sum(
            bool(row[field]) == automatic_match(row, role) for row in reviewed
        )
        agreement_text = f"{agreement / len(reviewed):.4f}" if reviewed else "--"
        lines.append(
            f"| {label} | {len(reviewed)} | {human_positive} | "
            f"{automatic_positive} | {agreement_text} |"
        )
    return "\n".join(lines) + "\n"


def show_row(row: dict[str, Any], index: int, total: int) -> None:
    targets = {target["role"]: target["text"] for target in row.get("targets", [])}
    print("\n" + "=" * 88)
    print(f"Row {index}/{total}  id={row['id']}")
    print("-" * 88)
    print(row.get("readout", ""))
    print("-" * 88)
    for role, label in (
        ("source_answer", "SOURCE"),
        ("gold_pdd", "GOLD"),
        ("category", "CATEGORY"),
    ):
        auto = automatic_match(row, role)
        print(f"{label:8s} auto={'Y' if auto else 'N'}  target={targets.get(role, '')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    source_rows = read_jsonl(args.input_jsonl)
    if args.output_jsonl.exists():
        rows = read_jsonl(args.output_jsonl)
        validate_resume(source_rows, rows)
    else:
        rows = source_rows
        write_jsonl_atomic(args.output_jsonl, rows)

    if not args.summary_only:
        pending = [index for index, row in enumerate(rows) if not is_complete(row)]
        for completed, row_index in enumerate(pending, start=1):
            row = rows[row_index]
            show_row(row, row_index + 1, len(rows))
            while True:
                try:
                    answer = input("source gold category [y/n y/n y/n; q saves]: ")
                    labels = parse_labels(answer)
                    break
                except (EOFError, KeyboardInterrupt):
                    print("\n[stopped] previously completed rows remain saved")
                    labels = None
                    break
                except ValueError as error:
                    print(error)
            if labels is None:
                break
            row["human_source"], row["human_gold"], row["human_category"] = labels
            row["human_reviewer"] = args.reviewer
            row["human_reviewed_at"] = datetime.now(timezone.utc).isoformat()
            write_jsonl_atomic(args.output_jsonl, rows)
            print(f"[saved] {completed}/{len(pending)} pending rows completed this session")

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(summary_markdown(rows), encoding="utf-8")
    print(summary_markdown(rows), end="")
    print(f"[reviewed] {args.output_jsonl}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
