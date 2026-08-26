"""Write the fixed DiReCT PDD label ontology used for candidate ranking.

The input manifest is private, but the output contains labels only. Building
the ontology separately prevents a validation-only input file from silently
shrinking the candidate set to labels that happened to occur in validation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.jsonl import read_jsonl, write_jsonl


def collect_labels(rows: list[dict], field: str) -> list[str]:
    labels = {
        str(row.get(field) or "").strip()
        for row in rows
        if str(row.get(field) or "").strip()
        and str(row.get(field) or "").strip() != "<unresolved>"
    }
    if not labels:
        raise ValueError(f"No resolved labels found in field {field!r}")
    return sorted(labels)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-md", type=Path, required=True)
    parser.add_argument("--label-field", default="canonical_pdd")
    args = parser.parse_args()

    rows = list(read_jsonl(args.manifest))
    labels = collect_labels(rows, args.label_field)
    counts = Counter(str(row.get(args.label_field) or "").strip() for row in rows)
    output_rows = [
        {
            "diagnosis_id": label,
            "diagnosis_name": label,
            "corpus_count": counts[label],
        }
        for label in labels
    ]
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_jsonl, output_rows)
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text(
        "# DiReCT Candidate Ontology\n\n"
        "Label-only artifact derived before locked-test scoring.\n\n"
        f"- source rows: **{len(rows)}**\n"
        f"- resolved candidate PDDs: **{len(labels)}**\n"
        f"- label field: `{args.label_field}`\n",
        encoding="utf-8",
    )
    print(f"[ontology] candidates={len(labels)} output={args.output_jsonl}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
