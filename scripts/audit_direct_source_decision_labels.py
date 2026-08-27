"""Audit whether free-form DiReCT source answers define a probe label space.

The report is aggregate-only. It checks exact normalized answer reuse and how
often an answer maps uniquely to a frozen PDD or disease-category ontology.
No probe should be trained on source-answer strings until this audit shows that
validation labels are represented in training and that ontology mapping is not
mostly unmatched or ambiguous.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, normalize
from src.jsonl import read_jsonl


SPLITS = ("train", "val_seen")


def identifier(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = identifier(row)
        if not key or key in result:
            raise ValueError(f"Missing or duplicate {label} ID: {key!r}")
        result[key] = row
    return result


def ontology(rows: list[dict[str, Any]], field: str) -> dict[str, list[str]]:
    aliases: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        label = str(row.get(field) or "").strip()
        if not label or label == "<unresolved>":
            continue
        aliases[label].add(label)
        if field == "canonical_pdd":
            for alias_field in ("annotation_root_diagnosis", "folder_pdd"):
                alias = str(row.get(alias_field) or "").strip()
                if alias:
                    aliases[label].add(alias)
    return {label: sorted(values) for label, values in sorted(aliases.items())}


def matches(answer: str, candidates: dict[str, list[str]]) -> list[str]:
    return [
        label
        for label, aliases in candidates.items()
        if is_correct(answer, label, [alias for alias in aliases if alias != label])
    ]


def split_stats(
    split: str,
    rows: list[dict[str, Any]],
    answers: dict[str, dict[str, Any]],
    pdds: dict[str, list[str]],
    categories: dict[str, list[str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    audited: list[dict[str, Any]] = []
    exact = Counter()
    counters = Counter()
    for row in rows:
        key = identifier(row)
        source = answers.get(key)
        if source is None:
            counters["missing_answer"] += 1
            continue
        answer = str(source.get("answer") or "").strip()
        if not answer:
            counters["empty_answer"] += 1
            continue
        normalized = normalize(answer)
        exact[normalized] += 1
        pdd_matches = matches(answer, pdds)
        category_matches = matches(answer, categories)
        audited.append(
            {
                "id": key,
                "split": split,
                "normalized_answer": normalized,
                "pdd_matches": pdd_matches,
                "category_matches": category_matches,
            }
        )
        counters["rows"] += 1
        pdd_key = len(pdd_matches) if len(pdd_matches) < 2 else "multiple"
        counters[f"pdd_match_{pdd_key}"] += 1
        counters[
            f"category_match_{len(category_matches) if len(category_matches) < 2 else 'multiple'}"
        ] += 1
    stats = {
        "split": split,
        "requested": len(rows),
        "audited": counters["rows"],
        "missing_answer": counters["missing_answer"],
        "empty_answer": counters["empty_answer"],
        "unique_normalized_answers": len(exact),
        "normalized_answer_frequency": {
            "at_least_2": sum(count >= 2 for count in exact.values()),
            "at_least_3": sum(count >= 3 for count in exact.values()),
            "at_least_5": sum(count >= 5 for count in exact.values()),
            "singleton_rows": sum(count for count in exact.values() if count == 1),
        },
        "pdd": {
            "unique": counters["pdd_match_1"],
            "unmatched": counters["pdd_match_0"],
            "ambiguous": counters["pdd_match_multiple"],
        },
        "category": {
            "unique": counters["category_match_1"],
            "unmatched": counters["category_match_0"],
            "ambiguous": counters["category_match_multiple"],
        },
    }
    return stats, audited


def write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# DiReCT Source-Decision Label Audit",
        "",
        "Aggregate-only audit. Free-text answers and case identifiers are not emitted.",
        "",
        f"- PDD ontology: **{report['ontology']['pdd']}** labels",
        f"- disease-category ontology: **{report['ontology']['category']}** labels",
        "",
        (
            "| split | n | unique strings | singleton rows | unique PDD | "
            "unmatched PDD | ambiguous PDD | unique category | unmatched category | "
            "ambiguous category |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = report["splits"][split]
        lines.append(
            f"| {split} | {item['audited']} | {item['unique_normalized_answers']} | "
            f"{item['normalized_answer_frequency']['singleton_rows']} | "
            f"{item['pdd']['unique']} | {item['pdd']['unmatched']} | {item['pdd']['ambiguous']} | "
            f"{item['category']['unique']} | {item['category']['unmatched']} | "
            f"{item['category']['ambiguous']} |"
        )
    coverage = report["validation_seen_in_train"]
    lines.extend(
        [
            "",
            "## Validation Label Coverage",
            "",
            "| label representation | covered validation rows | total validation rows | rate |",
            "|---|---:|---:|---:|",
            (
                f"| exact normalized answer | {coverage['exact']['covered']} | "
                f"{coverage['exact']['n']} | {coverage['exact']['rate']:.4f} |"
            ),
            (
                f"| uniquely mapped PDD | {coverage['pdd']['covered']} | "
                f"{coverage['pdd']['n']} | {coverage['pdd']['rate']:.4f} |"
            ),
            (
                f"| uniquely mapped category | {coverage['category']['covered']} | "
                f"{coverage['category']['n']} | {coverage['category']['rate']:.4f} |"
            ),
            "",
            (
                "A source-decision probe is admissible only for a representation fixed "
                "before locked-test scoring. Exact free-text labels require near-complete "
                "validation coverage; ontology labels additionally require high "
                "unique-mapping coverage. Otherwise the row remains an open-text "
                "Medical-NLA target rather than a closed-label probe."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def coverage(
    train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]], field: str
) -> dict[str, Any]:
    if field == "normalized_answer":
        train_labels = {row[field] for row in train_rows}
        eligible = val_rows
        covered = sum(row[field] in train_labels for row in eligible)
    else:
        match_field = f"{field}_matches"
        train_labels = {
            row[match_field][0] for row in train_rows if len(row[match_field]) == 1
        }
        eligible = [row for row in val_rows if len(row[match_field]) == 1]
        covered = sum(row[match_field][0] in train_labels for row in eligible)
    n = len(eligible)
    return {"covered": covered, "n": n, "rate": covered / n if n else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--answers", required=True, nargs="+", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    split_rows = {
        split: list(read_jsonl(args.split_dir / f"{split}.jsonl")) for split in SPLITS
    }
    all_clinical = [
        row
        for name in ("train", "val_seen", "test_seen", "test_pdd_heldout")
        for row in read_jsonl(args.split_dir / f"{name}.jsonl")
    ]
    answers = index_unique(
        [row for path in args.answers for row in read_jsonl(path)], "source answer"
    )
    pdds = ontology(all_clinical, "canonical_pdd")
    categories = ontology(all_clinical, "disease_category")
    report: dict[str, Any] = {
        "ontology": {"pdd": len(pdds), "category": len(categories)},
        "splits": {},
    }
    audited: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        stats, rows = split_stats(split, split_rows[split], answers, pdds, categories)
        report["splits"][split] = stats
        audited[split] = rows
    report["validation_seen_in_train"] = {
        "exact": coverage(audited["train"], audited["val_seen"], "normalized_answer"),
        "pdd": coverage(audited["train"], audited["val_seen"], "pdd"),
        "category": coverage(audited["train"], audited["val_seen"], "category"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.summary_md, report)
    print(args.summary_md.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
