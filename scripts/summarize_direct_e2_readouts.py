"""Compare DiReCT readout arms without emitting restricted clinical text."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.answer_matching import is_correct, normalize
from src.jsonl import read_jsonl


WORD = re.compile(r"[a-z0-9]+")


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name or not path:
        raise ValueError(f"Expected nonempty NAME=PATH, got {value!r}")
    return name, Path(path)


def base_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def trigrams(text: str) -> set[tuple[str, str, str]]:
    words = WORD.findall((text or "").casefold())
    return {tuple(words[index : index + 3]) for index in range(len(words) - 2)}


def deterministic_donors(
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, str]:
    by_category: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        case_id = base_id(row)
        by_category[str(row.get("disease_category") or "")].append(case_id)

    donors: dict[str, str] = {}
    for case_ids in by_category.values():
        ordered = sorted(set(case_ids))
        for case_id in ordered:
            own_answer = normalize(str(sources[case_id].get("answer") or ""))
            candidates = [
                other
                for other in ordered
                if other != case_id
                and normalize(str(sources[other].get("answer") or "")) != own_answer
            ]
            if candidates:
                donors[case_id] = candidates[0]
    return donors


def summarize_arm(
    rows: list[dict[str, Any]],
    sources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    donors = deterministic_donors(rows, sources)
    parsed = nonempty = 0
    lengths: list[int] = []
    normalized_outputs: Counter[str] = Counter()
    source_hits: list[bool] = []
    gold_hits: list[bool] = []
    category_hits: list[bool] = []
    own_donor_hits: list[bool] = []
    shuffled_donor_hits: list[bool] = []
    own_lexical: list[float] = []
    shuffled_lexical: list[float] = []
    p1_clean_hits: list[bool] = []

    by_id = {base_id(row): row for row in rows}
    for row in rows:
        case_id = base_id(row)
        source = sources[case_id]
        output = str(row.get("nla_output") or "")
        parsed += bool(row.get("parsed_explanation_tag"))
        nonempty += bool(output.strip())
        lengths.append(len(output))
        normalized_outputs[re.sub(r"\s+", " ", output).strip().casefold()] += 1

        source_hit = is_correct(output, str(source.get("answer") or ""), [])
        source_hits.append(source_hit)
        gold_hits.append(
            is_correct(
                output,
                str(row.get("canonical_pdd") or row.get("diagnosis_name") or ""),
                list(row.get("diagnosis_aliases") or []),
            )
        )
        category_hits.append(
            is_correct(output, str(row.get("disease_category") or ""), [])
        )
        if row.get("position_family") == "P1" and not bool(
            row.get("diagnosis_alias_in_reasoning")
        ):
            p1_clean_hits.append(source_hit)

        donor_id = donors.get(case_id)
        if donor_id is None:
            continue
        donor_source = sources[donor_id]
        own_donor_hits.append(source_hit)
        shuffled_donor_hits.append(
            is_correct(output, str(donor_source.get("answer") or ""), [])
        )
        output_trigrams = trigrams(output)
        if output_trigrams:
            own_prompt = trigrams(str(row.get("prompt") or ""))
            donor_prompt = trigrams(str(by_id[donor_id].get("prompt") or ""))
            own_lexical.append(len(output_trigrams & own_prompt) / len(output_trigrams))
            shuffled_lexical.append(
                len(output_trigrams & donor_prompt) / len(output_trigrams)
            )

    n = len(rows)
    deranged_n = len(own_donor_hits)
    own_source_rate = sum(own_donor_hits) / deranged_n if deranged_n else None
    donor_source_rate = sum(shuffled_donor_hits) / deranged_n if deranged_n else None
    own_lexical_mean = mean(own_lexical) if own_lexical else None
    shuffled_lexical_mean = mean(shuffled_lexical) if shuffled_lexical else None
    return {
        "n": n,
        "parsed_rate": parsed / n if n else None,
        "nonempty_rate": nonempty / n if n else None,
        "mean_chars": mean(lengths) if lengths else None,
        "unique_normalized": len(normalized_outputs),
        "rows_in_duplicate_groups": sum(
            count for count in normalized_outputs.values() if count > 1
        ),
        "source_answer_mention": sum(source_hits) / n if n else None,
        "gold_pdd_mention": sum(gold_hits) / n if n else None,
        "category_mention": sum(category_hits) / n if n else None,
        "derangement_n": deranged_n,
        "own_source_mention": own_source_rate,
        "donor_source_mention": donor_source_rate,
        "source_mention_gap": (
            own_source_rate - donor_source_rate
            if own_source_rate is not None and donor_source_rate is not None
            else None
        ),
        "lexical_n": len(own_lexical),
        "own_prompt_trigram_containment": own_lexical_mean,
        "donor_prompt_trigram_containment": shuffled_lexical_mean,
        "prompt_trigram_gap": (
            own_lexical_mean - shuffled_lexical_mean
            if own_lexical_mean is not None and shuffled_lexical_mean is not None
            else None
        ),
        "p1_leakage_free_n": len(p1_clean_hits),
        "p1_leakage_free_source_mention": (
            sum(p1_clean_hits) / len(p1_clean_hits) if p1_clean_hits else None
        ),
    }


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_summary(path: Path, summaries: dict[str, dict[str, Any]]) -> None:
    lines = [
        "# DiReCT E2 Readout Comparison",
        "",
        "Aggregate-only lexical and phrase-level diagnostics. These values do not replace the official DiReCT semantic evaluator.",
        "",
        "| arm | n | parsed | source answer | gold PDD | category | own-donor source gap | prompt trigram gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, result in summaries.items():
        lines.append(
            f"| {name} | {result['n']} | {format_rate(result['parsed_rate'])} | "
            f"{format_rate(result['source_answer_mention'])} | "
            f"{format_rate(result['gold_pdd_mention'])} | "
            f"{format_rate(result['category_mention'])} | "
            f"{format_rate(result['source_mention_gap'])} | "
            f"{format_rate(result['prompt_trigram_gap'])} |"
        )
    lines.extend(
        [
            "",
            "Source/gold/category columns are literal phrase-or-alias containment diagnostics, not semantic explanation scores.",
            "Own-donor pairs stay within disease category and require a different source answer.",
            "A positive gap is evidence of case specificity only for the named diagnostic.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", action="append", required=True, help="NAME=PATH")
    parser.add_argument("--source-answers", nargs="+", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    sources: dict[str, dict[str, Any]] = {}
    for path in args.source_answers:
        for row in read_jsonl(path):
            case_id = base_id(row)
            if not case_id:
                raise ValueError(f"Missing base_id in {path}")
            if case_id in sources:
                raise ValueError(f"Duplicate source answer for {case_id!r} across inputs")
            sources[case_id] = row
    named_rows: dict[str, list[dict[str, Any]]] = {}
    expected_ids: set[str] | None = None
    for value in args.readout:
        name, path = parse_named_path(value)
        if name in named_rows:
            raise ValueError(f"Duplicate arm name: {name}")
        rows = list(read_jsonl(path))
        ids = [base_id(row) for row in rows]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError(f"Missing or duplicate base_id in {path}")
        current_ids = set(ids)
        if expected_ids is None:
            expected_ids = current_ids
        elif current_ids != expected_ids:
            raise ValueError(
                f"Arm {name} has a different population: "
                f"missing={len(expected_ids - current_ids)} extra={len(current_ids - expected_ids)}"
            )
        missing_sources = current_ids - sources.keys()
        if missing_sources:
            raise ValueError(f"Arm {name} has {len(missing_sources)} rows without source answers")
        named_rows[name] = rows

    summaries = {
        name: summarize_arm(rows, sources) for name, rows in named_rows.items()
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.summary_md, summaries)
    print(f"[summary] arms={len(summaries)} n={len(expected_ids or set())}")
    print(f"[json] {args.output_json}")
    print(f"[markdown] {args.summary_md}")


if __name__ == "__main__":
    main()
