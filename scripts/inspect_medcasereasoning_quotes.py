"""Diagnose where MedCaseReasoning's reasoning text overlaps the case text.

The first ingestion attempt assumed the quoted spans inside
`diagnostic_reasoning` are verbatim excerpts of `case_prompt`. The observed
match rate (1.6%) says otherwise, so this script measures the alternatives
before we commit to an extraction rule:

  - are the quotes verbatim in `case_prompt`, in the full article `text`, or
    in neither (i.e. the reasoning's own words)?
  - if quotes are the wrong unit, do the reasoning *statements* instead
    overlap the case text at the n-gram level, which would let us anchor a
    cue by locating the longest shared span?

Prints a summary and writes samples so the actual strings can be read rather
than guessed at.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_medcasereasoning_cases import (
    QUOTE_PATTERN,
    normalize_quote,
    split_statements,
)


def read_records(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def longest_shared_span(statement: str, case: str, *, min_len: int) -> str | None:
    """Longest word n-gram of `statement` that appears verbatim in `case`.

    Greedy from the longest candidate down: if a statement quotes or closely
    tracks the case text, this recovers the anchor even when the quote marks
    are absent or the quote spans a sentence boundary.
    """
    words = statement.split()
    case_low = case.lower()
    for length in range(len(words), min_len - 1, -1):
        for start in range(0, len(words) - length + 1):
            candidate = " ".join(words[start : start + length])
            if candidate.lower() in case_low:
                return candidate
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-span-words", type=int, default=5)
    parser.add_argument("--out-samples", default=None, help="JSON path for readable samples.")
    args = parser.parse_args()

    totals = {
        "records": 0,
        "records_with_quotes": 0,
        "quotes": 0,
        "quote_in_case_prompt": 0,
        "quote_in_article_text": 0,
        "quote_in_neither": 0,
        "statements": 0,
        "statements_with_shared_span": 0,
    }
    span_word_counts: list[int] = []
    samples: list[dict[str, Any]] = []

    for record in read_records(Path(args.input), args.limit):
        totals["records"] += 1
        case = normalize_quote(str(record.get("case_prompt") or ""))
        article = normalize_quote(str(record.get("text") or ""))
        reasoning = str(record.get("diagnostic_reasoning") or "")
        if not case or not reasoning:
            continue

        quotes = [
            normalize_quote(m.group(1))
            for m in QUOTE_PATTERN.finditer(reasoning)
            if len(normalize_quote(m.group(1))) >= 12
        ]
        if quotes:
            totals["records_with_quotes"] += 1
        for quote in quotes:
            totals["quotes"] += 1
            in_case = quote.lower() in case.lower()
            in_article = bool(article) and quote.lower() in article.lower()
            totals["quote_in_case_prompt"] += int(in_case)
            totals["quote_in_article_text"] += int(in_article)
            totals["quote_in_neither"] += int(not in_case and not in_article)

        statements = split_statements(reasoning)
        record_spans: list[dict[str, str]] = []
        for statement in statements:
            totals["statements"] += 1
            span = longest_shared_span(
                normalize_quote(statement), case, min_len=args.min_span_words
            )
            if span:
                totals["statements_with_shared_span"] += 1
                span_word_counts.append(len(span.split()))
                record_spans.append({"statement": statement[:300], "shared_span": span})

        if len(samples) < args.samples:
            samples.append(
                {
                    "final_diagnosis": record.get("final_diagnosis"),
                    "case_prompt_head": case[:600],
                    "reasoning_head": reasoning[:600],
                    "quotes_found": quotes[:5],
                    "quotes_in_case_prompt": [q for q in quotes if q.lower() in case.lower()][:5],
                    "statement_shared_spans": record_spans[:5],
                }
            )

    def rate(num: int, den: int) -> float:
        return round(num / den, 4) if den else 0.0

    summary = {
        **totals,
        "quote_match_rate_case_prompt": rate(totals["quote_in_case_prompt"], totals["quotes"]),
        "quote_match_rate_article_text": rate(totals["quote_in_article_text"], totals["quotes"]),
        "quote_in_neither_rate": rate(totals["quote_in_neither"], totals["quotes"]),
        "statement_span_rate": rate(
            totals["statements_with_shared_span"], totals["statements"]
        ),
        "mean_shared_span_words": (
            round(sum(span_word_counts) / len(span_word_counts), 2) if span_word_counts else 0.0
        ),
        "mean_quotes_per_record": rate(totals["quotes"], totals["records"]),
        "mean_statements_per_record": rate(totals["statements"], totals["records"]),
    }
    print(json.dumps(summary, indent=2))

    if args.out_samples:
        out = Path(args.out_samples)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(samples, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[done] wrote {len(samples)} samples to {out}")


if __name__ == "__main__":
    main()
