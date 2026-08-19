"""Convert MedCaseReasoning into our case schema, using clinician quotes as cues.

MedCaseReasoning (Stanford, CC-BY-4.0, `zou-lab/MedCaseReasoning`) pairs a
patient presentation with numbered clinician-authored reasoning statements
that carry verbatim quotes from the case report, plus a gold diagnosis. The
quotes are what makes it usable here: each one is a span of the case text
that a clinician marked as evidence, which is exactly the role DDXPlus cue
strings play — except this is real case-report prose rather than assembled
questionnaire text.

Output rows match the DDXPlus case schema (`prompt`, `cue_targets`,
`diagnosis_id`, ...) so the existing cue-position, counterfactual and
readout pipelines run unchanged.

The quote-verification step is the analogue of DDXPlus's construction-exact
check: a quote is only kept as a cue when it appears verbatim in the prompt,
so every emitted cue has a well-defined character span. Rows and quotes that
fail are counted and reported rather than silently dropped, since that rate
is what tells us whether the dataset transfers.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.jsonl import write_jsonl

# Quotes appear inside the numbered reasoning statements, wrapped in quote
# marks. Case reports use several kinds, and the dataset is not normalized.
QUOTE_PATTERN = re.compile(r"[\"“‘']([^\"“”‘’']{12,400})[\"”’']")

# Statements are numbered "1." / "2)" / "Step 3:" etc.
STATEMENT_SPLIT = re.compile(r"(?:^|\n)\s*(?:step\s*)?\d+\s*[.):]\s*", re.IGNORECASE)


def normalize_quote(text: str) -> str:
    """Fold the typography that differs between quote and body text."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "unknown"


def split_statements(reasoning: str) -> list[str]:
    parts = [part.strip() for part in STATEMENT_SPLIT.split(reasoning) if part.strip()]
    return parts or ([reasoning.strip()] if reasoning.strip() else [])


def extract_quotes(reasoning: str) -> list[str]:
    """Quotes in reading order, de-duplicated, longest-first within a duplicate."""
    seen: set[str] = set()
    quotes: list[str] = []
    for statement in split_statements(reasoning):
        for match in QUOTE_PATTERN.finditer(statement):
            quote = normalize_quote(match.group(1))
            if len(quote) < 12:
                continue
            key = quote.lower()
            if key in seen:
                continue
            seen.add(key)
            quotes.append(quote)
    return quotes


def locate_quote(prompt_norm: str, quote: str) -> bool:
    return quote.lower() in prompt_norm.lower()


def drop_contained(quotes: list[str]) -> list[str]:
    """Drop quotes fully contained in a longer kept quote.

    Nested spans would make two cues resolve to overlapping token ranges,
    which the occurrence-index logic downstream cannot disambiguate.
    """
    ordered = sorted(quotes, key=len, reverse=True)
    kept: list[str] = []
    for quote in ordered:
        low = quote.lower()
        if any(low in other.lower() for other in kept):
            continue
        kept.append(quote)
    return [quote for quote in quotes if quote in kept]


def case_row(
    record: dict[str, Any],
    *,
    index: int,
    min_cues: int,
    max_cues: int | None,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats = {"quotes": 0, "matched": 0, "unmatched": 0}
    prompt = normalize_quote(str(record.get("case_prompt") or ""))
    reasoning = str(record.get("diagnostic_reasoning") or "")
    diagnosis = str(record.get("final_diagnosis") or "").strip()
    if not prompt or not reasoning or not diagnosis:
        return None, stats

    quotes = extract_quotes(reasoning)
    stats["quotes"] = len(quotes)
    matched = [quote for quote in quotes if locate_quote(prompt, quote)]
    stats["matched"] = len(matched)
    stats["unmatched"] = len(quotes) - len(matched)
    matched = drop_contained(matched)
    if len(matched) < min_cues:
        return None, stats
    if max_cues is not None:
        matched = matched[:max_cues]

    diagnosis_id = slug(diagnosis)
    base_id = f"mcr_{diagnosis_id}_{index:07d}"
    row = {
        "id": f"{base_id}__cues_all",
        "base_id": base_id,
        "source": "medcasereasoning",
        "case_id": str(record.get("case_id") or record.get("id") or index),
        "variant": "cue_count_all",
        "prompt": prompt,
        "diagnosis_id": diagnosis_id,
        "diagnosis_name": diagnosis,
        "diagnosis_aliases": [diagnosis],
        "cue_targets": matched,
        "cue_count": len(matched),
        "available_cue_count": stats["matched"],
        "cue_types": ["quote"] * len(matched),
        "reasoning_statement_count": len(split_statements(reasoning)),
    }
    return row, stats


def read_records(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        if path.suffix == ".json":
            data = json.load(handle)
            yield from (data if isinstance(data, list) else data.get("data", []))
            return
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="MedCaseReasoning split as JSONL/JSON "
        "(from `datasets.load_dataset('zou-lab/MedCaseReasoning')`, then to_json).",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-cues", type=int, default=3, help="Cases below this are dropped.")
    parser.add_argument("--max-cues", type=int, default=None, help="Cap cues kept per case.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON path for quote-match statistics (transfer diagnostics).",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    totals = {"cases_in": 0, "cases_out": 0, "quotes": 0, "matched": 0, "unmatched": 0}
    cue_counts: list[int] = []
    for index, record in enumerate(read_records(Path(args.input))):
        if args.limit is not None and totals["cases_in"] >= args.limit:
            break
        totals["cases_in"] += 1
        row, stats = case_row(record, index=index, min_cues=args.min_cues, max_cues=args.max_cues)
        for key in ("quotes", "matched", "unmatched"):
            totals[key] += stats[key]
        if row is not None:
            rows.append(row)
            cue_counts.append(row["cue_count"])
            totals["cases_out"] += 1

    if not rows:
        raise ValueError("No usable cases. Check --input format and --min-cues.")

    write_jsonl(Path(args.output), rows)
    match_rate = totals["matched"] / totals["quotes"] if totals["quotes"] else 0.0
    mean_cues = sum(cue_counts) / len(cue_counts)
    summary = {
        **totals,
        "quote_match_rate": round(match_rate, 4),
        "case_keep_rate": round(totals["cases_out"] / max(totals["cases_in"], 1), 4),
        "mean_cues_per_case": round(mean_cues, 2),
        "min_cues_per_case": min(cue_counts),
        "max_cues_per_case": max(cue_counts),
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[done] wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
