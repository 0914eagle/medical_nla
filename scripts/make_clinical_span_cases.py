"""Build case rows from clinical prose by segmenting the text into cue spans.

Our first attempt at MedCaseReasoning took the quoted spans inside
`diagnostic_reasoning` as evidence cues. Measurement refuted that: only 1.7%
of those quotes appear in `case_prompt`, 42.7% appear in the full article
(they quote the discussion, not the presentation), and the majority are the
reasoning's own words. So the dataset does not annotate which part of the
presentation is evidence.

It does not need to. What the readout experiments score is whether the
readout reproduces the content of the span whose activation was injected —
the span string is itself the gold. A cue therefore has to be a well-defined
piece of clinical text, not a human judgement about diagnostic relevance.
(Relevance ranking matters for the error-anatomy work, and that is where a
span-annotated corpus is still needed.)

So we cut cues out of the presentation itself: segment the case text into
clauses and keep the ones carrying clinical content. Because each cue is a
verbatim slice of the prompt, every cue has an exact character span by
construction — the property the DDXPlus pipeline relies on — while the text
stays real clinical prose.

Written against a generic (text field, label field) interface so the same
path serves other prose corpora.
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

# Abbreviations whose trailing period does not end a sentence. Splitting on
# them would cut a cue in half and leave an unmatchable span.
ABBREVIATIONS = (
    "dr",
    "mr",
    "mrs",
    "ms",
    "prof",
    "vs",
    "approx",
    "no",
    "fig",
    "e.g",
    "i.e",
    "etc",
    "yr",
    "yrs",
    "y.o",
    "wk",
    "hr",
    "min",
    "sec",
    "mg",
    "ml",
    "kg",
    "cm",
    "mm",
    "iu",
    "st",
    "b.i.d",
    "t.i.d",
    "q.d",
    "p.o",
    "i.v",
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
# Clause boundaries inside a long sentence: semicolons, and coordinations that
# reliably separate findings in clinical writing.
CLAUSE_SPLIT = re.compile(r";\s+|,\s+(?=(?:and|with|which|who|but)\s)|\s+—\s+")
HAS_LETTERS = re.compile(r"[A-Za-z]{3,}")


def normalize_text(text: str) -> str:
    """Collapse whitespace so prompt and cue slices share one representation."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = text.replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return value or "unknown"


def ends_with_abbreviation(fragment: str) -> bool:
    tail = fragment.rstrip().lower()
    if not tail.endswith("."):
        return False
    last = re.split(r"[\s(]", tail)[-1].rstrip(".")
    return last in ABBREVIATIONS


def split_sentences(text: str) -> list[str]:
    """Sentence split that re-joins splits caused by an abbreviation period."""
    parts = SENTENCE_END.split(text)
    merged: list[str] = []
    for part in parts:
        if merged and ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [part.strip() for part in merged if part.strip()]


def split_clauses(sentence: str, *, max_words: int) -> list[str]:
    """Split only sentences long enough that they bundle several findings."""
    if len(sentence.split()) <= max_words:
        return [sentence]
    parts = [part.strip(" ,;") for part in CLAUSE_SPLIT.split(sentence)]
    return [part for part in parts if part]


def clean_span(span: str) -> str:
    """Trim punctuation so the cue is a clean slice, still verbatim in text."""
    return span.strip().strip(".,;:!? ").strip()


def segment_cues(
    text: str,
    *,
    min_words: int,
    max_words: int,
    max_cues: int | None,
) -> list[str]:
    cues: list[str] = []
    seen: set[str] = set()
    for sentence in split_sentences(text):
        for clause in split_clauses(sentence, max_words=max_words):
            span = clean_span(clause)
            if not span or span not in text:
                continue
            words = span.split()
            if not (min_words <= len(words) <= max_words):
                continue
            if not HAS_LETTERS.search(span):
                continue
            key = span.lower()
            if key in seen:
                continue
            # Nested spans would resolve to overlapping token ranges.
            if any(key in kept.lower() or kept.lower() in key for kept in cues):
                continue
            seen.add(key)
            cues.append(span)
    if max_cues is not None:
        cues = cues[:max_cues]
    return cues


def case_row(
    record: dict[str, Any],
    *,
    index: int,
    text_field: str,
    label_field: str,
    source: str,
    min_cues: int,
    min_words: int,
    max_words: int,
    max_cues: int | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    text = normalize_text(str(record.get(text_field) or ""))
    label = str(record.get(label_field) or "").strip()
    if not text or not label:
        return None, []

    cues = segment_cues(text, min_words=min_words, max_words=max_words, max_cues=max_cues)
    if len(cues) < min_cues:
        return None, cues

    label_id = slug(label)
    base_id = f"{source}_{label_id}_{index:07d}"
    row = {
        "id": f"{base_id}__cues_all",
        "base_id": base_id,
        "source": source,
        "case_id": str(record.get("pmcid") or record.get("id") or index),
        "variant": "cue_count_all",
        "prompt": text,
        "diagnosis_id": label_id,
        "diagnosis_name": label,
        "diagnosis_aliases": [label],
        "cue_targets": cues,
        "cue_count": len(cues),
        "available_cue_count": len(cues),
        "cue_types": ["clinical_span"] * len(cues),
    }
    return row, cues


def read_records(path: Path, limit: int | None) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--text-field", default="case_prompt")
    parser.add_argument("--label-field", default="final_diagnosis")
    parser.add_argument("--source", default="mcr")
    parser.add_argument("--min-cues", type=int, default=3)
    parser.add_argument("--min-words", type=int, default=4)
    parser.add_argument("--max-words", type=int, default=25)
    parser.add_argument("--max-cues", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--print-samples",
        type=int,
        default=3,
        help="Print this many segmented cases so the cue text can be eyeballed.",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    totals = {"cases_in": 0, "cases_out": 0, "cues": 0}
    cue_counts: list[int] = []
    word_counts: list[int] = []
    printed = 0

    for index, record in enumerate(read_records(Path(args.input), args.limit)):
        totals["cases_in"] += 1
        row, cues = case_row(
            record,
            index=index,
            text_field=args.text_field,
            label_field=args.label_field,
            source=args.source,
            min_cues=args.min_cues,
            min_words=args.min_words,
            max_words=args.max_words,
            max_cues=args.max_cues,
        )
        if row is None:
            continue
        rows.append(row)
        totals["cases_out"] += 1
        totals["cues"] += len(cues)
        cue_counts.append(len(cues))
        word_counts.extend(len(cue.split()) for cue in cues)
        if printed < args.print_samples:
            printed += 1
            print(f"\n--- sample {printed}: {row['diagnosis_name']} ---")
            for cue in cues[:6]:
                print(f"  cue: {cue}")

    if not rows:
        raise ValueError("No usable cases. Check --text-field/--label-field and --min-cues.")

    # Every cue is a slice of its own prompt, so spans resolve by construction;
    # assert it rather than trust it, since downstream extraction depends on it.
    unresolved = sum(
        1 for row in rows for cue in row["cue_targets"] if cue not in row["prompt"]
    )

    write_jsonl(Path(args.output), rows)
    summary = {
        **totals,
        "case_keep_rate": round(totals["cases_out"] / max(totals["cases_in"], 1), 4),
        "mean_cues_per_case": round(sum(cue_counts) / len(cue_counts), 2),
        "min_cues_per_case": min(cue_counts),
        "max_cues_per_case": max(cue_counts),
        "mean_cue_words": round(sum(word_counts) / len(word_counts), 2),
        "unresolved_spans": unresolved,
    }
    print()
    print(json.dumps(summary, indent=2))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[done] wrote {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
