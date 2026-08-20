"""Acceptance gate for DDXPlus cue rendering, at both levels that can fail.

Eyeballing a handful of prompts does not establish that the corpus is sound.
Two exhaustive passes do:

1. Vocabulary. Every prompt is assembled from a finite set of (question, value)
   pairs, so rendering every pair and reviewing the result covers every prompt
   at the cue level. Pass --patients to enumerate the pairs the data actually
   contains; without it the enumeration comes from the evidence metadata, which
   omits most negative answers because their value ids are rarely labelled --
   precisely the renderings most likely to be wrong. The dump is written to TSV
   for reading in full; the mechanical failure classes are counted here.

2. Composition. What the vocabulary pass cannot see is what happens when cues
   are joined into a case: duplicates, cues that no longer appear verbatim in
   their own prompt, runaway cue counts. Those are checked on every case, not a
   sample, and the outliers are printed so the eyeballing that does happen is
   spent on the cases most likely to be wrong.

Exits non-zero when a hard violation is found, so it can gate a rebuild.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_ddxplus_probe_dataset import (
    cue_from_entry,
    is_malformed_cue,
    is_opaque_value_id,
    meta_text,
    read_json,
)
from src.jsonl import read_jsonl

# Renderings that pass the malformed check but still read badly enough to want
# a human decision. Soft: reported, not failed.
KNOWN_ABBREVIATIONS = {
    "HIV", "AIDS", "UC", "COPD", "DVT", "OSA", "BMI", "GERD", "NSTEMI", "STEMI",
    "TIA", "PE", "URTI", "SLE", "IBD", "CT", "MRI", "ECG", "EKG", "IV", "PO",
}

SUSPICIOUS_PATTERNS = {
    "leftover_question_mark": re.compile(r"\?"),
    "second_person": re.compile(r"\byou(r)?\b", re.I),
    "double_subject": re.compile(r"\bthe patient\b.*\bthe patient\b", re.I),
    # Unresolved identifiers: "V29", "V_29", and all-caps runs. Known medical
    # abbreviations match too, and are filtered by KNOWN_ABBREVIATIONS.
    "uppercase_code": re.compile(r"\b(?:[A-Z]{2,}\d*|[A-Z]_?\d+)\b"),
    "empty_parens": re.compile(r"\(\s*\)"),
}


def entries_in_use(patients_path: Path, limit: int | None = None) -> set[str]:
    """The (evidence, value) pairs the patient file actually contains.

    Enumerating `value_meaning` is not the same thing, and the difference is not
    academic: negative answers are usually recorded with a value id that has no
    entry there, so a metadata-derived enumeration silently omits exactly the
    renderings most likely to be wrong.
    """
    import ast
    import csv

    csv.field_size_limit(sys.maxsize)
    entries: set[str] = set()
    with patients_path.open(encoding="utf-8", newline="") as handle:
        for index, row in enumerate(csv.DictReader(handle)):
            if limit is not None and index >= limit:
                break
            raw = row.get("EVIDENCES") or row.get("evidences") or ""
            try:
                values = ast.literal_eval(raw) if raw else []
            except (ValueError, SyntaxError):
                continue
            entries.update(str(value) for value in values)
    return entries


def vocabulary_rows(
    evidence_meta: dict[str, Any],
    *,
    negative_cues: bool,
    entries_used: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Render every (question, value) pair that can reach a prompt."""
    rows: list[dict[str, Any]] = []
    by_evidence: dict[str, list[str]] = {}
    if entries_used:
        for entry in sorted(entries_used):
            by_evidence.setdefault(entry.split("_@_")[0], []).append(entry)

    for evidence_id, meta in evidence_meta.items():
        if not isinstance(meta, dict):
            continue
        if entries_used is not None:
            entries = by_evidence.get(evidence_id, [])
        else:
            value_meaning = meta.get("value_meaning")
            value_ids = list(value_meaning) if isinstance(value_meaning, dict) else []
            # Y/N are answered constantly but rarely labelled, so include them.
            entries = [f"{evidence_id}_@_{vid}" for vid in [*value_ids, "N", "Y"]] or []
            entries.append(evidence_id)
        for entry in entries:
            cue = cue_from_entry(
                entry, evidence_meta, clean_cues=True, negative_cues=negative_cues
            )
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "entry": entry,
                    "question": meta_text(meta, evidence_id),
                    "value_label": cue.get("value_label"),
                    "polarity": cue.get("cue_polarity"),
                    "excluded": bool(cue.get("excluded")),
                    "reason": cue.get("exclusion_reason") or "",
                    "cue_text": cue.get("cue_text") or "",
                }
            )
    return rows


def suspicious_flags(text: str) -> list[str]:
    found = []
    for name, pattern in SUSPICIOUS_PATTERNS.items():
        matches = pattern.findall(text)
        if not matches:
            continue
        if name == "uppercase_code" and all(
            (match if isinstance(match, str) else match[0]) in KNOWN_ABBREVIATIONS
            for match in matches
        ):
            # A named condition, not an unresolved code.
            continue
        found.append(name)
    return found


def audit_vocabulary(rows: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Hard-fail anything kept that the renderer itself calls malformed."""
    failures: list[str] = []
    reasons: Counter = Counter()
    flags: Counter = Counter()
    flagged_examples: list[dict[str, Any]] = []

    for row in rows:
        reasons[row["reason"] or f"KEPT({row['polarity']})"] += 1
        if row["excluded"]:
            continue
        cue = row["cue_text"]
        if not cue:
            failures.append(f"{row['entry']}: kept with empty cue text")
            continue
        if is_malformed_cue(cue):
            failures.append(f"{row['entry']}: kept but malformed -> {cue!r}")
        if is_opaque_value_id(str(row["value_label"] or "")):
            failures.append(f"{row['entry']}: kept with opaque value -> {cue!r}")
        found = suspicious_flags(cue)
        for name in found:
            flags[name] += 1
        if found:
            flagged_examples.append({**row, "flags": found})

    kept = sum(1 for row in rows if not row["excluded"])
    summary = {
        "renderings": len(rows),
        "kept": kept,
        "dropped": len(rows) - kept,
        "by_reason": dict(reasons),
        "suspicious": dict(flags),
        "flagged_examples": flagged_examples[:40],
    }
    return failures, summary


def audit_cases(cases: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Check every case, not a sample, for problems only assembly can create."""
    failures: list[str] = []
    cue_counts: list[int] = []
    prompt_words: list[int] = []
    polarity: Counter = Counter()
    flags: Counter = Counter()

    for case in cases:
        case_id = str(case.get("id") or case.get("base_id"))
        prompt = str(case.get("prompt") or "")
        cues = [str(cue) for cue in case.get("cue_targets") or []]
        cue_counts.append(len(cues))
        prompt_words.append(len(prompt.split()))
        for value in case.get("cue_polarities") or []:
            polarity[str(value)] += 1

        if not cues:
            failures.append(f"{case_id}: no cues")
        seen: set[str] = set()
        for cue in cues:
            # Extraction resolves cues by substring, so a cue that is not
            # verbatim in its own prompt cannot be located at all.
            if cue not in prompt:
                failures.append(f"{case_id}: cue not verbatim in prompt -> {cue!r}")
            if cue.lower() in seen:
                failures.append(f"{case_id}: duplicate cue -> {cue!r}")
            seen.add(cue.lower())
            if is_malformed_cue(cue):
                failures.append(f"{case_id}: malformed cue in prompt -> {cue!r}")
            for name in suspicious_flags(cue):
                flags[name] += 1

    def percentiles(values: list[int]) -> dict[str, float]:
        if not values:
            return {}
        ordered = sorted(values)
        pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
        return {
            "min": ordered[0],
            "p50": pick(0.50),
            "p90": pick(0.90),
            "p99": pick(0.99),
            "max": ordered[-1],
            "mean": round(sum(ordered) / len(ordered), 2),
        }

    summary = {
        "cases": len(cases),
        "cue_count": percentiles(cue_counts),
        "prompt_words": percentiles(prompt_words),
        "cue_polarity": dict(polarity),
        "suspicious_in_prompts": dict(flags),
    }
    return failures, summary


def longest_cases(cases: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """The cases most likely to read badly, so review time goes where it counts."""
    return sorted(cases, key=lambda case: -len(case.get("cue_targets") or []))[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidences", required=True)
    parser.add_argument(
        "--patients",
        default=None,
        help=(
            "Patient CSV. Given, the vocabulary pass renders exactly the "
            "(evidence, value) pairs the data contains, which is the only "
            "enumeration that truly covers every prompt."
        ),
    )
    parser.add_argument("--cases", default=None, help="Case JSONL to check for assembly problems.")
    parser.add_argument("--dump", default=None, help="Write every rendering to this TSV.")
    parser.add_argument("--negative-cues", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-longest", type=int, default=3)
    args = parser.parse_args()

    evidence_meta = read_json(Path(args.evidences))
    entries_used = None
    if args.patients:
        entries_used = entries_in_use(Path(args.patients))
        print(f"[scan] {len(entries_used):,} distinct evidence entries in {args.patients}")
    rows = vocabulary_rows(
        evidence_meta, negative_cues=args.negative_cues, entries_used=entries_used
    )
    vocab_failures, vocab_summary = audit_vocabulary(rows)

    if args.dump:
        dump_path = Path(args.dump)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with dump_path.open("w", encoding="utf-8") as handle:
            handle.write("entry\tpolarity\tkept\treason\tquestion\tvalue\tcue_text\n")
            for row in rows:
                handle.write(
                    "\t".join(
                        [
                            row["entry"],
                            str(row["polarity"]),
                            "no" if row["excluded"] else "yes",
                            row["reason"],
                            " ".join(str(row["question"]).split()),
                            str(row["value_label"] or ""),
                            row["cue_text"],
                        ]
                    )
                    + "\n"
                )
        print(f"[dump] {len(rows)} renderings -> {dump_path}")

    print("\n== vocabulary ==")
    print(json.dumps({k: v for k, v in vocab_summary.items() if k != "flagged_examples"}, indent=2))
    if vocab_summary["flagged_examples"]:
        print("\nflagged renderings (soft, review these):")
        for row in vocab_summary["flagged_examples"]:
            print(f"  {','.join(row['flags'])}: {row['cue_text']}")

    case_failures: list[str] = []
    if args.cases:
        cases = list(read_jsonl(args.cases))
        case_failures, case_summary = audit_cases(cases)
        print("\n== composition ==")
        print(json.dumps(case_summary, indent=2))
        for case in longest_cases(cases, args.show_longest):
            print(f"\nlongest case ({len(case.get('cue_targets') or [])} cues):")
            print(f"  {case.get('prompt')}")

    failures = vocab_failures + case_failures
    print(f"\n== hard violations: {len(failures)} ==")
    for failure in failures[:40]:
        print(f"  {failure}")
    if len(failures) > 40:
        print(f"  ... and {len(failures) - 40} more")
    if failures:
        raise SystemExit(1)
    print("clean")


if __name__ == "__main__":
    main()
