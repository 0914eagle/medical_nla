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
from collections import Counter, defaultdict
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


# Antecedents that a child does not have. DDXPlus's simulator samples
# antecedents without conditioning on age, so a 4-year-old can be given
# Parkinson's disease, heart failure, a prior stroke and COPD at once. Verified
# not to be a rendering fault: those four are four distinct binary evidences
# (E_95, E_106, E_107, E_123), not one multi-value question wrongly expanded.
#
# It is reported, never failed. The cue text is well-formed clinical English, so
# the readout task -- recover the cue from its activation -- is unaffected, and
# every contrast the paper draws is within-prompt (direct against chain of
# thought, counterfactual against original), which holds the implausibility
# constant. Only the absolute diagnostic accuracy is touched, and that is quoted
# descriptively.
ADULT_ONSET_ANTECEDENTS = (
    "parkinson",
    "heart failure",
    "stroke",
    "chronic obstructive pulmonary disease",
    "myocardial infarction",
    "atrial fibrillation",
    "smoke cigarettes",
)

PAEDIATRIC_AGE_LIMIT = 12


def adult_onset_antecedents_in_a_child(case: dict[str, Any]) -> list[str]:
    """Adult-onset history attributed to a child, if any."""
    try:
        age = int(float(case.get("age")))
    except (TypeError, ValueError):
        return []
    if age > PAEDIATRIC_AGE_LIMIT:
        return []
    return [
        cue
        for cue in (case.get("cue_targets") or [])
        if any(term in str(cue).lower() for term in ADULT_ONSET_ANTECEDENTS)
    ]


def suspicious_flag_matches(text: str) -> list[tuple[str, list[str]]]:
    """Soft flags with the strings that raised them.

    The count alone is not usable on prose. `uppercase_code` looks for
    unresolved identifiers ("V_29"), but case-report English is written in
    abbreviations, so it fires on 9.8% of MedCaseReasoning cues -- a rate that
    says nothing about whether any of them is an identifier. An allowlist
    cannot keep up with clinical prose; reporting *what* matched can, because a
    human scanning "CT, MRI, SpO2, BP" for a "V_29" needs one look.
    """
    found = []
    for name, pattern in SUSPICIOUS_PATTERNS.items():
        matches = [
            match if isinstance(match, str) else match[0] for match in pattern.findall(text)
        ]
        if not matches:
            continue
        if name == "uppercase_code":
            matches = [match for match in matches if match not in KNOWN_ABBREVIATIONS]
            if not matches:
                # A named condition, not an unresolved code.
                continue
        found.append((name, matches))
    return found


def suspicious_flags(text: str) -> list[str]:
    return [name for name, _ in suspicious_flag_matches(text)]


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


def audit_cases(
    cases: list[dict[str, Any]], *, check_malformed: bool = True
) -> tuple[list[str], dict[str, Any]]:
    """Check every case, not a sample, for problems only assembly can create.

    `check_malformed` applies the DDXPlus renderer's own shape check, which
    encodes assumptions about rendered questionnaire items: a leading auxiliary
    means an un-inverted question, `which` means an interrogative. Prose written
    by a clinician breaks both innocently ("was in moderate respiratory
    distress", "which resolved with self-stretching"), so corpora cut from prose
    turn it off. Everything else here is corpus-independent.
    """
    failures: list[str] = []
    cue_counts: list[int] = []
    cue_words: list[int] = []
    prompt_words: list[int] = []
    nested_cases = 0
    polarity: Counter = Counter()
    flags: Counter = Counter()
    flag_examples: dict[str, Counter] = defaultdict(Counter)
    paediatric_conflicts = 0
    paediatric_examples: Counter = Counter()

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
        lowered = [cue.lower() for cue in cues]
        if any(
            i != j and a in b for i, a in enumerate(lowered) for j, b in enumerate(lowered)
        ):
            # One cue inside another gives the readout credit for both.
            nested_cases += 1
        seen: set[str] = set()
        for cue in cues:
            cue_words.append(len(cue.split()))
            # Extraction resolves cues by substring, so a cue that is not
            # verbatim in its own prompt cannot be located at all.
            if cue not in prompt:
                failures.append(f"{case_id}: cue not verbatim in prompt -> {cue!r}")
            if cue.lower() in seen:
                failures.append(f"{case_id}: duplicate cue -> {cue!r}")
            seen.add(cue.lower())
            if check_malformed and is_malformed_cue(cue):
                failures.append(f"{case_id}: malformed cue in prompt -> {cue!r}")
            for name, matches in suspicious_flag_matches(cue):
                flags[name] += 1
                flag_examples[name].update(matches)
        implausible = adult_onset_antecedents_in_a_child(case)
        if implausible:
            paediatric_conflicts += 1
            paediatric_examples.update(implausible)

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
        "cue_words": percentiles(cue_words),
        "prompt_words": percentiles(prompt_words),
        "cue_polarity": dict(polarity),
        "cases_with_nested_cues": nested_cases,
        "suspicious_in_prompts": dict(flags),
        # What actually matched, so a flag that fires on a tenth of the corpus
        # can be dismissed or acted on rather than only counted.
        "adult_onset_antecedent_in_a_child": paediatric_conflicts,
        "adult_onset_antecedent_examples": [
            f"{cue} ({count})" for cue, count in paediatric_examples.most_common(8)
        ],
        "suspicious_examples": {
            name: [token for token, _ in counter.most_common(10)]
            for name, counter in sorted(flag_examples.items())
        },
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
    parser.add_argument(
        "--negative-cues",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Must match the flag the cases were generated with, or the vocabulary "
            "pass reports renderings the corpus does not contain."
        ),
    )
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
