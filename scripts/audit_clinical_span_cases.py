"""Acceptance gate for prose-derived case corpora (MedCaseReasoning).

The DDXPlus gate checks a rendering step this corpus does not have: cues here
are cut from text a clinician wrote, so there is no vocabulary to enumerate and
no question to un-invert. What can still go wrong is different, and until now
went unmeasured:

- the presentation naming the gold diagnosis, which turns an open-ended
  diagnosis into selection from a list;
- a cue that is not verbatim in its own prompt, or nested inside another, which
  breaks span resolution and double-counts a readout;
- near-constant cues, the failure that removed DDXPlus's negatives -- a cue in
  most cases carries no case detail and is free readout credit;
- formulaic "everything was normal" spans, which are legitimate findings but
  systematically easier to read back, so their share has to be known before a
  readout rate means anything.

Composition checks are imported from the DDXPlus gate rather than rewritten, so
the two corpora cannot drift apart on the checks they share.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.audit_ddxplus_cue_rendering import audit_cases
from scripts.make_clinical_span_cases import (
    READER_QUESTION,
    is_boilerplate,
    mentions_diagnosis,
)
from src.jsonl import read_jsonl

# A cue in more than this share of cases carries no case-specific information
# and can be emitted unconditionally for credit.
NEAR_CONSTANT_RATE = 0.5


def audit_prose_cases(
    cases: list[dict[str, Any]], *, near_constant_rate: float = NEAR_CONSTANT_RATE
) -> tuple[list[str], dict[str, Any]]:
    # The DDXPlus shape check assumes rendered questionnaire items and misreads
    # ordinary clinical prose ("was in moderate respiratory distress"), so it is
    # off here; the prose-specific checks below take its place.
    failures, summary = audit_cases(cases, check_malformed=False)
    n = len(cases) or 1

    asked = [
        str(case.get("id"))
        for case in cases
        if READER_QUESTION.search(str(case.get("presentation") or ""))
        or any(READER_QUESTION.search(f"{cue}?") for cue in case.get("cue_targets") or [])
    ]
    failures.extend(
        f"{case_id}: presentation still carries a question put to the reader"
        for case_id in asked[:20]
    )

    leaks = [
        str(case.get("id"))
        for case in cases
        if mentions_diagnosis(
            str(case.get("presentation") or case.get("prompt") or ""),
            str(case.get("diagnosis_name") or ""),
        )
    ]
    failures.extend(f"{case_id}: presentation names the gold diagnosis" for case_id in leaks[:20])

    frequency: Counter = Counter()
    for case in cases:
        for cue in case.get("cue_targets") or []:
            frequency[str(cue)] += 1
    near_constant = [
        (cue, count) for cue, count in frequency.most_common(20) if count / n >= near_constant_rate
    ]
    failures.extend(
        f"near-constant cue in {count / n:.0%} of cases: {cue!r}" for cue, count in near_constant
    )

    flags = [
        bool(value)
        for case in cases
        for value in (
            case.get("cue_is_boilerplate") or [is_boilerplate(cue) for cue in case.get("cue_targets") or []]
        )
    ]
    total_cues = sum(len(case.get("cue_targets") or []) for case in cases)

    summary.update(
        {
            "distinct_cues": len(frequency),
            "cue_occurrences": total_cues,
            "cues_seen_once_rate": round(
                sum(1 for count in frequency.values() if count == 1) / max(len(frequency), 1), 4
            ),
            "most_common_cue_rate": round(
                (frequency.most_common(1)[0][1] / n) if frequency else 0.0, 4
            ),
            "boilerplate_cue_rate": round(sum(flags) / max(total_cues, 1), 4),
            "diagnosis_named_in_presentation": len(leaks),
            "reader_question_in_presentation": len(asked),
        }
    )
    return failures, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--near-constant-rate", type=float, default=NEAR_CONSTANT_RATE)
    parser.add_argument("--show-frequent", type=int, default=12)
    parser.add_argument("--show-longest", type=int, default=1)
    args = parser.parse_args()

    cases = list(read_jsonl(args.cases))
    if not cases:
        raise SystemExit(f"no cases in {args.cases}")
    failures, summary = audit_prose_cases(cases, near_constant_rate=args.near_constant_rate)

    print(json.dumps(summary, indent=2))

    frequency: Counter = Counter()
    for case in cases:
        for cue in case.get("cue_targets") or []:
            frequency[str(cue)] += 1
    print(f"\nmost frequent cues (share of {len(cases):,} cases):")
    for cue, count in frequency.most_common(args.show_frequent):
        mark = "BP" if is_boilerplate(cue) else "  "
        print(f"  {count / len(cases):6.2%} {mark}  {cue[:88]}")

    for case in sorted(cases, key=lambda c: -len(c.get("cue_targets") or []))[: args.show_longest]:
        print(f"\nlongest case ({len(case.get('cue_targets') or [])} cues): {case.get('diagnosis_name')}")
        print(f"  {str(case.get('presentation') or '')[:600]}")

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
