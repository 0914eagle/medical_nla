"""Recompute Slide 20 on one canonical Direct-defined clean cohort.

The cohort is selected once from the referral/direct answer file:

1. the Direct no-note arm is canonically correct; and
2. the presentation does not explicitly name the gold diagnosis.

Every wording and CoT row is then restricted to those same base IDs. In
particular, CoT no-note correctness is *not* used for selection: doing that
would force both Direct and CoT baselines to one on different populations and
destroy the paired comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_hint_effect import group_by_case, summarize_population


def canonical_clean_ids(reference_cases: dict[str, dict[str, dict[str, Any]]]) -> set[str]:
    return {
        base_id
        for base_id, arms in reference_cases.items()
        if bool(arms.get("none", {}).get("source_correct"))
        and not bool(arms.get("none", {}).get("gold_in_prompt"))
    }


def restrict(
    cases: dict[str, dict[str, dict[str, Any]]], eligible_ids: set[str]
) -> dict[str, dict[str, dict[str, Any]]]:
    return {base_id: arms for base_id, arms in cases.items() if base_id in eligible_ids}


def metrics(cases: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    summary = summarize_population(cases)
    arms = summary["arms"]
    if "none" not in arms or "wrong" not in arms:
        raise ValueError("each Slide 20 input needs none and wrong arms")
    none_acc = float(arms["none"]["correct"])
    wrong_acc = float(arms["wrong"]["correct"])
    moved = summary.get("moved") or {}
    moved_n = int(moved.get("n") or 0)
    adopted = int(moved.get("to_suggestion") or 0)
    return {
        "n": int(summary["n"]),
        "none_accuracy": none_acc,
        "wrong_accuracy": wrong_acc,
        "note_cost": none_acc - wrong_acc,
        "moved": moved_n,
        "adopted_suggestion": adopted,
        "other_diagnosis": int(moved.get("to_third_diagnosis") or 0),
        "adoption_among_moved": adopted / moved_n if moved_n else None,
    }


def parse_named_path(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, path = value.split("=", 1)
    if not label.strip() or not path.strip():
        raise argparse.ArgumentTypeError("expected non-empty LABEL=PATH")
    return label.strip(), path.strip()


def fmt(value: float | None, digits: int = 4) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def write_summary(path: str, result: dict[str, Any]) -> None:
    lines = [
        "# Slide 20 Canonical-Cohort Robustness",
        "",
        "The cohort is defined once by canonical Direct no-note correctness and",
        "`gold_in_prompt=false`. CoT correctness is not an eligibility criterion.",
        "",
        f"- eligible clean base IDs: **{result['eligible_clean_n']:,}**",
        "",
        "## Wording robustness",
        "",
        "| wording | n | no-note acc. | wrong-note acc. | note cost (pp) | moved | adopted | adoption / moved |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in result["wordings"].items():
        lines.append(
            f"| {label} | {row['n']:,} | {fmt(row['none_accuracy'])} | "
            f"{fmt(row['wrong_accuracy'])} | {row['note_cost'] * 100:.2f} | "
            f"{row['moved']:,} | {row['adopted_suggestion']:,} | "
            f"{fmt(row['adoption_among_moved'])} |"
        )
    lines.extend([
        "",
        "## Direct versus CoT on identical IDs",
        "",
        f"- paired n: **{result['direct_vs_cot']['n']:,}**",
        "",
        "| generation | no-note acc. | wrong-note acc. | note cost (pp) | moved | adopted | adoption / moved |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for label in ("Direct", "CoT"):
        row = result["direct_vs_cot"][label.lower()]
        lines.append(
            f"| {label} | {fmt(row['none_accuracy'])} | {fmt(row['wrong_accuracy'])} | "
            f"{row['note_cost'] * 100:.2f} | {row['moved']:,} | "
            f"{row['adopted_suggestion']:,} | {fmt(row['adoption_among_moved'])} |"
        )
    lines.extend([
        "",
        "Direct no-note accuracy must be 1.0 by construction. CoT no-note accuracy",
        "need not be 1.0 because the cohort was selected by Direct, not by CoT.",
        "",
    ])
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-answers",
        required=True,
        help="Canonically rescored referral/direct answers carrying none and wrong arms.",
    )
    parser.add_argument(
        "--wording",
        action="append",
        type=parse_named_path,
        default=[],
        metavar="LABEL=PATH",
        help="A canonically rescored wording answer file. Repeat for each wording.",
    )
    parser.add_argument("--cot-answers", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    reference = group_by_case(args.reference_answers)
    eligible = canonical_clean_ids(reference)
    if not eligible:
        raise SystemExit("no canonical Direct-clean cases")

    wordings: dict[str, Any] = {"Referral": metrics(restrict(reference, eligible))}
    coverage: dict[str, Any] = {"Referral": len(restrict(reference, eligible))}
    for label, path in args.wording:
        cases = group_by_case(path)
        subset = restrict(cases, eligible)
        if not subset:
            raise SystemExit(f"{label}: no base IDs overlap the reference cohort")
        wordings[label] = metrics(subset)
        coverage[label] = len(subset)

    cot = group_by_case(args.cot_answers)
    shared = eligible & set(cot)
    if not shared:
        raise SystemExit("CoT answers do not overlap the canonical Direct-clean cohort")
    direct_paired = metrics(restrict(reference, shared))
    cot_paired = metrics(restrict(cot, shared))
    if abs(direct_paired["none_accuracy"] - 1.0) > 1e-12:
        raise SystemExit("Direct no-note accuracy is not 1.0 on its own eligibility cohort")

    result = {
        "reference_answers": args.reference_answers,
        "eligible_clean_n": len(eligible),
        "wording_coverage": coverage,
        "wordings": wordings,
        "direct_vs_cot": {
            "n": len(shared),
            "direct": direct_paired,
            "cot": cot_paired,
        },
    }
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary(args.summary_md, result)
    print(f"[cohort] canonical Direct-clean IDs: {len(eligible):,}")
    for label, row in wordings.items():
        print(
            f"[wording] {label:<12} n={row['n']:,} none={row['none_accuracy']:.4f} "
            f"wrong={row['wrong_accuracy']:.4f} moved={row['moved']:,} "
            f"adopted={row['adopted_suggestion']:,}"
        )
    print(
        f"[paired] Direct/CoT n={len(shared):,}  "
        f"direct={direct_paired['none_accuracy']:.4f}/{direct_paired['wrong_accuracy']:.4f}  "
        f"cot={cot_paired['none_accuracy']:.4f}/{cot_paired['wrong_accuracy']:.4f}"
    )
    print(f"[json] {args.output_json}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
