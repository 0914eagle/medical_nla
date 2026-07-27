"""Score Medical-NLA activation-grounding controls."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.score_medical_nla_v2_readouts import clean_text, hit_terms, score_row
from src.jsonl import read_jsonl, write_jsonl


def donor_aliases(row: dict[str, Any]) -> list[str]:
    aliases = [clean_text(value) for value in row.get("donor_diagnosis_aliases") or [] if clean_text(value)]
    name = clean_text(row.get("donor_diagnosis_name"))
    if name:
        aliases.insert(0, name)
    diagnosis_id = clean_text(row.get("donor_diagnosis_id"))
    if diagnosis_id:
        aliases.append(diagnosis_id.replace("_", " "))
    deduped = []
    seen = set()
    for alias in aliases:
        key = alias.lower()
        if alias and key not in seen:
            deduped.append(alias)
            seen.add(key)
    return deduped


def score_control_row(row: dict[str, Any]) -> dict[str, Any]:
    scored = score_row(row)
    donor_terms = donor_aliases(row)
    donor_answer_hits = hit_terms(scored.get("answer_readout") or "", donor_terms)
    donor_output_hits = hit_terms(scored.get("nla_output") or "", donor_terms)
    scored["donor_diagnosis_aliases_scored"] = donor_terms
    scored["donor_answer_hits"] = donor_answer_hits
    scored["donor_output_hits"] = donor_output_hits
    scored["donor_answer_hit"] = bool(donor_answer_hits)
    scored["donor_output_hit"] = bool(donor_output_hits)
    return scored


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    by_control: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_control[str(row.get("control_type") or "unknown")].append(row)

    with path.open("w", encoding="utf-8") as f:
        f.write("# Activation Control Readout Summary\n\n")
        f.write("Lexical alias scoring of Medical-NLA structured readouts.\n\n")
        f.write(f"- n: {len(rows)}\n\n")
        f.write("| control_type | n | original_answer_hit | donor_answer_hit | parsed_answer | mean_cue_recall |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for control_type, items in sorted(by_control.items()):
            original_hits = sum(bool(row.get("answer_hit")) for row in items)
            donor_applicable = [row for row in items if row.get("donor_diagnosis_id")]
            donor_hits = sum(bool(row.get("donor_answer_hit")) for row in donor_applicable)
            parsed = sum(bool(row.get("parsed_answer")) for row in items)
            recalls = [float(row["cue_recall"]) for row in items if row.get("cue_recall") is not None]
            f.write(
                f"| {control_type} | {len(items)} | "
                f"{original_hits / len(items) if items else 0:.4f} ({original_hits}) | "
                f"{donor_hits / len(donor_applicable) if donor_applicable else 0:.4f} ({donor_hits}) | "
                f"{parsed / len(items) if items else 0:.4f} ({parsed}) | "
                f"{mean(recalls) if recalls else 0:.4f} |\n"
            )

        f.write("\n## Mismatched Direction Counts\n\n")
        f.write("| category | n |\n")
        f.write("|---|---:|\n")
        direction_counts = Counter()
        for row in rows:
            if row.get("control_type") != "mismatched_activation":
                continue
            original = bool(row.get("answer_hit"))
            donor = bool(row.get("donor_answer_hit"))
            if donor and not original:
                category = "follows_donor_only"
            elif original and not donor:
                category = "follows_original_only"
            elif donor and original:
                category = "hits_both_original_and_donor"
            else:
                category = "hits_neither"
            direction_counts[category] += 1
        for category, count in direction_counts.most_common():
            f.write(f"| {category} | {count} |\n")

        f.write("\n## Examples\n\n")
        f.write("| id | control | original_gold | donor_gold | answer | original_hit | donor_hit |\n")
        f.write("|---|---|---|---|---|---:|---:|\n")
        for row in rows[:120]:
            answer = clean_text(row.get("answer_readout") or "-")
            if len(answer) > 120:
                answer = answer[:117] + "..."
            f.write(
                f"| {row.get('id')} | {row.get('control_type')} | "
                f"{row.get('diagnosis_name') or row.get('gold_diagnosis_name')} | "
                f"{row.get('donor_diagnosis_name') or '-'} | {answer} | "
                f"{'Y' if row.get('answer_hit') else 'N'} | "
                f"{'Y' if row.get('donor_answer_hit') else 'N'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    scored = [score_control_row(row) for row in read_jsonl(args.input)]
    if not scored:
        raise ValueError("No rows to score.")
    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, scored)
    write_summary(summary_path, scored)
    print(f"[done] wrote {len(scored)} rows to {output_path}")
    print(f"[done] wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
