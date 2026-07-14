"""Score specificity-shift NLA outputs with simple lexical probes.

This is a lightweight, reproducible screening metric. It does not replace
claim-level groundedness; it turns the current manual read into a table.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


SPECIFIC_ALIASES = {
    "shift_001": ["meningitis", "meningeal", "neck stiffness", "neck rigidity"],
    "shift_002": ["wernicke", "thiamine", "vitamin b1", "b1 deficiency"],
    "shift_003": ["aortic dissection", "dissection", "aortic", "mediastinum"],
    "shift_004": ["pheochromocytoma", "adrenal", "adrenergic", "hypertension"],
    "shift_005": ["deep vein thrombosis", "dvt", "thrombosis", "deep veins"],
    "shift_006": ["endocarditis", "blood culture", "murmur", "iv drug"],
    "shift_007": ["pulmonary embolism", "embolism", "pleuritic", "tachycardia"],
    "shift_008": ["appendicitis", "right lower quadrant", "rebound"],
    "shift_009": ["subarachnoid", "sah", "thunderclap", "worst headache"],
    "shift_010": ["stroke", "aphasia", "facial droop", "unilateral"],
    "shift_011": ["pneumonia", "lobar", "consolidation"],
    "shift_012": ["hemolysis", "hemolytic", "unconjugated", "ldh", "anemia"],
}

NONSPECIFIC_ALIASES = {
    "shift_001": ["migraine", "light sensitivity", "photophobia"],
    "shift_002": ["myasthenia", "cranial nerve", "ocular", "ophthalmoplegia"],
    "shift_003": ["acute coronary", "acs", "mi", "chest pain"],
    "shift_004": ["anxiety", "arrhythmia", "palpitations"],
    "shift_005": ["injury", "edema", "swelling"],
    "shift_006": ["infection", "inflammation", "fever"],
    "shift_007": ["asthma", "anxiety", "respiratory", "dyspnea"],
    "shift_008": ["gastrointestinal", "abdominal pain", "nonspecific"],
    "shift_009": ["migraine", "tension", "headache"],
    "shift_010": ["fatigue", "weakness", "neuromuscular"],
    "shift_011": ["uri", "bronchitis", "cough"],
    "shift_012": ["liver", "biliary", "jaundice"],
}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def variant(row: dict) -> str:
    return row.get("variant") or row["id"].split("__", 1)[1]


def contains_term(text: str, term: str) -> bool:
    text_l = text.lower()
    term_l = term.lower()
    if re.search(r"\s", term_l):
        return term_l in text_l
    return re.search(r"(?<![a-z0-9])" + re.escape(term_l) + r"(?![a-z0-9])", text_l) is not None


def hits(text: str, aliases: list[str]) -> list[str]:
    return [term for term in aliases if contains_term(text, term)]


def classify_case(case_rows: dict[str, dict]) -> str:
    nonspecific_full = case_rows.get("specific_full_nonspecific_cue", {})
    full_format = case_rows.get("specific_full_format", {})
    specific_cues = [
        row for key, row in case_rows.items() if key.startswith("specific_full_specific_cue_")
    ]
    nonspecific_shifted = bool(nonspecific_full.get("specific_hit"))
    format_shifted = bool(full_format.get("specific_hit"))
    specific_cue_shifted = any(row.get("specific_hit") for row in specific_cues)

    if format_shifted:
        return "format_shift"
    if nonspecific_shifted:
        return "nonspecific_cue_shift"
    if specific_cue_shifted:
        return "specific_cue_only"
    return "no_specific_signal"


def write_summary(path: Path, scored: list[dict]) -> None:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    by_role: dict[str, list[dict]] = defaultdict(list)
    by_case: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in scored:
        by_variant[variant(row)].append(row)
        by_role[row.get("target_role", "unknown")].append(row)
        by_case[row["base_id"]][variant(row)] = row

    case_classes = {base_id: classify_case(rows) for base_id, rows in by_case.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("# Specificity Shift Summary\n\n")
        f.write("Lexical screening only; not claim-level groundedness.\n\n")
        f.write("## By Variant\n\n")
        f.write("| variant | n | specific_hit | nonspecific_hit | mean_activation_norm |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for key in sorted(by_variant):
            rows = by_variant[key]
            f.write(
                f"| {key} | {len(rows)} | "
                f"{sum(r['specific_hit'] for r in rows)} | "
                f"{sum(r['nonspecific_hit'] for r in rows)} | "
                f"{mean(float(r['activation_norm']) for r in rows):.1f} |\n"
            )
        f.write("\n## By Target Role\n\n")
        f.write("| target_role | n | specific_hit | nonspecific_hit |\n")
        f.write("|---|---:|---:|---:|\n")
        for key in sorted(by_role):
            rows = by_role[key]
            f.write(
                f"| {key} | {len(rows)} | "
                f"{sum(r['specific_hit'] for r in rows)} | "
                f"{sum(r['nonspecific_hit'] for r in rows)} |\n"
            )
        f.write("\n## Case Classes\n\n")
        counts = Counter(case_classes.values())
        for key in sorted(counts):
            f.write(f"- {key}: {counts[key]}\n")
        f.write("\n## Per Case\n\n")
        f.write("| case | class | nonspecific_cue_full | specific_cues | format |\n")
        f.write("|---|---|---|---|---|\n")
        for base_id in sorted(by_case):
            rows = by_case[base_id]
            ns = rows.get("specific_full_nonspecific_cue", {})
            fmt = rows.get("specific_full_format", {})
            specific_cues = [
                rows[key] for key in sorted(rows) if key.startswith("specific_full_specific_cue_")
            ]
            cue_bits = ", ".join(
                f"{row.get('target_text')}:{'Y' if row.get('specific_hit') else 'N'}"
                for row in specific_cues
            )
            f.write(
                f"| {base_id} | {case_classes[base_id]} | "
                f"{'Y' if ns.get('specific_hit') else 'N'} | "
                f"{cue_bits} | "
                f"{'Y' if fmt.get('specific_hit') else 'N'} |\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    scored = []
    for row in read_jsonl(Path(args.input)):
        base_id = row["base_id"]
        output = row.get("nla_output", "")
        specific_hits = hits(output, SPECIFIC_ALIASES.get(base_id, []))
        nonspecific_hits = hits(output, NONSPECIFIC_ALIASES.get(base_id, []))
        scored.append(
            {
                **row,
                "variant": variant(row),
                "specific_aliases": SPECIFIC_ALIASES.get(base_id, []),
                "nonspecific_aliases": NONSPECIFIC_ALIASES.get(base_id, []),
                "specific_hits": specific_hits,
                "nonspecific_hits": nonspecific_hits,
                "specific_hit": bool(specific_hits),
                "nonspecific_hit": bool(nonspecific_hits),
            }
        )
    write_jsonl(Path(args.output_jsonl), scored)
    write_summary(Path(args.summary_md), scored)
    print(f"wrote {len(scored)} scored rows to {args.output_jsonl}")
    print(f"wrote summary to {args.summary_md}")


if __name__ == "__main__":
    main()
