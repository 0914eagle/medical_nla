"""Assemble Figure 4 values from one canonical-eligible DDXPlus cohort.

This is intentionally a join over per-case artifacts, not a transcription of
printed summaries. It prevents the detection panel, correction panel, and
legend counts from silently referring to different matcher-era populations.
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

from scripts.analyze_cot_monitor import parse_probability
from scripts.analyze_hint_effect import (
    group_by_case,
    require_canonical_no_note_correct,
)
from scripts.compare_channels_on_attribution import stratified_auroc
from src.jsonl import read_jsonl


ANSWER = "answer states the suspicion (output-only)"
CHAIN = "chain dwells on the suspicion"
READOUT = "answer omits the internal conclusion (containment)"


def load_rows(path: str) -> dict[str, dict[str, Any]]:
    return {
        str(row["base_id"]): row
        for row in read_jsonl(path)
        if row.get("base_id")
    }


def score(rows: list[dict[str, Any]], field: str) -> float:
    usable = [row for row in rows if row.get(field) is not None]
    value, _ = stratified_auroc(
        [float(row[field]) for row in usable],
        [bool(row["moved"]) for row in usable],
        [str(row["diagnosis_name"]) for row in usable],
    )
    if value != value:
        raise ValueError(f"within-diagnosis AUROC is undefined for {field}")
    return value


def rung_name(row: dict[str, Any], path: str) -> str:
    value = str(row.get("ladder_rung") or "").lower().strip()
    if value and not value.startswith("r"):
        value = f"r{value}"
    if value in {"r3", "r4", "r5", "r6"}:
        return value
    for candidate in ("r3", "r4", "r5", "r6"):
        if candidate in Path(path).stem.lower():
            return candidate
    raise ValueError(f"cannot identify ladder rung from {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--cases")
    parser.add_argument("--channel-scores", required=True)
    parser.add_argument("--monitor", nargs="+", required=True)
    parser.add_argument("--probe-verdicts", required=True)
    parser.add_argument("--rungs", nargs="+", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-md", required=True)
    args = parser.parse_args()

    all_cases = group_by_case(args.answers, args.cases)
    eligible_cases = require_canonical_no_note_correct(all_cases)
    eligible = set(eligible_cases)
    channels = load_rows(args.channel_scores)
    probe = load_rows(args.probe_verdicts)

    monitor: dict[str, float] = {}
    for path in args.monitor:
        for row in read_jsonl(path):
            value = parse_probability(str(row.get("response") or ""))
            if value is not None:
                monitor[str(row.get("id") or row.get("base_id") or "")] = value

    shared = sorted(eligible & set(channels) & set(probe))
    if len(shared) != len(eligible):
        raise SystemExit(
            "canonical cohort does not join completely: "
            f"eligible={len(eligible):,}, shared={len(shared):,}"
        )

    detection_rows: list[dict[str, Any]] = []
    for base_id in shared:
        row = dict(channels[base_id])
        if base_id not in monitor:
            raise SystemExit(f"missing LLM monitor verdict for {base_id}")
        row["llm_monitor"] = monitor[base_id]
        row["linear_probe"] = float(probe[base_id]["probe_flag"])
        detection_rows.append(row)

    silent_rows = [row for row in detection_rows if not row["answer_is_suggestion"]]
    detection = {
        "channels": [
            "Answer heuristic",
            "Rule-based CoT",
            "LLM monitor",
            "AV readout",
            "Linear probe",
        ],
        "all": [
            score(detection_rows, ANSWER),
            score(detection_rows, CHAIN),
            score(detection_rows, "llm_monitor"),
            score(detection_rows, READOUT),
            score(detection_rows, "linear_probe"),
        ],
        # Output-only and literal chain rules are deliberately omitted in the
        # subset defined to remove the literal answer=suggestion shortcut.
        "silent": [
            None,
            None,
            score(silent_rows, "llm_monitor"),
            score(silent_rows, READOUT),
            score(silent_rows, "linear_probe"),
        ],
        "n_all": len(detection_rows),
        "n_silent": len(silent_rows),
        "n_moved": sum(bool(row["moved"]) for row in detection_rows),
        "n_silent_moved": sum(bool(row["moved"]) for row in silent_rows),
    }

    rung_rows: dict[str, dict[str, dict[str, Any]]] = {}
    for path in args.rungs:
        rows = [row for row in read_jsonl(path) if str(row.get("base_id") or "") in eligible]
        if not rows:
            raise SystemExit(f"no canonical rows in ladder file {path}")
        name = rung_name(rows[0], path)
        rung_rows[name] = {str(row["base_id"]): row for row in rows}
    missing_rungs = [name for name in ("r3", "r4", "r5", "r6") if name not in rung_rows]
    if missing_rungs:
        raise SystemExit(f"missing ladder rungs: {', '.join(missing_rungs)}")

    common = set(eligible)
    for rows in rung_rows.values():
        common &= set(rows)
    if common != eligible:
        raise SystemExit(
            f"ladder join is incomplete: eligible={len(eligible):,}, common={len(common):,}"
        )

    ordered = sorted(common)
    first = [rung_rows["r3"][base_id] for base_id in ordered]
    moved_ids = [base_id for base_id in ordered if bool(rung_rows["r3"][base_id].get("moved"))]
    correction = {
        "stages": ["First answer", "r3", "r4", "r5", "r6"],
        "overall": [
            sum(bool(row.get("first_correct")) for row in first) / len(first),
            *[
                sum(bool(rung_rows[name][base_id].get("source_correct")) for base_id in ordered)
                / len(ordered)
                for name in ("r3", "r4", "r5", "r6")
            ],
        ],
        "moved": [
            sum(bool(rung_rows["r3"][base_id].get("first_correct")) for base_id in moved_ids)
            / len(moved_ids),
            *[
                sum(bool(rung_rows[name][base_id].get("source_correct")) for base_id in moved_ids)
                / len(moved_ids)
                for name in ("r3", "r4", "r5", "r6")
            ],
        ],
        "n_all": len(ordered),
        "n_moved": len(moved_ids),
    }

    output = {
        "cohort": {
            "definition": "canonical no-note correct",
            "source_rows": len(all_cases),
            "eligible_rows": len(eligible),
        },
        "detection": detection,
        "correction": correction,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Canonical-Eligible Figure 4 Values",
        "",
        f"- cohort: canonical no-note correct, {len(eligible):,}/{len(all_cases):,}",
        f"- moved: {detection['n_moved']:,}",
        f"- silent: {detection['n_silent']:,}",
        f"- silent moved: {detection['n_silent_moved']:,}",
        "",
        "## Detection",
        "",
        "| channel | all | silent |",
        "|---|---:|---:|",
    ]
    for name, all_value, silent_value in zip(
        detection["channels"], detection["all"], detection["silent"], strict=True
    ):
        silent_text = "—" if silent_value is None else f"{silent_value:.4f}"
        lines.append(f"| {name} | {all_value:.4f} | {silent_text} |")
    lines.extend([
        "",
        "## Correction",
        "",
        "| stage | overall | moved |",
        "|---|---:|---:|",
    ])
    for name, overall, moved_value in zip(
        correction["stages"], correction["overall"], correction["moved"], strict=True
    ):
        lines.append(f"| {name} | {overall:.4f} | {moved_value:.4f} |")
    summary_path = Path(args.summary_md)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[json] {output_path}")
    print(f"[summary] {summary_path}")


if __name__ == "__main__":
    main()
