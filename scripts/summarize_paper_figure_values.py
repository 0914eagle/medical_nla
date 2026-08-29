"""Combine canonical Figure 2/3 value artifacts into one audit summary."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any


LAYERS = (16, 24, 32)
FIGURE2_SERIES = (
    "DiReCT category top-1",
    "DiReCT PDD top-1",
    "DDXPlus finding micro F1",
    "DDXPlus native-value accuracy",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_rate(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} is not a finite rate: {value!r}")
    return number


def validate_figure2(payload: dict[str, Any]) -> dict[str, list[float]]:
    if payload.get("population") != "validation":
        raise ValueError("Figure 2 values are not validation results")
    if tuple(payload.get("layers") or []) != LAYERS:
        raise ValueError(f"Figure 2 layers must be {LAYERS}")
    series = payload.get("series") or {}
    if set(series) != set(FIGURE2_SERIES):
        raise ValueError(f"Figure 2 series mismatch: {sorted(series)}")
    return {
        label: [
            require_rate(value, f"{label}/HS{layer}")
            for layer, value in zip(LAYERS, series[label], strict=True)
        ]
        for label in FIGURE2_SERIES
    }


def validate_figure3(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("population") != "locked_test":
        raise ValueError("Figure 3 values are not locked-test results")
    metrics = payload.get("metrics") or {}
    if int(metrics.get("layer", -1)) != 24:
        raise ValueError("Figure 3 must use the frozen HS24 probe")
    deletion = metrics.get("deletion") or {}
    retained = metrics.get("retained") or {}
    edit = metrics.get("value_edit") or {}
    if int(deletion.get("eligible", -1)) != 4540:
        raise ValueError("Figure 3 deletion denominator must be 4,540")
    if int(edit.get("eligible", -1)) != 539:
        raise ValueError("Figure 3 value-edit denominator must be 539")
    for field in ("original_hit", "phantom", "removal_given_original_hit"):
        require_rate(deletion.get(field), f"deletion/{field}")
    require_rate(retained.get("preservation_given_original_hit"), "retained/preservation")
    for field in ("replacement_hit", "old_persistence", "clean_switch"):
        require_rate(edit.get(field), f"value_edit/{field}")
    return metrics


def fmt(value: Any, *, signed: bool = False) -> str:
    number = float(value)
    return f"{number:+.4f}" if signed else f"{number:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure2-values", required=True, type=Path)
    parser.add_argument("--figure3-values", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    figure2_payload = json.loads(args.figure2_values.read_text(encoding="utf-8"))
    figure3_payload = json.loads(args.figure3_values.read_text(encoding="utf-8"))
    figure2 = validate_figure2(figure2_payload)
    figure3 = validate_figure3(figure3_payload)
    report = {
        "schema_version": 1,
        "figure2": figure2,
        "figure3": figure3,
        "sources": {
            "figure2_values": {
                "path": str(args.figure2_values),
                "sha256": sha256_file(args.figure2_values),
            },
            "figure3_values": {
                "path": str(args.figure3_values),
                "sha256": sha256_file(args.figure3_values),
            },
        },
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    deletion = figure3["deletion"]
    retained = figure3["retained"]
    edit = figure3["value_edit"]
    deletion_n = deletion["eligible"]
    edit_n = edit["eligible"]
    lines = [
        "# Paper Figure 2/3 Numeric Summary",
        "",
        "Figure 2 is validation layer sensitivity. Figure 3 is the frozen DDXPlus locked test.",
        "",
        "## Figure 2 — P0 Layer Sensitivity",
        "",
        "| target | HS16 | HS24 | HS32 |",
        "|---|---:|---:|---:|",
    ]
    for label in FIGURE2_SERIES:
        scores = figure2[label]
        lines.append(
            f"| {label} | {scores[0]:.4f} | {scores[1]:.4f} | {scores[2]:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Figure 3 — DDXPlus Counterfactual Response",
            "",
            "| metric | value | denominator |",
            "|---|---:|---:|",
            "| deleted-cue probability before | "
            f"{fmt(deletion['mean_probability_before'])} | {deletion_n} |",
            "| deleted-cue probability after | "
            f"{fmt(deletion['mean_probability_after'])} | {deletion_n} |",
            "| deleted-cue probability drop | "
            f"{fmt(deletion['mean_probability_drop'], signed=True)} | {deletion_n} |",
            f"| deletion original hit | {fmt(deletion['original_hit'])} | {deletion_n} |",
            f"| deletion phantom | {fmt(deletion['phantom'])} | {deletion_n} |",
            "| removal given original hit | "
            f"{fmt(deletion['removal_given_original_hit'])} | conditional |",
            "| untouched finding retention | "
            f"{fmt(retained['preservation_given_original_hit'])} | "
            f"{retained['conditional_denominator']} |",
            f"| value-edit replacement hit | {fmt(edit['replacement_hit'])} | {edit_n} |",
            f"| value-edit old persistence | {fmt(edit['old_persistence'])} | {edit_n} |",
            "| clean value switch | "
            f"{fmt(edit['clean_switch'])} | {edit['clean_switch_denominator']} |",
            "| new-minus-old value margin change | "
            f"{fmt(edit['mean_new_minus_old_margin_change'], signed=True)} | {edit_n} |",
            "",
            "The script verifies deletion n=4,540, value-edit n=539, HS24, and source hashes.",
            "",
        ]
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"[json] {args.output_json}")
    print(f"[summary] {args.summary_md}")


if __name__ == "__main__":
    main()
