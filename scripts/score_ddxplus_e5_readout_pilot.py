"""Score paired DDXPlus E5 readout counterfactuals on validation outputs.

This is a lexical validation diagnostic, not the locked semantic evaluation.
Rows are paired by base_id across original, cue-deleted, and native value-edited
activations. A cue is counted as read when at least ``threshold`` of its content
tokens occur in the emitted NLA text.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.summarize_cue_position_readouts import content_tokens, gold_token_recall
from src.jsonl import read_jsonl


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected METHOD=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("expected non-empty METHOD=PATH")
    return name.strip(), Path(raw_path).expanduser()


def emitted_text(row: dict[str, Any]) -> str:
    return " ".join(str(row.get("nla_output") or row.get("raw_nla_output") or "").split())


def cue_hit(cue: str, text: str, threshold: float) -> bool:
    recall = gold_token_recall(cue, text)
    return recall is not None and recall >= threshold


def contrastive_cue_hit(cue: str, other: str, text: str, threshold: float) -> bool:
    """Match only tokens that distinguish one native value from the other."""
    cue_tokens = content_tokens(cue)
    discriminative = cue_tokens - content_tokens(other)
    target_tokens = discriminative or cue_tokens
    if not target_tokens:
        return False
    observed = content_tokens(text)
    return len(target_tokens & observed) / len(target_tokens) >= threshold


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


def score_method(rows: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    by_base: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    variants: Counter[str] = Counter()
    current_recalls: list[float] = []
    parsed = 0

    for row in rows:
        base_id = str(row.get("base_id") or "")
        variant = str(row.get("variant") or "")
        if not base_id or not variant:
            raise ValueError(f"Missing base_id/variant on row {row.get('id')!r}")
        if variant in by_base[base_id]:
            raise ValueError(f"Duplicate {base_id}/{variant}")
        by_base[base_id][variant] = row
        variants[variant] += 1
        parsed += bool(row.get("parsed_explanation_tag"))
        text = emitted_text(row)
        cues = [str(cue) for cue in row.get("cue_targets") or [] if str(cue).strip()]
        if cues:
            current_recalls.append(
                sum(cue_hit(cue, text, threshold) for cue in cues) / len(cues)
            )

    deletion = Counter()
    value_edit = Counter()
    untouched = {
        "deletion": Counter(),
        "value_edit": Counter(),
    }

    for base_id, group in by_base.items():
        original = group.get("original")
        if original is None:
            raise ValueError(f"Base case {base_id} has no original row")
        original_text = emitted_text(original)
        original_cues = [str(cue) for cue in original.get("cue_targets") or []]

        deleted = group.get("cue_deleted")
        if deleted is not None:
            deletion["pairs"] += 1
            target = str(deleted.get("cf_original_cue") or "")
            deleted_text = emitted_text(deleted)
            original_hit = cue_hit(target, original_text, threshold)
            phantom = cue_hit(target, deleted_text, threshold)
            deletion["original_target_hit"] += original_hit
            deletion["deleted_target_phantom"] += phantom
            deletion["removed_after_original_hit"] += original_hit and not phantom
            for cue in deleted.get("cue_targets") or []:
                cue = str(cue)
                before = cue_hit(cue, original_text, threshold)
                after = cue_hit(cue, deleted_text, threshold)
                untouched["deletion"]["items"] += 1
                untouched["deletion"]["original_hits"] += before
                untouched["deletion"]["derived_hits"] += after
                untouched["deletion"]["preserved_original_hits"] += before and after

        edited = group.get("value_edited")
        if edited is not None:
            value_edit["pairs"] += 1
            old = str(edited.get("cf_original_cue") or "")
            new = str(edited.get("cf_replacement_cue") or "")
            edited_text = emitted_text(edited)
            old_before = contrastive_cue_hit(old, new, original_text, threshold)
            old_after = contrastive_cue_hit(old, new, edited_text, threshold)
            new_after = contrastive_cue_hit(new, old, edited_text, threshold)
            value_edit["original_target_hit"] += old_before
            value_edit["replacement_hit"] += new_after
            value_edit["original_persistence"] += old_after
            value_edit["clean_switch"] += new_after and not old_after
            value_edit["clean_switch_after_original_hit"] += (
                old_before and new_after and not old_after
            )
            for cue in original_cues:
                if cue == old:
                    continue
                before = cue_hit(cue, original_text, threshold)
                after = cue_hit(cue, edited_text, threshold)
                untouched["value_edit"]["items"] += 1
                untouched["value_edit"]["original_hits"] += before
                untouched["value_edit"]["derived_hits"] += after
                untouched["value_edit"]["preserved_original_hits"] += before and after

    def untouched_result(name: str) -> dict[str, Any]:
        counts = untouched[name]
        return {
            "items": counts["items"],
            "original_hit_rate": safe_rate(counts["original_hits"], counts["items"]),
            "derived_hit_rate": safe_rate(counts["derived_hits"], counts["items"]),
            "preservation_given_original_hit": safe_rate(
                counts["preserved_original_hits"], counts["original_hits"]
            ),
        }

    return {
        "n": len(rows),
        "base_cases": len(by_base),
        "variant_counts": dict(sorted(variants.items())),
        "parsed_explanation_rate": safe_rate(parsed, len(rows)),
        "mean_current_finding_recall": mean_or_none(current_recalls),
        "deletion": {
            "pairs": deletion["pairs"],
            "original_target_hit_rate": safe_rate(
                deletion["original_target_hit"], deletion["pairs"]
            ),
            "deleted_target_phantom_rate": safe_rate(
                deletion["deleted_target_phantom"], deletion["pairs"]
            ),
            "removal_success_given_original_hit": safe_rate(
                deletion["removed_after_original_hit"],
                deletion["original_target_hit"],
            ),
            "untouched": untouched_result("deletion"),
        },
        "value_edit": {
            "pairs": value_edit["pairs"],
            "original_target_hit_rate": safe_rate(
                value_edit["original_target_hit"], value_edit["pairs"]
            ),
            "replacement_hit_rate": safe_rate(
                value_edit["replacement_hit"], value_edit["pairs"]
            ),
            "original_persistence_rate": safe_rate(
                value_edit["original_persistence"], value_edit["pairs"]
            ),
            "clean_switch_rate": safe_rate(
                value_edit["clean_switch"], value_edit["pairs"]
            ),
            "clean_switch_given_original_hit": safe_rate(
                value_edit["clean_switch_after_original_hit"],
                value_edit["original_target_hit"],
            ),
            "untouched": untouched_result("value_edit"),
        },
    }


def format_rate(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def write_summary(path: Path, results: dict[str, dict[str, Any]], threshold: float) -> None:
    lines = [
        "# DDXPlus E5 Readout Pilot",
        "",
        "Validation-only lexical diagnostic; not a locked-test or semantic score.",
        "",
        f"- content-token recall threshold: **{threshold:.2f}**",
        "",
        (
            "| method | n | bases | parsed | current finding recall | deletion phantom | "
            "removal success* | replacement hit | old-value persistence | clean switch |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, result in results.items():
        deletion = result["deletion"]
        value = result["value_edit"]
        lines.append(
            f"| {method} | {result['n']} | {result['base_cases']} | "
            f"{format_rate(result['parsed_explanation_rate'])} | "
            f"{format_rate(result['mean_current_finding_recall'])} | "
            f"{format_rate(deletion['deleted_target_phantom_rate'])} | "
            f"{format_rate(deletion['removal_success_given_original_hit'])} | "
            f"{format_rate(value['replacement_hit_rate'])} | "
            f"{format_rate(value['original_persistence_rate'])} | "
            f"{format_rate(value['clean_switch_rate'])} |"
        )
    lines.extend(
        [
            "",
            "`*` Removal success is conditional on the target cue being read in the original arm.",
            "",
            "## Untouched Finding Retention",
            "",
            "| method | intervention | items | original hit | derived hit | preservation* |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for method, result in results.items():
        for intervention in ("deletion", "value_edit"):
            item = result[intervention]["untouched"]
            lines.append(
                f"| {method} | {intervention} | {item['items']} | "
                f"{format_rate(item['original_hit_rate'])} | "
                f"{format_rate(item['derived_hit_rate'])} | "
                f"{format_rate(item['preservation_given_original_hit'])} |"
            )
    lines.extend(
        [
            "",
            "`*` Preservation is conditional on the unchanged cue being read in the original arm.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readout", action="append", required=True, type=parse_named_path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    if not 0 <= args.threshold <= 1:
        raise ValueError("--threshold must be between zero and one")

    paths_by_method: dict[str, list[Path]] = defaultdict(list)
    for method, path in args.readout:
        paths_by_method[method].append(path)

    rows_by_method: dict[str, list[dict[str, Any]]] = {}
    id_sets: dict[str, set[str]] = {}
    for method, paths in paths_by_method.items():
        rows = [row for path in paths for row in read_jsonl(path)]
        ids = [str(row.get("id") or "") for row in rows]
        if not ids or any(not row_id for row_id in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"{method}: empty, missing, or duplicate row IDs")
        rows_by_method[method] = rows
        id_sets[method] = set(ids)

    reference_method = next(iter(id_sets))
    reference_ids = id_sets[reference_method]
    for method, ids in id_sets.items():
        if ids != reference_ids:
            raise ValueError(
                f"Population mismatch {reference_method} vs {method}: "
                f"missing={len(reference_ids - ids)} extra={len(ids - reference_ids)}"
            )

    results = {
        method: score_method(rows, args.threshold)
        for method, rows in rows_by_method.items()
    }
    payload = {"threshold": args.threshold, "methods": results}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.summary_md, results, args.threshold)
    print(
        f"[score] methods={len(results)} rows={len(reference_ids)} "
        f"summary={args.summary_md}",
        flush=True,
    )


if __name__ == "__main__":
    main()
