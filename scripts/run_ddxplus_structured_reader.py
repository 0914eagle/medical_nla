"""Freeze and evaluate a deterministic DDXPlus probe-guided reader.

This is a structured-monitor baseline, not an open-ended NLA.  A frozen
finding/value probe selects evidence IDs and values from one activation.  A
train-only lexicon then renders those structured predictions as clinical
bullets.  Evaluation never reads prompt text to construct a prediction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from scripts.evaluate_ddxplus_finding_value_probes import normalize_slices
from scripts.train_ddxplus_finding_value_probes import base_id, predict_linear
from src.jsonl import read_jsonl, write_jsonl


LOCKED_CONFIRMATION = "I_ACCEPT_DDXPLUS_STRUCTURED_READER_LOCKED_TEST"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def parse_path_maps(values: Iterable[str]) -> list[tuple[str, str]]:
    result = []
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError("--path-map must be OLD=NEW")
        old, new = value.split("=", 1)
        if not old:
            raise argparse.ArgumentTypeError("--path-map OLD prefix cannot be empty")
        result.append((old, new))
    return result


def mapped_path(value: Any, path_maps: list[tuple[str, str]]) -> Path:
    raw = str(value or "")
    for old, new in path_maps:
        if raw.startswith(old):
            raw = new + raw[len(old) :]
            break
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def aligned_cues(row: dict[str, Any]) -> list[tuple[str, str, str]]:
    texts = list(row.get("cue_targets") or [])
    evidence_ids = list(row.get("cue_evidence_ids") or [])
    value_ids = list(row.get("cue_value_ids") or [])
    if not (len(texts) == len(evidence_ids) == len(value_ids)):
        raise ValueError(f"Cue fields are not aligned for {base_id(row)!r}")
    return [
        (str(evidence), str(value or ""), clean(text))
        for evidence, value, text in zip(evidence_ids, value_ids, texts, strict=True)
        if evidence and clean(text)
    ]


def canonical_phrase(counts: Counter[str]) -> tuple[str, int]:
    if not counts:
        raise ValueError("Cannot select a canonical phrase from an empty counter")
    maximum = max(counts.values())
    phrase = min(text for text, count in counts.items() if count == maximum)
    return phrase, maximum


def build_lexicon(
    rows: list[dict[str, Any]],
    finding_labels: list[str],
    values_by_evidence: dict[str, list[str]],
) -> dict[str, Any]:
    finding_set = set(finding_labels)
    value_set = {
        evidence: set(values) for evidence, values in values_by_evidence.items()
    }
    evidence_texts: dict[str, Counter[str]] = defaultdict(Counter)
    value_texts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for row in rows:
        for evidence, value, text in aligned_cues(row):
            if evidence not in finding_set:
                continue
            evidence_texts[evidence][text] += 1
            if value in value_set.get(evidence, set()):
                value_texts[(evidence, value)][text] += 1

    missing = sorted(finding_set - set(evidence_texts))
    if missing:
        raise ValueError(f"Train cases cannot render {len(missing)} finding labels: {missing}")

    findings = {}
    for evidence in finding_labels:
        phrase, count = canonical_phrase(evidence_texts[evidence])
        findings[evidence] = {
            "text": phrase,
            "train_count": sum(evidence_texts[evidence].values()),
            "canonical_count": count,
        }

    values = {}
    for evidence, candidates in values_by_evidence.items():
        for value in candidates:
            counts = value_texts.get((evidence, value), Counter())
            if not counts:
                raise ValueError(
                    f"Train cases cannot render probe value {evidence!r}/{value!r}"
                )
            phrase, count = canonical_phrase(counts)
            values[f"{evidence}\0{value}"] = {
                "text": phrase,
                "train_count": sum(counts.values()),
                "canonical_count": count,
            }
    return {"findings": findings, "values": values}


def freeze_protocol(args: argparse.Namespace) -> None:
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    layer = int(artifact["layer"])
    if layer != args.expected_layer:
        raise ValueError(f"Expected HS{args.expected_layer}, got HS{layer}")
    train_rows = list(read_jsonl(args.train_cases))
    if not train_rows:
        raise ValueError("No train cases")
    labels = [str(value) for value in artifact["finding_labels"]]
    values_by_evidence = {
        str(evidence): [str(value) for value in values]
        for evidence, values in artifact["values_by_evidence"].items()
    }
    lexicon = build_lexicon(train_rows, labels, values_by_evidence)
    protocol = {
        "schema_version": 1,
        "method": "probe_guided_structured_reader",
        "method_class": "structured monitor; not open-ended NLA",
        "layer": layer,
        "finding_threshold": float(artifact["finding_threshold"]),
        "artifact": str(args.artifact),
        "artifact_sha256": sha256_file(args.artifact),
        "train_cases": str(args.train_cases),
        "train_cases_sha256": sha256_file(args.train_cases),
        "finding_labels": labels,
        "values_by_evidence": values_by_evidence,
        "lexicon": lexicon,
        "selection": "all finding probabilities >= frozen probe threshold",
        "ordering": "descending finding probability, then evidence_id",
        "rendering": "train-only modal exact cue phrase; lexical tie broken ascending",
        "prompt_text_used_for_prediction": False,
        "validation_or_test_rows_read": False,
        "locked_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[freeze] labels={len(labels)} values={len(lexicon['values'])}")
    print(f"[protocol] {args.output}")


def load_rows(path: Path, path_maps: list[tuple[str, str]]) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No manifest rows in {path}")
    seen = set()
    for row in rows:
        identifier = str(row.get("id") or "")
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate row id: {identifier!r}")
        seen.add(identifier)
        if str(row.get("position_family") or "P0") != "P0":
            raise ValueError(f"Non-P0 row: {identifier}")
        row["_activation_path"] = str(mapped_path(row.get("activation_path"), path_maps))
    return rows


def render_claims(
    row: dict[str, Any],
    finding_probabilities: torch.Tensor,
    value_logits: torch.Tensor,
    protocol: dict[str, Any],
    slices: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], str]:
    labels = protocol["finding_labels"]
    threshold = float(protocol["finding_threshold"])
    values_by_evidence = protocol["values_by_evidence"]
    lexicon = protocol["lexicon"]
    claims = []
    for column, evidence in enumerate(labels):
        probability = float(finding_probabilities[column])
        if probability < threshold:
            continue
        value_id = None
        value_probability = None
        value_key = None
        if evidence in slices:
            start, end = slices[evidence]
            conditional = value_logits[start:end].softmax(dim=0)
            offset = int(conditional.argmax())
            value_id = values_by_evidence[evidence][offset]
            value_probability = float(conditional[offset])
            candidate = f"{evidence}\0{value_id}"
            if candidate in lexicon["values"]:
                value_key = candidate
        entry = lexicon["values"].get(value_key) if value_key else None
        if entry is None:
            entry = lexicon["findings"][evidence]
        claims.append(
            {
                "evidence_id": evidence,
                "finding_probability": probability,
                "value_id": value_id,
                "value_probability": value_probability,
                "text": entry["text"],
                "rendering": "value" if value_key else "finding",
            }
        )
    claims.sort(key=lambda item: (-item["finding_probability"], item["evidence_id"]))
    bullets = "\n".join(f"- {item['text']}" for item in claims)
    observed = f"<observed>\n{bullets}\n</observed>" if bullets else "<observed>\n</observed>"
    return claims, observed


def selected_set(readout: dict[str, Any]) -> set[str]:
    return {str(item["evidence_id"]) for item in readout["selected_claims"]}


def micro_f1(predictions: list[set[str]], targets: list[set[str]]) -> float:
    tp = sum(len(pred & gold) for pred, gold in zip(predictions, targets, strict=True))
    fp = sum(len(pred - gold) for pred, gold in zip(predictions, targets, strict=True))
    fn = sum(len(gold - pred) for pred, gold in zip(predictions, targets, strict=True))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def hard_shuffle_metrics(
    originals: dict[str, dict[str, Any]],
    readouts: dict[str, dict[str, Any]],
    hard_pairs: Path | None,
    label_set: set[str],
) -> dict[str, Any] | None:
    if hard_pairs is None:
        return None
    own_predictions = []
    own_targets = []
    donor_targets = []
    for pair in read_jsonl(hard_pairs):
        if not pair.get("primary_pair_eligible", True):
            continue
        own = str(pair.get("own_base_id") or "")
        donor = str(pair.get("donor_base_id") or "")
        if own not in readouts or own not in originals or donor not in originals:
            continue
        own_predictions.append(selected_set(readouts[own]))
        own_targets.append(set(map(str, originals[own].get("cue_evidence_ids") or [])) & label_set)
        donor_targets.append(
            set(map(str, originals[donor].get("cue_evidence_ids") or []))
            & label_set
        )
    if not own_predictions:
        raise ValueError("No hard-shuffle pair joined the manifest")
    own = micro_f1(own_predictions, own_targets)
    shuffled = micro_f1(own_predictions, donor_targets)
    return {
        "pairs": len(own_predictions),
        "own_f1": own,
        "shuffled_f1": shuffled,
        "gap": own - shuffled,
    }


def value_prediction(readout: dict[str, Any], evidence: str) -> str | None:
    for claim in readout["selected_claims"]:
        if claim["evidence_id"] == evidence:
            return claim.get("value_id")
    return None


def evaluate_readouts(
    rows: list[dict[str, Any]],
    outputs: list[dict[str, Any]],
    protocol: dict[str, Any],
    hard_pairs: Path | None,
) -> dict[str, Any]:
    labels = set(map(str, protocol["finding_labels"]))
    values = {
        key: set(map(str, items))
        for key, items in protocol["values_by_evidence"].items()
    }
    by_row = {str(row["id"]): output for row, output in zip(rows, outputs, strict=True)}
    originals = {
        base_id(row): row
        for row in rows
        if str(row.get("variant") or "original") == "original"
    }
    original_outputs = {
        identifier: by_row[str(row["id"])] for identifier, row in originals.items()
    }
    predictions = [selected_set(original_outputs[key]) for key in originals]
    targets = [
        set(map(str, originals[key].get("cue_evidence_ids") or [])) & labels
        for key in originals
    ]

    value_total = value_correct = value_emitted = 0
    for identifier, row in originals.items():
        for evidence, value, _ in aligned_cues(row):
            if value not in values.get(evidence, set()):
                continue
            value_total += 1
            predicted = value_prediction(original_outputs[identifier], evidence)
            value_emitted += predicted is not None
            value_correct += predicted == value

    deletion_total = deletion_original_hit = deletion_phantom = deletion_removed = 0
    retained_total = retained_original_hit = retained_preserved = 0
    edit_total = edit_replacement = edit_old_persistence = edit_clean_switch = edit_clean_n = 0
    for row in rows:
        variant = str(row.get("variant") or "original")
        if variant == "original" or base_id(row) not in original_outputs:
            continue
        original = original_outputs[base_id(row)]
        derived = by_row[str(row["id"])]
        if variant == "cue_deleted":
            changed = str(row.get("cf_original_evidence_id") or "")
            if changed in labels:
                deletion_total += 1
                before = changed in selected_set(original)
                after = changed in selected_set(derived)
                deletion_original_hit += before
                deletion_phantom += after
                deletion_removed += before and not after
            original_cues = (
                set(
                    map(
                        str,
                        originals[base_id(row)].get("cue_evidence_ids") or [],
                    )
                )
                & labels
            )
            derived_cues = set(map(str, row.get("cue_evidence_ids") or [])) & labels
            for evidence in original_cues & derived_cues:
                retained_total += 1
                before = evidence in selected_set(original)
                retained_original_hit += before
                retained_preserved += before and evidence in selected_set(derived)
        elif variant == "value_edited":
            evidence = str(row.get("cf_original_evidence_id") or "")
            old = str(row.get("cf_original_value_id") or "")
            new = str(row.get("cf_replacement_value_id") or "")
            if new not in values.get(evidence, set()) or old not in values.get(evidence, set()):
                continue
            edit_total += 1
            before = value_prediction(original, evidence)
            after = value_prediction(derived, evidence)
            edit_replacement += after == new
            edit_old_persistence += after == old
            if before == old:
                edit_clean_n += 1
                edit_clean_switch += after == new

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    return {
        "original_cases": len(originals),
        "finding": {
            "micro_f1": micro_f1(predictions, targets),
            "mean_claims": (
                float(np.mean([len(value) for value in predictions]))
                if predictions
                else 0.0
            ),
            "hard_shuffle": hard_shuffle_metrics(originals, original_outputs, hard_pairs, labels),
        },
        "value": {
            "eligible_targets": value_total,
            "emission_coverage": ratio(value_emitted, value_total),
            "end_to_end_accuracy": ratio(value_correct, value_total),
            "accuracy_given_emitted": ratio(value_correct, value_emitted),
        },
        "deletion": {
            "eligible": deletion_total,
            "original_hit": ratio(deletion_original_hit, deletion_total),
            "deleted_phantom": ratio(deletion_phantom, deletion_total),
            "removal_success_given_original_hit": ratio(deletion_removed, deletion_original_hit),
        },
        "retained": {
            "eligible": retained_total,
            "original_hit": ratio(retained_original_hit, retained_total),
            "preservation_given_original_hit": ratio(retained_preserved, retained_original_hit),
        },
        "value_edit": {
            "eligible": edit_total,
            "replacement_hit": ratio(edit_replacement, edit_total),
            "old_value_persistence": ratio(edit_old_persistence, edit_total),
            "clean_switch_given_original_old": ratio(edit_clean_switch, edit_clean_n),
            "conditional_denominator": edit_clean_n,
        },
    }


def fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "N/A"
    return f"{float(value):.4f}"


def write_summary(path: Path, population: str, result: dict[str, Any]) -> None:
    finding = result["finding"]
    shuffled = finding["hard_shuffle"]
    lines = [
        "# DDXPlus Probe-Guided Structured Reader",
        "",
        "Deterministic structured-monitor baseline; not an open-ended NLA.",
        "Claims are selected from the activation by a frozen probe and rendered "
        "with a train-only lexicon.",
        "",
        f"- population: **{population}**",
        f"- original cases: **{result['original_cases']}**",
        f"- mean emitted claims: **{finding['mean_claims']:.4f}**",
        "- prompt text used to construct predictions: **no**",
        "",
        "| metric | value | denominator/control |",
        "|---|---:|---|",
        f"| finding micro F1 | {finding['micro_f1']:.4f} | original cases |",
    ]
    if shuffled:
        lines.extend(
            [
                "| same-diagnosis shuffled F1 | "
                f"{shuffled['shuffled_f1']:.4f} | n={shuffled['pairs']} |",
                f"| own-shuffled finding gap | {shuffled['gap']:+.4f} | n={shuffled['pairs']} |",
            ]
        )
    lines.extend(
        [
            "| native value end-to-end accuracy | "
            f"{fmt(result['value']['end_to_end_accuracy'])} | "
            f"n={result['value']['eligible_targets']} |",
            "| native value emission coverage | "
            f"{fmt(result['value']['emission_coverage'])} | "
            f"n={result['value']['eligible_targets']} |",
            f"| deletion original hit | {fmt(result['deletion']['original_hit'])} | "
            f"n={result['deletion']['eligible']} |",
            f"| deletion phantom | {fmt(result['deletion']['deleted_phantom'])} | "
            f"n={result['deletion']['eligible']} |",
            "| removal success given original hit | "
            f"{fmt(result['deletion']['removal_success_given_original_hit'])} | "
            "conditional |",
            "| retained finding preservation | "
            f"{fmt(result['retained']['preservation_given_original_hit'])} | "
            f"n={result['retained']['eligible']} |",
            "| value-edit replacement hit | "
            f"{fmt(result['value_edit']['replacement_hit'])} | "
            f"n={result['value_edit']['eligible']} |",
            "| value-edit old persistence | "
            f"{fmt(result['value_edit']['old_value_persistence'])} | "
            f"n={result['value_edit']['eligible']} |",
            f"| clean value switch | {fmt(result['value_edit']['clean_switch_given_original_old'])} | "
            f"n={result['value_edit']['conditional_denominator']} |",
            "",
            "Finding selection is mathematically identical to the frozen probe. "
            "The new diagnostic is whether that selected state can be rendered "
            "without a free-generating AV decoder.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def validate_locked_access(
    population: str,
    confirmation: str | None,
    validation_results: Path | None,
    protocol_sha256: str,
) -> None:
    if population != "locked_test":
        return
    if confirmation != LOCKED_CONFIRMATION:
        raise ValueError(f"Locked test requires --confirmation {LOCKED_CONFIRMATION}")
    if validation_results is None:
        raise ValueError("Locked test requires --validation-results")
    validation = json.loads(validation_results.read_text(encoding="utf-8"))
    if validation.get("population") != "validation":
        raise ValueError("--validation-results is not a validation report")
    if validation.get("protocol_sha256") != protocol_sha256:
        raise ValueError("Locked-test protocol differs from validated protocol")


def evaluate(args: argparse.Namespace) -> None:
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha256 = sha256_file(args.protocol)
    validate_locked_access(
        args.population,
        args.confirmation,
        args.validation_results,
        protocol_sha256,
    )
    if sha256_file(args.artifact) != protocol["artifact_sha256"]:
        raise ValueError("Artifact hash does not match frozen protocol")
    artifact = torch.load(args.artifact, map_location="cpu", weights_only=True)
    if int(artifact["layer"]) != int(protocol["layer"]):
        raise ValueError("Artifact layer does not match protocol")
    rows = load_rows(args.manifest, args.path_maps)
    features = torch.stack(
        [
            torch.load(
                row["_activation_path"], map_location="cpu", weights_only=True
            )
            .flatten()
            .float()
            for row in rows
        ]
    )
    features = (features - artifact["feature_mean"]) / artifact["feature_std"]
    device = torch.device(args.device)
    finding_logits = predict_linear(artifact["finding_state_dict"], features, device)
    value_logits = predict_linear(artifact["value_state_dict"], features, device)
    finding_probabilities = finding_logits.sigmoid()
    slices = normalize_slices(artifact["value_slices"])
    outputs = []
    for index, row in enumerate(rows):
        claims, observed = render_claims(
            row, finding_probabilities[index], value_logits[index], protocol, slices
        )
        outputs.append(
            {
                "id": row["id"],
                "base_id": base_id(row),
                "variant": str(row.get("variant") or "original"),
                "diagnosis_id": row.get("diagnosis_id"),
                "position_family": "P0",
                "layer": int(protocol["layer"]),
                "selected_claims": claims,
                "observed": observed,
                "parsed_observed": True,
                "prompt_text_in_output": False,
            }
        )
    result = evaluate_readouts(rows, outputs, protocol, args.hard_pairs)
    report = {
        "schema_version": 1,
        "method": protocol["method"],
        "method_class": protocol["method_class"],
        "population": args.population,
        "protocol": str(args.protocol),
        "protocol_sha256": protocol_sha256,
        "validation_results": (
            str(args.validation_results) if args.validation_results else None
        ),
        "artifact_sha256": sha256_file(args.artifact),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "locked_test_read": args.population == "locked_test",
        "metrics": result,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "readouts.jsonl", outputs)
    (args.out_dir / "results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(args.out_dir / "summary.md", args.population, result)
    print(f"[readouts] {len(outputs)} -> {args.out_dir / 'readouts.jsonl'}")
    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--artifact", required=True, type=Path)
    freeze.add_argument("--train-cases", required=True, type=Path)
    freeze.add_argument("--output", required=True, type=Path)
    freeze.add_argument("--expected-layer", type=int, default=24)

    run = subparsers.add_parser("evaluate")
    run.add_argument("--protocol", required=True, type=Path)
    run.add_argument("--artifact", required=True, type=Path)
    run.add_argument("--manifest", required=True, type=Path)
    run.add_argument("--hard-pairs", type=Path)
    run.add_argument("--out-dir", required=True, type=Path)
    run.add_argument("--population", choices=["validation", "locked_test"], required=True)
    run.add_argument("--confirmation")
    run.add_argument("--validation-results", type=Path)
    run.add_argument("--path-map", action="append", default=[])
    run.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    if args.command == "freeze":
        freeze_protocol(args)
    else:
        args.path_maps = parse_path_maps(args.path_map)
        evaluate(args)


if __name__ == "__main__":
    main()
