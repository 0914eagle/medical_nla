"""Apply frozen DiReCT linear probes once to the frozen test-seen population."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.train_direct_linear_probe import balanced_weights, label_value, metrics
from src.jsonl import read_jsonl, write_jsonl


LOCKED_CONFIRMATION = "I_ACCEPT_DIRECT_LOCKED_PROBE_EVALUATION"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "").strip()


def remap_path(path: str, mappings: list[tuple[str, str]]) -> Path:
    value = path
    for source, destination in mappings:
        if value.startswith(source):
            value = destination + value[len(source) :]
            break
    return Path(value)


def path_map(value: str) -> tuple[str, str]:
    source, separator, destination = value.partition("=")
    if not separator or not source or not destination:
        raise argparse.ArgumentTypeError("expected OLD=NEW")
    return source, destination


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1 - fraction) + ordered[high] * fraction


def cluster_ci(
    rows: list[dict[str, Any]], values: list[float], *, replicates: int, seed: int
) -> list[float]:
    by_group: dict[str, list[float]] = defaultdict(list)
    for row, value in zip(rows, values, strict=True):
        by_group[str(row.get("patient_group") or case_id(row))].append(value)
    groups = sorted(by_group)
    rng = random.Random(seed)
    estimates = []
    for _ in range(replicates):
        sampled = [rng.choice(groups) for _ in groups]
        sample = [value for group in sampled for value in by_group[group]]
        estimates.append(sum(sample) / len(sample))
    return [percentile(estimates, 0.025), percentile(estimates, 0.975)]


def load_protocol(path: Path) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "shuffle_seed",
        "control_init_seed",
        "bootstrap_seed",
        "bootstrap_replicates",
        "shuffle_unit",
    }
    missing = required - protocol.keys()
    if missing:
        raise ValueError(f"Control protocol misses: {sorted(missing)}")
    if protocol["shuffle_unit"] != "patient_group":
        raise ValueError("Only patient_group label shuffle is supported")
    if int(protocol["bootstrap_replicates"]) < 100:
        raise ValueError("bootstrap_replicates must be at least 100")
    return protocol


def shuffled_targets(
    rows: list[dict[str, Any]], labels: torch.Tensor, *, seed: int
) -> torch.Tensor:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[str(row.get("patient_group") or case_id(row))].append(index)
    ordered = sorted(groups)
    if len(ordered) < 2:
        raise ValueError("Label-shuffle control needs at least two patient groups")
    cycle = ordered.copy()
    random.Random(seed).shuffle(cycle)
    donor_by_group = {
        group: cycle[(index + 1) % len(cycle)] for index, group in enumerate(cycle)
    }
    output = labels.clone()
    for target_group in ordered:
        donor_group = donor_by_group[target_group]
        target_indices = groups[target_group]
        donor_indices = groups[donor_group]
        donor_values = [int(labels[index]) for index in donor_indices]
        for offset, target_index in enumerate(target_indices):
            output[target_index] = donor_values[offset % len(donor_values)]
    return output


def load_vectors(
    rows: list[dict[str, Any]], mappings: list[tuple[str, str]]
) -> torch.Tensor:
    vectors = []
    for row in rows:
        if str(row.get("position_family") or "") != "P0":
            raise ValueError(f"Non-P0 row: {case_id(row)}")
        activation = remap_path(str(row.get("activation_path") or ""), mappings)
        if not activation.is_file():
            raise FileNotFoundError(activation)
        vector = torch.load(activation, map_location="cpu", weights_only=True)
        vectors.append(vector.flatten().float())
    return torch.stack(vectors)


def train_shuffled_control(
    artifact: dict[str, Any],
    train_features: torch.Tensor,
    shuffled_labels: torch.Tensor,
    *,
    device: torch.device,
    seed: int,
) -> torch.nn.Module:
    selected = artifact["selected"]
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(train_features.shape[1], len(artifact["classes"])).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(selected["learning_rate"]),
        weight_decay=float(selected["weight_decay"]),
    )
    weights = None
    if bool(selected["class_balanced"]):
        weights = balanced_weights(shuffled_labels, len(artifact["classes"])).to(device)
    x = train_features.to(device)
    y = shuffled_labels.to(device)
    for _ in range(int(selected["best_epoch"])):
        model.train()
        loss = torch.nn.functional.cross_entropy(model(x), y, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def evaluate_artifact(
    artifact_path: Path,
    train_manifest_path: Path,
    manifest_path: Path,
    mappings: list[tuple[str, str]],
    protocol: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    layer = int(artifact["layer"])
    label_field = str(artifact["label_field"])
    if layer != 24:
        raise ValueError(f"Primary locked probe must be HS24, got HS{layer}")
    rows = list(read_jsonl(manifest_path))
    if len(rows) != 72:
        raise ValueError(f"test_seen manifest has {len(rows)} rows; expected 72")
    ids = [case_id(row) for row in rows]
    if not all(ids) or len(ids) != len(set(ids)):
        raise ValueError("Missing or duplicate test_seen IDs")
    classes = [str(value) for value in artifact["classes"]]
    class_to_index = {label: index for index, label in enumerate(classes)}
    unseen = sorted({label_value(row, label_field) for row in rows} - class_to_index.keys())
    if unseen:
        raise ValueError(f"test_seen contains unseen {label_field} labels: {unseen}")
    features = load_vectors(rows, mappings)
    features = (features - artifact["feature_mean"]) / artifact["feature_std"]
    model = torch.nn.Linear(features.shape[1], len(classes)).to(device)
    model.load_state_dict(artifact["state_dict"])
    model.eval()
    with torch.inference_mode():
        logits = model(features.to(device)).cpu()
    labels = torch.tensor(
        [class_to_index[label_value(row, label_field)] for row in rows], dtype=torch.long
    )
    primary = metrics(logits, labels)
    train_rows = list(read_jsonl(train_manifest_path))
    if len(train_rows) != 266:
        raise ValueError(f"Train manifest has {len(train_rows)} rows; expected 266")
    train_labels = torch.tensor(
        [class_to_index[label_value(row, label_field)] for row in train_rows],
        dtype=torch.long,
    )
    shuffled_train_labels = shuffled_targets(
        train_rows, train_labels, seed=int(protocol["shuffle_seed"])
    )
    train_features = load_vectors(train_rows, mappings)
    train_features = (train_features - artifact["feature_mean"]) / artifact["feature_std"]
    control_model = train_shuffled_control(
        artifact,
        train_features,
        shuffled_train_labels,
        device=device,
        seed=int(protocol["control_init_seed"]),
    )
    with torch.inference_mode():
        control_logits = control_model(features.to(device)).cpu()
    control = metrics(control_logits, labels)
    predicted = logits.argmax(dim=-1)
    control_predicted = control_logits.argmax(dim=-1)
    own_correct = (predicted == labels).float().tolist()
    shuffle_correct = (control_predicted == labels).float().tolist()
    gaps = [own - control for own, control in zip(own_correct, shuffle_correct, strict=True)]
    replicates = int(protocol["bootstrap_replicates"])
    bootstrap_seed = int(protocol["bootstrap_seed"])
    result = {
        "label_field": label_field,
        "layer": layer,
        "classes": len(classes),
        "n": len(rows),
        "patient_groups": len({str(row.get('patient_group') or case_id(row)) for row in rows}),
        "own": primary,
        "label_shuffle": control,
        "own_minus_shuffle": primary["acc1"] - control["acc1"],
        "accuracy_cluster_ci95": cluster_ci(
            rows, own_correct, replicates=replicates, seed=bootstrap_seed
        ),
        "gap_cluster_ci95": cluster_ci(
            rows, gaps, replicates=replicates, seed=bootstrap_seed
        ),
        "artifact": str(artifact_path),
        "artifact_sha256": sha256_file(artifact_path),
        "train_manifest": str(train_manifest_path),
        "train_manifest_sha256": sha256_file(train_manifest_path),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    case_rows = [
        {
            "id": case_id(row),
            "patient_group": str(row.get("patient_group") or case_id(row)),
            "label_field": label_field,
            "gold": classes[int(labels[index])],
            "prediction": classes[int(predicted[index])],
            "label_shuffle_prediction": classes[int(control_predicted[index])],
            "correct": bool(own_correct[index]),
            "shuffle_correct": bool(shuffle_correct[index]),
        }
        for index, row in enumerate(rows)
    ]
    return result, case_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", action="append", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--control-protocol", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--path-map", action="append", default=[], type=path_map)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--confirmation", required=True)
    args = parser.parse_args()
    if args.confirmation != LOCKED_CONFIRMATION:
        raise ValueError(f"Locked probe evaluation requires --confirmation {LOCKED_CONFIRMATION}")
    protocol = load_protocol(args.control_protocol)
    fields = []
    private_rows = []
    for artifact in args.artifact:
        result, rows = evaluate_artifact(
            artifact,
            args.train_manifest,
            args.manifest,
            args.path_map,
            protocol,
            torch.device(args.device),
        )
        fields.append(result)
        private_rows.extend(rows)
    if {item["label_field"] for item in fields} != {"canonical_pdd", "disease_category"}:
        raise ValueError("Both canonical_pdd and disease_category HS24 artifacts are required")
    report = {
        "schema_version": 1,
        "population": "test_seen",
        "selection_performed_on_test": False,
        "locked_test_read": True,
        "control_protocol": str(args.control_protocol),
        "control_protocol_sha256": sha256_file(args.control_protocol),
        "results": sorted(fields, key=lambda item: item["label_field"]),
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_jsonl(args.out_dir / "private_case_scores.jsonl", private_rows)
    lines = [
        "# DiReCT Frozen Test-Seen Linear Probes",
        "",
        "Validation-selected HS24 probes applied once to 72 frozen test-seen rows.",
        "",
        "| target | classes | n | acc1 | cluster 95% CI | shuffled acc1 | gap | "
        "gap cluster 95% CI |",
        "|---|---:|---:|---:|---|---:|---:|---|",
    ]
    for item in report["results"]:
        ci, gap_ci = item["accuracy_cluster_ci95"], item["gap_cluster_ci95"]
        lines.append(
            f"| {item['label_field']} | {item['classes']} | {item['n']} | "
            f"{item['own']['acc1']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | "
            f"{item['label_shuffle']['acc1']:.4f} | {item['own_minus_shuffle']:+.4f} | "
            f"[{gap_ci[0]:+.4f}, {gap_ci[1]:+.4f}] |"
        )
    (args.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] {args.out_dir}")


if __name__ == "__main__":
    main()
