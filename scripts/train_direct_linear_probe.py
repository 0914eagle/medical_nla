"""Fit validation-selected linear probes on frozen DiReCT P0 activations.

Only train and validation manifests are read. Test manifests are intentionally
absent from the interface so layer and regularization choices cannot inspect
the locked evaluation pools.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.jsonl import read_jsonl


def case_id(row: dict[str, Any]) -> str:
    return str(row.get("base_id") or row.get("id") or "")


def label_value(row: dict[str, Any], field: str) -> str:
    if field == "canonical_pdd":
        return str(row.get("canonical_pdd") or row.get("diagnosis_name") or "")
    return str(row.get(field) or "")


def load_split(path: Path, label_field: str) -> list[dict[str, Any]]:
    rows = list(read_jsonl(path))
    if not rows:
        raise ValueError(f"No rows in {path}")
    seen: set[str] = set()
    for row in rows:
        identifier = case_id(row)
        if not identifier or identifier in seen:
            raise ValueError(f"Missing or duplicate case id in {path}: {identifier!r}")
        seen.add(identifier)
        if str(row.get("position_family") or "") != "P0":
            raise ValueError(f"Non-P0 row in primary probe manifest: {row.get('id')}")
        if not label_value(row, label_field):
            raise ValueError(f"Missing {label_field} for {identifier}")
        if not Path(str(row.get("activation_path") or "")).is_file():
            raise FileNotFoundError(str(row.get("activation_path")))
    return rows


def load_matrix(
    rows: list[dict[str, Any]], label_field: str, class_to_index: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    vectors = [
        torch.load(row["activation_path"], map_location="cpu", weights_only=True)
        .flatten()
        .float()
        for row in rows
    ]
    labels = [class_to_index[label_value(row, label_field)] for row in rows]
    return torch.stack(vectors), torch.tensor(labels, dtype=torch.long)


def metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    logits = logits.cpu()
    labels = labels.cpu()
    order = logits.argsort(dim=-1, descending=True)
    ranks = (order == labels[:, None]).nonzero(as_tuple=False)[:, 1] + 1
    predictions = order[:, 0]
    per_class = []
    for value in sorted(set(labels.tolist())):
        mask = labels == value
        per_class.append((predictions[mask] == labels[mask]).float().mean().item())
    return {
        "nll": F.cross_entropy(logits, labels).item(),
        "acc1": (predictions == labels).float().mean().item(),
        "acc5": (order[:, : min(5, logits.shape[1])] == labels[:, None])
        .any(dim=1)
        .float()
        .mean()
        .item(),
        "mrr": (1.0 / ranks.float()).mean().item(),
        "macro_recall": sum(per_class) / len(per_class),
    }


def predict(model: torch.nn.Module, features: torch.Tensor, device: torch.device) -> torch.Tensor:
    model.eval()
    with torch.inference_mode():
        return model(features.to(device)).cpu()


def balanced_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels, minlength=n_classes).float().clamp_min(1)
    weights = labels.numel() / (n_classes * counts)
    return weights / weights.mean()


def train_candidate(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    n_classes: int,
    learning_rate: float,
    weight_decay: float,
    class_balanced: bool,
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = torch.nn.Linear(train_x.shape[1], n_classes).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    weights = balanced_weights(train_y, n_classes).to(device) if class_balanced else None
    x_device = train_x.to(device)
    y_device = train_y.to(device)
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float] | None = None
    best_epoch = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(x_device)
        loss = F.cross_entropy(logits, y_device, weight=weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        val_metrics = metrics(predict(model, val_x, device), val_y)
        if best_metrics is None or val_metrics["nll"] < best_metrics["nll"] - 1e-8:
            best_metrics = val_metrics
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is None or best_metrics is None:
        raise RuntimeError("Probe training produced no checkpoint")
    return best_state, best_epoch, best_metrics


def patient_disjoint(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> bool:
    train_groups = {str(row.get("patient_group") or case_id(row)) for row in train_rows}
    val_groups = {str(row.get("patient_group") or case_id(row)) for row in val_rows}
    return not bool(train_groups & val_groups)


def fit_layer_label(
    train_path: Path,
    val_path: Path,
    *,
    layer: int,
    label_field: str,
    learning_rates: list[float],
    weight_decays: list[float],
    class_balance_options: list[bool],
    epochs: int,
    patience: int,
    device: torch.device,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    train_rows = load_split(train_path, label_field)
    val_rows = load_split(val_path, label_field)
    if not patient_disjoint(train_rows, val_rows):
        raise ValueError(f"Patient overlap between train and validation at HS{layer}")

    classes = sorted({label_value(row, label_field) for row in train_rows})
    class_to_index = {label: index for index, label in enumerate(classes)}
    unseen_val = sorted(
        {label_value(row, label_field) for row in val_rows} - class_to_index.keys()
    )
    if unseen_val:
        raise ValueError(f"Validation contains unseen {label_field} labels: {unseen_val}")

    train_x, train_y = load_matrix(train_rows, label_field, class_to_index)
    val_x, val_y = load_matrix(val_rows, label_field, class_to_index)
    feature_mean = train_x.mean(dim=0, keepdim=True)
    feature_std = train_x.std(dim=0, keepdim=True).clamp_min(1e-6)
    train_x = (train_x - feature_mean) / feature_std
    val_x = (val_x - feature_mean) / feature_std

    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    selected_state: dict[str, torch.Tensor] | None = None
    for learning_rate in learning_rates:
        for weight_decay in weight_decays:
            for class_balanced in class_balance_options:
                state, best_epoch, val_metrics = train_candidate(
                    train_x,
                    train_y,
                    val_x,
                    val_y,
                    n_classes=len(classes),
                    learning_rate=learning_rate,
                    weight_decay=weight_decay,
                    class_balanced=class_balanced,
                    epochs=epochs,
                    patience=patience,
                    device=device,
                    seed=seed,
                )
                candidate = {
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "class_balanced": class_balanced,
                    "best_epoch": best_epoch,
                    "validation": val_metrics,
                }
                candidates.append(candidate)
                if selected is None or (
                    val_metrics["nll"], -val_metrics["acc1"], weight_decay
                ) < (
                    selected["validation"]["nll"],
                    -selected["validation"]["acc1"],
                    selected["weight_decay"],
                ):
                    selected = candidate
                    selected_state = state

    if selected is None or selected_state is None:
        raise RuntimeError("No probe candidate was selected")
    majority_label = Counter(train_y.tolist()).most_common(1)[0][0]
    majority_acc = (val_y == majority_label).float().mean().item()
    artifact_path = out_dir / f"{label_field}_hs{layer}.pt"
    torch.save(
        {
            "state_dict": selected_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "classes": classes,
            "label_field": label_field,
            "layer": layer,
            "selected": selected,
            "train_manifest": str(train_path),
            "val_manifest": str(val_path),
        },
        artifact_path,
    )
    return {
        "layer": layer,
        "label_field": label_field,
        "n_classes": len(classes),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "patient_disjoint": True,
        "majority_val_acc": majority_acc,
        "selected": selected,
        "candidates": candidates,
        "artifact": str(artifact_path),
    }


def write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# DiReCT P0 Linear Probe Validation",
        "",
        "Train/validation only. Locked test manifests were not read.",
        "",
        "| label | HS | classes | train | val | majority | acc1 | acc5 | MRR | macro recall | val NLL |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        val = result["selected"]["validation"]
        lines.append(
            f"| {result['label_field']} | {result['layer']} | {result['n_classes']} | "
            f"{result['n_train']} | {result['n_val']} | {result['majority_val_acc']:.4f} | "
            f"{val['acc1']:.4f} | {val['acc5']:.4f} | {val['mrr']:.4f} | "
            f"{val['macro_recall']:.4f} | {val['nll']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Hyperparameters and stopping epoch are selected by validation NLL; accuracy is reported, not optimized separately.",
            "HS32 remains the primary NLA index because the public AV/AR checkpoint was trained for HS32. Other probe layers are sensitivity analyses.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--activation-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--layers", nargs="+", type=int, default=[16, 24, 32])
    parser.add_argument(
        "--label-fields", nargs="+", default=["canonical_pdd", "disease_category"]
    )
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[3e-4, 1e-3])
    parser.add_argument(
        "--weight-decays", nargs="+", type=float, default=[0.0, 1e-4, 1e-3, 1e-2]
    )
    parser.add_argument(
        "--class-balance", choices=("both", "off", "on"), default="both"
    )
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    balance_options = {
        "both": [False, True],
        "off": [False],
        "on": [True],
    }[args.class_balance]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for label_field in args.label_fields:
        for layer in args.layers:
            base = args.activation_root / f"layer{layer}" / "last_token"
            print(f"[probe] label={label_field} HS{layer}", flush=True)
            result = fit_layer_label(
                base / "manifest_train.jsonl",
                base / "manifest_val_seen.jsonl",
                layer=layer,
                label_field=label_field,
                learning_rates=args.learning_rates,
                weight_decays=args.weight_decays,
                class_balance_options=balance_options,
                epochs=args.epochs,
                patience=args.patience,
                device=device,
                seed=args.seed,
                out_dir=args.out_dir,
            )
            results.append(result)
            val = result["selected"]["validation"]
            print(
                f"[probe] label={label_field} HS{layer} "
                f"acc1={val['acc1']:.4f} nll={val['nll']:.4f}",
                flush=True,
            )

    (args.out_dir / "validation_results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.out_dir / "summary.md", results)
    print(f"[done] {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
