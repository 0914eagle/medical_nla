"""Train linear diagnosis probes on DDXPlus activation manifests.

The manifest is expected to come from `scripts/make_ddxplus_probe_dataset.py`
followed by `python -m src.extract_activations`. Probes are trained separately
for each position variant so we can compare where diagnosis information is
linearly readable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


DEFAULT_GROUPS = [
    "single_cue",
    "single_format",
    "multi_cue_1",
    "multi_cue_2",
    "multi_cue_3",
    "multi_cue_all",
    "multi_format",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_cases(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    train_frac: float,
    val_frac: float,
) -> dict[str, str]:
    by_case: dict[str, dict[str, str]] = {}
    for row in rows:
        base_id = str(row["base_id"])
        diagnosis = str(row["diagnosis_id"])
        if base_id in by_case and by_case[base_id]["diagnosis_id"] != diagnosis:
            raise ValueError(f"Case {base_id} has inconsistent diagnosis labels.")
        by_case[base_id] = {"base_id": base_id, "diagnosis_id": diagnosis}

    by_diagnosis: dict[str, list[str]] = defaultdict(list)
    for case in by_case.values():
        by_diagnosis[case["diagnosis_id"]].append(case["base_id"])

    rng = random.Random(seed)
    split_map = {}
    for diagnosis, case_ids in sorted(by_diagnosis.items()):
        case_ids = list(case_ids)
        rng.shuffle(case_ids)
        n = len(case_ids)
        n_train = max(1, int(round(n * train_frac)))
        n_val = max(1, int(round(n * val_frac)))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1
        for case_id in case_ids[:n_train]:
            split_map[case_id] = "train"
        for case_id in case_ids[n_train : n_train + n_val]:
            split_map[case_id] = "val"
        for case_id in case_ids[n_train + n_val :]:
            split_map[case_id] = "test"
    return split_map


def rows_for_group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    if group == "multi_cue_all":
        return [row for row in rows if str(row.get("variant", "")).startswith("multi_cue_")]
    return [row for row in rows if row.get("variant") == group]


def load_matrix(rows: list[dict[str, Any]], class_to_idx: dict[str, int]) -> tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    for row in rows:
        tensor = torch.load(row["activation_path"], map_location="cpu")
        xs.append(tensor.flatten().float())
        ys.append(class_to_idx[str(row["diagnosis_id"])])
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def accuracy(logits: torch.Tensor, y: torch.Tensor, *, k: int = 1) -> float:
    k = min(k, logits.shape[-1])
    pred = logits.topk(k, dim=-1).indices
    return pred.eq(y[:, None]).any(dim=-1).float().mean().item()


def evaluate(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, batch_size: int) -> dict[str, float]:
    model.eval()
    logits = []
    with torch.inference_mode():
        for start in range(0, x.shape[0], batch_size):
            logits.append(model(x[start : start + batch_size]).cpu())
    all_logits = torch.cat(logits, dim=0)
    loss = F.cross_entropy(all_logits, y.cpu()).item()
    return {
        "loss": loss,
        "acc1": accuracy(all_logits, y.cpu(), k=1),
        "acc5": accuracy(all_logits, y.cpu(), k=5),
    }


def train_group(
    *,
    group: str,
    rows: list[dict[str, Any]],
    split_map: dict[str, str],
    class_to_idx: dict[str, int],
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    standardize: bool,
    seed: int,
) -> dict[str, Any]:
    group_rows = rows_for_group(rows, group)
    if not group_rows:
        raise ValueError(f"No rows for group {group}")

    split_rows = {
        split: [row for row in group_rows if split_map[str(row["base_id"])] == split]
        for split in ("train", "val", "test")
    }
    x_train, y_train = load_matrix(split_rows["train"], class_to_idx)
    x_val, y_val = load_matrix(split_rows["val"], class_to_idx)
    x_test, y_test = load_matrix(split_rows["test"], class_to_idx)

    if standardize:
        mean = x_train.mean(dim=0, keepdim=True)
        std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
        x_train = (x_train - mean) / std
        x_val = (x_val - mean) / std
        x_test = (x_test - mean) / std

    generator = torch.Generator()
    generator.manual_seed(seed)
    model = torch.nn.Linear(x_train.shape[1], len(class_to_idx)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val_acc = -1.0
    best_epoch = 0

    x_train_device = x_train.to(device)
    y_train_device = y_train.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(x_train_device.shape[0], generator=generator).to(device)
        for start in range(0, order.numel(), batch_size):
            idx = order[start : start + batch_size]
            logits = model(x_train_device[idx])
            loss = F.cross_entropy(logits, y_train_device[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        val_metrics = evaluate(model, x_val.to(device), y_val, batch_size)
        if val_metrics["acc1"] > best_val_acc:
            best_val_acc = val_metrics["acc1"]
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate(model, x_train.to(device), y_train, batch_size)
    val_metrics = evaluate(model, x_val.to(device), y_val, batch_size)
    test_metrics = evaluate(model, x_test.to(device), y_test, batch_size)

    return {
        "group": group,
        "n_train": int(y_train.numel()),
        "n_val": int(y_val.numel()),
        "n_test": int(y_test.numel()),
        "best_epoch": best_epoch,
        "train": train_metrics,
        "val": val_metrics,
        "test": test_metrics,
    }


def markdown_summary(results: list[dict[str, Any]], *, n_classes: int) -> str:
    lines = [
        "# DDXPlus Linear Probe",
        "",
        f"- classes: {n_classes}",
        f"- chance_acc1: {1 / n_classes:.4f}",
        f"- chance_acc5: {min(5, n_classes) / n_classes:.4f}",
        "",
        "| group | n_train | n_val | n_test | best_epoch | test_acc1 | test_acc5 | val_acc1 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            "| {group} | {n_train} | {n_val} | {n_test} | {best_epoch} | "
            "{test_acc1:.4f} | {test_acc5:.4f} | {val_acc1:.4f} |".format(
                group=result["group"],
                n_train=result["n_train"],
                n_val=result["n_val"],
                n_test=result["n_test"],
                best_epoch=result["best_epoch"],
                test_acc1=result["test"]["acc1"],
                test_acc5=result["test"]["acc5"],
                val_acc1=result["val"]["acc1"],
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--no-standardize", action="store_true")
    args = parser.parse_args()

    rows = read_jsonl(Path(args.manifest))
    rows = [row for row in rows if row.get("diagnosis_id") and row.get("variant")]
    if not rows:
        raise ValueError("No probe rows with diagnosis_id and variant found.")

    classes = sorted({str(row["diagnosis_id"]) for row in rows})
    class_to_idx = {name: idx for idx, name in enumerate(classes)}
    split_map = split_cases(
        rows,
        seed=args.seed,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
    )

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_rows = [
        {"base_id": base_id, "split": split}
        for base_id, split in sorted(split_map.items())
    ]
    write_jsonl(out_dir / "splits.jsonl", split_rows)

    metadata = {
        "manifest": str(Path(args.manifest)),
        "groups": args.groups,
        "classes": classes,
        "class_counts": Counter(str(row["diagnosis_id"]) for row in rows),
        "device": str(device),
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "standardize": not args.no_standardize,
    }
    write_json(out_dir / "metadata.json", metadata)

    results = []
    for group in args.groups:
        print(f"[probe] training {group}", flush=True)
        result = train_group(
            group=group,
            rows=rows,
            split_map=split_map,
            class_to_idx=class_to_idx,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            standardize=not args.no_standardize,
            seed=args.seed,
        )
        results.append(result)
        print(
            f"[probe] {group} test_acc1={result['test']['acc1']:.4f} "
            f"test_acc5={result['test']['acc5']:.4f}",
            flush=True,
        )

    write_json(out_dir / "results.json", results)
    (out_dir / "summary.md").write_text(
        markdown_summary(results, n_classes=len(classes)),
        encoding="utf-8",
    )
    print(f"[done] wrote probe results to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
