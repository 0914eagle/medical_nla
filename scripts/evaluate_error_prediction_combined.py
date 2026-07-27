"""Cross-validated combined error-prediction models.

This compares source confidence features against the gold-free Medical-NLA
disagreement signal, then tests whether combining them improves prediction of
source-model errors.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.evaluate_error_prediction import as_bool, as_float, auroc, average_precision
from src.jsonl import read_jsonl, write_jsonl


SOURCE_CONFIDENCE_FEATURES = [
    ("source_low_top1_prob", "source_top1_prob", -1.0),
    ("source_low_top1_top2_prob_margin", "source_top1_top2_prob_margin", -1.0),
    ("source_low_top1_top2_margin", "source_top1_top2_margin", -1.0),
    ("source_entropy", "source_candidate_entropy_norm", 1.0),
]


def nla_disagree_feature(row: dict[str, Any]) -> float | None:
    agree = as_bool(row.get("source_nla_answer_agree"))
    if agree is None:
        return None
    return 0.0 if agree else 1.0


def source_feature_values(row: dict[str, Any]) -> list[float] | None:
    values = []
    for _, field, direction in SOURCE_CONFIDENCE_FEATURES:
        value = as_float(row.get(field))
        if value is None:
            return None
        values.append(direction * value)
    return values


def collect_dataset(rows: list[dict[str, Any]], feature_set: str) -> tuple[list[str], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    names: list[str]
    if feature_set == "source_confidence":
        names = [name for name, _, _ in SOURCE_CONFIDENCE_FEATURES]
    elif feature_set == "nla_disagree":
        names = ["source_nla_disagree"]
    elif feature_set == "combined":
        names = [name for name, _, _ in SOURCE_CONFIDENCE_FEATURES] + ["source_nla_disagree"]
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    kept_rows = []
    x_rows = []
    y = []
    for row in rows:
        label = as_bool(row.get("is_error"))
        if label is None:
            continue
        source_values = source_feature_values(row)
        disagree = nla_disagree_feature(row)
        if feature_set == "source_confidence":
            if source_values is None:
                continue
            features = source_values
        elif feature_set == "nla_disagree":
            if disagree is None:
                continue
            features = [disagree]
        else:
            if source_values is None or disagree is None:
                continue
            features = source_values + [disagree]
        kept_rows.append(row)
        x_rows.append(features)
        y.append(1.0 if label else 0.0)
    return names, np.asarray(x_rows, dtype=np.float64), np.asarray(y, dtype=np.float64), kept_rows


def stratified_folds(y: np.ndarray, n_folds: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    positives = [idx for idx, value in enumerate(y) if value >= 0.5]
    negatives = [idx for idx, value in enumerate(y) if value < 0.5]
    rng.shuffle(positives)
    rng.shuffle(negatives)
    folds = [[] for _ in range(n_folds)]
    for idx, item in enumerate(positives):
        folds[idx % n_folds].append(item)
    for idx, item in enumerate(negatives):
        folds[idx % n_folds].append(item)
    return folds


def diagnosis_folds(rows: list[dict[str, Any]], n_folds: int, seed: int) -> list[list[int]]:
    by_dx: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        by_dx.setdefault(str(row.get("gold_diagnosis_id") or "-"), []).append(idx)
    diagnoses = sorted(by_dx)
    random.Random(seed).shuffle(diagnoses)
    folds = [[] for _ in range(n_folds)]
    for idx, diagnosis in enumerate(diagnoses):
        folds[idx % n_folds].extend(by_dx[diagnosis])
    return folds


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean_vec = x_train.mean(axis=0)
    std_vec = x_train.std(axis=0)
    std_vec[std_vec < 1e-8] = 1.0
    return (x_train - mean_vec) / std_vec, (x_test - mean_vec) / std_vec


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-values))


def fit_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    lr: float,
    l2: float,
) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_train.shape[0], 1)), x_train], axis=1)
    weights = np.zeros(x_aug.shape[1], dtype=np.float64)
    for _ in range(epochs):
        probs = sigmoid(x_aug @ weights)
        grad = (x_aug.T @ (probs - y_train)) / max(len(y_train), 1)
        grad[1:] += l2 * weights[1:]
        weights -= lr * grad
    return weights


def predict_logistic(x_test: np.ndarray, weights: np.ndarray) -> np.ndarray:
    x_aug = np.concatenate([np.ones((x_test.shape[0], 1)), x_test], axis=1)
    return sigmoid(x_aug @ weights)


def cross_validated_scores(
    *,
    x: np.ndarray,
    y: np.ndarray,
    rows: list[dict[str, Any]],
    n_folds: int,
    split: str,
    seed: int,
    epochs: int,
    lr: float,
    l2: float,
) -> np.ndarray:
    if split == "diagnosis":
        folds = diagnosis_folds(rows, n_folds, seed)
    else:
        folds = stratified_folds(y, n_folds, seed)
    scores = np.zeros(len(y), dtype=np.float64)
    all_indices = set(range(len(y)))
    for test_indices in folds:
        test_set = set(test_indices)
        train_indices = sorted(all_indices - test_set)
        test_indices = sorted(test_indices)
        x_train, x_test = x[train_indices], x[test_indices]
        y_train = y[train_indices]
        x_train, x_test = standardize_train_test(x_train, x_test)
        weights = fit_logistic(x_train, y_train, epochs=epochs, lr=lr, l2=l2)
        scores[test_indices] = predict_logistic(x_test, weights)
    return scores


def write_summary(path: Path, result_rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("# Combined Error Prediction Evaluation\n\n")
        f.write("Cross-validated logistic models; higher score predicts source-model error.\n\n")
        f.write("| model | n | split | auroc | average_precision | features |\n")
        f.write("|---|---:|---|---:|---:|---|\n")
        for row in result_rows:
            features = ", ".join(row["feature_names"])
            auroc_text = "-" if row["auroc"] is None else f"{row['auroc']:.4f}"
            ap_text = "-" if row["average_precision"] is None else f"{row['average_precision']:.4f}"
            f.write(
                f"| {row['model']} | {row['n']} | {row['split']} | "
                f"{auroc_text} | {ap_text} | {features} |\n"
            )
        by_name = {row["model"]: row for row in result_rows}
        if (
            "source_confidence" in by_name
            and "combined" in by_name
            and by_name["combined"]["auroc"] is not None
            and by_name["source_confidence"]["auroc"] is not None
        ):
            delta = by_name["combined"]["auroc"] - by_name["source_confidence"]["auroc"]
            f.write(f"\n- combined_minus_source_confidence_auroc: {delta:+.4f}\n")
        if (
            "nla_disagree" in by_name
            and "combined" in by_name
            and by_name["combined"]["auroc"] is not None
            and by_name["nla_disagree"]["auroc"] is not None
        ):
            delta = by_name["combined"]["auroc"] - by_name["nla_disagree"]["auroc"]
            f.write(f"- combined_minus_nla_disagree_auroc: {delta:+.4f}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-md", required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--split", choices=["stratified", "diagnosis"], default="stratified")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--l2", type=float, default=1e-3)
    args = parser.parse_args()

    rows = list(read_jsonl(args.input))
    result_rows = []
    for feature_set in ("source_confidence", "nla_disagree", "combined"):
        names, x, y, kept_rows = collect_dataset(rows, feature_set)
        if len(y) == 0:
            result_rows.append(
                {
                    "model": feature_set,
                    "n": 0,
                    "split": args.split,
                    "auroc": None,
                    "average_precision": None,
                    "feature_names": names,
                }
            )
            continue
        if feature_set == "nla_disagree":
            scores = x[:, 0]
        else:
            scores = cross_validated_scores(
                x=x,
                y=y,
                rows=kept_rows,
                n_folds=args.folds,
                split=args.split,
                seed=args.seed,
                epochs=args.epochs,
                lr=args.lr,
                l2=args.l2,
            )
        labels = [bool(value) for value in y]
        result_rows.append(
            {
                "model": feature_set,
                "n": len(labels),
                "split": args.split,
                "folds": args.folds,
                "seed": args.seed,
                "auroc": auroc(labels, list(scores)),
                "average_precision": average_precision(labels, list(scores)),
                "feature_names": names,
            }
        )

    output_path = Path(args.output_jsonl)
    summary_path = Path(args.summary_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_path, result_rows)
    write_summary(summary_path, result_rows)
    print(f"[done] wrote {output_path}")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
