"""Render canonical HS16/24/32 probe sensitivity from validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LAYERS = (16, 24, 32)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def direct_values(paths: Path | list[Path]) -> dict[str, list[float]]:
    if isinstance(paths, Path):
        paths = [paths]
    rows = [
        row
        for path in paths
        for row in json.loads(path.read_text(encoding="utf-8"))
    ]
    indexed = {(str(row["label_field"]), int(row["layer"])): row for row in rows}
    result = {}
    for field, label in (
        ("disease_category", "DiReCT category top-1"),
        ("canonical_pdd", "DiReCT PDD top-1"),
    ):
        missing = [layer for layer in LAYERS if (field, layer) not in indexed]
        if missing:
            raise ValueError(f"Direct validation results miss {field} layers {missing}")
        result[label] = [
            float(indexed[(field, layer)]["selected"]["validation"]["acc1"])
            for layer in LAYERS
        ]
    return result


def ddxplus_values(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    indexed = {int(row["layer"]): row for row in payload["results"]}
    missing = [layer for layer in LAYERS if layer not in indexed]
    if missing:
        raise ValueError(f"DDXPlus validation results miss layers {missing}")
    return {
        "DDXPlus finding micro F1": [
            float(indexed[layer]["finding"]["selected_threshold"]["micro_f1"])
            for layer in LAYERS
        ],
        "DDXPlus native-value accuracy": [
            float(indexed[layer]["value"]["validation"]["accuracy"])
            for layer in LAYERS
        ],
    }


def validate(values: dict[str, list[float]]) -> None:
    if len(values) != 4:
        raise ValueError("Expected four sensitivity curves")
    for label, scores in values.items():
        if len(scores) != len(LAYERS) or not all(0 <= value <= 1 for value in scores):
            raise ValueError(f"Invalid scores for {label}: {scores}")


def write_values(
    path: Path, values: dict[str, list[float]], sources: dict[str, Path]
) -> None:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "population": "validation",
        "layers": list(LAYERS),
        "series": values,
        "sources": {
            name: {"path": str(source), "sha256": sha256_file(source)}
            for name, source in sources.items()
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def render(output: Path, values: dict[str, list[float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.15), sharex=True)
    panels = (
        (
            axes[0],
            ("DiReCT category top-1", "DiReCT PDD top-1"),
            "(a) DiReCT diagnostic state",
            "top-1 accuracy",
        ),
        (
            axes[1],
            ("DDXPlus finding micro F1", "DDXPlus native-value accuracy"),
            "(b) DDXPlus clinical state",
            "validation score",
        ),
    )
    styles = (("black", "o", "-"), ("0.45", "s", "--"))
    for axis, labels, title, ylabel in panels:
        for label, (color, marker, linestyle) in zip(labels, styles, strict=True):
            scores = values[label]
            axis.plot(
                LAYERS,
                scores,
                color=color,
                marker=marker,
                linestyle=linestyle,
                linewidth=1.5,
                markersize=5,
                markerfacecolor="white" if marker == "s" else color,
                label=label.split(" ", 1)[1],
            )
            for layer, score in zip(LAYERS, scores, strict=True):
                axis.text(layer, score + 0.025, f"{score:.3f}", ha="center", fontsize=6.5)
        axis.axvline(24, color="0.82", linewidth=0.8, linestyle=":")
        axis.set_xticks(LAYERS, [f"HS{layer}" for layer in LAYERS])
        axis.set_ylim(0.3 if axis is axes[0] else 0.64, 1.03)
        axis.set_xlabel("backbone layer", fontsize=8)
        axis.set_ylabel(ylabel, fontsize=8)
        axis.tick_params(labelsize=7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_title(title, fontsize=8.5)
        axis.legend(frameon=False, fontsize=6.6, loc="lower center")
    fig.subplots_adjust(left=0.09, right=0.99, top=0.88, bottom=0.17, wspace=0.3)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-results", action="append", required=True, type=Path)
    parser.add_argument("--ddxplus-results", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--values-json", required=True, type=Path)
    args = parser.parse_args()
    values = {**direct_values(args.direct_results), **ddxplus_values(args.ddxplus_results)}
    validate(values)
    write_values(
        args.values_json,
        values,
        {
            **{
                f"direct_{index}": path
                for index, path in enumerate(args.direct_results, start=1)
            },
            "ddxplus": args.ddxplus_results,
        },
    )
    render(args.output, values)
    print(f"[figure] {args.output}")
    print(f"[values] {args.values_json}")


if __name__ == "__main__":
    main()
