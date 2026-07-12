from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.reconstruction_scoring import (
    confab_regex,
    cjk_frac_for_row,
    high_norm_flag,
    infer_domain,
    load_activation,
    load_nla_critic_class,
    read_jsonl,
    resolve_ar_checkpoint,
    summarize_scored_rows,
    torch_dtype,
    validate_mse,
    write_jsonl,
)


def plot_rows(rows: list[dict], output: str | Path) -> None:
    rng = np.random.default_rng(17)
    colors = {"medical": "#c43c39", "general": "#2b6cb0"}

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    for row in rows:
        x = float(row["recon_mse"])
        y = (1.0 if row["confab_regex"] else 0.0) + float(rng.normal(0.0, 0.025))
        marker = "x" if row["confab_regex"] else "o"
        edge = "black" if row["high_norm_flag"] else colors[row["domain"]]
        linewidth = 1.4 if row["high_norm_flag"] else 0.8
        ax.scatter(
            x,
            y,
            c=colors[row["domain"]],
            marker=marker,
            edgecolors=edge,
            linewidths=linewidth,
            alpha=0.82,
            s=48,
        )

    ax.axvline(2.0, linestyle="--", color="0.35", linewidth=1.0, label="orthogonal MSE=2")
    ax.set_xlabel("recon_mse (lower = better reconstruction)")
    ax.set_ylabel("confab_regex")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["False", "True"])
    ax.set_xlim(left=0)
    ax.set_ylim(-0.2, 1.2)

    from matplotlib.lines import Line2D

    legend_items = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["general"], label="general"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=colors["medical"], label="medical"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", label="non-confab"),
        Line2D([0], [0], marker="x", color="black", label="confab"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", label="high-norm edge"),
        Line2D([0], [0], linestyle="--", color="0.35", label="MSE=2"),
    ]
    ax.legend(handles=legend_items, loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--ar", default="kitft/nla-gemma3-12b-L32-ar")
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--cache-dir", default="/data1/heejae/hf_cache")
    parser.add_argument("--nla-inference-path", default=None)
    parser.add_argument("--mse-equivalent-margin", type=float, default=0.1)
    parser.add_argument(
        "--high-norm-threshold",
        type=float,
        default=12000.0,
        help=(
            "Rows above this activation_norm are marked high_norm_flag and excluded from "
            "summary stats. The original 12000 default is conservative for Qwen-style "
            "runs; Gemma-3 L32 often needs a much higher threshold such as 120000."
        ),
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ar_dir = resolve_ar_checkpoint(args.ar, args.cache_dir)
    critic_cls = load_nla_critic_class(args.nla_inference_path)
    critic = critic_cls(ar_dir, device=args.device, dtype=torch_dtype(args.dtype))

    scored: list[dict] = []
    for input_path in args.inputs:
        domain = infer_domain(input_path)
        for row in read_jsonl(input_path):
            row_id = str(row.get("id", ""))
            explanation = str(row.get("nla_output") or "")
            activation = load_activation(row["activation_path"])
            mse, cos = critic.score(explanation, activation)
            mse = float(mse)
            validate_mse(mse, row_id)
            enriched = dict(row)
            enriched.update(
                {
                    "domain": domain,
                    "recon_mse": mse,
                    "recon_cos": float(cos),
                    "confab_regex": confab_regex(str(row.get("prompt", "")), explanation),
                    "high_norm_flag": high_norm_flag(row, threshold=args.high_norm_threshold),
                    "high_norm_threshold": args.high_norm_threshold,
                    "cjk_frac": cjk_frac_for_row(row),
                }
            )
            scored.append(enriched)
            print(
                f"[score] {domain}/{row_id} mse={mse:.4f} cos={float(cos):.4f} "
                f"confab={enriched['confab_regex']} high_norm={enriched['high_norm_flag']}",
                flush=True,
            )

    write_jsonl(out_dir / "scored.jsonl", scored)
    plot_rows(scored, out_dir / "mse_vs_confab.png")
    (out_dir / "summary.md").write_text(
        summarize_scored_rows(
            scored,
            margin=args.mse_equivalent_margin,
            high_norm_threshold=args.high_norm_threshold,
        ),
        encoding="utf-8",
    )

    shutil.copy2(__file__, out_dir / "score_reconstruction_mse.py")
    print(f"[done] wrote {len(scored)} rows to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
