"""Appendix Figure A1 -- AV layer sweeps at cue and final-prompt positions.

The two sweeps are deliberately drawn as separate panels. They use the same
lexical recall family, but not the same training recipe or held-out axis:

* cue-token: per-cue reader, held-out cue strings (438 rows per layer);
* final-prompt-token: cue-first reader, held-out diagnoses (727 seen and 800
  held-out rows per layer).

A 2x3 heatmap invited an invalid vertical comparison that changed position,
recipe, and held-out definition at once. Separate panels preserve the useful
within-sweep comparisons without implying a controlled position ablation.

Values are transcribed from the frozen 2026-08-17 sweep reports. A JSON file
can override them after a rerun:

    {
      "cue_heldout": [0.510, 0.658, 0.589],
      "final_seen": [0.3597, 0.6839, 0.6251],
      "final_heldout": [0.1883, 0.2490, 0.1876]
    }
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = ["L16", "L24", "L32"]
DEFAULT = {
    "cue_heldout": [0.510, 0.658, 0.589],
    "final_seen": [0.3597, 0.6839, 0.6251],
    "final_heldout": [0.1883, 0.2490, 0.1876],
}


def annotate(ax, xs: list[int], ys: list[float], *, dy: float = 0.025) -> None:
    for x, value in zip(xs, ys):
        ax.text(x, value + dy, f"{value:.3f}", ha="center", fontsize=7)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--values", help="JSON with cue_heldout/final_seen/final_heldout.")
    args = parser.parse_args()

    values = DEFAULT
    if args.values:
        values = json.loads(Path(args.values).read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = list(range(len(LAYERS)))
    fig, (ax_cue, ax_final) = plt.subplots(1, 2, figsize=(7.2, 2.7), sharey=True)

    cue = values["cue_heldout"]
    ax_cue.plot(xs, cue, "-o", color="black", lw=1.5, ms=5)
    annotate(ax_cue, xs, cue)
    ax_cue.set_title("(a) Cue-token reader\nheld-out cue strings", fontsize=9)
    ax_cue.set_ylabel("Mean lexical cue recall", fontsize=8)
    ax_cue.text(
        0.02,
        0.04,
        "one readout per cue; n=438 per layer",
        transform=ax_cue.transAxes,
        fontsize=6.5,
        color="0.25",
    )

    seen = values["final_seen"]
    heldout = values["final_heldout"]
    ax_final.plot(xs, seen, "-o", color="black", lw=1.5, ms=5, label="seen diagnoses")
    ax_final.plot(
        xs,
        heldout,
        "--s",
        color="0.35",
        lw=1.3,
        ms=4.5,
        label="held-out diagnoses",
    )
    annotate(ax_final, xs, seen)
    annotate(ax_final, xs, heldout, dy=-0.055)
    ax_final.set_title("(b) Final-prompt-token reader\ndiagnosis-held-out split", fontsize=9)
    ax_final.legend(frameon=False, fontsize=7, loc="upper left")
    ax_final.text(
        0.02,
        0.04,
        "cue-first readout; n=727 seen / 800 held-out",
        transform=ax_final.transAxes,
        fontsize=6.5,
        color="0.25",
    )

    for ax in (ax_cue, ax_final):
        ax.set_xticks(xs, LAYERS, fontsize=8)
        ax.set_xlabel("Layer", fontsize=8)
        ax.set_ylim(0, 0.82)
        ax.tick_params(labelsize=7)
        ax.grid(axis="y", color="0.88", lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
