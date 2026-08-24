"""Figure 2 -- where the internals are readable: layer x position map.

Six cells: {cue token, answer-forming token} x {L16, L24, L32}, unseen-cue
readability. The top row is an inverted U peaking at L24; the bottom row is
flat and low at every depth. The division of labour the paper builds on is
this picture: evidence detail lives at cue tokens mid-stack, and by the
answer position it has folded into a conclusion that only a different
instrument (the class probe, or the conclusion readout) can read.

One scorer for all six cells -- the v2 lexical scorer's mean token recall on
heldout cues -- for two reasons, and the second is the binding one.

The weaker reason: the manually scored A+B rate exists only for the cue row
(438 rows x 3 layers), and a heatmap whose two rows were measured with
different rubrics is not a heatmap.

The binding reason: the 438-row semantic scoring is held open. Its numbers
(.340 / .731 / .557, same shape with a sharper peak) stand, but the paper
does not state who produced them -- that slot waits for an external judge to
re-score the same rows, and no second human pass or self-agreement rate fills
it in the meantime. A figure annotation has no room for that caveat, and an
annotation reading "hand-labeled" would assert exactly the attribution the
decision defers. So the figure carries one scorer and says so; the deferred
numbers live in the text, where the placeholder can be stated.

Values are transcribed constants, not a dump: both sweeps are period-1
results whose analyzers no longer run end to end, and their numbers are
frozen in docs/results/results_2026-08-17_{layer,format_position}_sweep.md.
--values pointing at a JSON of the same shape overrides them if the sweeps
are ever re-run.

    python scripts/make_figure_readout_map.py --output figure2_readout_map.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

LAYERS = ["L16", "L24", "L32"]
# docs/results/results_2026-08-17_layer_sweep.md (heldout mean token recall)
# and results_2026-08-17_format_position_sweep.md (heldout mean cue_recall,
# same v2 lexical scorer).
DEFAULT = {
    "cue token": [0.510, 0.658, 0.589],
    "answer-forming token": [0.188, 0.249, 0.188],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--values", help="JSON {row_label: [L16, L24, L32]} to override.")
    args = parser.parse_args()

    values = DEFAULT
    if args.values:
        values = json.loads(Path(args.values).read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(values)
    grid = [values[r] for r in rows]

    fig, ax = plt.subplots(figsize=(4.0, 2.2))
    im = ax.imshow(grid, cmap="Greys", vmin=0, vmax=1.0, aspect="auto")
    for ri, row in enumerate(grid):
        for ci, v in enumerate(row):
            ax.text(ci, ri, f"{v:.3f}".lstrip("0"), ha="center", va="center",
                    fontsize=9, color="white" if v > 0.55 else "black")
    ax.set_xticks(range(len(LAYERS)), LAYERS, fontsize=8)
    ax.set_yticks(range(len(rows)), rows, fontsize=8)
    ax.set_title("unseen-cue readability (held-out cue strings)", fontsize=8.5)
    fig.colorbar(im, ax=ax, shrink=0.85).ax.tick_params(labelsize=7)
    ax.set_xlabel("layer, mean token recall (one scorer, all cells)", fontsize=7)
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
