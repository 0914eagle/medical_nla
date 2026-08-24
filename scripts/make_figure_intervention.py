"""Figure 3 -- the four-arm intervention, drawn from analyze_hint_effect --dump.

Grouped bars: one cluster per corpus (or per run -- a CoT run is just another
dump), four bars per cluster in the arm order none / neutral / wrong /
correct. The wrong bar is the black one; the figure's message is that one
bar, and the neutral bar beside it is what licenses reading the drop as
suggestion-specific rather than as the cost of inserting a sentence.

Drawn from dump JSON rather than by re-scoring answers, so the plotted values
are the reported values by construction. Population defaults to `clean`
(charts that do not name the gold -- Table 2's population); pass
--population all to draw the full set instead.

    python scripts/analyze_hint_effect.py --answers ... --dump ddx.json
    python scripts/analyze_hint_effect.py --answers ... --dump mcr.json
    python scripts/make_figure_intervention.py \
        --dumps ddx.json mcr.json --labels DDXPlus MedCaseReasoning \
        --output figure3_intervention.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ARM_ORDER = ["none", "neutral", "wrong", "correct"]
ARM_LABEL = {
    "none": "no note",
    "neutral": "neutral note",
    "wrong": "wrong suspicion",
    "correct": "correct suspicion",
}
# Greyscale + hatch, so the wrong arm stays the darkest thing on a b/w print.
ARM_FACE = {"none": "white", "neutral": "0.85", "wrong": "black", "correct": "0.55"}
ARM_HATCH = {"none": "//", "neutral": "", "wrong": "", "correct": ""}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dumps", nargs="+", required=True,
                        help="JSON files from analyze_hint_effect --dump, one per cluster.")
    parser.add_argument("--labels", nargs="+", required=True,
                        help="Cluster label per dump, e.g. DDXPlus MedCaseReasoning.")
    parser.add_argument("--population", default="clean", choices=["clean", "all", "leaky"])
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.dumps) != len(args.labels):
        raise SystemExit("one --labels entry per --dumps entry")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    clusters = []
    for path, label in zip(args.dumps, args.labels):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        pop = data.get(args.population)
        if pop is None:
            raise SystemExit(f"{path} has no '{args.population}' population "
                             f"(has: {sorted(data)})")
        # A dump missing an arm draws a cluster with fewer bars, which reads as
        # a finished figure rather than a missing run. Say so: the arms of one
        # run are often in separate answer files, and analyze_hint_effect
        # merges them only if all of them were passed to it.
        missing = [a for a in ARM_ORDER if a not in pop["arms"]]
        if missing:
            print(f"[warn] {label}: no {', '.join(missing)} arm in {path} -- "
                  f"that cluster will be drawn with {4 - len(missing)} bars.\n"
                  f"       Re-dump with every answer file for this run: "
                  f"analyze_hint_effect.py --answers a.jsonl b.jsonl --dump ...")
        clusters.append((label, pop))

    fig, ax = plt.subplots(figsize=(0.6 + 2.4 * len(clusters), 2.9))
    width = 0.19
    for ci, (label, pop) in enumerate(clusters):
        for ai, arm in enumerate(ARM_ORDER):
            stats = pop["arms"].get(arm)
            if not stats:
                continue
            x = ci + (ai - 1.5) * width
            acc = stats["correct"]
            ax.bar(
                x, acc, width * 0.92, facecolor=ARM_FACE[arm], edgecolor="black",
                hatch=ARM_HATCH[arm], lw=0.8,
                label=ARM_LABEL[arm] if ci == 0 else None,
            )
            ax.text(x, acc + 0.012, f"{acc:.3f}".lstrip("0"), ha="center",
                    fontsize=6.2, rotation=0)

    ax.set_xticks(range(len(clusters)))
    ax.set_xticklabels(
        [f"{label}\n(n = {pop['n']:,})" for label, pop in clusters], fontsize=8
    )
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    # Above the axes, one row: inside the plot it lands on the tallest bars.
    ax.legend(fontsize=6.5, frameon=False, ncols=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
