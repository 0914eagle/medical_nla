"""Figure 2 -- behavioral effect of the referral-note intervention.

Panel (a) draws intervention-arm accuracy for the clean population whose chart
does not name the gold diagnosis. The paper's primary panel (b) decomposes
causally affected cases in the same clean population into causal suggestion
adoption and gold loss without suggestion adoption. An all-eligible sensitivity
version can still be rendered explicitly with ``--destination-population all``.
Both denominators are printed in the plot.

Drawn from dump JSON rather than by re-scoring answers, so the plotted values
are the reported values by construction. Older dumps must be regenerated so
they include the `moved` block added to analyze_hint_effect.py.

    python scripts/analyze_hint_effect.py --answers ... --dump ddx.json
    python scripts/analyze_hint_effect.py --answers ... --dump mcr.json
    python scripts/make_figure_intervention.py \
        --dumps ddx.json mcr.json --labels DDXPlus MedCaseReasoning \
        --output figure2_behavior.pdf
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
    parser.add_argument(
        "--accuracy-population", default="clean", choices=["clean", "all", "leaky"],
        help="Population for panel (a); clean is the paper's primary accuracy cohort.",
    )
    parser.add_argument(
        "--destination-population", default="clean", choices=["clean", "all", "leaky"],
        help="Population for panel (b); clean is the paper's primary behavior cohort.",
    )
    parser.add_argument(
        "--population", choices=["clean", "all", "leaky"],
        help="Legacy shortcut: use one population for both panels.",
    )
    parser.add_argument(
        "--omit-no-note",
        action="store_true",
        help="Omit the no-note accuracy bar and draw a 1.0 reference line. "
        "Use only with dumps restricted to canonically correct no-note cases.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if len(args.dumps) != len(args.labels):
        raise SystemExit("one --labels entry per --dumps entry")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    accuracy_population = args.population or args.accuracy_population
    destination_population = args.population or args.destination_population
    clusters = []
    for path, label in zip(args.dumps, args.labels):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        accuracy_pop = data.get(accuracy_population)
        destination_pop = data.get(destination_population)
        if accuracy_pop is None:
            raise SystemExit(f"{path} has no '{accuracy_population}' population "
                             f"(has: {sorted(data)})")
        if destination_pop is None:
            raise SystemExit(f"{path} has no '{destination_population}' population "
                             f"(has: {sorted(data)})")
        if "moved" not in destination_pop:
            raise SystemExit(
                f"{path} has no moved decomposition; regenerate it with the current "
                "analyze_hint_effect.py --dump"
            )
        # A dump missing an arm draws a cluster with fewer bars, which reads as
        # a finished figure rather than a missing run. Say so: the arms of one
        # run are often in separate answer files, and analyze_hint_effect
        # merges them only if all of them were passed to it.
        missing = [a for a in ARM_ORDER if a not in accuracy_pop["arms"]]
        if missing:
            print(f"[warn] {label}: no {', '.join(missing)} arm in {path} -- "
                  f"that cluster will be drawn with {4 - len(missing)} bars.\n"
                  f"       Re-dump with every answer file for this run: "
                  f"analyze_hint_effect.py --answers a.jsonl b.jsonl --dump ...")
        clusters.append((label, accuracy_pop, destination_pop))

    fig, (ax, ax_dest) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.45, 1.0]}
    )
    display_arms = [a for a in ARM_ORDER if not (args.omit_no_note and a == "none")]
    width = 0.24 if len(display_arms) == 3 else 0.19
    for ci, (label, pop, _) in enumerate(clusters):
        for ai, arm in enumerate(display_arms):
            stats = pop["arms"].get(arm)
            if not stats:
                continue
            x = ci + (ai - (len(display_arms) - 1) / 2) * width
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
        [f"{label}\n(n = {pop['n']:,})" for label, pop, _ in clusters], fontsize=8
    )
    if args.omit_no_note:
        for label, pop, _ in clusters:
            no_note = pop["arms"].get("none", {}).get("correct")
            if no_note is None or abs(float(no_note) - 1.0) > 1e-12:
                raise SystemExit(
                    f"{label}: --omit-no-note requires no-note accuracy 1.0; got {no_note}"
                )
        ax.axhline(
            1.0, color="0.45", linestyle=":", linewidth=0.9,
            label="no-note reference = 1.0 (selection criterion)", zorder=0,
        )
    ax.set_ylim(0.6, 1.02)
    ax.set_ylabel("accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    # Above the axes, one row: inside the plot it lands on the tallest bars.
    ax.legend(fontsize=6.5, frameon=False, ncols=4,
              loc="lower center", bbox_to_anchor=(0.5, 1.20))
    ax.set_title("(a) Accuracy by referral-note arm", fontsize=8)

    xs = list(range(len(clusters)))
    suggestion = [dest["moved"]["to_suggestion"] for _, _, dest in clusters]
    third = [dest["moved"]["to_third_diagnosis"] for _, _, dest in clusters]
    totals = [dest["moved"]["n"] for _, _, dest in clusters]
    ax_dest.bar(xs, suggestion, color="black", width=0.62,
                label="adopted suggestion")
    ax_dest.bar(
        xs, third, bottom=suggestion, color="0.72", edgecolor="black",
        linewidth=0.6, hatch="///", width=0.62,
        label="lost gold; other diagnosis",
    )
    for x, to_hint, to_third, total in zip(xs, suggestion, third, totals):
        if to_hint:
            ax_dest.text(x, to_hint / 2, str(to_hint), color="white", ha="center",
                         va="center", fontsize=6.5)
        if to_third:
            ax_dest.text(x, to_hint + to_third / 2, str(to_third), ha="center",
                         va="center", fontsize=6.5)
        ax_dest.text(x, total, f"total {total}", ha="center", va="bottom", fontsize=6.5)
    ax_dest.set_xticks(
        xs,
        [f"{label}\n(n = {dest['n']:,})" for label, _, dest in clusters],
        fontsize=8,
    )
    ax_dest.set_ylabel("causally affected cases", fontsize=8)
    ax_dest.tick_params(labelsize=7)
    ax_dest.spines[["top", "right"]].set_visible(False)
    ax_dest.legend(fontsize=6.2, frameon=False, loc="upper left")
    ax_dest.set_title("(b) Causally affected cases", fontsize=8)

    fig.subplots_adjust(left=0.08, right=0.99, top=0.74, bottom=0.20, wspace=0.32)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
