"""Figure 4 -- the anchoring trajectory, drawn from analyze_trajectory --dump.

Two panels. (a) p(gold) across the six prompt landmarks for the three
behavioural groups, each with its own no-note counterfactual as a grey twin;
the vertical gap between a line and its twin is the note's internal cost, and
the figure's message is that the gap never brings any group's gold below the
suggestion. (b) the flip-point histogram, dominated by the `never` bar
(268 of 324 moved cases).

The expected shape written into the first spec -- the suggestion's mass
overtaking the gold somewhere mid-prompt -- is NOT what was measured, and the
figure must not be drawn to suggest it. What was measured is sustain + rift:
every curve stays gold-dominated to the final token while the emitted answer
is wrong by construction of the population. The crossing the reader expects
to find is the one thing the figure shows does not happen.

Drawn from the dump JSON rather than by re-running the probes, so the plotted
values are the reported values by construction. Black-and-white safe: groups
are distinguished by line style and marker, not colour alone.

    python scripts/analyze_trajectory.py ... --dump $ART/results/trajectory_curve.json
    python scripts/make_figure_trajectory.py \
        --dump $ART/results/trajectory_curve.json \
        --output $ART/results/figure_trajectory.pdf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GROUP_STYLE = {
    # label, linestyle, marker
    "kept": ("answer unchanged", "-", "o"),
    "moved-lost-gold": ("lost the gold", "--", "s"),
    "moved-onto-hint": ("adopted the suggestion", "-.", "^"),
}
LANDMARK_LABEL = {
    "last_cue": "last\nfinding",
    "note": "referral\nnote",
    "question": "question",
    "constraint": "constraint",
    "format": "format",
    "final": "final\ntoken",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", required=True, help="JSON from analyze_trajectory --dump.")
    parser.add_argument("--output", required=True, help=".pdf or .png; drawn at paper width.")
    parser.add_argument(
        "--suggestion-line",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also draw p(suggestion) for the adopted group -- the line that "
        "stays at the bottom, which is the point.",
    )
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    landmarks: list[str] = data["landmarks"]
    xs = list(range(len(landmarks)))

    fig, (ax_curve, ax_flip) = plt.subplots(
        1, 2, figsize=(8.0, 3.1), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )

    for group, (label, ls, marker) in GROUP_STYLE.items():
        cells = data["groups"].get(group, {})
        ys = [cells[r]["p_gold"] if r in cells else None for r in landmarks]
        ax_curve.plot(xs, ys, ls, marker=marker, ms=4, lw=1.4, color="black", label=label)
        # The counterfactual twin. The note landmark has no twin -- the note
        # does not exist in the no-note arm -- so the grey line skips it.
        cf = [cells[r].get("cf_p_gold") if r in cells else None for r in landmarks]
        ax_curve.plot(xs, cf, ls, lw=0.9, color="0.65")
    if args.suggestion_line:
        # A different marker from the adopted group's p(gold) line, or the two
        # lines that must be read against each other become indistinguishable
        # in the legend.
        cells = data["groups"].get("moved-onto-hint", {})
        ys = [cells[r]["p_hint"] if r in cells else None for r in landmarks]
        ax_curve.plot(
            xs, ys, ":", marker="v", ms=3.5, lw=1.1, color="black",
            markerfacecolor="white", label="p(suggestion), adopted",
        )

    ax_curve.set_xticks(xs, [LANDMARK_LABEL.get(r, r) for r in landmarks], fontsize=7)
    ax_curve.set_ylim(0, 1.0)
    ax_curve.set_ylabel("probe mass on the gold diagnosis", fontsize=8)
    ax_curve.tick_params(labelsize=7)
    ax_curve.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.set_title("(a) the state holds the gold; grey = same cases, no note", fontsize=8)

    flips = data["flips"]
    order = [*landmarks, "never"]
    counts = [flips.get(r, 0) for r in order]
    bars = ax_flip.bar(
        range(len(order)), counts,
        color=["0.55"] * len(landmarks) + ["black"], width=0.7,
    )
    for bar, count in zip(bars, counts):
        if count:
            ax_flip.text(
                bar.get_x() + bar.get_width() / 2, count, str(count),
                ha="center", va="bottom", fontsize=6.5,
            )
    # Single-line labels rotated, not the two-line labels of panel (a): seven
    # narrow bars cannot host them side by side without collisions.
    flat = {"last_cue": "last finding", "note": "note", "final": "final token"}
    ax_flip.set_xticks(range(len(order)))
    ax_flip.set_xticklabels(
        [flat.get(r, LANDMARK_LABEL.get(r, r).replace("\n", " ")) for r in order],
        fontsize=6.5, rotation=40, ha="right",
    )
    ax_flip.set_ylabel("moved cases", fontsize=8)
    ax_flip.tick_params(labelsize=7)
    ax_flip.spines[["top", "right"]].set_visible(False)
    ax_flip.set_title("(b) first landmark reading the suggestion", fontsize=8)

    fig.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
