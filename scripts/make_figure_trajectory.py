"""Figure 4 -- decoded trajectory under the wrong referral note.

Three panels separate three claims that the old two-panel figure conflated:

* absolute decoded gold (and adopted-group suggestion) probability;
* the paired note effect, wrong-note minus no-note, on the same cases;
* the first landmark where the suggestion becomes probe top-1, with cases
  where it never does split into gold-throughout and third-diagnosis paths.

The final stacked `suggestion never top-1` bar explicitly separates cases
where gold was top-1 throughout from cases where a third diagnosis became
top-1. This script always reads the counts from the dump and never hard-codes
the trajectory denominator.

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

    fig, (ax_curve, ax_delta, ax_flip) = plt.subplots(
        1, 3, figsize=(10.2, 3.0), gridspec_kw={"width_ratios": [1.8, 1.8, 1.15]}
    )

    for group, (label, ls, marker) in GROUP_STYLE.items():
        cells = data["groups"].get(group, {})
        ys = [cells[r]["p_gold"] if r in cells else None for r in landmarks]
        ax_curve.plot(xs, ys, ls, marker=marker, ms=4, lw=1.4, color="black", label=label)
        delta = [
            cells[r]["p_gold"] - cells[r]["cf_p_gold"]
            if r in cells and "cf_p_gold" in cells[r]
            else None
            for r in landmarks
        ]
        ax_delta.plot(
            xs, delta, ls, marker=marker, ms=4, lw=1.4, color="black", label=label
        )
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
    ax_curve.set_ylabel("Mean probe probability", fontsize=8)
    ax_curve.tick_params(labelsize=7)
    ax_curve.legend(
        fontsize=6.2,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=2,
    )
    ax_curve.spines[["top", "right"]].set_visible(False)
    ax_curve.set_title("(a) Decoded signal under the wrong note", fontsize=8)

    ax_delta.axhline(0, color="0.65", lw=0.8)
    ax_delta.set_xticks(xs, [LANDMARK_LABEL.get(r, r) for r in landmarks], fontsize=7)
    ax_delta.set_ylabel(r"Note effect on gold: $p_{wrong}-p_{none}$", fontsize=8)
    ax_delta.tick_params(labelsize=7)
    ax_delta.legend(fontsize=6.5, frameon=False, loc="lower left")
    ax_delta.spines[["top", "right"]].set_visible(False)
    ax_delta.set_title("(b) Paired internal cost of the note", fontsize=8)

    # There is no note-position token in the no-note arm, so a paired delta at
    # that landmark is undefined rather than zero.
    if "note" in landmarks:
        note_x = landmarks.index("note")
        y0, y1 = ax_delta.get_ylim()
        ax_delta.text(
            note_x,
            y1 - 0.08 * (y1 - y0),
            "N/A",
            ha="center",
            va="top",
            fontsize=6.2,
            color="0.35",
        )

    flips = data["flips"]
    order = list(landmarks)
    counts = [flips.get(r, 0) for r in order]
    bars = ax_flip.bar(
        range(len(order)), counts, color="0.55", width=0.7,
    )
    zero_label_y = max([*counts, 1]) * 0.012
    for bar, count in zip(bars, counts):
        if count:
            ax_flip.text(
                bar.get_x() + bar.get_width() / 2, count, str(count),
                ha="center", va="bottom", fontsize=6.5,
            )
        else:
            # A missing label is easy to misread as a missing landmark. In the
            # canonical trajectory the referral-note count is a measured zero.
            ax_flip.text(
                bar.get_x() + bar.get_width() / 2,
                zero_label_y,
                "0",
                ha="center",
                va="bottom",
                fontsize=6.5,
                color="0.35",
            )
    # `never suggestion` is not synonymous with `gold throughout`. Show the
    # two mutually exclusive paths as a stacked final bar so the visual cannot
    # support that stronger, invalid reading.
    top1_paths = data.get("top1_paths", {})
    gold_all = int(top1_paths.get("gold_all_landmarks", 0))
    other_never = int(top1_paths.get("other_top1_but_never_suggestion", 0))
    never_total = int(flips.get("never", gold_all + other_never))
    never_x = len(order)
    if gold_all + other_never == never_total and never_total:
        ax_flip.bar(
            never_x,
            gold_all,
            color="black",
            width=0.7,
            label="gold top-1 throughout",
        )
        ax_flip.bar(
            never_x,
            other_never,
            bottom=gold_all,
            color="0.72",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            width=0.7,
            label="other top-1; suggestion never top-1",
        )
        if gold_all:
            ax_flip.text(
                never_x,
                gold_all / 2,
                str(gold_all),
                ha="center",
                va="center",
                fontsize=6.2,
                color="white",
            )
        if other_never:
            ax_flip.text(
                never_x,
                gold_all + other_never / 2,
                str(other_never),
                ha="center",
                va="center",
                fontsize=6.2,
            )
        ax_flip.text(
            never_x,
            never_total,
            f"total {never_total}",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
        ax_flip.legend(fontsize=5.4, frameon=False, loc="upper left")
    else:
        # Backward compatibility for pre-audit dumps. The label remains
        # explicit, but camera-ready figures should use a dump with top1_paths.
        ax_flip.bar(never_x, never_total, color="black", width=0.7)
        if never_total:
            ax_flip.text(
                never_x,
                never_total,
                str(never_total),
                ha="center",
                va="bottom",
                fontsize=6.5,
            )

    # Single-line labels rotated, not the two-line labels of panel (a): the
    # narrow bars cannot host them side by side without collisions.
    flat = {
        "last_cue": "last finding",
        "note": "note",
        "final": "final token",
    }
    ax_flip.set_xticks(range(len(order) + 1))
    ax_flip.set_xticklabels(
        [flat.get(r, LANDMARK_LABEL.get(r, r).replace("\n", " ")) for r in order]
        + ["suggestion never top-1"],
        fontsize=6.5, rotation=40, ha="right",
    )
    ax_flip.set_ylabel("moved cases", fontsize=8)
    ax_flip.tick_params(labelsize=7)
    ax_flip.spines[["top", "right"]].set_visible(False)
    ax_flip.set_title("(c) First suggestion top-1 landmark", fontsize=8)

    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.31, wspace=0.32)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
