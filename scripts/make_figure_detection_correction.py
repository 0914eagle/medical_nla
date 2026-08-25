"""Figure 4 -- single-run detection and conditional correction.

Panel (a) compares channels by within-diagnosis AUROC for detecting cases
causally moved by a wrong referral note. `Silent` removes cases whose answer
names the note's suggestion, so performance there cannot come from literal
answer copying. Panel (b) separates overall second-pass accuracy from recovery
on the moved subset. The contrast is the policy result: internal feedback can
repair moved cases, but applying reconsideration indiscriminately damages the
much larger kept population.

Values must be supplied from the canonical-eligible reaggregation. The old
matcher-era 1,747-case constants were removed so a bare plotting command
cannot silently resurrect a deprecated population:

Use `build_canonical_figure4_values.py` to create this schema; do not hand-edit
or copy values from an older summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def validate(values: dict[str, Any]) -> None:
    detection = values["detection"]
    correction = values["correction"]
    n_channels = len(detection["channels"])
    if len(detection["all"]) != n_channels or len(detection["silent"]) != n_channels:
        raise ValueError("detection channels/all/silent must have equal lengths")
    n_stages = len(correction["stages"])
    if len(correction["overall"]) != n_stages or len(correction["moved"]) != n_stages:
        raise ValueError("correction stages/overall/moved must have equal lengths")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, help="Output .png or .pdf path.")
    parser.add_argument(
        "--values",
        required=True,
        help="Canonical-eligible JSON from build_canonical_figure4_values.py.",
    )
    args = parser.parse_args()

    values = json.loads(Path(args.values).read_text(encoding="utf-8"))
    validate(values)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    detection = values["detection"]
    correction = values["correction"]
    fig, (ax_detect, ax_correct) = plt.subplots(1, 2, figsize=(7.2, 3.15))

    channels = detection["channels"]
    ys = list(range(len(channels)))
    all_scores = detection["all"]
    silent_scores = detection["silent"]
    ax_detect.axvline(0.5, color="0.75", lw=0.8, linestyle=":")
    ax_detect.scatter(
        all_scores, ys, color="black", s=23, marker="o",
        label=f"all (n={detection['n_all']:,})", zorder=3,
    )
    silent_x = [score for score in silent_scores if score is not None]
    silent_y = [y for y, score in zip(ys, silent_scores) if score is not None]
    ax_detect.scatter(
        silent_x, silent_y, edgecolor="0.4", facecolor="white", s=24, marker="s",
        label=f"silent (n={detection['n_silent']:,})", zorder=3,
    )
    for y, all_score, silent_score in zip(ys, all_scores, silent_scores):
        if silent_score is not None:
            ax_detect.plot([all_score, silent_score], [y, y], color="0.7", lw=0.8, zorder=1)
    for x, y in zip(all_scores, ys):
        ax_detect.text(x + 0.012, y - 0.12, f"{x:.3f}", fontsize=6.2)
    for x, y in zip(silent_x, silent_y):
        dx = -0.055 if x > 0.95 else 0.012
        ax_detect.text(x + dx, y + 0.22, f"{x:.3f}", fontsize=6.2, color="0.3")
    ax_detect.set_yticks(ys, channels, fontsize=7)
    ax_detect.invert_yaxis()
    ax_detect.set_xlim(0.48, 1.02)
    ax_detect.set_xlabel("within-diagnosis AUROC", fontsize=8)
    ax_detect.tick_params(axis="x", labelsize=7)
    ax_detect.spines[["top", "right"]].set_visible(False)
    ax_detect.legend(
        frameon=False, fontsize=6.3, loc="upper center",
        bbox_to_anchor=(0.5, -0.13), ncol=2,
    )
    ax_detect.set_title("(a) Detecting causal movement", fontsize=8.5)

    stages = correction["stages"]
    xs = list(range(len(stages)))
    overall = correction["overall"]
    moved = correction["moved"]
    ax_correct.plot(xs, overall, "-o", color="black", lw=1.4, ms=4.5, label="overall")
    ax_correct.plot(
        xs, moved, "--s", color="0.4", lw=1.3, ms=4.2,
        markerfacecolor="white", label="moved subset",
    )
    for x, value in zip(xs, overall):
        label_x = x - 0.08 if x else x
        ax_correct.text(label_x, value + 0.055, f"{value:.3f}", ha="center", fontsize=6.2)
    for x, value in zip(xs, moved):
        dy = -0.075 if value > 0.08 else 0.035
        label_x = x + 0.08 if x else x
        ax_correct.text(label_x, value + dy, f"{value:.3f}", ha="center", fontsize=6.2,
                        color="0.3")
    ax_correct.set_xticks(xs, stages, fontsize=7)
    ax_correct.set_ylim(0, 0.92)
    ax_correct.set_ylabel("diagnostic accuracy", fontsize=8)
    ax_correct.tick_params(axis="y", labelsize=7)
    ax_correct.spines[["top", "right"]].set_visible(False)
    ax_correct.legend(frameon=False, fontsize=6.5, loc="upper center", ncol=2)
    ax_correct.set_title("(b) Correction requires selection", fontsize=8.5)
    fig.subplots_adjust(left=0.17, right=0.99, top=0.88, bottom=0.24, wspace=0.38)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    print(f"[figure] {args.output}")


if __name__ == "__main__":
    main()
