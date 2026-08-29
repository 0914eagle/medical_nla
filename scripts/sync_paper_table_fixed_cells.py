"""Materialize or verify branch-independent approved paper-table cells.

This script does not calculate new experimental results and never reads a
locked-test artifact.  It only keeps the canonical paper-table Markdown in
sync with values that were already approved and recorded in the experiment
ledger.  Locked DiReCT cells and conditional generative rows are deliberately
outside its scope.
"""

from __future__ import annotations

import argparse
from pathlib import Path


FIXED_ROWS = {
    "| Finding presence | Multi-label probe |": (
        "| Finding presence | Multi-label probe | 91 frozen evidence IDs | .9607 | "
        "**.9562** | same-diagnosis shuffled .7938; gap +.1624 [.1576,.1672] |"
    ),
    "| Finding value | Conditional probe |": (
        "| Finding value | Conditional probe | 6 evidence tasks / 32 native values | "
        ".7700 | **.7659** | same-diagnosis shuffled .5791; gap +.1868 "
        "[.1650,.2091] |"
    ),
    "| closed decoder | Frozen probe |": (
        "| closed decoder | Frozen probe | .9562 | .7938 | "
        "+.1624 [.1576, .1672] | .7659 |"
    ),
    "| structured monitor | Probe-guided reader | .9587 |": (
        "| structured monitor | Probe-guided reader | .9587 | .7938 | +.1624 | .7654 |"
    ),
    "| structured monitor | Probe-guided reader | .3593 |": (
        "| structured monitor | Probe-guided reader | .3593 | .6407 | .9987 | "
        ".1466 | .5955 | .0804 |"
    ),
    "| Full-data SFT | DiReCT Obscomp |": (
        "| Full-data SFT | DiReCT Obscomp | > .2130 | .0301/.0296, fail |"
    ),
    "| D10 1x2, 20 steps |": (
        "| D10 1x2, 20 steps | ranking−control changed-gap delta, seeds 17/29/43 | "
        "each ≥ .05, CI > 0, specificity | +.0005/+.0028/+.0030, fail |"
    ),
    "| D14 K=5 OOF teacher |": (
        "| D14 K=5 OOF teacher | original cue precision | ≥ .90 + 6 calibration gates | "
        ".8881, fail |"
    ),
    "| D16 soft bottleneck |": (
        "| D16 soft bottleneck | proposed−control DiReCT alignment delta | "
        "each ≥ .005, CI > 0 | −.001137/−.001476/+.001433, fail |"
    ),
    "| D16 frozen-z |": (
        "| D16 frozen-z | auxiliary−control finding F1 | positive across seeds | "
        "−.0009/−.0007/−.0016, fail |"
    ),
}


def sync_rows(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    changed: list[str] = []
    for prefix, expected in FIXED_ROWS.items():
        matches = [index for index, line in enumerate(lines) if line.startswith(prefix)]
        if len(matches) != 1:
            raise ValueError(
                f"Expected exactly one Markdown row starting with {prefix!r}; "
                f"found {len(matches)}"
            )
        index = matches[0]
        if lines[index] != expected:
            lines[index] = expected
            changed.append(prefix)
    return "\n".join(lines) + "\n", changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        type=Path,
        default=Path("docs/paper/tables_and_figures.md"),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    args = parser.parse_args()

    original = args.paper.read_text(encoding="utf-8")
    rendered, changed = sync_rows(original)
    if args.check:
        if changed:
            raise SystemExit(
                "Paper fixed cells are stale: " + ", ".join(changed)
            )
        print(f"[ok] {len(FIXED_ROWS)} fixed rows in {args.paper}")
        return
    if changed:
        args.paper.write_text(rendered, encoding="utf-8")
    print(f"[write] changed={len(changed)} paper={args.paper}")


if __name__ == "__main__":
    main()
