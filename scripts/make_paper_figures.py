"""Render every current paper figure except the experimental pipeline.

Figure 1 is a conceptual pipeline and intentionally excluded. This command
renders main Figures 2--4 plus Appendix Figure A1 from frozen analysis dumps.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> None:
    print("[run]", " ".join(args), flush=True)
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ddx-dump", required=True, help="DDXPlus analyze_hint_effect dump.")
    parser.add_argument("--mcr-dump", required=True, help="MCR analyze_hint_effect dump.")
    parser.add_argument("--trajectory-dump", required=True, help="Canonical trajectory dump.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--readout-values", help="Optional Appendix Figure A1 values JSON.")
    parser.add_argument(
        "--detection-values",
        required=True,
        help="Canonical-eligible Figure 4 values JSON.",
    )
    parser.add_argument("--format", choices=["png", "pdf"], default="png")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ext = args.format

    run(
        sys.executable, str(root / "scripts/make_figure_intervention.py"),
        "--dumps", args.ddx_dump, args.mcr_dump,
        "--labels", "DDXPlus", "MedCaseReasoning",
        "--accuracy-population", "clean", "--destination-population", "all",
        "--omit-no-note",
        "--output", str(out / f"figure2_behavior.{ext}"),
    )
    run(
        sys.executable, str(root / "scripts/make_figure_trajectory.py"),
        "--dump", args.trajectory_dump,
        "--output", str(out / f"figure3_trajectory.{ext}"),
    )
    detection_cmd = [
        sys.executable, str(root / "scripts/make_figure_detection_correction.py"),
        "--output", str(out / f"figure4_detection_correction.{ext}"),
        "--values", args.detection_values,
    ]
    run(*detection_cmd)

    readout_cmd = [
        sys.executable, str(root / "scripts/make_figure_readout_map.py"),
        "--output", str(out / f"appendix_figure_a1_readout_map.{ext}"),
    ]
    if args.readout_values:
        readout_cmd.extend(["--values", args.readout_values])
    run(*readout_cmd)


if __name__ == "__main__":
    main()
