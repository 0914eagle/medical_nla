#!/usr/bin/env python3
"""Check that current paper documents use the canonical analysis cohorts.

This is intentionally a documentation audit, not a result recomputation. It
guards the high-risk denominators and headline values that have changed across
matcher and eligibility rebuilds. Historical values are allowed when they are
explicitly labelled as fixed-cohort or audit results.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED: dict[str, tuple[str, ...]] = {
    "docs/experiments/POPULATION_REGISTRY_2026-08-25.md": (
        "1,204",
        "1,729",
        "1,628",
        "1,452",
        "319 = 89 adopted + 230 other diagnosis",
        "287 = 86 adopted + 201 other diagnosis",
    ),
    "docs/paper/table_camera_ready_2026-08-25.md": (
        "All: n=1,729",
        "Silent: n=1,628",
        "**.7305**",
        "**.8319**",
        "**.9881**",
    ),
    "docs/professor/paper_presentation_full_2026-08-25.md": (
        "canonical silent 1,628",
        "**.9032**",
        "**−.0998**",
        "n=716/channel",
    ),
    "docs/paper/experiment_summary_2026-08-25.md": (
        "canonical 전체 **.9330**, 침묵 **.9881**",
        "canonical overall **.4170/.4147/.4083/.4552**",
    ),
}

# These phrases were once valid fixed-cohort results but are invalid when
# presented as current/canonical values. Historical sections may retain the
# numbers under an explicit audit label, so only the misleading phrase is
# forbidden.
FORBIDDEN: dict[str, tuple[str, ...]] = {
    "docs/experiments": (
        "canonical n=1,641",
        "정본 침묵 정의(1,641)",
        "08-24 정본 재채점, n=1,747",
    ),
    "docs/paper": (
        "canonical AUROC **.7233 / 침묵 .6829**",
        "Silent (canonical n=1,641)",
        "canonical-eligible clean n=2,137 재집계됐다",
    ),
    "docs/professor/paper_presentation_full_2026-08-25.md": (
        "| No account | .8235",
        "| Probe label | **.8951**",
        "| AV readout | .7301",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: inferred from this script).",
    )
    return parser.parse_args()


def markdown_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(path.rglob("*.md"))


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    failures: list[str] = []

    for relative, needles in REQUIRED.items():
        path = root / relative
        if not path.exists():
            failures.append(f"missing required document: {relative}")
            continue
        text = path.read_text(encoding="utf-8")
        for needle in needles:
            if needle not in text:
                failures.append(f"{relative}: missing canonical marker: {needle!r}")

    for relative, phrases in FORBIDDEN.items():
        base = root / relative
        if not base.exists():
            failures.append(f"missing audit path: {relative}")
            continue
        for path in markdown_files(base):
            text = path.read_text(encoding="utf-8")
            for phrase in phrases:
                if phrase in text:
                    rel = path.relative_to(root)
                    failures.append(f"{rel}: stale canonical phrase: {phrase!r}")

    if failures:
        print("[population-audit] FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)

    print("[population-audit] OK")
    print("- canonical DDXPlus: clean=1,204 all=1,729 silent=1,628 moved=319")
    print("- canonical MCR behavior: n=1,452 moved=427")
    print("- current paper and presentation markers are synchronized")


if __name__ == "__main__":
    main()
