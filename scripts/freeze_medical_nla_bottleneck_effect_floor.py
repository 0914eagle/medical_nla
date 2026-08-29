"""Freeze the D16 effect floor after all three control audits and before proposed."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.nla_bottleneck import sha256_file


def effect_floor(gaps: list[float]) -> tuple[float, float]:
    if len(gaps) != 3 or not all(math.isfinite(value) for value in gaps):
        raise ValueError("D16 effect floor requires three finite control gaps")
    spread = max(gaps) - min(gaps)
    return spread, max(2.0 * spread, 0.005)


def parse_seed_paths(values: list[str], name: str) -> dict[int, Path]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"{name} must be SEED=PATH")
        raw_seed, raw_path = value.split("=", 1)
        seed = int(raw_seed)
        if seed in parsed:
            raise ValueError(f"Duplicate {name} seed {seed}")
        parsed[seed] = Path(raw_path)
    if set(parsed) != {17, 29, 43}:
        raise ValueError(f"{name} seeds must be exactly 17,29,43")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-audit", action="append", required=True)
    parser.add_argument("--control-adapter", action="append", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite immutable floor: {args.output}")

    audits = parse_seed_paths(args.control_audit, "--control-audit")
    adapters = parse_seed_paths(args.control_adapter, "--control-adapter")
    records = {}
    gaps = []
    for seed in (17, 29, 43):
        audit_path = audits[seed]
        adapter_path = adapters[seed]
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        metadata = json.loads((adapter_path / "best.json").read_text(encoding="utf-8"))
        if int(audit.get("seed") or -1) != seed:
            raise ValueError(f"Audit seed mismatch for {audit_path}")
        if metadata.get("arm") != "control" or int(metadata.get("seed") or -1) != seed:
            raise ValueError(f"Control adapter metadata mismatch for {adapter_path}")
        if audit.get("bottleneck_sha256") != metadata.get("bottleneck_sha256"):
            raise ValueError(f"Audit did not use the control bottleneck for seed {seed}")
        private_scores = Path(str(audit.get("private_scores") or ""))
        if not private_scores.is_file() or audit.get("private_scores_sha256") != sha256_file(
            private_scores
        ):
            raise ValueError(f"Control private-score hash mismatch for seed {seed}")
        gap = float(audit["symmetric_cross"]["cross_minus_matched"])
        if not math.isfinite(gap):
            raise ValueError(f"Non-finite control gap for seed {seed}")
        gaps.append(gap)
        records[str(seed)] = {
            "gap": gap,
            "audit": str(audit_path),
            "audit_sha256": sha256_file(audit_path),
            "adapter": str(adapter_path),
            "best_sha256": sha256_file(adapter_path / "best.json"),
            "bottleneck_sha256": metadata["bottleneck_sha256"],
            "private_scores": str(private_scores),
            "private_scores_sha256": sha256_file(private_scores),
        }
    spread, floor = effect_floor(gaps)
    report = {
        "decision": "D16",
        "frozen_before_proposed": True,
        "primary_metric": "Direct_validation_symmetric_alignment_gap",
        "formula": "max(2*(max_control_gap-min_control_gap),0.005)",
        "control_gaps": records,
        "control_gap_spread": spread,
        "effect_floor": floor,
        "proposed_checkpoint_read": False,
        "locked_test_read": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = args.output.with_name("effect_floor_summary.md")
    summary.write_text(
        "\n".join(
            [
                "# D16 Immutable Control-First Effect Floor",
                "",
                f"- control gaps: **{', '.join(f'{value:+.6f}' for value in gaps)}**",
                f"- control gap spread: **{spread:.6f}**",
                "- frozen formula: `max(2 * spread, .005)`",
                f"- frozen effect floor: **{floor:.6f}**",
                "- proposed checkpoint read: **no**",
                "- locked test read: **no**",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    main()
