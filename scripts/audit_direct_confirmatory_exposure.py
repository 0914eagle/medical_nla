"""Audit how a frozen DiReCT confirmatory split overlaps prior pilot artifacts.

Only aggregate counts and hashes are emitted. Input JSONLs may contain restricted
clinical text and must remain outside Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val_seen", "test_seen", "test_pdd_heldout")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    value = row.get("base_id") or row.get("id")
    if value is None:
        raise ValueError("A row lacks both base_id and id")
    return str(value)


def read_split_assignments(root: Path) -> dict[str, str]:
    assignments: dict[str, str] = {}
    for split in SPLITS:
        for row in read_jsonl(root / f"{split}.jsonl"):
            identifier = row_id(row)
            if identifier in assignments:
                raise ValueError(
                    f"Duplicate ID across splits in {root}: {identifier}"
                )
            assignments[identifier] = split
    return assignments


def id_hash(values: set[str]) -> str:
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/file.jsonl")
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/file.jsonl")
    return name, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-split-dir", type=Path, required=True)
    parser.add_argument("--confirmatory-split-dir", type=Path, required=True)
    parser.add_argument(
        "--observed-jsonl",
        action="append",
        default=[],
        type=parse_named_path,
        metavar="NAME=PATH",
        help="Prior materialized output; may be repeated.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--summary-md", type=Path)
    args = parser.parse_args()

    pilot = read_split_assignments(args.pilot_split_dir)
    confirmatory = read_split_assignments(args.confirmatory_split_dir)
    if set(pilot) != set(confirmatory):
        raise ValueError(
            "Pilot and confirmatory eligible populations differ: "
            f"pilot={len(pilot)}, confirmatory={len(confirmatory)}, "
            f"pilot_only={len(set(pilot) - set(confirmatory))}, "
            f"confirmatory_only={len(set(confirmatory) - set(pilot))}"
        )

    confirmatory_heldout = {
        identifier
        for identifier, split in confirmatory.items()
        if split == "test_pdd_heldout"
    }
    origin_counts = Counter(pilot[identifier] for identifier in confirmatory_heldout)

    artifacts = []
    union_materialized: set[str] = set()
    for name, path in args.observed_jsonl:
        ids = {row_id(row) for row in read_jsonl(path)}
        eligible_ids = ids & set(confirmatory)
        heldout_ids = ids & confirmatory_heldout
        union_materialized.update(heldout_ids)
        artifacts.append(
            {
                "name": name,
                "path": str(path),
                "rows": len(ids),
                "eligible_rows": len(eligible_ids),
                "confirmatory_heldout_rows": len(heldout_ids),
                "confirmatory_heldout_id_sha256": id_hash(heldout_ids),
                "heldout_rows_by_pilot_split": dict(
                    sorted(Counter(pilot[i] for i in heldout_ids).items())
                ),
            }
        )

    result = {
        "pilot_rows": len(pilot),
        "confirmatory_rows": len(confirmatory),
        "same_eligible_population": True,
        "confirmatory_heldout_rows": len(confirmatory_heldout),
        "confirmatory_heldout_id_sha256": id_hash(confirmatory_heldout),
        "confirmatory_heldout_by_pilot_split": dict(sorted(origin_counts.items())),
        "materialized_artifacts": artifacts,
        "materialized_confirmatory_heldout_union": len(union_materialized),
        "materialized_confirmatory_heldout_union_sha256": id_hash(
            union_materialized
        ),
    }

    lines = [
        "# DiReCT Confirmatory Exposure Audit",
        "",
        "Aggregate-only report; no clinical text or case identifier is emitted.",
        "",
        f"- pilot / confirmatory eligible rows: **{len(pilot)} / {len(confirmatory)}**",
        f"- confirmatory PDD-heldout rows: **{len(confirmatory_heldout)}**",
        "- confirmatory heldout origins in the old pilot split: "
        f"`{dict(sorted(origin_counts.items()))}`",
        "- confirmatory heldout rows with any supplied prior output: "
        f"**{len(union_materialized)}/{len(confirmatory_heldout)}**",
        "",
        "## Prior Materialized Artifacts",
        "",
        "| artifact | rows | eligible | confirmatory heldout | old-split origins |",
        "|---|---:|---:|---:|---|",
    ]
    for artifact in artifacts:
        lines.append(
            "| {name} | {rows} | {eligible_rows} | "
            "{confirmatory_heldout_rows} | `{heldout_rows_by_pilot_split}` |".format(
                **artifact
            )
        )
    if not artifacts:
        lines.append("| (none supplied) | 0 | 0 | 0 | `{}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A row appearing in a prior artifact means its output was materialized, not "
            "necessarily inspected case-by-case. Any nonzero overlap prevents calling this "
            "a dataset-level untouched test. The split can still be frozen prospectively "
            "for downstream Medical-NLA training, selection, and evaluation if no "
            "confirmatory readout result is used to revise the method.",
        ]
    )

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.summary_md:
        args.summary_md.parent.mkdir(parents=True, exist_ok=True)
        args.summary_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
