"""Create leakage-resistant patient- and PDD-disjoint DiReCT splits.

Input and split JSONLs contain restricted note text and must remain private.
The Markdown summary is aggregate-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stable_fraction(seed: int, value: str) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
    return int(digest[:16], 16) / float(16**16)


class DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[max(root_left, root_right)] = min(root_left, root_right)


def select_eligible_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    exclusions: dict[str, str] = {}
    eligible: list[dict[str, Any]] = []
    duplicate_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if not row.get("canonical_pdd_resolved"):
            exclusions[row["id"]] = "label_conflict"
        elif not row.get("patient_id_parsed"):
            exclusions[row["id"]] = "unparsed_patient"
        else:
            duplicate_groups[row["input_digest"]].append(row)

    for group_rows in duplicate_groups.values():
        ordered = sorted(group_rows, key=lambda row: row["id"])
        eligible.append(ordered[0])
        for duplicate in ordered[1:]:
            exclusions[duplicate["id"]] = "duplicate_copy"

    return sorted(eligible, key=lambda row: row["id"]), exclusions


def pdd_components(rows: list[dict[str, Any]]) -> list[set[str]]:
    labels = {str(row["canonical_pdd"]) for row in rows}
    dsu = DisjointSet(labels)
    patient_labels: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        patient_labels[row["patient_group"]].add(str(row["canonical_pdd"]))
    for linked_labels in patient_labels.values():
        ordered = sorted(linked_labels)
        for label in ordered[1:]:
            dsu.union(ordered[0], label)
    components: dict[str, set[str]] = defaultdict(set)
    for label in labels:
        components[dsu.find(label)].add(label)
    return list(components.values())


def choose_heldout_components(
    rows: list[dict[str, Any]],
    components: list[set[str]],
    fraction: float,
    min_label_rows: int,
    min_remaining_category_rows: int,
    seed: int,
) -> list[set[str]]:
    label_counts = Counter(str(row["canonical_pdd"]) for row in rows)
    category_counts = Counter(row["disease_category"] for row in rows)
    component_rows: list[tuple[set[str], list[dict[str, Any]]]] = []
    for component in components:
        component_subset = [
            row for row in rows if str(row["canonical_pdd"]) in component
        ]
        component_category_counts = Counter(
            row["disease_category"] for row in component_subset
        )
        if any(label_counts[label] < min_label_rows for label in component):
            continue
        if any(
            category_counts[category] - count < min_remaining_category_rows
            for category, count in component_category_counts.items()
        ):
            continue
        component_rows.append((component, component_subset))

    rng = random.Random(seed)
    rng.shuffle(component_rows)
    target = round(len(rows) * fraction)
    selected: list[set[str]] = []
    selected_rows = 0
    selected_category_counts: Counter[str] = Counter()
    covered_categories: set[str] = set()
    remaining = component_rows[:]
    while remaining and selected_rows < target:
        feasible = []
        for item in remaining:
            _, subset = item
            candidate_counts = Counter(row["disease_category"] for row in subset)
            if all(
                category_counts[category]
                - selected_category_counts[category]
                - count
                >= min_remaining_category_rows
                for category, count in candidate_counts.items()
            ):
                feasible.append(item)
        if not feasible:
            break

        def score(item: tuple[set[str], list[dict[str, Any]]]) -> tuple[int, int, float]:
            component, subset = item
            categories = {row["disease_category"] for row in subset}
            new_categories = len(categories - covered_categories)
            distance = abs(target - (selected_rows + len(subset)))
            tie = stable_fraction(seed, "|".join(sorted(component)))
            return (-new_categories, distance, tie)

        component, subset = min(feasible, key=score)
        remaining.remove((component, subset))
        selected.append(component)
        selected_rows += len(subset)
        selected_category_counts.update(row["disease_category"] for row in subset)
        covered_categories.update(row["disease_category"] for row in subset)
    return selected


def assign_seen_patients(
    rows: list[dict[str, Any]], seed: int, train_fraction: float, val_fraction: float
) -> dict[str, str]:
    patient_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        patient_rows[row["patient_group"]].append(row)

    assignments: dict[str, str] = {}
    for patient in patient_rows:
        value = stable_fraction(seed, patient)
        if value < train_fraction:
            assignments[patient] = "train"
        elif value < train_fraction + val_fraction:
            assignments[patient] = "val_seen"
        else:
            assignments[patient] = "test_seen"

    labels = {str(row["canonical_pdd"]) for row in rows}
    while True:
        train_labels = {
            str(row["canonical_pdd"])
            for patient, group_rows in patient_rows.items()
            if assignments[patient] == "train"
            for row in group_rows
        }
        missing = labels - train_labels
        if not missing:
            break
        moved = False
        for label in sorted(missing):
            candidates = [
                (patient, group_rows)
                for patient, group_rows in patient_rows.items()
                if assignments[patient] != "train"
                and any(str(row["canonical_pdd"]) == label for row in group_rows)
            ]
            if not candidates:
                continue
            patient, _ = min(
                candidates,
                key=lambda item: (
                    len(item[1]),
                    stable_fraction(seed + 1, item[0]),
                ),
            )
            assignments[patient] = "train"
            moved = True
        if not moved:
            raise ValueError("Could not ensure every seen PDD appears in train")
    return assignments


def build_splits(
    rows: list[dict[str, Any]],
    seed: int,
    heldout_fraction: float,
    train_fraction: float,
    val_fraction: float,
    min_heldout_label_rows: int,
    min_remaining_category_rows: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, str], list[set[str]]]:
    eligible, exclusions = select_eligible_rows(rows)
    components = pdd_components(eligible)
    heldout_components = choose_heldout_components(
        eligible,
        components,
        heldout_fraction,
        min_heldout_label_rows,
        min_remaining_category_rows,
        seed,
    )
    heldout_labels = set().union(*heldout_components) if heldout_components else set()
    heldout = [
        row for row in eligible if str(row["canonical_pdd"]) in heldout_labels
    ]
    seen = [
        row for row in eligible if str(row["canonical_pdd"]) not in heldout_labels
    ]
    seen_assignments = assign_seen_patients(
        seen, seed + 10_000, train_fraction, val_fraction
    )

    splits: dict[str, list[dict[str, Any]]] = {
        "train": [],
        "val_seen": [],
        "test_seen": [],
        "test_pdd_heldout": heldout,
    }
    for row in seen:
        splits[seen_assignments[row["patient_group"]]].append(row)

    patient_sets = {
        split: {row["patient_group"] for row in split_rows}
        for split, split_rows in splits.items()
    }
    for left, left_patients in patient_sets.items():
        for right, right_patients in patient_sets.items():
            if left < right and left_patients & right_patients:
                raise ValueError(f"Patient leakage between {left} and {right}")
    train_labels = {str(row["canonical_pdd"]) for row in splits["train"]}
    heldout_label_check = {
        str(row["canonical_pdd"]) for row in splits["test_pdd_heldout"]
    }
    if train_labels & heldout_label_check:
        raise ValueError("PDD leakage between train and heldout test")
    return splits, exclusions, heldout_components


def write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    splits: dict[str, list[dict[str, Any]]],
    exclusions: dict[str, str],
    heldout_components: list[set[str]],
    seed: int,
) -> None:
    exclusion_counts = Counter(exclusions.values())
    lines = [
        "# DiReCT Leakage-Resistant Split Summary",
        "",
        "Aggregate-only summary. Split JSONLs contain restricted note text.",
        "",
        f"- seed: **{seed}**",
        f"- input rows: **{len(rows)}**",
        f"- eligible unique rows: **{sum(len(items) for items in splits.values())}**",
        f"- exclusions: `{dict(exclusion_counts)}`",
        f"- held-out PDD connected components: **{len(heldout_components)}**",
        "",
        "## Split Sizes",
        "",
        "| split | rows | patient groups | PDDs | categories |",
        "|---|---:|---:|---:|---:|",
    ]
    for split, split_rows in splits.items():
        lines.append(
            f"| {split} | {len(split_rows)} | "
            f"{len({row['patient_group'] for row in split_rows})} | "
            f"{len({row['canonical_pdd'] for row in split_rows})} | "
            f"{len({row['disease_category'] for row in split_rows})} |"
        )
    heldout_counts = Counter(
        str(row["canonical_pdd"]) for row in splits["test_pdd_heldout"]
    )
    lines.extend(
        [
            "",
            "## Held-Out PDDs",
            "",
            "| PDD | n |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {label} | {count} |" for label, count in heldout_counts.most_common())
    lines.extend(
        [
            "",
            "## Held-Out Connected Components",
            "",
        ]
    )
    lines.extend(
        f"- component {index}: `{', '.join(sorted(component))}`"
        for index, component in enumerate(heldout_components, start=1)
    )
    lines.extend(
        [
            "",
            "## Invariants",
            "",
            "- Patient groups are disjoint across all four splits.",
            "- Held-out PDD connected components do not appear in train.",
            "- Every seen PDD appears at least once in train.",
            "- Unresolved labels, unparsed patient IDs, and duplicate copies are excluded.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--heldout-fraction", type=float, default=0.20)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--min-heldout-label-rows", type=int, default=3)
    parser.add_argument("--min-remaining-category-rows", type=int, default=3)
    args = parser.parse_args()

    if not 0 < args.heldout_fraction < 1:
        raise ValueError("--heldout-fraction must be between 0 and 1")
    if not 0 < args.train_fraction < 1:
        raise ValueError("--train-fraction must be between 0 and 1")
    if not 0 <= args.val_fraction < 1 - args.train_fraction:
        raise ValueError("--val-fraction leaves no room for test_seen")

    rows = read_jsonl(args.manifest)
    splits, exclusions, heldout_components = build_splits(
        rows,
        seed=args.seed,
        heldout_fraction=args.heldout_fraction,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        min_heldout_label_rows=args.min_heldout_label_rows,
        min_remaining_category_rows=args.min_remaining_category_rows,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        write_jsonl(args.out_dir / f"{split}.jsonl", split_rows)
    assignments = [
        {"id": row["id"], "split": split}
        for split, split_rows in splits.items()
        for row in split_rows
    ]
    assignments.extend(
        {"id": row_id, "split": f"excluded_{reason}"}
        for row_id, reason in exclusions.items()
    )
    write_jsonl(
        args.out_dir / "assignments.jsonl",
        sorted(assignments, key=lambda row: row["id"]),
    )
    write_summary(
        args.out_dir / "summary.md",
        rows,
        splits,
        exclusions,
        heldout_components,
        args.seed,
    )
    print(
        "[split] "
        + " ".join(f"{name}={len(items)}" for name, items in splits.items())
    )
    print(f"[split] excluded={len(exclusions)} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
