import json

import torch

from scripts.prepare_ddxplus_probe_train import cue_statistics
from scripts.train_ddxplus_finding_value_probes import (
    binary_auroc,
    donor_targets,
    finding_targets,
    finding_vocabulary,
    value_metrics,
    value_ontology,
    value_slices,
    value_targets,
)


def row(identifier, evidence_ids, value_ids):
    return {
        "id": f"{identifier}__original__cot_p0",
        "base_id": identifier,
        "variant": "original",
        "position_family": "P0",
        "cue_evidence_ids": evidence_ids,
        "cue_value_ids": value_ids,
    }


def test_train_only_finding_and_value_ontologies_apply_count_floors():
    rows = [
        row("a", ["fever", "pain"], ["high", "chest"]),
        row("b", ["fever", "pain"], ["low", "chest"]),
        row("c", ["fever", "rare"], ["high", "yes"]),
    ]

    findings, counts = finding_vocabulary(rows, min_count=2)
    values, pair_counts = value_ontology(rows, min_value_count=1)

    assert findings == ["fever", "pain"]
    assert counts["rare"] == 1
    assert values == {"fever": ["high", "low"]}
    assert pair_counts[("pain", "chest")] == 2


def test_value_metric_conditions_candidates_on_evidence_id():
    rows = [
        row("a", ["fever", "pain"], ["high", "chest"]),
        row("b", ["fever", "pain"], ["low", "abdomen"]),
    ]
    ontology = {"fever": ["high", "low"], "pain": ["abdomen", "chest"]}
    targets = value_targets(rows, ontology)
    slices = value_slices(ontology)
    logits = torch.tensor(
        [
            [5.0, 0.0, 0.0, 5.0],
            [0.0, 5.0, 5.0, 0.0],
        ]
    )

    metrics = value_metrics(logits, targets, slices)

    assert metrics["targets"] == 4
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_recall"] == 1.0


def test_hard_shuffle_uses_donor_labels_from_same_population(tmp_path):
    rows = [
        row("a", ["fever"], ["high"]),
        row("b", ["pain"], ["chest"]),
    ]
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(
        json.dumps(
            {
                "own_base_id": "a",
                "donor_base_id": "b",
                "primary_pair_eligible": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    indices, donor_findings, donor_values = donor_targets(
        rows,
        pairs,
        ["fever", "pain"],
        {"fever": ["high", "low"], "pain": ["abdomen", "chest"]},
    )

    assert indices == [0]
    assert donor_findings.tolist() == [[0.0, 1.0]]
    assert donor_values["pain"][0].tolist() == [0]
    assert donor_values["pain"][1].tolist() == [1]


def test_binary_auroc_and_population_statistics():
    labels = torch.tensor([0, 0, 1, 1]).numpy()
    scores = torch.tensor([0.1, 0.2, 0.8, 0.9]).numpy()
    assert binary_auroc(labels, scores) == 1.0

    rows = [
        row("a", ["fever", "pain"], ["high", "chest"]),
        row("b", ["fever"], ["low"]),
    ]
    stats = cue_statistics(rows)
    assert stats["cue_occurrences"] == 3
    assert stats["unique_evidence_ids"] == 2
    assert stats["unique_evidence_value_pairs"] == 3

    targets = finding_targets(rows, ["fever", "pain"])
    assert targets.tolist() == [[1.0, 1.0], [1.0, 0.0]]
