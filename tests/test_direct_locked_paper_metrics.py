import json

import torch

from scripts.evaluate_direct_locked_probes import (
    cluster_ci,
    shuffled_targets,
)
from scripts.reindex_and_score_direct_locked_source_outputs import (
    annotate,
    exact_mcnemar,
    paired_summary,
    summarize,
)


def frozen(identifier, group, pdd="HFrEF", category="Heart Failure"):
    return {
        "id": identifier,
        "patient_group": group,
        "canonical_pdd": pdd,
        "disease_category": category,
    }


def answer(identifier, value, aliases=None):
    return {
        "id": identifier,
        "answer": value,
        "answer_parsed": True,
        "diagnosis_name": "old label",
        "diagnosis_aliases": aliases or [],
    }


def test_source_annotation_uses_frozen_labels_and_legitimate_aliases():
    row = annotate(
        "cot",
        "test_seen",
        frozen("a", "p1"),
        answer("a", "HFrEF due to heart failure", ["HFrEF"]),
    )
    assert row["strict_correct"] is True
    assert row["category_correct"] is True


def test_source_summary_and_paired_cluster_bootstrap_are_deterministic():
    direct = [
        {**annotate("direct", "test_seen", frozen("a", "p1"), answer("a", "HFrEF"))},
        {**annotate("direct", "test_seen", frozen("b", "p2"), answer("b", "wrong"))},
    ]
    cot = [
        {**annotate("cot", "test_seen", frozen("a", "p1"), answer("a", "wrong"))},
        {**annotate("cot", "test_seen", frozen("b", "p2"), answer("b", "HFrEF"))},
    ]
    summary = summarize(direct, replicates=200, seed=17)
    assert summary["strict_pdd_accuracy"] == 0.5
    paired = paired_summary(
        direct, cot, "strict_correct", replicates=200, seed=17
    )
    assert paired["right_minus_left"] == 0.0
    assert paired["mcnemar_exact_p"] == 1.0
    assert exact_mcnemar(0, 2) == 0.5


def test_probe_shuffle_keeps_rows_and_cluster_ci_bounds():
    rows = [
        {"id": "a", "patient_group": "p1"},
        {"id": "b", "patient_group": "p1"},
        {"id": "c", "patient_group": "p2"},
        {"id": "d", "patient_group": "p3"},
    ]
    labels = torch.tensor([0, 0, 1, 2])
    shuffled = shuffled_targets(rows, labels, seed=17)
    assert shuffled.shape == labels.shape
    assert not torch.equal(shuffled, labels)
    interval = cluster_ci(rows, [1.0, 1.0, 0.0, 0.0], replicates=200, seed=17)
    assert 0 <= interval[0] <= interval[1] <= 1


def test_probe_control_protocol_schema_example(tmp_path):
    protocol = {
        "shuffle_unit": "patient_group",
        "shuffle_seed": 17,
        "control_init_seed": 17,
        "bootstrap_seed": 17,
        "bootstrap_replicates": 10000,
    }
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    assert json.loads(path.read_text())["shuffle_unit"] == "patient_group"
