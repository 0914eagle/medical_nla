import json
import sys
from pathlib import Path

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
from scripts.summarize_direct_locked_paper_tables import main as summarize_tables
from scripts.validate_direct_locked_baseline_recipe import (
    load_json,
    validate_decision,
    validate_recipe,
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


def test_frozen_baseline_recipe_validates_against_decisions():
    root = Path(__file__).resolve().parents[1]
    d19_path = root / "configs/decisions/d19_d10_budget1552_fail_v1.json"
    d21_path = root / "configs/decisions/d21_d20_specificity_anchor_fail_v1.json"
    recipe_path = root / "configs/decisions/direct_locked_baseline_only_v1.json"
    d19, d21, recipe = map(load_json, (d19_path, d21_path, recipe_path))
    validate_decision(d19, decision_id="D19", method_fragment="D10")
    validate_decision(d21, decision_id="D21", method_fragment="D20")
    validate_recipe(recipe, d19_path, d21_path)


def test_paper_table_summary_combines_completed_batch(tmp_path, monkeypatch):
    source = {
        "arms": {
            arm: {
                split: {
                    "n": n,
                    "strict_pdd_accuracy": 0.5,
                    "category_accuracy": 0.75,
                    "mean_token_f1": 0.6,
                }
                for split, n in (("test_seen", 72), ("test_pdd_heldout", 106))
            }
            for arm in ("direct", "cot")
        }
    }
    probe = {
        "population": "test_seen",
        "locked_test_read": True,
        "results": [
            {
                "label_field": target,
                "n": 72,
                "own": {"acc1": 0.6},
                "label_shuffle": {"acc1": 0.2},
                "own_minus_shuffle": 0.4,
            }
            for target in ("canonical_pdd", "disease_category")
        ],
    }
    (tmp_path / "table1a_source").mkdir()
    (tmp_path / "table1b_probes").mkdir()
    (tmp_path / "table1a_source/results.json").write_text(json.dumps(source))
    (tmp_path / "table1b_probes/results.json").write_text(json.dumps(probe))
    for split, n in (("test_seen", 72), ("test_pdd_heldout", 106)):
        pool = tmp_path / split
        (pool / "reports").mkdir(parents=True)
        audit = []
        for method in ("cot", "vanilla"):
            for index in range(n):
                audit.append(
                    {
                        "method": method,
                        "accepted_claims": [{}] if index == 0 else [],
                        "parse_error": None,
                    }
                )
            report = {
                "population": {"expected_predictions": n, "zero_scored": 0},
                "metrics": {
                    field: {"mean": 0.25} for field in (
                        "acc_diag", "comp_pre", "comp_re", "comp_coverage", "faith_ob", "faith_all"
                    )
                },
            }
            (pool / "reports" / f"{method}.json").write_text(json.dumps(report))
        (pool / "private_extraction_audit.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in audit)
        )
    output_json = tmp_path / "paper_tables_summary.json"
    output_md = tmp_path / "paper_tables_summary.md"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_direct_locked_paper_tables.py",
            "--root", str(tmp_path),
            "--output-json", str(output_json),
            "--summary-md", str(output_md),
        ],
    )
    summarize_tables()
    result = json.loads(output_json.read_text())
    assert result["table_2"]["test_seen"]["cot"]["extraction_coverage"] == 1 / 72
    assert "Medical-NLA locked row" in output_md.read_text()
