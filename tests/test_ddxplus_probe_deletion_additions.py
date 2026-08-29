import pytest
import torch

from scripts.evaluate_ddxplus_finding_value_probes import paired_counterfactual_metrics


def test_counterfactual_metrics_report_new_absent_labels_after_deletion() -> None:
    rows = [
        {
            "id": "case__original",
            "base_id": "case",
            "variant": "original",
            "cue_evidence_ids": ["A"],
        },
        {
            "id": "case__deleted",
            "base_id": "case",
            "variant": "cue_deleted",
            "cf_original_evidence_id": "A",
            "cue_evidence_ids": [],
        },
    ]
    finding_logits = torch.tensor([[4.0, -4.0], [-4.0, 4.0]])
    report = paired_counterfactual_metrics(
        rows,
        finding_logits,
        torch.empty((2, 0)),
        ["A", "B"],
        {},
        {},
        0.5,
    )
    deletion = report["deletion"]
    assert deletion["mean_original_selected"] == pytest.approx(1.0)
    assert deletion["mean_deleted_selected"] == pytest.approx(1.0)
    assert deletion["mean_newly_added_after_deletion"] == pytest.approx(1.0)
    assert deletion["newly_added_absent_from_deleted_input"] == pytest.approx(1.0)
