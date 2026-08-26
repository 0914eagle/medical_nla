import torch

from scripts.train_direct_linear_probe import metrics, patient_disjoint


def test_metrics_perfect_ranking() -> None:
    logits = torch.tensor([[4.0, 1.0], [0.0, 3.0]])
    labels = torch.tensor([0, 1])
    result = metrics(logits, labels)
    assert result["acc1"] == 1.0
    assert result["acc5"] == 1.0
    assert result["mrr"] == 1.0
    assert result["macro_recall"] == 1.0


def test_patient_overlap_is_rejected() -> None:
    train = [{"base_id": "a", "patient_group": "patient-1"}]
    val = [{"base_id": "b", "patient_group": "patient-1"}]
    assert not patient_disjoint(train, val)
