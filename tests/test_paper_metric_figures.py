import json

import torch

from scripts.make_ddxplus_counterfactual_figure import compute
from scripts.make_medical_nla_probe_layer_figure import (
    ddxplus_values,
    direct_values,
    validate,
)


def test_probe_layer_values_are_extracted_from_result_artifacts(tmp_path):
    direct = []
    for field in ("canonical_pdd", "disease_category"):
        for layer, score in zip((16, 24, 32), (0.4, 0.5, 0.45), strict=True):
            direct.append(
                {
                    "label_field": field,
                    "layer": layer,
                    "selected": {"validation": {"acc1": score}},
                }
            )
    direct_path = tmp_path / "direct.json"
    direct_path.write_text(json.dumps(direct))
    ddx = {
        "results": [
            {
                "layer": layer,
                "finding": {"selected_threshold": {"micro_f1": finding}},
                "value": {"validation": {"accuracy": value}},
            }
            for layer, finding, value in (
                (16, 0.96, 0.76),
                (24, 0.95, 0.77),
                (32, 0.94, 0.70),
            )
        ]
    }
    ddx_path = tmp_path / "ddx.json"
    ddx_path.write_text(json.dumps(ddx))
    values = {**direct_values(direct_path), **ddxplus_values(ddx_path)}
    validate(values)
    assert values["DiReCT PDD top-1"] == [0.4, 0.5, 0.45]
    assert values["DDXPlus native-value accuracy"] == [0.76, 0.77, 0.7]


def test_counterfactual_figure_metrics_follow_paired_activations(tmp_path):
    artifact = tmp_path / "probe.pt"
    torch.save(
        {
            "layer": 24,
            "feature_mean": torch.zeros((1, 2)),
            "feature_std": torch.ones((1, 2)),
            "finding_labels": ["fever", "cough"],
            "finding_threshold": 0.5,
            "finding_state_dict": {
                "weight": torch.tensor([[10.0, 0.0], [0.0, 10.0]]),
                "bias": torch.tensor([-5.0, -5.0]),
            },
            "values_by_evidence": {"fever": ["high", "low"]},
            "value_slices": {"fever": (0, 2)},
            "value_state_dict": {
                "weight": torch.tensor([[10.0, 0.0], [-10.0, 0.0]]),
                "bias": torch.zeros(2),
            },
        },
        artifact,
    )
    rows = []
    specs = [
        ("a__original", "original", torch.tensor([1.0, 1.0])),
        ("a__deleted", "cue_deleted", torch.tensor([0.0, 1.0])),
        ("a__edited", "value_edited", torch.tensor([-1.0, 1.0])),
    ]
    for identifier, variant, vector in specs:
        activation = tmp_path / f"{identifier}.pt"
        torch.save(vector, activation)
        row = {
            "id": identifier,
            "base_id": "a",
            "variant": variant,
            "activation_path": str(activation),
            "cue_evidence_ids": ["cough"] if variant == "cue_deleted" else ["fever", "cough"],
        }
        if variant != "original":
            row["cf_original_evidence_id"] = "fever"
        if variant == "value_edited":
            row["cf_original_value_id"] = "high"
            row["cf_replacement_value_id"] = "low"
        rows.append(row)
    metrics, distributions = compute(artifact, rows, torch.device("cpu"))
    assert metrics["deletion"]["phantom"] == 0.0
    assert metrics["deletion"]["removal_given_original_hit"] == 1.0
    assert metrics["retained"]["preservation_given_original_hit"] == 1.0
    assert metrics["value_edit"]["clean_switch"] == 1.0
    assert len(distributions["deletion_drop"]) == 1
