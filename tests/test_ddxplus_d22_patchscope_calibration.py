from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.calibrate_ddxplus_d22_patchscope import (
    CLINICAL_PROMPT,
    IDENTITY_PROMPT,
    distribution_metrics,
    select_ddx_cases,
)
from src.jsonl import write_jsonl


def test_paper_style_prompts_end_at_the_patched_marker() -> None:
    assert IDENTITY_PROMPT.endswith("; foo")
    assert CLINICAL_PROMPT.endswith("; foo")
    assert not IDENTITY_PROMPT.endswith("foo ->")
    assert not CLINICAL_PROMPT.endswith("foo ->")


def test_distribution_metrics_reports_patch_lift_and_rank() -> None:
    patched = torch.tensor([0.0, 3.0, 1.0])
    baseline = torch.tensor([2.0, 0.0, 1.0])
    result = distribution_metrics(patched, baseline, target_id=1)
    assert result["patched_top1_id"] == 1
    assert result["no_patch_top1_id"] == 0
    assert result["target_rank_patched"] == 1
    assert result["target_rank_no_patch"] == 3
    assert result["target_logprob_lift"] > 0
    assert result["kl_patched_to_no_patch"] > 0


def test_select_ddx_cases_reuses_frozen_v1_population(tmp_path: Path) -> None:
    validation = tmp_path / "validation.jsonl"
    protocol = tmp_path / "protocol.json"
    generations = tmp_path / "generations.jsonl"
    activation = tmp_path / "real.pt"
    donor = tmp_path / "donor.pt"
    mean = tmp_path / "mean.pt"
    for path in (activation, donor, mean):
        torch.save(torch.tensor([1.0, 2.0]), path)
    write_jsonl(
        validation,
        [
            {
                "id": "case_original",
                "base_id": "case",
                "variant": "original",
                "activation_path": str(activation),
                "chat_text": "rendered source",
                "position": "7",
            }
        ],
    )
    protocol.write_text(json.dumps({"selected_base_ids": ["case"]}))
    write_jsonl(
        generations,
        [
            {
                "base_id": "case",
                "variant": "original",
                "condition": "real",
                "patch_activation_path": str(activation),
            },
            {
                "base_id": "case",
                "variant": "original",
                "condition": "same_diagnosis_shuffled",
                "patch_activation_path": str(donor),
            },
            {
                "base_id": "case",
                "variant": "shared",
                "condition": "train_mean",
                "patch_activation_path": str(mean),
            },
        ],
    )
    rows = select_ddx_cases(validation, protocol, generations, [], cases=1)
    assert rows == [
        {
            "base_id": "case",
            "chat_text": "rendered source",
            "position": 7,
            "real": str(activation),
            "same_diagnosis_shuffled": str(donor),
            "train_mean": str(mean),
        }
    ]
