"""The uncertainty audit must agree with the pilot scorer's point estimates."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analyze_cf_readout_uncertainty import (
    aggregate,
    base_stats,
    bootstrap_method,
    claim_texts,
    load_bases,
    paired_comparisons,
)
from scripts.score_ddxplus_e5_readout_pilot import score_method


def readout(cues: list[str]) -> str:
    body = "\n".join(f"- {cue}" for cue in cues)
    return f"<explanation>\n<readout>\n<observed>\n{body}\n</observed>\n</readout>\n</explanation>"


def make_rows() -> list[dict]:
    return [
        {
            "id": "b1-orig",
            "base_id": "b1",
            "variant": "original",
            "cue_targets": ["persistent dry cough", "fever above thirty eight"],
            "cf_original_cue": "",
            "nla_output": readout(
                ["persistent dry cough", "fever above thirty eight"]
            ),
        },
        {
            "id": "b1-del",
            "base_id": "b1",
            "variant": "cue_deleted",
            "cue_targets": ["fever above thirty eight"],
            "cf_original_cue": "persistent dry cough",
            # phantom: still reads the deleted cue
            "nla_output": readout(
                ["persistent dry cough", "fever above thirty eight"]
            ),
        },
        {
            "id": "b2-orig",
            "base_id": "b2",
            "variant": "original",
            "cue_targets": ["sharp left knee pain", "swollen ankle joint"],
            "cf_original_cue": "",
            "nla_output": readout(["sharp left knee pain", "unrelated fabricated claim"]),
        },
        {
            "id": "b2-del",
            "base_id": "b2",
            "variant": "cue_deleted",
            "cue_targets": ["swollen ankle joint"],
            "cf_original_cue": "sharp left knee pain",
            # clean removal: deleted cue is gone
            "nla_output": readout(["swollen ankle joint"]),
        },
        {
            "id": "b2-edit",
            "base_id": "b2",
            "variant": "value_edited",
            "cue_targets": ["sharp right knee pain", "swollen ankle joint"],
            "cf_original_cue": "sharp left knee pain",
            "cf_replacement_cue": "sharp right knee pain",
            # clean switch: reads new value, not old
            "nla_output": readout(["sharp right knee pain", "swollen ankle joint"]),
        },
    ]


def build_stats(threshold: float = 0.5) -> dict[str, dict]:
    rows = make_rows()
    by_base: dict[str, dict[str, dict]] = {}
    for row in rows:
        by_base.setdefault(row["base_id"], {})[row["variant"]] = row
    return {
        base_id: base_stats(group, threshold) for base_id, group in by_base.items()
    }


def test_point_estimates_match_pilot_scorer():
    rows = make_rows()
    expected = score_method(rows, 0.5)
    actual = aggregate(list(build_stats(0.5).values()))

    assert actual["mean_current_finding_recall"] == expected["mean_current_finding_recall"]
    assert actual["deletion_pairs"] == expected["deletion"]["pairs"]
    assert (
        actual["original_target_hit_rate"]
        == expected["deletion"]["original_target_hit_rate"]
    )
    assert (
        actual["deleted_target_phantom_rate"]
        == expected["deletion"]["deleted_target_phantom_rate"]
    )
    assert (
        actual["removal_success_given_original_hit"]
        == expected["deletion"]["removal_success_given_original_hit"]
    )
    assert actual["replacement_hit_rate"] == expected["value_edit"]["replacement_hit_rate"]
    assert actual["clean_switch_rate"] == expected["value_edit"]["clean_switch_rate"]


def test_fixture_encodes_one_phantom_and_one_clean_removal():
    actual = aggregate(list(build_stats(0.5).values()))
    assert actual["original_target_hit_rate"] == 1.0
    assert actual["deleted_target_phantom_rate"] == 0.5
    assert actual["deletion_contrast"] == 0.5
    assert actual["clean_switch_rate"] == 1.0


def test_verbosity_counts_bullet_claims():
    rows = make_rows()
    assert claim_texts(rows[0]) == [
        "persistent dry cough",
        "fever above thirty eight",
    ]
    actual = aggregate(list(build_stats(0.5).values()))
    assert actual["mean_claims_per_readout"] == 9 / 5
    # 9 claims total; unsupported are b2-orig's fabricated claim and
    # b1-del's phantom (its cue is no longer in the current cue set)
    assert actual["unsupported_claim_rate"] == 2 / 9


def test_bootstrap_ci_brackets_point_and_is_deterministic():
    bases = list(build_stats(0.5).values())
    first = bootstrap_method(bases, draws=200, seed=17)
    second = bootstrap_method(bases, draws=200, seed=17)
    assert first == second
    for metric in ("mean_current_finding_recall", "deletion_contrast"):
        lo, hi = first["ci95"][metric]
        assert lo <= first["point"][metric] <= hi


def test_self_comparison_delta_is_zero():
    stats = build_stats(0.5)
    comparisons = paired_comparisons(
        {"a": stats, "b": stats}, [("a", "b")], 0.5, draws=100, seed=17
    )
    assert comparisons[0]["shared_base_cases"] == 2
    for delta in comparisons[0]["deltas"].values():
        assert delta["delta"] == 0.0
        assert not delta["excludes_zero"]


def test_load_bases_rejects_duplicate_variant(tmp_path):
    rows = make_rows()
    duplicated = rows + [rows[0]]
    path = tmp_path / "rows.jsonl"
    import json

    path.write_text(
        "\n".join(json.dumps(row) for row in duplicated) + "\n", encoding="utf-8"
    )
    try:
        load_bases(path)
    except ValueError as error:
        assert "Duplicate" in str(error)
    else:
        raise AssertionError("expected duplicate variant to raise")
