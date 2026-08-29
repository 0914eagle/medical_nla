import pytest

from scripts.summarize_paper_figure_values import validate_figure2, validate_figure3


def figure2_payload():
    return {
        "population": "validation",
        "layers": [16, 24, 32],
        "series": {
            "DiReCT category top-1": [0.5, 0.6, 0.52],
            "DiReCT PDD top-1": [0.38, 0.44, 0.38],
            "DDXPlus finding micro F1": [0.96, 0.96, 0.96],
            "DDXPlus native-value accuracy": [0.76, 0.77, 0.70],
        },
    }


def figure3_payload():
    return {
        "population": "locked_test",
        "metrics": {
            "layer": 24,
            "deletion": {
                "eligible": 4540,
                "original_hit": 1.0,
                "phantom": 0.36,
                "removal_given_original_hit": 0.64,
            },
            "retained": {"preservation_given_original_hit": 0.99},
            "value_edit": {
                "eligible": 539,
                "replacement_hit": 0.15,
                "old_persistence": 0.60,
                "clean_switch": 0.08,
            },
        },
    }


def test_summary_validators_accept_canonical_populations():
    assert validate_figure2(figure2_payload())["DiReCT category top-1"][1] == 0.6
    assert validate_figure3(figure3_payload())["layer"] == 24


def test_figure3_rejects_wrong_denominator():
    payload = figure3_payload()
    payload["metrics"]["deletion"]["eligible"] = 4523
    with pytest.raises(ValueError, match="4,540"):
        validate_figure3(payload)
