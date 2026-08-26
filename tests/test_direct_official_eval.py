import pytest

from scripts.make_direct_oracle_predictions import official_oracle_prediction
from scripts.run_direct_official_evaluator import is_yes as evaluator_is_yes
from scripts.score_direct_official_eval import score_record


def test_oracle_prediction_matches_official_schema():
    row = {
        "disease_category": "Cardiology",
        "annotation_chain": ["NSTEMI", "NSTE-ACS", "Suspected ACS"],
        "gold_deductions": [
            {
                "observation": "elevated troponin",
                "rationale": "supports myocardial injury",
                "annotated_source_section": "input6",
                "diagnosis": "NSTEMI",
            }
        ],
    }

    prediction, duplicate_count = official_oracle_prediction(row)

    assert duplicate_count == 0
    assert prediction["elevated troponin"] == [
        "supports myocardial injury",
        "input6",
        "NSTEMI",
    ]
    assert prediction["chain"] == [
        "Cardiology",
        "Suspected ACS",
        "NSTE-ACS",
        "NSTEMI",
    ]


def test_duplicate_oracle_observation_is_audited():
    row = {
        "disease_category": "Cardiology",
        "annotation_chain": ["NSTEMI"],
        "gold_deductions": [
            {
                "observation": "chest pain",
                "rationale": "first",
                "annotated_source_section": "input1",
                "diagnosis": "NSTEMI",
            },
            {
                "observation": "chest pain",
                "rationale": "second",
                "annotated_source_section": "input2",
                "diagnosis": "NSTEMI",
            },
        ],
    }

    prediction, duplicate_count = official_oracle_prediction(row)

    assert duplicate_count == 1
    assert prediction["chest pain"][0] == "second"


@pytest.mark.parametrize(
    ("response", "mode", "expected"),
    [
        ("Yes", "official", True),
        ("Yes.", "official", False),
        (" yes. ", "strip-casefold", True),
        ("No", "strip-casefold", False),
    ],
)
def test_judge_response_modes(response, mode, expected):
    assert evaluator_is_yes(response, mode) is expected


def test_score_record_reproduces_official_plus_one_denominators():
    record = {
        "chain_gt": ["Suspected ACS", "NSTEMI"],
        "chain_pred": ["Cardiology", "Suspected ACS", "NSTEMI"],
        "len_ob_gt": 2,
        "len_ob_pred": 2,
        "ob_record_paired": {
            "[0, 0]": [
                "NSTEMI",
                "NSTEMI",
                "gold rationale",
                "predicted rationale",
                "Yes",
            ]
        },
    }

    scores = score_record(record, "official")

    assert scores["acc_cat"] == 1.0
    assert scores["acc_diag"] == 1.0
    assert scores["comp_pre"] == pytest.approx(1 / 3)
    assert scores["comp_re"] == pytest.approx(1 / 3)
    assert scores["comp_coverage"] == pytest.approx(1 / 3)
    assert scores["faith_ob"] == 1.0
    assert scores["faith_all"] == pytest.approx(1 / 3)
    assert scores["ob_precision_unsmoothed"] == 0.5
    assert scores["ob_recall_unsmoothed"] == 0.5
