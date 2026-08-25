from scripts.evaluate_error_prediction import (
    auroc,
    collect_scores,
    paired_disagree_auroc,
)
from scripts.make_error_prediction_table import answer_agree, id_agree


def test_id_agree_is_exact_match_in_id_space():
    assert id_agree("bronchitis", "bronchitis") is True
    assert id_agree("Acute_COPD_exacerbation", "acute copd exacerbation") is True
    assert id_agree("bronchitis", "acute bronchitis") is False
    assert id_agree(None, "bronchitis") is None
    assert id_agree("bronchitis", "") is None
    # answer_agree stays substring-based for free-text NLA answers.
    assert answer_agree("bronchitis", "acute bronchitis") is True


def test_collect_scores_probe_disagree():
    rows = [
        {"is_error": True, "source_probe_answer_agree": False},
        {"is_error": False, "source_probe_answer_agree": True},
        {"is_error": True, "source_probe_answer_agree": None},
        {"is_error": None, "source_probe_answer_agree": False},
    ]
    labels, scores = collect_scores(
        rows, name="source_probe_disagree", field="source_probe_answer_agree", direction=-1
    )
    assert labels == [True, False]
    assert scores == [1.0, 0.0]
    assert auroc(labels, scores) == 1.0


def test_paired_disagree_auroc_uses_intersection_only():
    rows = [
        # Paired rows: NLA separates perfectly, probe does not.
        {"is_error": True, "source_nla_answer_agree": False, "source_probe_answer_agree": True},
        {"is_error": False, "source_nla_answer_agree": True, "source_probe_answer_agree": True},
        # Unpaired rows must be excluded from both sides.
        {"is_error": True, "source_nla_answer_agree": False, "source_probe_answer_agree": None},
        {"is_error": False, "source_nla_answer_agree": None, "source_probe_answer_agree": False},
    ]
    paired = paired_disagree_auroc(rows)
    assert paired is not None
    assert paired["n"] == 2
    assert paired["nla_auroc"] == 1.0
    assert paired["probe_auroc"] == 0.5

    assert paired_disagree_auroc([{"is_error": True, "source_nla_answer_agree": False}]) is None


def test_limiting_the_run_must_not_narrow_the_candidate_set():
    """--limit 200 on a diagnosis-grouped file left two labels in play, and the
    resulting two-way choice scored top1 200/200 and mrr 1.0000 against a
    source model that answers these cases at 0.3724. Candidates are collected
    before any limit, and the limit samples rather than truncates."""
    import random

    import pytest

    pytest.importorskip("torch")
    from scripts.score_source_diagnosis_logprobs import collect_candidates

    rows = [
        {"prompt": f"case {i}", "diagnosis_id": f"dx_{i // 100}", "diagnosis_name": f"Dx {i // 100}"}
        for i in range(490)
    ]
    # The order the bug depended on: a hundred cases of each label in a block.
    assert [row["diagnosis_id"] for row in rows[:200]] == ["dx_0"] * 100 + ["dx_1"] * 100

    candidates = collect_candidates(rows)
    assert len(candidates) == 5

    limited = random.Random(17).sample(rows, 200)
    assert len({row["diagnosis_id"] for row in limited}) == 5
    # The candidate set is the file's, not the sample's.
    assert len(collect_candidates(rows)) == 5


def test_external_candidate_file_is_deduplicated(tmp_path):
    import json

    import pytest

    pytest.importorskip("torch")
    from scripts.score_source_diagnosis_logprobs import read_candidates

    path = tmp_path / "candidates.jsonl"
    rows = [
        {"diagnosis_id": "pneumonia", "diagnosis_name": "Pneumonia"},
        {"diagnosis_id": "pneumonia", "diagnosis_name": "Pneumonia"},
        {"diagnosis_id": "pulmonary_embolism", "diagnosis_name": "Pulmonary embolism"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    candidates = read_candidates(str(path), [])
    assert [row["diagnosis_id"] for row in candidates] == [
        "pneumonia",
        "pulmonary_embolism",
    ]


def test_output_head_features_use_wrong_run_only():
    from scripts.evaluate_output_head_attribution import output_head_features

    case = {
        "none": {"diagnosis_name": "Pneumonia", "answer": "Pneumonia"},
        "wrong": {
            "answer": "Pneumonia",
            "hint_diagnosis_name": "Pulmonary embolism",
        },
    }
    row = {
        "id": "case__wrong",
        "num_candidates": 2,
        "top_candidates": [
            {
                "diagnosis_id": "pulmonary_embolism",
                "diagnosis_name": "Pulmonary embolism",
                "first_token_logprob": -0.1,
            },
            {
                "diagnosis_id": "pneumonia",
                "diagnosis_name": "Pneumonia",
                "first_token_logprob": -2.0,
            },
        ],
    }
    features = output_head_features(case, row, "first_token_logprob")
    assert features["output-head top1 is suggestion"] == 1.0
    assert features["output-head top1 disagrees with generated answer"] == 1.0
    assert features["output-head probability of suggestion"] > 0.8
    assert features["[analysis only] suggestion minus gold probability"] > 0.0
