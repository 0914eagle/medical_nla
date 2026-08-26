"""Aggregate DiReCT evaluator JSONs with the official statistics.py formulas.

The emitted JSON and Markdown contain aggregate metrics only. Evaluation JSONs
contain restricted text and must remain in private storage.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


OFFICIAL_METRICS = (
    "acc_cat",
    "acc_diag",
    "comp_pre",
    "comp_re",
    "comp_coverage",
    "faith_ob",
    "faith_all",
)


def capitalize_first_letter(value: Any) -> str:
    text = str(value)
    return text[:1].upper() + text[1:]


def is_yes(response: Any, mode: str) -> bool:
    text = str(response)
    if mode == "official":
        return text == "Yes"
    if mode == "strip-casefold":
        return text.strip().rstrip(".").casefold() == "yes"
    raise ValueError(f"Unknown response mode: {mode}")


def score_record(data: dict[str, Any], rationale_response_mode: str) -> dict[str, float]:
    chain_gt = data["chain_gt"]
    chain_pred = data["chain_pred"]
    len_ob_gt = int(data["len_ob_gt"])
    len_ob_pred = int(data["len_ob_pred"])
    paired = data["ob_record_paired"]
    paired_count = len(paired)

    union_count = len_ob_gt + len_ob_pred - paired_count
    acc_cat = float(
        len(chain_gt) >= 1
        and len(chain_pred) >= 2
        and capitalize_first_letter(chain_gt[0])
        == capitalize_first_letter(chain_pred[1])
    )
    acc_diag = float(
        bool(chain_gt)
        and bool(chain_pred)
        and capitalize_first_letter(chain_gt[-1])
        == capitalize_first_letter(chain_pred[-1])
    )

    rationale_and_diagnosis_matches = 0
    for value in paired.values():
        if value[0] is None or value[1] is None:
            continue
        diagnosis_match = capitalize_first_letter(value[0]) == capitalize_first_letter(
            value[1]
        )
        if diagnosis_match and is_yes(value[-1], rationale_response_mode):
            rationale_and_diagnosis_matches += 1

    return {
        "acc_cat": acc_cat,
        "acc_diag": acc_diag,
        "comp_pre": paired_count / (len_ob_pred + 1),
        "comp_re": paired_count / (len_ob_gt + 1),
        "comp_coverage": paired_count / union_count if union_count else 0.0,
        "faith_ob": (
            rationale_and_diagnosis_matches / paired_count if paired_count else 0.0
        ),
        "faith_all": (
            rationale_and_diagnosis_matches / union_count if union_count else 0.0
        ),
        "ob_precision_unsmoothed": paired_count / len_ob_pred if len_ob_pred else 0.0,
        "ob_recall_unsmoothed": paired_count / len_ob_gt if len_ob_gt else 0.0,
    }


def zero_score() -> dict[str, float]:
    return {
        **{metric: 0.0 for metric in OFFICIAL_METRICS},
        "ob_precision_unsmoothed": 0.0,
        "ob_recall_unsmoothed": 0.0,
    }


def mirrored_json_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    }


def aggregate(scores: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not scores:
        return {}
    output: dict[str, dict[str, float]] = {}
    for metric in scores[0]:
        values = [score[metric] for score in scores]
        output[metric] = {
            "mean": statistics.fmean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument(
        "--rationale-response-mode",
        choices=("official", "strip-casefold"),
        default="official",
    )
    args = parser.parse_args()

    prediction_files = mirrored_json_files(args.prediction_root)
    eval_files = mirrored_json_files(args.eval_root)
    if not prediction_files:
        raise ValueError(f"No prediction JSON files under {args.prediction_root}")

    scores: list[dict[str, float]] = []
    invalid = 0
    missing = 0
    for relative_path in sorted(prediction_files):
        eval_path = eval_files.get(relative_path)
        if eval_path is None:
            missing += 1
            scores.append(zero_score())
            continue
        try:
            data = json.loads(eval_path.read_text(encoding="utf-8"))
            score = score_record(data, args.rationale_response_mode)
            if not all(math.isfinite(value) for value in score.values()):
                raise ValueError("Non-finite metric")
            scores.append(score)
        except Exception:
            invalid += 1
            scores.append(zero_score())

    metrics = aggregate(scores)
    result = {
        "population": {
            "expected_predictions": len(prediction_files),
            "eval_files": len(eval_files),
            "missing_eval": missing,
            "invalid_eval": invalid,
            "zero_scored": missing + invalid,
        },
        "rationale_response_mode": args.rationale_response_mode,
        "metrics": metrics,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    aliases = {
        "acc_cat": "Acccat",
        "acc_diag": "Accdiag",
        "comp_pre": "Obspre (official +1)",
        "comp_re": "Obsrec (official +1)",
        "comp_coverage": "Obscomp",
        "faith_ob": "Expcom",
        "faith_all": "Expall",
        "ob_precision_unsmoothed": "Observation precision (unsmoothed)",
        "ob_recall_unsmoothed": "Observation recall (unsmoothed)",
    }
    lines = [
        "# DiReCT Official Evaluator Aggregate",
        "",
        "Aggregate-only output. Missing or invalid eval files receive zero, matching the official",
        "statistics pipeline's effective behavior.",
        "",
        f"- expected predictions: **{len(prediction_files)}**",
        f"- eval files found: **{len(eval_files)}**",
        f"- missing / invalid / zero-scored: **{missing} / {invalid} / {missing + invalid}**",
        f"- rationale response mode: `{args.rationale_response_mode}`",
        "",
        "| implementation field | presentation alias | mean | std | n |",
        "|---|---|---:|---:|---:|",
    ]
    for metric, values in metrics.items():
        lines.append(
            f"| `{metric}` | {aliases[metric]} | {values['mean']:.4f} | "
            f"{values['std']:.4f} | {values['n']} |"
        )
    lines.extend(
        [
            "",
            "`comp_pre` and `comp_re` use the official `+1` denominators. The unsmoothed",
            "rows are sensitivity metrics and must not be reported as official DiReCT scores.",
            "",
        ]
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    args.summary_md.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"[score] expected={len(prediction_files)} eval={len(eval_files)} "
        f"missing={missing} invalid={invalid}"
    )
    print(f"[score] summary={args.summary_md}")


if __name__ == "__main__":
    main()
