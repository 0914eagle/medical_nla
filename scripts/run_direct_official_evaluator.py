"""Run the DiReCT semantic evaluator without its hard-coded GPU selection.

This preserves the official greedy matching and exact ``Yes`` decision rule,
while recording raw judge responses and failures for auditability. Inputs and
outputs contain restricted clinical text and must stay in private storage.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


def load_official_modules(official_repo: Path) -> tuple[Any, Any, Any, Any, Any]:
    repo = str(official_repo.resolve())
    if repo not in sys.path:
        sys.path.insert(0, repo)
    data_analysis = importlib.import_module("utils.data_analysis")
    data_extraction = importlib.import_module("utils.data_extraction")
    llama_module = importlib.import_module("llama")
    return (
        data_analysis.cal_a_json,
        data_analysis.deduction_assemble,
        data_extraction.discriminate_similarity_observation,
        data_extraction.discriminate_similarity_reason,
        llama_module.Llama,
    )


def judge_once(
    generator: Any,
    prompt: str,
    max_gen_len: int | None,
    temperature: float,
    top_p: float,
) -> str:
    results = generator.chat_completion(
        [[{"role": "user", "content": prompt}]],
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )
    return str(results[0]["generation"]["content"])


def is_yes(response: str, mode: str) -> bool:
    if mode == "official":
        return response == "Yes"
    if mode == "strip-casefold":
        return response.strip().rstrip(".").casefold() == "yes"
    raise ValueError(f"Unknown response mode: {mode}")


def evaluate_one(
    gold_path: Path,
    prediction_path: Path,
    generator: Any,
    cal_a_json: Any,
    deduction_assemble: Any,
    observation_prompt: Any,
    rationale_prompt: Any,
    max_gen_len: int | None,
    temperature: float,
    top_p: float,
    response_mode: str,
) -> dict[str, Any]:
    record_node, _, chain_gt = cal_a_json(str(gold_path))
    gold = deduction_assemble(record_node)
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    chain_pred = prediction.pop("chain")
    gold_observations = list(gold.keys())
    predicted_observations = list(prediction.keys())

    matched_prediction_indices: set[int] = set()
    observation_pairs: list[tuple[int, int]] = []
    observation_attempts: list[dict[str, Any]] = []
    for gold_index, gold_observation in enumerate(gold_observations):
        for prediction_index, predicted_observation in enumerate(predicted_observations):
            if prediction_index in matched_prediction_indices:
                continue
            prompt = observation_prompt(gold_observation, predicted_observation)
            response = judge_once(
                generator,
                prompt,
                max_gen_len,
                temperature,
                top_p,
            )
            accepted = is_yes(response, response_mode)
            observation_attempts.append(
                {
                    "gold_index": gold_index,
                    "prediction_index": prediction_index,
                    "response": response,
                    "accepted": accepted,
                }
            )
            if accepted:
                observation_pairs.append((gold_index, prediction_index))
                matched_prediction_indices.add(prediction_index)
                break

    chain_gt = list(reversed(chain_gt))
    record: dict[str, Any] = {
        "chain_gt": chain_gt,
        "chain_pred": chain_pred,
        "len_ob_gt": len(gold_observations),
        "len_ob_pred": len(predicted_observations),
        "ob_record_paired": {},
        "GT_observation": gold_observations,
        "predict_observation": predicted_observations,
        "observation_attempts": observation_attempts,
        "judge_response_mode": response_mode,
    }

    for gold_index, prediction_index in observation_pairs:
        gold_observation = gold_observations[gold_index]
        predicted_observation = predicted_observations[prediction_index]
        rationale_gold = gold[gold_observation][0]
        diagnosis_gold = gold[gold_observation][2]
        rationale_pred = prediction[predicted_observation][0]
        diagnosis_pred = prediction[predicted_observation][2]
        prompt = rationale_prompt(rationale_gold, rationale_pred)
        response = judge_once(
            generator,
            prompt,
            max_gen_len,
            temperature,
            top_p,
        )
        record["ob_record_paired"][str([gold_index, prediction_index])] = [
            diagnosis_gold,
            diagnosis_pred,
            rationale_gold,
            rationale_pred,
            response,
        ]
    return record


def mirrored_json_files(root: Path) -> dict[Path, Path]:
    return {
        path.relative_to(root): path
        for path in sorted(root.rglob("*.json"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--samples-root", required=True, type=Path)
    parser.add_argument("--prediction-root", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--ckpt-dir", required=True, type=Path)
    parser.add_argument("--tokenizer-path", required=True, type=Path)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--max-batch-size", type=int, default=4)
    parser.add_argument("--max-gen-len", type=int)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--response-mode",
        choices=("official", "strip-casefold"),
        default="official",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--error-jsonl", required=True, type=Path)
    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    for required_path in (
        args.official_repo,
        args.samples_root,
        args.prediction_root,
        args.ckpt_dir,
        args.tokenizer_path,
    ):
        if not required_path.exists():
            raise FileNotFoundError(required_path)

    (
        cal_a_json,
        deduction_assemble,
        observation_prompt,
        rationale_prompt,
        llama_class,
    ) = load_official_modules(args.official_repo)
    generator = llama_class.build(
        ckpt_dir=str(args.ckpt_dir),
        tokenizer_path=str(args.tokenizer_path),
        max_seq_len=args.max_seq_len,
        max_batch_size=args.max_batch_size,
    )

    gold_files = mirrored_json_files(args.samples_root)
    prediction_files = mirrored_json_files(args.prediction_root)
    shared_paths = sorted(gold_files.keys() & prediction_files.keys())
    if args.limit is not None:
        shared_paths = shared_paths[: args.limit]
    args.eval_root.mkdir(parents=True, exist_ok=True)
    args.error_jsonl.parent.mkdir(parents=True, exist_ok=True)

    processed = skipped = failed = 0
    errors: list[dict[str, str]] = []
    for index, relative_path in enumerate(shared_paths, start=1):
        eval_path = args.eval_root / relative_path
        if eval_path.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            record = evaluate_one(
                gold_files[relative_path],
                prediction_files[relative_path],
                generator,
                cal_a_json,
                deduction_assemble,
                observation_prompt,
                rationale_prompt,
                args.max_gen_len,
                args.temperature,
                args.top_p,
                args.response_mode,
            )
            eval_path.parent.mkdir(parents=True, exist_ok=True)
            eval_path.write_text(
                json.dumps(record, ensure_ascii=False),
                encoding="utf-8",
            )
            processed += 1
        except Exception as exc:  # Keep the full private audit trail.
            failed += 1
            errors.append(
                {
                    "relative_path": relative_path.as_posix(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
        print(
            f"[evaluate] {index}/{len(shared_paths)} "
            f"processed={processed} skipped={skipped} failed={failed}",
            flush=True,
        )

    with args.error_jsonl.open("w", encoding="utf-8") as handle:
        for error in errors:
            handle.write(json.dumps(error, ensure_ascii=False) + "\n")
    print(
        f"[done] expected={len(shared_paths)} processed={processed} "
        f"skipped={skipped} failed={failed} eval_root={args.eval_root}"
    )


if __name__ == "__main__":
    main()
