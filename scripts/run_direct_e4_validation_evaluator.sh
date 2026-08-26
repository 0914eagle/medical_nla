#!/usr/bin/env bash
set -euo pipefail

# Validation-only E4 extraction and official evaluation. All generated artifacts
# contain private clinical text and stay under restricted/direct/e4.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae on server 62}"
GPU="${GPU:-2}"
RUN_NAME="${RUN_NAME:-validation_full_v1}"
LIMIT_CASES="${LIMIT_CASES:-0}"
EXPECTED_CASES="${EXPECTED_CASES:-50}"
OVERWRITE_EVAL="${OVERWRITE_EVAL:-0}"
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-codex}"
EXTRACTOR_MODEL="${EXTRACTOR_MODEL:-}"
CODEX_CMD="${CODEX_CMD:-codex}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

if [[ "${DATA_ROOT}" != "/data/heejae" ]]; then
  echo "[error] gather all E4 validation outputs on server 62 under /data/heejae" >&2
  exit 2
fi

DIRECT="${DATA_ROOT}/restricted/direct"
OFFICIAL="${DIRECT}/official_repo"
SPLITS="${DIRECT}/splits/direct_patient_pdd_confirmatory_v1"
E1="${DIRECT}/e1"
E3="${DIRECT}/e3/direct_e3_sft_v1"
READOUTS="${DIRECT}/e4/validation_readouts_v1"
OUT="${DIRECT}/e4/${RUN_NAME}"
JUDGE="${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original"

mkdir -p "${OUT}" "${DATA_ROOT}/medical_nla/logs"

for path in \
  "${E3}/sft_val.jsonl" \
  "${SPLITS}/val_seen.jsonl" \
  "${DIRECT}/manifests/direct_canonical_v3_private.jsonl" \
  "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl" \
  "${E1}/direct_e1_test_v1/source_cot_answers.jsonl" \
  "${READOUTS}/vanilla.jsonl" \
  "${READOUTS}/medical_nla_seed17.jsonl" \
  "${READOUTS}/medical_nla_seed29.jsonl" \
  "${READOUTS}/medical_nla_seed43.jsonl" \
  "${JUDGE}/consolidated.00.pth" \
  "${JUDGE}/tokenizer.model"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

limit_args=()
effective_cases="${EXPECTED_CASES}"
if [[ "${LIMIT_CASES}" -gt 0 ]]; then
  limit_args=(--limit-cases "${LIMIT_CASES}")
  effective_cases="${LIMIT_CASES}"
fi

echo "[stage 1/5] build ${effective_cases}-case x 5-method extraction requests"
python scripts/make_direct_e4_claim_requests.py \
  --cohort "${E3}/sft_val.jsonl" \
  --case-manifest "${SPLITS}/val_seen.jsonl" \
  --candidate-manifest "${DIRECT}/manifests/direct_canonical_v3_private.jsonl" \
  --source-answers \
    "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl" \
    "${E1}/direct_e1_test_v1/source_cot_answers.jsonl" \
  --readout "vanilla=${READOUTS}/vanilla.jsonl" \
  --readout "medical_nla_seed17=${READOUTS}/medical_nla_seed17.jsonl" \
  --readout "medical_nla_seed29=${READOUTS}/medical_nla_seed29.jsonl" \
  --readout "medical_nla_seed43=${READOUTS}/medical_nla_seed43.jsonl" \
  --requests "${OUT}/extraction_requests.jsonl" \
  --private-index "${OUT}/private_index.jsonl" \
  --summary-md "${OUT}/requests_summary.md" \
  --expected-cases "${EXPECTED_CASES}" \
  "${limit_args[@]}"

echo "[stage 2/5] quote-constrained extraction via ${EXTRACTOR_BACKEND}"
if [[ "${EXTRACTOR_BACKEND}" == "codex" ]]; then
  command -v "${CODEX_CMD%% *}" >/dev/null || {
    echo "[error] Codex command not found: ${CODEX_CMD}" >&2
    exit 2
  }
  model_args=()
  if [[ -n "${EXTRACTOR_MODEL}" ]]; then
    model_args=(--model "${EXTRACTOR_MODEL}")
  fi
  python scripts/run_judge.py \
    --requests "${OUT}/extraction_requests.jsonl" \
    --out "${OUT}/extraction_judgements.jsonl" \
    --backend codex \
    --codex-cmd "${CODEX_CMD}" \
    --timeout 300 \
    "${model_args[@]}"
elif [[ "${EXTRACTOR_BACKEND}" == "local_llama" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
    scripts/run_direct_local_llama_judge.py \
    --requests "${OUT}/extraction_requests.jsonl" \
    --out "${OUT}/extraction_judgements.jsonl" \
    --official-repo "${OFFICIAL}" \
    --ckpt-dir "${JUDGE}" \
    --tokenizer-path "${JUDGE}/tokenizer.model" \
    --max-seq-len 8192 \
    --max-batch-size 1 \
    --max-gen-len 768 \
    --temperature 0 \
    --top-p 1
else
  echo "[error] EXTRACTOR_BACKEND must be codex or local_llama" >&2
  exit 2
fi

echo "[stage 3/5] validate quotes and create official-schema predictions"
python scripts/apply_direct_e4_claim_extractions.py \
  --private-index "${OUT}/private_index.jsonl" \
  --judgements "${OUT}/extraction_judgements.jsonl" \
  --candidate-manifest "${DIRECT}/manifests/direct_canonical_v3_private.jsonl" \
  --prediction-root "${OUT}/predictions" \
  --audit-jsonl "${OUT}/private_extraction_audit.jsonl" \
  --summary-md "${OUT}/extraction_summary.md" \
  --expected-cases "${effective_cases}"

echo "[stage 4/5] official semantic matching"
for method in cot vanilla medical_nla_seed17 medical_nla_seed29 medical_nla_seed43; do
  overwrite_args=()
  if [[ "${OVERWRITE_EVAL}" == "1" ]]; then
    overwrite_args=(--overwrite)
  fi
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
    scripts/run_direct_official_evaluator.py \
    --official-repo "${OFFICIAL}" \
    --samples-root "${DIRECT}/samples" \
    --prediction-root "${OUT}/predictions/${method}" \
    --eval-root "${OUT}/evaluations/${method}" \
    --ckpt-dir "${JUDGE}" \
    --tokenizer-path "${JUDGE}/tokenizer.model" \
    --max-seq-len 8192 \
    --max-batch-size 4 \
    --temperature 0 \
    --top-p 1 \
    --response-mode official \
    --error-jsonl "${OUT}/private_errors_${method}.jsonl" \
    "${overwrite_args[@]}"
done

echo "[stage 5/5] aggregate official metrics"
for method in cot vanilla medical_nla_seed17 medical_nla_seed29 medical_nla_seed43; do
  python scripts/score_direct_official_eval.py \
    --prediction-root "${OUT}/predictions/${method}" \
    --eval-root "${OUT}/evaluations/${method}" \
    --output-json "${OUT}/reports/${method}.json" \
    --summary-md "${OUT}/reports/${method}.md"
done

echo "[done] ${OUT}"
echo "[first] cat ${OUT}/extraction_summary.md"
echo "[metrics] cat ${OUT}/reports/{cot,vanilla,medical_nla_seed17,medical_nla_seed29,medical_nla_seed43}.md"
