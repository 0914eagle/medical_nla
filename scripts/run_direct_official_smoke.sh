#!/usr/bin/env bash
set -euo pipefail

# Gold-oracle smoke test for the official DiReCT semantic evaluator.
# Prepare private oracle files without a model:
#   PREPARE_ONLY=1 bash scripts/run_direct_official_smoke.sh
# Run after Meta-Llama-3-8B-Instruct/original has been downloaded:
#   GPU=0 nohup bash scripts/run_direct_official_smoke.sh > "$ART/logs/direct_eval_smoke.log" 2>&1 &

CODE_ROOT="${CODE_ROOT:-/home/eagle0914/medical_nla}"
DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
PRIVATE_ROOT="${PRIVATE_ROOT:-${DATA_ROOT}/restricted/direct}"
OFFICIAL_REPO="${OFFICIAL_REPO:-${PRIVATE_ROOT}/official_repo}"
SAMPLES_ROOT="${SAMPLES_ROOT:-${PRIVATE_ROOT}/samples}"
MANIFEST="${MANIFEST:-${PRIVATE_ROOT}/manifests/direct_canonical_v2_private.jsonl}"
MODEL_ROOT="${MODEL_ROOT:-${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-10}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"
FORCE="${FORCE:-0}"

RUN_ROOT="${RUN_ROOT:-${PRIVATE_ROOT}/evaluator_smoke/oracle_${LIMIT}}"
PREDICTION_ROOT="${RUN_ROOT}/predictions"
EVAL_ROOT="${RUN_ROOT}/evaluations"
REPORT_ROOT="${RUN_ROOT}/reports"
ERROR_JSONL="${RUN_ROOT}/private_errors.jsonl"

cd "${CODE_ROOT}"
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
export PYTHONPATH="${CODE_ROOT}"
mkdir -p "${PREDICTION_ROOT}" "${EVAL_ROOT}" "${REPORT_ROOT}" "${DATA_ROOT}/medical_nla/logs"

for path in "${OFFICIAL_REPO}" "${SAMPLES_ROOT}" "${MANIFEST}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[error] missing required path: ${path}" >&2
    exit 1
  fi
done

echo "[stage 1/3] build ${LIMIT}-row official-schema oracle predictions"
python scripts/make_direct_oracle_predictions.py \
  --manifest "${MANIFEST}" \
  --official-repo "${OFFICIAL_REPO}" \
  --samples-root "${SAMPLES_ROOT}" \
  --output-root "${PREDICTION_ROOT}" \
  --limit "${LIMIT}" \
  --summary-md "${REPORT_ROOT}/oracle_summary.md"

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "[done] PREPARE_ONLY=1; oracle files are ready under ${PREDICTION_ROOT}"
  exit 0
fi

for filename in consolidated.00.pth params.json tokenizer.model; do
  if [[ ! -s "${MODEL_ROOT}/${filename}" ]]; then
    echo "[error] missing judge model file: ${MODEL_ROOT}/${filename}" >&2
    echo "Download: hf download meta-llama/Meta-Llama-3-8B-Instruct --include 'original/*' --local-dir ${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct" >&2
    exit 1
  fi
done

if ! python -c 'import fairscale, tiktoken' >/dev/null 2>&1; then
  echo "[error] missing native Llama evaluator dependencies: fairscale and/or tiktoken" >&2
  echo "Install: uv pip install 'fairscale>=0.4.13' 'tiktoken>=0.7'" >&2
  exit 1
fi

echo "[stage 2/3] run official Llama-3-8B semantic matching on GPU ${GPU}"
EVAL_ARGS=(
  --official-repo "${OFFICIAL_REPO}"
  --samples-root "${SAMPLES_ROOT}"
  --prediction-root "${PREDICTION_ROOT}"
  --eval-root "${EVAL_ROOT}"
  --ckpt-dir "${MODEL_ROOT}"
  --tokenizer-path "${MODEL_ROOT}/tokenizer.model"
  --response-mode official
  --limit "${LIMIT}"
  --error-jsonl "${ERROR_JSONL}"
)
if [[ "${FORCE}" == "1" ]]; then
  EVAL_ARGS+=(--overwrite)
fi
CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
  scripts/run_direct_official_evaluator.py "${EVAL_ARGS[@]}"

echo "[stage 3/3] aggregate official statistics.py-compatible metrics"
python scripts/score_direct_official_eval.py \
  --prediction-root "${PREDICTION_ROOT}" \
  --eval-root "${EVAL_ROOT}" \
  --output-json "${REPORT_ROOT}/official_metrics.json" \
  --summary-md "${REPORT_ROOT}/official_metrics_summary.md"

echo "[done] smoke test completed"
echo "[summary] ${REPORT_ROOT}/official_metrics_summary.md"
cat "${REPORT_ROOT}/official_metrics_summary.md"
