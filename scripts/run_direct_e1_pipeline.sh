#!/usr/bin/env bash
set -euo pipefail

# Private DiReCT E1 pipeline: source CoT -> exact transcript -> P0/P1/P2 states.
# Start with LIMIT=10. Set LIMIT=0 only after inspecting the smoke outputs.

CODE_ROOT="${CODE_ROOT:-/home/eagle0914/medical_nla}"
DATA_ROOT="${DATA_ROOT:-${MEDICAL_NLA_DATA_ROOT:-/data/heejae}}"
PRIVATE_ROOT="${PRIVATE_ROOT:-${DATA_ROOT}/restricted/direct}"
SPLIT_DIR="${SPLIT_DIR:-${PRIVATE_ROOT}/splits/direct_patient_pdd_v1}"
CONFIG="${CONFIG:-configs/default.yaml}"
GPUS="${GPUS:-2,3}"
LIMIT="${LIMIT:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
LAYERS="${LAYERS:-16 24 32}"
FORCE="${FORCE:-0}"

if [[ "${LIMIT}" == "0" ]]; then
  RUN_NAME="${RUN_NAME:-direct_e1_full_v1}"
else
  RUN_NAME="${RUN_NAME:-direct_e1_smoke${LIMIT}_v1}"
fi
RUN_ROOT="${RUN_ROOT:-${PRIVATE_ROOT}/e1/${RUN_NAME}}"
CASES="${RUN_ROOT}/cases.jsonl"
SOURCE_ANSWERS="${RUN_ROOT}/source_cot_answers.jsonl"
SOURCE_SUMMARY="${RUN_ROOT}/source_cot_summary.json"
POSITION_ROWS="${RUN_ROOT}/activation_rows.jsonl"
ACTIVATION_DIR="${RUN_ROOT}/activations"
REPORT_DIR="${RUN_ROOT}/reports"

cd "${CODE_ROOT}"
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
export PYTHONPATH="${CODE_ROOT}"
export MEDICAL_NLA_DATA_ROOT="${DATA_ROOT}"
export HF_HOME="${DATA_ROOT}/hf_cache"
unset TRANSFORMERS_CACHE
export CUDA_VISIBLE_DEVICES="${GPUS}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${RUN_ROOT}" "${REPORT_DIR}" "${DATA_ROOT}/medical_nla/logs"
chmod -R go-rwx "${PRIVATE_ROOT}"

for path in "${SPLIT_DIR}" "${DATA_ROOT}/hf_cache/hub/models--google--gemma-3-12b-it"; do
  if [[ ! -e "${path}" ]]; then
    echo "[error] missing required path: ${path}" >&2
    exit 1
  fi
done

LIMIT_ARGS=()
if [[ "${LIMIT}" != "0" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}" --sample-seed 17)
fi

echo "[stage 1/4] build private DiReCT source cases"
python scripts/make_direct_e1_cases.py \
  --split-dir "${SPLIT_DIR}" \
  --output-jsonl "${CASES}" \
  --summary-md "${REPORT_DIR}/cases_summary.md"

if [[ "${FORCE}" == "1" || ! -s "${SOURCE_ANSWERS}" ]]; then
  echo "[stage 2/4] generate source CoT and diagnosis on visible GPUs ${GPUS}"
  python scripts/run_source_answers.py \
    --config "${CONFIG}" \
    --cases "${CASES}" \
    --condition cot \
    --no-force-answer \
    --batch-size "${BATCH_SIZE}" \
    --output-jsonl "${SOURCE_ANSWERS}" \
    --summary-json "${SOURCE_SUMMARY}" \
    "${LIMIT_ARGS[@]}"
else
  echo "[stage 2/4] reuse existing source answers: ${SOURCE_ANSWERS}"
fi

echo "[stage 3/4] build teacher-forced P0/P1/P2 extraction rows"
python scripts/make_direct_transcript_activation_rows.py \
  --source-answers "${SOURCE_ANSWERS}" \
  --output-jsonl "${POSITION_ROWS}" \
  --summary-md "${REPORT_DIR}/activation_rows_summary.md"

RESUME_ARG=--resume
if [[ "${FORCE}" == "1" ]]; then
  RESUME_ARG=--no-resume
fi

echo "[stage 4/4] extract layers ${LAYERS} at P0/P1/P2"
# shellcheck disable=SC2086
python -m src.extract_activations \
  --config "${CONFIG}" \
  --input "${POSITION_ROWS}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${ACTIVATION_DIR}" \
  --layers ${LAYERS} \
  --batch-size "${BATCH_SIZE}" \
  "${RESUME_ARG}"

echo "[done] private E1 run: ${RUN_ROOT}"
echo "[safe summary] ${REPORT_DIR}/cases_summary.md"
echo "[safe summary] ${REPORT_DIR}/activation_rows_summary.md"
