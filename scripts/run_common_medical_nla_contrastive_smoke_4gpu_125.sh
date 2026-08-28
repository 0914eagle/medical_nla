#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
BASE_RUN="${BASE_RUN:-common_medical_nla_full_sft_v1}"
RUN_NAME="${RUN_NAME:-common_medical_nla_contrastive_smoke20_v2}"
MAX_STEPS="${MAX_STEPS:-20}"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs"
TRAIN="${DATA_ROOT}/restricted/direct/e3/${BASE_RUN}/dataset/sft_train.jsonl"
VAL="${DATA_ROOT}/restricted/direct/e3/${BASE_RUN}/dataset/sft_val.jsonl"
INIT="${DATA_ROOT}/restricted/direct/e3/${BASE_RUN}/adapters/${BASE_RUN}_seed29"
ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
ALIGN="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_alignment_val_v1"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla
mkdir -p "${ROOT}" "${ALIGN}" "${LOG_ROOT}"

for path in "${TRAIN}" "${VAL}" "${INIT}/best.json"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
for pair in 0,1 2,3; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

run_arm() {
  local label="$1"
  local weight="$2"
  local gpus="$3"
  local adapter="${ROOT}/${label}"
  local train_log="${LOG_ROOT}/${RUN_NAME}_${label}_train.log"
  local align_log="${LOG_ROOT}/${RUN_NAME}_${label}_alignment.log"
  test ! -e "${adapter}" || { echo "[error] output exists: ${adapter}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/train_medical_nla_contrastive.py \
    --config configs/default.yaml \
    --train-jsonl "${TRAIN}" \
    --init-adapter "${INIT}" \
    --out-dir "${adapter}" \
    --max-steps "${MAX_STEPS}" \
    --pairs-per-batch 1 \
    --grad-accum-steps 4 \
    --max-pairs-per-source 124 \
    --pair-loss-weight "${weight}" \
    --pair-temperature 0.1 \
    --lr 5e-5 \
    --seed 29 \
    >"${train_log}" 2>&1

  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/audit_medical_nla_target_alignment.py \
    --config configs/default.yaml \
    --manifest "${VAL}" \
    --adapter "${adapter}" \
    --output-jsonl "${ALIGN}/${label}_private_scores.jsonl" \
    --output-json "${ALIGN}/${label}.json" \
    --summary-md "${ALIGN}/${label}_summary.md" \
    --source-dataset direct \
    --batch-size 4 \
    --seed 17 \
    >"${align_log}" 2>&1
}

echo "[launch] lambda_0p1 on GPUs 0,1"
run_arm lambda_0p1 0.1 0,1 &
pid_low=$!
echo "[launch] lambda_1p0 on GPUs 2,3"
run_arm lambda_1p0 1.0 2,3 &
pid_high=$!

status_low=0
status_high=0
wait "${pid_low}" || status_low=$?
wait "${pid_high}" || status_high=$?
echo "[workers] lambda_0p1=${status_low} lambda_1p0=${status_high}"
if [[ "${status_low}" -ne 0 || "${status_high}" -ne 0 ]]; then
  exit 1
fi
cat "${ALIGN}/lambda_0p1_summary.md"
cat "${ALIGN}/lambda_1p0_summary.md"
echo "[done] ${ROOT}"
