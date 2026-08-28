#!/usr/bin/env bash
set -euo pipefail

# Server 125 owns /data1/heejae and exposes four RTX 4090s. The trainer uses
# model parallelism rather than data parallelism, so two independent 2-GPU
# workers provide more throughput than spreading one 12B LoRA run over all four.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
RUN_NAME="${RUN_NAME:-common_medical_nla_full_sft_v1}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_VALIDATION="${RUN_VALIDATION:-1}"
RUN_GROUNDING="${RUN_GROUNDING:-1}"
GROUNDING_MANIFEST="${GROUNDING_MANIFEST:-${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl}"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
mkdir -p "${LOG_ROOT}"

DATA_ROOT="${DATA_ROOT}" RUN_NAME="${RUN_NAME}" PREPARE_ONLY=1 \
  bash scripts/run_common_medical_nla_full_sft.sh

for pair in 0,1 2,3; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

run_worker() {
  local seed="$1"
  local gpus="$2"
  local train_log="${LOG_ROOT}/${RUN_NAME}_seed${seed}_train.log"
  local val_log="${LOG_ROOT}/${RUN_NAME}_seed${seed}_validation.log"
  local grounding_log="${LOG_ROOT}/${RUN_NAME}_seed${seed}_grounding.log"

  DATA_ROOT="${DATA_ROOT}" GPUS="${gpus}" SEEDS="${seed}" \
    EPOCHS="${EPOCHS}" MAX_STEPS="${MAX_STEPS}" \
    RUN_NAME="${RUN_NAME}" PREPARE_ONLY=0 \
    bash scripts/run_common_medical_nla_full_sft.sh >"${train_log}" 2>&1

  if [[ "${RUN_VALIDATION}" == "1" ]]; then
    DATA_ROOT="${DATA_ROOT}" GPUS="${gpus}" SEEDS="${seed}" \
      RUN_NAME="${RUN_NAME}" RUN_VANILLA=0 \
      bash scripts/run_common_medical_nla_validation.sh >"${val_log}" 2>&1
  fi

  if [[ "${RUN_GROUNDING}" == "1" ]]; then
    test -s "${GROUNDING_MANIFEST}" || {
      echo "[error] missing server-local grounding manifest ${GROUNDING_MANIFEST}" >&2
      return 2
    }
    SOURCE_MANIFEST="${GROUNDING_MANIFEST}" DATA_ROOT="${DATA_ROOT}" \
      GPUS="${gpus}" COMMON_RUN_NAME="${RUN_NAME}" SEEDS="${seed}" \
      GROUNDING_RUN_NAME="${RUN_NAME}_seed${seed}_ddx_grounding_val_v1" \
      bash scripts/run_overnight_common_ddx_grounding.sh >"${grounding_log}" 2>&1
  fi
}

echo "[launch] seed17 on GPUs 0,1"
run_worker 17 0,1 &
pid17=$!
echo "[launch] seed29 on GPUs 2,3"
run_worker 29 2,3 &
pid29=$!

status17=0
status29=0
wait "${pid17}" || status17=$?
wait "${pid29}" || status29=$?
echo "[workers] seed17=${status17} seed29=${status29}"
if [[ "${status17}" -ne 0 || "${status29}" -ne 0 ]]; then
  exit 1
fi
echo "[done] four-GPU full-data queue completed"
