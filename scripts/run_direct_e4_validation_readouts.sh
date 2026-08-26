#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-17}"
RUN_VANILLA="${RUN_VANILLA:-0}"
EXPECTED_ROWS="${EXPECTED_ROWS:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E3="${DATA_ROOT}/restricted/direct/e3"
VAL="${E3}/direct_e3_sft_v1/sft_val.jsonl"
OUT="${DATA_ROOT}/restricted/direct/e4/validation_readouts_v1"
mkdir -p "${OUT}" "${DATA_ROOT}/medical_nla/logs"

test -s "${VAL}"
val_rows="$(wc -l < "${VAL}")"
if [[ "${val_rows}" -ne "${EXPECTED_ROWS}" ]]; then
  echo "[error] validation population=${val_rows}; expected ${EXPECTED_ROWS}" >&2
  exit 2
fi
echo "[population] validation=${val_rows}"

run_readout() {
  local label="$1"
  local adapter="$2"
  local output="${OUT}/${label}.jsonl"
  if [[ -s "${output}" ]] && [[ "$(wc -l < "${output}")" -eq "${EXPECTED_ROWS}" ]]; then
    echo "[skip] ${label}: complete ${EXPECTED_ROWS}-row output"
    return
  fi

  local adapter_args=()
  if [[ -n "${adapter}" ]]; then
    test -s "${adapter}/best.json"
    cat "${adapter}/best.json"
    adapter_args=(--adapter-id "${adapter}")
  fi

  echo "[readout] ${label} GPUs=${GPUS}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "${VAL}" \
    --output "${output}" \
    "${adapter_args[@]}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}"

  rows="$(wc -l < "${output}")"
  if [[ "${rows}" -ne "${EXPECTED_ROWS}" ]]; then
    echo "[error] ${label}: output=${rows}; expected ${EXPECTED_ROWS}" >&2
    exit 2
  fi
  echo "[done] ${label}: ${rows} rows"
}

if [[ "${RUN_VANILLA}" == "1" ]]; then
  run_readout "vanilla" ""
fi

for seed in ${SEEDS}; do
  adapter="${E3}/adapters/direct_e3_sft_v1_seed${seed}"
  run_readout "medical_nla_seed${seed}" "${adapter}"
done

echo "[done] validation readouts under ${OUT}"
