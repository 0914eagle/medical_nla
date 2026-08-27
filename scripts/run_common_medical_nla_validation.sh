#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-17}"
RUN_VANILLA="${RUN_VANILLA:-0}"
RUN_NAME="${RUN_NAME:-common_medical_nla_pilot_v1}"
EXPECTED_ROWS="${EXPECTED_ROWS:-100}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E3="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
VAL="${E3}/dataset/sft_val.jsonl"
OUT="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_validation_v1"
PROMPT="prompt_templates/common_p0_clinical_state_readout.txt"
mkdir -p "${OUT}" "${DATA_ROOT}/medical_nla/logs"

test -s "${VAL}" || { echo "[error] missing ${VAL}" >&2; exit 2; }
rows="$(wc -l < "${VAL}")"
if [[ "${rows}" -ne "${EXPECTED_ROWS}" ]]; then
  echo "[error] validation population=${rows}; expected ${EXPECTED_ROWS}" >&2
  exit 2
fi

run_readout() {
  local label="$1"
  local adapter="$2"
  local output="${OUT}/${label}.jsonl"
  local scored="${OUT}/${label}_scored.jsonl"
  local summary="${OUT}/${label}_summary.md"
  if [[ ! -s "${output}" ]] || [[ "$(wc -l < "${output}")" -ne "${EXPECTED_ROWS}" ]]; then
    local adapter_args=()
    if [[ -n "${adapter}" ]]; then
      test -s "${adapter}/best.json" || { echo "[error] incomplete ${adapter}" >&2; exit 2; }
      adapter_args=(--adapter-id "${adapter}")
    fi
    echo "[readout] ${label} rows=${EXPECTED_ROWS} GPUs=${GPUS}"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.run_nla \
      --config configs/default.yaml \
      --manifest "${VAL}" \
      --output "${output}" \
      --actor-prompt-template-file "${PROMPT}" \
      "${adapter_args[@]}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --batch-size "${BATCH_SIZE}"
  else
    echo "[skip] ${label}: complete output"
  fi
  python scripts/score_medical_nla_v2_readouts.py \
    --input "${output}" \
    --output-jsonl "${scored}" \
    --summary-md "${summary}"
  cat "${summary}"
}

if [[ "${RUN_VANILLA}" == "1" ]]; then
  run_readout vanilla ""
fi

for seed in ${SEEDS}; do
  adapter="${E3}/adapters/${RUN_NAME}_seed${seed}"
  run_readout "medical_nla_seed${seed}" "${adapter}"
done

echo "[done] validation readouts under ${OUT}"
