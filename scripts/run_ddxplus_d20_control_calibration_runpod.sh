#!/usr/bin/env bash
set -euo pipefail

# Read-only D20 calibration. It reuses the completed D10 controls, generates a
# fixed validation pilot for claim-count spread, writes an unapproved gate
# recommendation, and stops. It never trains D20 or reads locked test data.

DATA_ROOT="/data1/heejae"
REPO_DIR="${REPO_DIR:-/home/eagle0914/medical_nla}"
CONFIG="${CONFIG:-configs/runpod.yaml}"
CONTROL_RUN_NAME="${CONTROL_RUN_NAME:-ddxplus_d10_budget1552_v1}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
BATCH_SIZE="${BATCH_SIZE:-4}"

cd "${REPO_DIR}"
if [[ -f "${DATA_ROOT}/uv/medical_nla/bin/activate" ]]; then
  source "${DATA_ROOT}/uv/medical_nla/bin/activate"
fi
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH="${REPO_DIR}"

E5_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
VAL_MANIFEST="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
CONTROL_ROOT="${DATA_ROOT}/restricted/direct/e3/${CONTROL_RUN_NAME}"
CONTROL_EVAL="${DATA_ROOT}/restricted/direct/e4/${CONTROL_RUN_NAME}_validation_v1/step001552"
OUT="${DATA_ROOT}/restricted/direct/e4/ddxplus_d20_control_calibration_v1"
SHARDS="${OUT}/manifest_shards"
PILOT="${OUT}/paired_validation_pilot.jsonl"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/ddxplus_d20_control_calibration_v1"
mkdir -p "${OUT}" "${LOG_ROOT}"

test -s "${CONFIG}" || { echo "[error] missing ${CONFIG}" >&2; exit 2; }
test -s "${VAL_MANIFEST}" || { echo "[error] missing ${VAL_MANIFEST}" >&2; exit 2; }
for seed in 17 29 43; do
  test -s "${CONTROL_ROOT}/original_only_seed${seed}/checkpoint-step001552/adapter_config.json" || {
    echo "[error] missing D10 control checkpoint seed ${seed}" >&2; exit 2;
  }
  test -s "${CONTROL_EVAL}/original_only_seed${seed}_private_scores.jsonl" || {
    echo "[error] missing D10 control scores seed ${seed}" >&2; exit 2;
  }
done

if [[ ! -s "${PILOT}" ]]; then
  python scripts/shard_jsonl_by_key.py \
    --input "${VAL_MANIFEST}" --out-dir "${SHARDS}" --num-shards 40 --key base_id
  merge_args=()
  expected=0
  for shard in 0 1 2 3; do
    printf -v path '%s/shard_%03d_of_040.jsonl' "${SHARDS}" "${shard}"
    merge_args+=(--input "${path}")
    expected=$((expected + $(wc -l < "${path}")))
  done
  python scripts/merge_jsonl_files.py "${merge_args[@]}" \
    --output "${PILOT}" --expected-rows "${expected}"
fi
pilot_rows="$(wc -l < "${PILOT}")"
echo "[population] validation pilot rows=${pilot_rows}"

generate_control() {
  local seed="$1" gpu="$2"
  local adapter="${CONTROL_ROOT}/original_only_seed${seed}/checkpoint-step001552"
  local output="${OUT}/control_seed${seed}.jsonl"
  local log="${LOG_ROOT}/control_seed${seed}_generation.log"
  if [[ -s "${output}" && "$(wc -l < "${output}")" -eq "${pilot_rows}" ]]; then
    echo "[skip] control seed ${seed} generation"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python -m src.run_nla \
    --config "${CONFIG}" \
    --manifest "${PILOT}" \
    --output "${output}" \
    --adapter-id "${adapter}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" >"${log}" 2>&1
  test "$(wc -l < "${output}")" -eq "${pilot_rows}"
}

echo "[stage 1/2] generate three frozen D10 controls"
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  generate_control 17 "${GPU_A}"
  generate_control 29 "${GPU_A}"
else
  generate_control 17 "${GPU_A}" & p17=$!
  generate_control 29 "${GPU_B}" & p29=$!
  s17=0; s29=0
  wait "${p17}" || s17=$?
  wait "${p29}" || s29=$?
  if [[ "${s17}" -ne 0 || "${s29}" -ne 0 ]]; then
    echo "[error] control generation seed17=${s17} seed29=${s29}" >&2
    exit 1
  fi
fi
generate_control 43 "${GPU_A}"

echo "[stage 2/2] write unapproved spread recommendation"
python scripts/audit_ddxplus_d20_control_spread.py \
  --control-score "17=${CONTROL_EVAL}/original_only_seed17_private_scores.jsonl" \
  --control-score "29=${CONTROL_EVAL}/original_only_seed29_private_scores.jsonl" \
  --control-score "43=${CONTROL_EVAL}/original_only_seed43_private_scores.jsonl" \
  --control-readout "17=${OUT}/control_seed17.jsonl" \
  --control-readout "29=${OUT}/control_seed29.jsonl" \
  --control-readout "43=${OUT}/control_seed43.jsonl" \
  --output-json "${OUT}/recommendation_unapproved.json" \
  --summary-md "${OUT}/summary.md"

echo "[stop] paste ${OUT}/summary.md; D20 training remains locked"
