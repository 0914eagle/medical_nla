#!/usr/bin/env bash
set -euo pipefail

# Extract only the development-side DDXPlus E5 activations. The locked official
# test split is deliberately absent from this wrapper.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
CONDITION="${CONDITION:-cot}"
LAYERS="${LAYERS:-16 24 32}"
BATCH_SIZE="${BATCH_SIZE:-1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
case "${CONDITION}" in
  cot)
    DEFAULT_INPUT="${E5}/activation_rows_validation.jsonl"
    DEFAULT_RUN_NAME="ddxplus_e5_validation_cot_p0_v1"
    ;;
  direct)
    DEFAULT_INPUT="${E5}/activation_rows_validation_direct_control.jsonl"
    DEFAULT_RUN_NAME="ddxplus_e5_validation_direct_p0_v1"
    ;;
  *)
    echo "[error] CONDITION must be cot or direct" >&2
    exit 2
    ;;
esac
INPUT="${INPUT_FILE:-${DEFAULT_INPUT}}"
RUN_NAME="${RUN_NAME:-${DEFAULT_RUN_NAME}}"
OUT="${OUT_DIR:-${E5}/activations/${RUN_NAME}}"

test -s "${INPUT}" || { echo "[error] missing ${INPUT}" >&2; exit 2; }
mkdir -p "${OUT}" "${DATA_ROOT}/medical_nla/logs"

echo "[population] condition=${CONDITION} rows=$(wc -l < "${INPUT}")"
echo "[extract] layers=${LAYERS} GPUs=${GPUS} output=${OUT}"
CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.extract_activations \
  --config configs/default.yaml \
  --input "${INPUT}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT}" \
  --layers ${LAYERS} \
  --batch-size "${BATCH_SIZE}" \
  --resume

for layer in ${LAYERS}; do
  manifest="${OUT}/layer${layer}/last_token/manifest.jsonl"
  test -s "${manifest}" || { echo "[error] missing ${manifest}" >&2; exit 2; }
  echo "[manifest] HS${layer} $(wc -l < "${manifest}") rows"
done
echo "[done] ${OUT}"
