#!/usr/bin/env bash
set -euo pipefail

# Extract train-only CoT-P0 activations for finding/value probe fitting.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
LAYERS="${LAYERS:-16 24 32}"
BATCH_SIZE="${BATCH_SIZE:-1}"
INPUT_FILE="${INPUT_FILE:?Set INPUT_FILE to one train activation-row JSONL shard}"
RUN_NAME="${RUN_NAME:?Set a unique RUN_NAME for this shard}"
OUT_DIR="${OUT_DIR:?Set OUT_DIR under the train-only probe population}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${INPUT_FILE}" || { echo "[error] missing ${INPUT_FILE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}" "${DATA_ROOT}/medical_nla/logs"

echo "[population] train-only rows=$(wc -l < "${INPUT_FILE}")"
echo "[extract] layers=${LAYERS} GPUs=${GPUS} output=${OUT_DIR}"
CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.extract_activations \
  --config configs/default.yaml \
  --input "${INPUT_FILE}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --layers ${LAYERS} \
  --batch-size "${BATCH_SIZE}" \
  --resume

for layer in ${LAYERS}; do
  manifest="${OUT_DIR}/layer${layer}/last_token/manifest.jsonl"
  test -s "${manifest}" || { echo "[error] missing ${manifest}" >&2; exit 2; }
  echo "[manifest] HS${layer} $(wc -l < "${manifest}") rows"
done
echo "[done] ${OUT_DIR}"
