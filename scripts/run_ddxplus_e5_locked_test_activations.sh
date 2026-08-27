#!/usr/bin/env bash
set -euo pipefail

# Extract locked E5 test activations only after a validation-selected artifact
# exists. The artifact is a gate, not an input to backbone extraction.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
INPUT_FILE="${INPUT_FILE:?Set one base_id-sharded E5 test JSONL}"
RUN_NAME="${RUN_NAME:?Set a unique locked-test run name}"
OUT_DIR="${OUT_DIR:?Set the shard activation output directory}"
FROZEN_ARTIFACT="${FROZEN_ARTIFACT:?Set the validation-selected probe artifact}"
BATCH_SIZE="${BATCH_SIZE:-1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${INPUT_FILE}" || { echo "[error] missing ${INPUT_FILE}" >&2; exit 2; }
test -s "${FROZEN_ARTIFACT}" || {
  echo "[error] no frozen validation artifact: ${FROZEN_ARTIFACT}" >&2
  exit 2
}
mkdir -p "${OUT_DIR}" "${DATA_ROOT}/medical_nla/logs"

LAYER=$(python - "${FROZEN_ARTIFACT}" <<'PY'
import sys
import torch

artifact = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
print(int(artifact["layer"]))
PY
)
test "${LAYER}" = "24" || {
  echo "[error] expected frozen HS24 from validation, got HS${LAYER}" >&2
  exit 2
}

echo "[population] locked test rows=$(wc -l < "${INPUT_FILE}")"
echo "[freeze] HS${LAYER} artifact=${FROZEN_ARTIFACT}"
CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.extract_activations \
  --config configs/default.yaml \
  --input "${INPUT_FILE}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --layers "${LAYER}" \
  --batch-size "${BATCH_SIZE}" \
  --resume

MANIFEST="${OUT_DIR}/layer${LAYER}/last_token/manifest.jsonl"
test -s "${MANIFEST}" || { echo "[error] missing ${MANIFEST}" >&2; exit 2; }
echo "[manifest] HS${LAYER} $(wc -l < "${MANIFEST}") rows"
echo "[done] ${OUT_DIR}"
