#!/usr/bin/env bash
set -euo pipefail

# Extract one complete base_id shard of the DDXPlus locked CoT-P0 population
# at HS32. HS32 is fixed by the public Vanilla AV checkpoint, not selected on
# the locked population.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the server-local data root}"
GPUS="${GPUS:?Set the two visible GPUs, for example 0,1}"
INPUT_FILE="${INPUT_FILE:?Set one base_id-complete locked JSONL shard}"
OUT_DIR="${OUT_DIR:?Set the activation output directory for this shard}"
RUN_NAME="${RUN_NAME:?Set a unique extraction run name}"
CONFIRMATION="${CONFIRMATION:-}"
BATCH_SIZE="${BATCH_SIZE:-1}"

if [[ "${CONFIRMATION}" != "I_ACCEPT_DDXPLUS_HS32_READOUT_EXTRACTION" ]]; then
  echo "[error] missing exact HS32 readout-extraction confirmation" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${INPUT_FILE}" || { echo "[error] missing ${INPUT_FILE}" >&2; exit 2; }
mkdir -p "${OUT_DIR}/provenance"
expected_rows="$(wc -l < "${INPUT_FILE}")"

python scripts/validate_ddxplus_locked_population.py \
  --input "${INPUT_FILE}" \
  --expected-rows "${expected_rows}" \
  --report "${OUT_DIR}/provenance/input_population.json"

CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.extract_activations \
  --config configs/default.yaml \
  --input "${INPUT_FILE}" \
  --run-name "${RUN_NAME}" \
  --output-dir "${OUT_DIR}" \
  --layers 32 \
  --batch-size "${BATCH_SIZE}" \
  --resume

manifest="${OUT_DIR}/layer32/last_token/manifest.jsonl"
test -s "${manifest}" || { echo "[error] missing ${manifest}" >&2; exit 2; }
python scripts/validate_ddxplus_locked_population.py \
  --input "${manifest}" \
  --expected-rows "${expected_rows}" \
  --expected-layer 32 \
  --require-activation-files \
  --report "${OUT_DIR}/provenance/output_population.json"

echo "[done] HS32 rows=${expected_rows} output=${OUT_DIR}"
