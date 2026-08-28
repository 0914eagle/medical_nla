#!/usr/bin/env bash
set -euo pipefail

# Unattended validation-only semantic evaluation of the common-schema pilot.
# This never reads a locked test split and never launches text patching.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data1/heejae on server 125}"
GPU="${GPU:-0}"
COMMON_RUN_NAME="${COMMON_RUN_NAME:-common_medical_nla_pilot_v1}"
EVAL_RUN_NAME="${EVAL_RUN_NAME:-${COMMON_RUN_NAME}_direct_semantic_val_v1}"
READOUT_METHODS="${READOUT_METHODS:-vanilla medical_nla_seed17 medical_nla_seed29 medical_nla_seed43}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

READOUTS="${DATA_ROOT}/restricted/direct/e4/${COMMON_RUN_NAME}_validation_v1"
for method in ${READOUT_METHODS}; do
  path="${READOUTS}/${method}.jsonl"
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
  rows="$(wc -l < "${path}")"
  if [[ "${rows}" -ne 100 ]]; then
    echo "[error] ${method} has ${rows} rows; expected 100" >&2
    exit 2
  fi
done

read -r -a semantic_methods <<< "${READOUT_METHODS}"
echo "[queue] Direct semantic validation: 50 cases x CoT plus ${#semantic_methods[@]} NLA methods"
DATA_ROOT="${DATA_ROOT}" \
GPU="${GPU}" \
RUN_NAME="${EVAL_RUN_NAME}" \
READOUTS_DIR="${READOUTS}" \
READOUT_SOURCE_DATASET=direct \
READOUT_METHODS="${READOUT_METHODS}" \
EXTRACTOR_BACKEND=codex \
bash scripts/run_direct_e4_validation_evaluator.sh

ROOT="${DATA_ROOT}/restricted/direct/e4/${EVAL_RUN_NAME}"
echo "[done] ${ROOT}"
cat "${ROOT}/extraction_summary.md"
for method in cot ${READOUT_METHODS}; do
  echo "===== ${method} ====="
  cat "${ROOT}/reports/${method}.md"
done
