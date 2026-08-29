#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPU="${GPU:-0}"
MODE="${MODE:-validation}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

TRAIN="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
PROBE_DIR="${DATA_ROOT}/medical_nla/results/ddxplus_finding_value_probe_val_v1"
ARTIFACT="${ARTIFACT:-${PROBE_DIR}/finding_value_hs24.pt}"
PROTOCOL="${PROTOCOL:-${PROBE_DIR}/structured_reader_hs24_protocol_v1.json}"

test -s "${ARTIFACT}" || { echo "[error] missing ${ARTIFACT}" >&2; exit 2; }
test -s "${TRAIN}/cases_train.jsonl" || { echo "[error] missing train cases" >&2; exit 2; }

if [[ ! -s "${PROTOCOL}" ]]; then
  python scripts/run_ddxplus_structured_reader.py freeze \
    --artifact "${ARTIFACT}" \
    --train-cases "${TRAIN}/cases_train.jsonl" \
    --output "${PROTOCOL}" \
    --expected-layer 24
fi

if [[ "${MODE}" == "validation" ]]; then
  MANIFEST="${MANIFEST:-${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer24/last_token/manifest.jsonl}"
  HARD_PAIRS="${HARD_PAIRS:-${E5}/hard_shuffle_pairs_validation.jsonl}"
  OUT_DIR="${OUT_DIR:-${DATA_ROOT}/medical_nla/results/ddxplus_structured_reader_validation_v1}"
  POPULATION=validation
  CONFIRMATION_ARGS=()
  VALIDATION_ARGS=()
elif [[ "${MODE}" == "locked_test" ]]; then
  MANIFEST="${MANIFEST:-${E5}/activations/ddxplus_e5_test_cot_p0_hs24_merged_v1/layer24/last_token/manifest.jsonl}"
  HARD_PAIRS="${HARD_PAIRS:-${E5}/hard_shuffle_pairs_test.jsonl}"
  OUT_DIR="${OUT_DIR:-${DATA_ROOT}/medical_nla/results/ddxplus_structured_reader_locked_test_v1}"
  POPULATION=locked_test
  CONFIRMATION_ARGS=(--confirmation I_ACCEPT_DDXPLUS_STRUCTURED_READER_LOCKED_TEST)
  VALIDATION_RESULTS="${VALIDATION_RESULTS:-${DATA_ROOT}/medical_nla/results/ddxplus_structured_reader_validation_v1/results.json}"
  test -s "${VALIDATION_RESULTS}" || { echo "[error] missing validation receipt ${VALIDATION_RESULTS}" >&2; exit 2; }
  VALIDATION_ARGS=(--validation-results "${VALIDATION_RESULTS}")
else
  echo "[error] MODE must be validation or locked_test" >&2
  exit 2
fi

test -s "${MANIFEST}" || { echo "[error] missing ${MANIFEST}" >&2; exit 2; }
test -s "${HARD_PAIRS}" || { echo "[error] missing ${HARD_PAIRS}" >&2; exit 2; }

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/run_ddxplus_structured_reader.py evaluate \
  --protocol "${PROTOCOL}" \
  --artifact "${ARTIFACT}" \
  --manifest "${MANIFEST}" \
  --hard-pairs "${HARD_PAIRS}" \
  --out-dir "${OUT_DIR}" \
  --population "${POPULATION}" \
  "${VALIDATION_ARGS[@]}" \
  "${CONFIRMATION_ARGS[@]}"
