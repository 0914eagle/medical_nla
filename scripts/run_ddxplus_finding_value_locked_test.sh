#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPU="${GPU:-0}"
ARTIFACT="${ARTIFACT:?Set frozen HS24 finding/value artifact}"
MANIFEST="${MANIFEST:?Set merged locked-test HS24 manifest}"
HARD_PAIRS="${HARD_PAIRS:?Set locked-test same-diagnosis hard pairs}"
OUT_DIR="${OUT_DIR:?Set locked-test result directory}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${ARTIFACT}" || { echo "[error] missing ${ARTIFACT}" >&2; exit 2; }
test -s "${MANIFEST}" || { echo "[error] missing ${MANIFEST}" >&2; exit 2; }
test -s "${HARD_PAIRS}" || { echo "[error] missing ${HARD_PAIRS}" >&2; exit 2; }
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/evaluate_ddxplus_finding_value_probes.py \
  --artifact "${ARTIFACT}" \
  --manifest "${MANIFEST}" \
  --hard-pairs "${HARD_PAIRS}" \
  --out-dir "${OUT_DIR}" \
  --bootstrap-replicates "${BOOTSTRAP_REPLICATES:-2000}" \
  --seed "${SEED:-17}"
