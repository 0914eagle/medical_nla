#!/usr/bin/env bash
set -euo pipefail

# Fit train-only finding/value heads and select their settings on validation.
# There is intentionally no official-test path in this wrapper.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPU="${GPU:-0}"
TRAIN_ROOT="${TRAIN_ROOT:?Set merged train activation root}"
VALIDATION_ROOT="${VALIDATION_ROOT:?Set merged E5 validation activation root}"
VALIDATION_HARD_PAIRS="${VALIDATION_HARD_PAIRS:?Set E5 validation hard-shuffle JSONL}"
OUT_DIR="${OUT_DIR:?Set probe artifact output directory}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -d "${TRAIN_ROOT}" || { echo "[error] missing ${TRAIN_ROOT}" >&2; exit 2; }
test -d "${VALIDATION_ROOT}" || { echo "[error] missing ${VALIDATION_ROOT}" >&2; exit 2; }
test -s "${VALIDATION_HARD_PAIRS}" || {
  echo "[error] missing ${VALIDATION_HARD_PAIRS}" >&2
  exit 2
}
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/train_ddxplus_finding_value_probes.py \
  --train-root "${TRAIN_ROOT}" \
  --validation-root "${VALIDATION_ROOT}" \
  --validation-hard-pairs "${VALIDATION_HARD_PAIRS}" \
  --out-dir "${OUT_DIR}" \
  --layers 16 24 32 \
  --min-finding-train-count "${MIN_FINDING_TRAIN_COUNT:-20}" \
  --min-value-train-count "${MIN_VALUE_TRAIN_COUNT:-10}" \
  --epochs "${EPOCHS:-80}" \
  --patience "${PATIENCE:-8}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --seed "${SEED:-17}"

cat "${OUT_DIR}/summary.md"
