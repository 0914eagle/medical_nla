#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
LIMIT_PER_ARM="${LIMIT_PER_ARM:-20}"
RUN_NAME="${RUN_NAME:-medical_nla_d22_public_ar_geometry20_v1}"
OUT="${OUT:-${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}}"
MODE="${MODE:-all}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this validation/private wrapper is frozen for server 125" >&2
  exit 2
fi
if [[ "${MODE}" != "all" && "${MODE}" != "audit" ]]; then
  echo "[error] MODE must be all or audit" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
DDX_TRAIN_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
DIRECT="${DATA_ROOT}/restricted/direct"
DDX_VAL="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
DDX_TRAIN="${DDX_TRAIN_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
DIRECT_VAL="${DIRECT}/e3/direct_e3_sft_v1/sft_val.jsonl"
DIRECT_TRAIN="${DIRECT}/e3/direct_e3_sft_v1/sft_train.jsonl"

for path in "${DDX_VAL}" "${DDX_TRAIN}" "${DIRECT_VAL}" "${DIRECT_TRAIN}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

if [[ "${MODE}" == "all" ]]; then
  echo "[stage 1/2] reconstruct and persist the frozen 160-vector diagnostic"
  OUT="${OUT}" LIMIT_PER_ARM="${LIMIT_PER_ARM}" DATA_ROOT="${DATA_ROOT}" GPU="${GPU}" \
    MODE=all bash scripts/run_medical_nla_d22_ar_diagnostic_125.sh
else
  test -s "${OUT}/private_scores.jsonl" || {
    echo "[error] MODE=audit requires ${OUT}/private_scores.jsonl" >&2
    exit 2
  }
  echo "[stage 1/2] reuse persisted reconstruction vectors"
fi

echo "[stage 2/2] CPU A1-A5 geometry audit"
python scripts/audit_medical_nla_d22_geometry.py \
  --scores "${OUT}/private_scores.jsonl" \
  --ddx-validation-manifest "${DDX_VAL}" \
  --direct-validation-manifest "${DIRECT_VAL}" \
  --ddx-train-manifest "${DDX_TRAIN}" \
  --direct-train-manifest "${DIRECT_TRAIN}" \
  --path-map /data/heejae=/data1/heejae \
  --out-dir "${OUT}/geometry_audit" \
  --summary-md "${OUT}/geometry_audit/summary.md"

echo "[done] ${OUT}/geometry_audit"
