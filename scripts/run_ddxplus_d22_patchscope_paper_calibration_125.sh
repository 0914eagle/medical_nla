#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPUS="${GPUS:-2,3}"
CASES="${CASES:-5}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_d22_patchscope_paper_calibration5_v2}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this validation calibration is frozen for server 125" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
VALIDATION_ROOT="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1"
VALIDATION_MANIFEST="${VALIDATION_ROOT}/layer32/last_token/manifest.jsonl"
V1="${DATA_ROOT}/medical_nla/results/ddxplus_d22_patchscope50_v1"

for path in \
  "${VALIDATION_MANIFEST}" \
  "${V1}/population_protocol.json" \
  "${V1}/generation_manifest.jsonl"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

mkdir -p "${OUT}"
echo "[calibration] paper-style token identity, entity description, then ${CASES} clinical cases"
CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/calibrate_ddxplus_d22_patchscope.py \
  --config configs/default.yaml \
  --validation-manifest "${VALIDATION_MANIFEST}" \
  --v1-protocol "${V1}/population_protocol.json" \
  --v1-generation-manifest "${V1}/generation_manifest.jsonl" \
  --path-map /data/heejae=/data1/heejae \
  --cases "${CASES}" \
  --out-dir "${OUT}" \
  --summary-md "${OUT}/summary.md"
