#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPUS="${GPUS:-2,3}"
CASES="${CASES:-5}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_d22_patchscope_same_layer5_v1}"

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
TRAIN="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
VROOT="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1"
TROOT="${TRAIN}/activations/ddxplus_probe_train_cot_p0_merged_v1"
V1="${DATA_ROOT}/medical_nla/results/ddxplus_d22_patchscope50_v1"

required=("${V1}/population_protocol.json" "${V1}/generation_manifest.jsonl")
for layer in 16 24 32; do
  required+=("${VROOT}/layer${layer}/last_token/manifest.jsonl")
  required+=("${TROOT}/layer${layer}/last_token/manifest.jsonl")
done
for path in "${required[@]}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

mkdir -p "${OUT}"
echo "[same-layer sweep] controls select HS16/24/32 before ${CASES} clinical cases"
CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/calibrate_ddxplus_d22_patchscope_same_layer.py \
  --config configs/default.yaml \
  --validation-layer-manifest "16=${VROOT}/layer16/last_token/manifest.jsonl" \
  --validation-layer-manifest "24=${VROOT}/layer24/last_token/manifest.jsonl" \
  --validation-layer-manifest "32=${VROOT}/layer32/last_token/manifest.jsonl" \
  --train-layer-manifest "16=${TROOT}/layer16/last_token/manifest.jsonl" \
  --train-layer-manifest "24=${TROOT}/layer24/last_token/manifest.jsonl" \
  --train-layer-manifest "32=${TROOT}/layer32/last_token/manifest.jsonl" \
  --v1-protocol "${V1}/population_protocol.json" \
  --v1-generation-manifest "${V1}/generation_manifest.jsonl" \
  --path-map /data/heejae=/data1/heejae \
  --cases "${CASES}" \
  --out-dir "${OUT}" \
  --summary-md "${OUT}/summary.md"
