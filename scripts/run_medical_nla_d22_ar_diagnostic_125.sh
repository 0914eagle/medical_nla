#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
MODE="${MODE:-all}"
LIMIT_PER_ARM="${LIMIT_PER_ARM:-}"
OUT="${OUT:-${DATA_ROOT}/restricted/direct/e4/medical_nla_d22_public_ar_diagnostic_v1}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this wrapper is frozen for server 125 (/data1/heejae)" >&2
  exit 2
fi
if [[ "${MODE}" != "prepare" && "${MODE}" != "score" && "${MODE}" != "summarize" && "${MODE}" != "all" ]]; then
  echo "[error] MODE must be prepare, score, summarize, or all" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DATA_ROOT}/restricted/direct"
DIRECT_MANIFEST="${DIRECT}/e3/direct_e3_sft_v1/sft_val.jsonl"
DIRECT_BUNDLE="${DIRECT}/e4/sft_raw_audit50_v2/private_bundle.jsonl"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
DDX_MANIFEST="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
READER="${DATA_ROOT}/medical_nla/results/ddxplus_structured_reader_validation_v1/readouts.jsonl"
AR="${AR:-kitft/nla-gemma3-12b-L32-ar}"
NLA_INFERENCE="${NLA_INFERENCE:-${DATA_ROOT}/nla-inference}"

for path in "${DIRECT_MANIFEST}" "${DIRECT_BUNDLE}" "${DDX_MANIFEST}" "${READER}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
test -s "${NLA_INFERENCE}/nla_inference.py" || {
  echo "[error] missing ${NLA_INFERENCE}/nla_inference.py" >&2
  exit 2
}
mkdir -p "${OUT}"

if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
  limit_args=()
  if [[ -n "${LIMIT_PER_ARM}" ]]; then
    limit_args=(--limit-per-arm "${LIMIT_PER_ARM}")
  fi
  echo "[stage 1/3] validation-only private manifest and same-diagnosis donors"
  python scripts/audit_medical_nla_ar_roundtrip.py prepare \
    --direct-manifest "${DIRECT_MANIFEST}" \
    --direct-private-bundle "${DIRECT_BUNDLE}" \
    --ddx-manifest "${DDX_MANIFEST}" \
    --structured-reader "${READER}" \
    --path-map /data/heejae=/data1/heejae \
    "${limit_args[@]}" \
    --out-dir "${OUT}"
fi

if [[ "${MODE}" == "score" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/private_manifest.jsonl" || { echo "[error] prepare first" >&2; exit 2; }
  echo "[stage 2/3] released HS32 AR reconstruction on GPU ${GPU}"
  CUDA_VISIBLE_DEVICES="${GPU}" python scripts/audit_medical_nla_ar_roundtrip.py score \
    --manifest "${OUT}/private_manifest.jsonl" \
    --output "${OUT}/private_scores.jsonl" \
    --ar "${AR}" \
    --device cuda:0 \
    --dtype bfloat16 \
    --cache-dir "${HF_HOME}" \
    --nla-inference-path "${NLA_INFERENCE}"
fi

if [[ "${MODE}" == "summarize" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/private_scores.jsonl" || { echo "[error] score first" >&2; exit 2; }
  echo "[stage 3/3] aggregate matched-over-shuffled report"
  python scripts/audit_medical_nla_ar_roundtrip.py summarize \
    --scored "${OUT}/private_scores.jsonl" \
    --output-json "${OUT}/results.json" \
    --summary-md "${OUT}/summary.md"
fi

echo "[done] ${OUT}"
