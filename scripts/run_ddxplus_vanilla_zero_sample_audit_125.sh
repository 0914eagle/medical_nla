#!/usr/bin/env bash
set -euo pipefail

# Post-hoc private error analysis only. This never rewrites the frozen score.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
AUDITOR_MODEL="${AUDITOR_MODEL:-gpt-5.4}"
CODEX_CMD="${CODEX_CMD:-codex}"
CASES="${CASES:-20}"
SEED="${SEED:-17}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_vanilla_locked_semantic_v1/private_sample20_audit_v1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

MANIFEST="${MANIFEST:-${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_test_cot_p0_hs32_merged_v1/layer32/last_token/manifest.jsonl}"
READOUTS="${READOUTS:-${DATA_ROOT}/medical_nla/results/ddxplus_vanilla_locked_generation_v1/vanilla_locked.jsonl}"

for path in "${MANIFEST}" "${READOUTS}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}"

echo "[stage 1/3] prepare ${CASES}-case diagnosis-stratified private audit"
python scripts/audit_ddxplus_vanilla_zero_sample.py prepare \
  --manifest "${MANIFEST}" \
  --readouts "${READOUTS}" \
  --out-dir "${OUT}" \
  --cases "${CASES}" \
  --seed "${SEED}"

echo "[stage 2/3] independent quote-constrained audit via ${AUDITOR_MODEL}"
python scripts/run_judge.py \
  --requests "${OUT}/requests.jsonl" \
  --out "${OUT}/judgements.jsonl" \
  --backend codex \
  --model "${AUDITOR_MODEL}" \
  --codex-cmd "${CODEX_CMD}" \
  --timeout 300

echo "[stage 3/3] validate quotes and write aggregate summary"
python scripts/audit_ddxplus_vanilla_zero_sample.py finalize \
  --private-index "${OUT}/private_index.jsonl" \
  --judgements "${OUT}/judgements.jsonl" \
  --out-dir "${OUT}"

echo "[private] case-level text remains under ${OUT}/private_case_audit.jsonl"
echo "[done] ${OUT}"
