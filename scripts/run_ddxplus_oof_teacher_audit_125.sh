#!/usr/bin/env bash
set -euo pipefail

# D14 post-materialization CPU audit. No student target or validation/test row
# is read. GPU is not required.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:?Set PROBE_ARTIFACT to finding_value_hs32.pt}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D14 teacher audit is frozen to server 125 (/data1/heejae)" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
CF_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
ORIGINAL_MANIFEST="${ORIGINAL_MANIFEST:-${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl}"
COUNTERFACTUAL_MANIFEST="${COUNTERFACTUAL_MANIFEST:-${CF_ROOT}/activations/ddxplus_counterfactual_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl}"
TEACHER="${TEACHER_DIR:-${CF_ROOT}/oof_finding_teacher_hs32_v1}"
OUT="${OUT_DIR:-${TEACHER}/calibration_audit_v1}"

for path in "${ORIGINAL_MANIFEST}" "${COUNTERFACTUAL_MANIFEST}" \
  "${PROBE_ARTIFACT}" "${TEACHER}/private_teacher_scores.jsonl" \
  "${TEACHER}/report.json"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}"

python scripts/audit_ddxplus_oof_teacher_calibration.py \
  --teacher-jsonl "${TEACHER}/private_teacher_scores.jsonl" \
  --teacher-report "${TEACHER}/report.json" \
  --original-manifest "${ORIGINAL_MANIFEST}" \
  --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
  --probe-artifact "${PROBE_ARTIFACT}" \
  --output-json "${OUT}/report.json" \
  --label-prevalence-jsonl "${OUT}/private_label_prevalence.jsonl" \
  --summary-md "${OUT}/summary.md" \
  --device cpu

cat "${OUT}/summary.md"
echo "[done] ${OUT}"
echo "[stop] do not build targets until this calibration audit is reviewed"
