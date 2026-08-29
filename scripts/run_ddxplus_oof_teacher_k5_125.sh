#!/usr/bin/env bash
set -euo pipefail

# One-shot D14 K=5 OOF teacher and preregistered calibration gate. K=2 output
# remains archived. This queue does not build student targets or read
# validation/locked test.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:?Set PROBE_ARTIFACT to finding_value_hs32.pt}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D14 K=5 queue is frozen to server 125 (/data1/heejae)" >&2
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
OUT="${OUT_DIR:-${CF_ROOT}/oof_finding_teacher_hs32_k5_v2}"
AUDIT="${OUT}/calibration_audit_v1"

for path in "${ORIGINAL_MANIFEST}" "${COUNTERFACTUAL_MANIFEST}" "${PROBE_ARTIFACT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}" "${AUDIT}"

echo "[stage 1/2] one-shot K=5 full-label OOF teacher"
CUDA_VISIBLE_DEVICES="${GPU}" python scripts/materialize_ddxplus_oof_finding_teacher.py \
  --original-manifest "${ORIGINAL_MANIFEST}" \
  --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
  --probe-artifact "${PROBE_ARTIFACT}" \
  --output-jsonl "${OUT}/private_teacher_scores.jsonl" \
  --output-json "${OUT}/report.json" \
  --summary-md "${OUT}/summary.md" \
  --num-folds 5 \
  --batch-size "${BATCH_SIZE:-512}" \
  --seed 17

rows="$(wc -l < "${OUT}/private_teacher_scores.jsonl")"
if [[ "${rows}" -ne 9310 ]]; then
  echo "[error] expected 9310 teacher rows, got ${rows}" >&2
  exit 2
fi

echo "[stage 2/2] CPU calibration audit and frozen gate"
python scripts/audit_ddxplus_oof_teacher_calibration.py \
  --teacher-jsonl "${OUT}/private_teacher_scores.jsonl" \
  --teacher-report "${OUT}/report.json" \
  --original-manifest "${ORIGINAL_MANIFEST}" \
  --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
  --probe-artifact "${PROBE_ARTIFACT}" \
  --output-json "${AUDIT}/report.json" \
  --label-prevalence-jsonl "${AUDIT}/private_label_prevalence.jsonl" \
  --summary-md "${AUDIT}/summary.md" \
  --device cpu

cat "${AUDIT}/summary.md"
echo "[done] ${OUT}"
echo "[stop] inspect the K=5 gate before target building"
echo "[rule] failure forbids further K/threshold sweeps; pass still requires P2-P4 approval"
