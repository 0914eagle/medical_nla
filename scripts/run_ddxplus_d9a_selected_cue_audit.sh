#!/usr/bin/env bash
set -euo pipefail

# Read-only D9a audit on server 125. This trains two small cross-fitted linear
# probes and writes selected changed-cue scores; it does not build SFT targets.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:?Set PROBE_ARTIFACT to finding_value_hs32.pt}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this wrapper is frozen for server 125 (/data1/heejae)" >&2
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
OUT_DIR="${OUT_DIR:-${CF_ROOT}/d9a_selected_changed_cue_audit_v1}"

for path in "${ORIGINAL_MANIFEST}" "${COUNTERFACTUAL_MANIFEST}" "${PROBE_ARTIFACT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT_DIR}"

CUDA_VISIBLE_DEVICES="${GPU}" python scripts/score_ddxplus_selected_changed_cues.py \
  --original-manifest "${ORIGINAL_MANIFEST}" \
  --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
  --probe-artifact "${PROBE_ARTIFACT}" \
  --output-jsonl "${OUT_DIR}/private_scores.jsonl" \
  --output-json "${OUT_DIR}/report.json" \
  --summary-md "${OUT_DIR}/summary.md" \
  --max-donors "${MAX_DONORS:-5}" \
  --min-fold-positive-count "${MIN_FOLD_POSITIVE_COUNT:-5}" \
  --batch-size "${BATCH_SIZE:-512}" \
  --seed "${SEED:-17}"

cat "${OUT_DIR}/summary.md"
