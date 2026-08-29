#!/usr/bin/env bash
set -euo pipefail

# Staged D9a pipeline. MODE=audit is read-only. MODE=select requires an
# explicit post-audit grid and writes an unapproved recommendation. MODE=build
# requires a separately human-approved protocol. No mode opens locked test.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
MODE="${MODE:-audit}"
PROBE_ARTIFACT="${PROBE_ARTIFACT:?Set PROBE_ARTIFACT to finding_value_hs32.pt}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D9a is frozen to server 125 (/data1/heejae)" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
CF_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
E5_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
ORIGINAL_MANIFEST="${ORIGINAL_MANIFEST:-${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl}"
COUNTERFACTUAL_MANIFEST="${COUNTERFACTUAL_MANIFEST:-${CF_ROOT}/activations/ddxplus_counterfactual_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl}"
VALIDATION_MANIFEST="${VALIDATION_MANIFEST:-${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl}"
ROOT="${OUT_ROOT:-${CF_ROOT}/d9a_selected_changed_cue_v1}"
TRAIN_AUDIT="${ROOT}/train_audit"
VAL_AUDIT="${ROOT}/validation_null_audit"
CUTS="${ROOT}/cut_selection"
PAIRS="${ROOT}/approved_pairs"
mkdir -p "${ROOT}"

for path in "${ORIGINAL_MANIFEST}" "${COUNTERFACTUAL_MANIFEST}" \
  "${VALIDATION_MANIFEST}" "${PROBE_ARTIFACT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

case "${MODE}" in
  audit)
    mkdir -p "${TRAIN_AUDIT}" "${VAL_AUDIT}"
    echo "[stage 1/2] official-train OOF selected changed-cue audit"
    CUDA_VISIBLE_DEVICES="${GPU}" python scripts/score_ddxplus_selected_changed_cues.py \
      --original-manifest "${ORIGINAL_MANIFEST}" \
      --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
      --probe-artifact "${PROBE_ARTIFACT}" \
      --output-jsonl "${TRAIN_AUDIT}/private_scores.jsonl" \
      --output-json "${TRAIN_AUDIT}/report.json" \
      --summary-md "${TRAIN_AUDIT}/summary.md" \
      --max-donors 5 \
      --min-fold-positive-count 5 \
      --batch-size "${BATCH_SIZE:-512}" \
      --seed 17

    echo "[stage 2/2] validation positive/null read-only audit"
    CUDA_VISIBLE_DEVICES="${GPU}" python scripts/score_ddxplus_validation_changed_cues.py \
      --manifest "${VALIDATION_MANIFEST}" \
      --probe-artifact "${PROBE_ARTIFACT}" \
      --output-jsonl "${VAL_AUDIT}/private_scores.jsonl" \
      --output-json "${VAL_AUDIT}/report.json" \
      --summary-md "${VAL_AUDIT}/summary.md" \
      --max-donors 5

    cat "${TRAIN_AUDIT}/summary.md"
    cat "${VAL_AUDIT}/summary.md"
    echo "[stop] inspect both distributions before supplying a threshold grid"
    ;;
  select)
    : "${PRESENCE_THRESHOLDS:?Set explicit space-separated PRESENCE_THRESHOLDS after audit review}"
    : "${DELETION_THRESHOLDS:?Set explicit space-separated DELETION_THRESHOLDS after audit review}"
    : "${DONOR_THRESHOLDS:?Set explicit space-separated DONOR_THRESHOLDS after audit review}"
    test -s "${VAL_AUDIT}/private_scores.jsonl" || {
      echo "[error] run MODE=audit first" >&2
      exit 2
    }
    mkdir -p "${CUTS}"
    # Intentional word splitting turns each reviewed space-separated grid into
    # separate argparse values. No candidate values are embedded in this file.
    # shellcheck disable=SC2086
    python scripts/select_ddxplus_d9a_support_thresholds.py \
      --validation-scores "${VAL_AUDIT}/private_scores.jsonl" \
      --presence-thresholds ${PRESENCE_THRESHOLDS} \
      --deletion-thresholds ${DELETION_THRESHOLDS} \
      --donor-thresholds ${DONOR_THRESHOLDS} \
      --max-false-support-rate 0.05 \
      --min-fold-positive-count 5 \
      --candidates-jsonl "${CUTS}/private_candidates.jsonl" \
      --recommendation-json "${CUTS}/recommendation_unapproved.json" \
      --summary-md "${CUTS}/summary.md"
    cat "${CUTS}/summary.md"
    echo "[stop] recommendation is unapproved; do not build targets yet"
    ;;
  build)
    APPROVED_PROTOCOL="${APPROVED_PROTOCOL:?Set APPROVED_PROTOCOL after explicit human approval}"
    test -s "${TRAIN_AUDIT}/private_scores.jsonl" || {
      echo "[error] run MODE=audit first" >&2
      exit 2
    }
    test -s "${APPROVED_PROTOCOL}" || {
      echo "[error] missing approved protocol ${APPROVED_PROTOCOL}" >&2
      exit 2
    }
    mkdir -p "${PAIRS}"
    python scripts/make_ddxplus_d9a_supported_pairs.py \
      --train-scores "${TRAIN_AUDIT}/private_scores.jsonl" \
      --validation-scores "${VAL_AUDIT}/private_scores.jsonl" \
      --original-manifest "${ORIGINAL_MANIFEST}" \
      --counterfactual-manifest "${COUNTERFACTUAL_MANIFEST}" \
      --approved-protocol "${APPROVED_PROTOCOL}" \
      --output-jsonl "${PAIRS}/pairs_train.jsonl" \
      --protocol-json "${PAIRS}/protocol.json" \
      --summary-md "${PAIRS}/summary.md"
    cat "${PAIRS}/summary.md"
    echo "[done] D9a pairs rebuilt with the approved D10 retained-cue control"
    echo "[next] bash scripts/run_ddxplus_d10_1x2_smoke_4gpu_125.sh"
    ;;
  *)
    echo "[error] MODE must be audit, select, or build" >&2
    exit 2
    ;;
esac
