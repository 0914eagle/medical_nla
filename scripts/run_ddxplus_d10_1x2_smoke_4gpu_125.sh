#!/usr/bin/env bash
set -euo pipefail

# Human-approved D10 mechanism smoke. Each seed launches original-only and
# ranking arms concurrently on two 4090 pairs. Locked DDXPlus test is never read.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
MAX_STEPS="${MAX_STEPS:-20}"
SEEDS="${SEEDS:-17 29 43}"
RUN_NAME="${RUN_NAME:-ddxplus_d10_1x2_smoke20_v1}"
INIT_ADAPTER_TEMPLATE="${INIT_ADAPTER_TEMPLATE:-}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D10 is frozen to server 125 (/data1/heejae)" >&2
  exit 2
fi
if [[ "${SEEDS}" != "17 29 43" ]]; then
  echo "[error] D10 freezes seeds to exactly: 17 29 43" >&2
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
D9A="${CF_ROOT}/d9a_selected_changed_cue_v1"
TRAIN_SCORES="${D9A}/train_audit/private_scores.jsonl"
VAL_SCORES="${D9A}/validation_null_audit/private_scores.jsonl"
APPROVED="${D9A}/cut_selection/protocol_approved.json"
TRAIN_MANIFEST="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
CF_MANIFEST="${CF_ROOT}/activations/ddxplus_counterfactual_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
VAL_MANIFEST="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
TRAIN_PAIRS="${D9A}/approved_pairs/pairs_train.jsonl"
VAL_PAIRS="${D9A}/approved_pairs/pairs_validation.jsonl"
ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
EVAL="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_validation_v1"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs"
mkdir -p "${D9A}/approved_pairs" "${ROOT}" "${EVAL}" "${LOG_ROOT}"

for path in "${TRAIN_SCORES}" "${VAL_SCORES}" "${APPROVED}" \
  "${TRAIN_MANIFEST}" "${CF_MANIFEST}" "${VAL_MANIFEST}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

echo "[stage 1/4] rebuild approved train pairs with frozen retained-cue controls"
python scripts/make_ddxplus_d9a_supported_pairs.py \
  --train-scores "${TRAIN_SCORES}" \
  --validation-scores "${VAL_SCORES}" \
  --original-manifest "${TRAIN_MANIFEST}" \
  --counterfactual-manifest "${CF_MANIFEST}" \
  --approved-protocol "${APPROVED}" \
  --output-jsonl "${TRAIN_PAIRS}" \
  --protocol-json "${D9A}/approved_pairs/protocol.json" \
  --summary-md "${D9A}/approved_pairs/summary.md"

echo "[stage 2/4] build frozen validation pairs"
python scripts/make_ddxplus_d10_validation_pairs.py \
  --validation-scores "${VAL_SCORES}" \
  --validation-manifest "${VAL_MANIFEST}" \
  --approved-protocol "${APPROVED}" \
  --output-jsonl "${VAL_PAIRS}" \
  --report-json "${D9A}/approved_pairs/validation_protocol.json" \
  --summary-md "${D9A}/approved_pairs/validation_summary.md"

for pair in 0,1 2,3; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

run_arm() {
  local seed="$1"
  local arm="$2"
  local ranking_weight="$3"
  local gpus="$4"
  local adapter="${ROOT}/${arm}_seed${seed}"
  local score_prefix="${EVAL}/${arm}_seed${seed}"
  local train_log="${LOG_ROOT}/${RUN_NAME}_${arm}_seed${seed}_train.log"
  local eval_log="${LOG_ROOT}/${RUN_NAME}_${arm}_seed${seed}_specificity.log"
  local init_args=()

  if [[ -n "${INIT_ADAPTER_TEMPLATE}" ]]; then
    local init_adapter="${INIT_ADAPTER_TEMPLATE//\{seed\}/${seed}}"
    test -s "${init_adapter}/best.json" || {
      echo "[error] missing init adapter ${init_adapter}" >&2
      return 2
    }
    init_args=(--init-adapter "${init_adapter}")
  fi

  if [[ ! -s "${adapter}/best.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpus}" python scripts/train_ddxplus_d10_1x2.py \
      --config configs/default.yaml \
      --train-jsonl "${TRAIN_PAIRS}" \
      --out-dir "${adapter}" \
      "${init_args[@]}" \
      --max-steps "${MAX_STEPS}" \
      --grad-accum-steps 4 \
      --ranking-weight "${ranking_weight}" \
      --temperature 1.0 \
      --lr 2e-4 \
      --seed "${seed}" \
      >"${train_log}" 2>&1
  else
    echo "[resume] adapter exists: ${adapter}"
  fi

  if [[ ! -s "${score_prefix}.json" ]]; then
    CUDA_VISIBLE_DEVICES="${gpus}" python scripts/evaluate_ddxplus_d10_specificity.py \
      --config configs/default.yaml \
      --pairs "${VAL_PAIRS}" \
      --adapter "${adapter}" \
      --output-jsonl "${score_prefix}_private_scores.jsonl" \
      --output-json "${score_prefix}.json" \
      --summary-md "${score_prefix}_summary.md" \
      --batch-size 4 \
      --seed "${seed}" \
      >"${eval_log}" 2>&1
  else
    echo "[resume] evaluation exists: ${score_prefix}.json"
  fi
}

echo "[stage 3/4] train and evaluate matched arms, one seed per wave"
for seed in ${SEEDS}; do
  echo "[launch] seed ${seed}: control GPUs 0,1; ranking GPUs 2,3"
  run_arm "${seed}" original_only 0.0 0,1 &
  pid_control=$!
  run_arm "${seed}" ranking 1.0 2,3 &
  pid_ranking=$!
  status_control=0
  status_ranking=0
  wait "${pid_control}" || status_control=$?
  wait "${pid_ranking}" || status_ranking=$?
  echo "[workers] seed=${seed} control=${status_control} ranking=${status_ranking}"
  if [[ "${status_control}" -ne 0 || "${status_ranking}" -ne 0 ]]; then
    exit 1
  fi
done

echo "[stage 4/4] aggregate frozen teacher-forced gates"
python scripts/summarize_ddxplus_d10_arms.py \
  --arm "17=control=${EVAL}/original_only_seed17_private_scores.jsonl" \
  --arm "17=ranking=${EVAL}/ranking_seed17_private_scores.jsonl" \
  --arm "29=control=${EVAL}/original_only_seed29_private_scores.jsonl" \
  --arm "29=ranking=${EVAL}/ranking_seed29_private_scores.jsonl" \
  --arm "43=control=${EVAL}/original_only_seed43_private_scores.jsonl" \
  --arm "43=ranking=${EVAL}/ranking_seed43_private_scores.jsonl" \
  --output-json "${EVAL}/paired_arm_comparison.json" \
  --summary-md "${EVAL}/paired_arm_comparison_summary.md"

cat "${EVAL}/paired_arm_comparison_summary.md"
echo "[done] ${RUN_NAME}"
