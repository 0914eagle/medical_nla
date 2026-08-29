#!/usr/bin/env bash
set -euo pipefail

# RunPod variant of run_ddxplus_d10_budget_4gpu_125.sh. The experiment is
# byte-identical (same trainer, data, frozen budget/checkpoints/seeds); only
# host guard, venv activation, and GPU placement differ. All six runs must
# execute on this same pod hardware; record the GPU model in the report.
#
#   Single 80GB GPU:  bash scripts/run_ddxplus_d10_budget_runpod.sh
#   Two GPUs:         GPUS_CONTROL=0 GPUS_RANKING=1 bash scripts/run_ddxplus_d10_budget_runpod.sh

DATA_ROOT="/data1/heejae"
MAX_STEPS="${MAX_STEPS:-1552}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-20 194 388 776 1164 1552}"
SEEDS="${SEEDS:-17 29 43}"
RUN_NAME="${RUN_NAME:-ddxplus_d10_budget1552_v1}"
INIT_ADAPTER_TEMPLATE="${INIT_ADAPTER_TEMPLATE:-}"
GPUS_CONTROL="${GPUS_CONTROL:-0}"
GPUS_RANKING="${GPUS_RANKING:-0}"
REPO_DIR="${REPO_DIR:-/home/eagle0914/medical_nla}"
CONFIG="${CONFIG:-configs/runpod.yaml}"

if [[ "${MAX_STEPS}" != "1552" || "${CHECKPOINT_STEPS}" != "20 194 388 776 1164 1552" ]]; then
  echo "[error] frozen budget is 1,552 with checkpoints 20 194 388 776 1164 1552" >&2
  exit 2
fi
if [[ "${SEEDS}" != "17 29 43" ]]; then
  echo "[error] frozen seeds are exactly 17 29 43" >&2
  exit 2
fi

cd "${REPO_DIR}"
test -s "${CONFIG}" || { echo "[error] missing ${CONFIG} - run setup_runpod_d10.sh first" >&2; exit 2; }
if [[ -f "${DATA_ROOT}/uv/medical_nla/bin/activate" ]]; then
  source "${DATA_ROOT}/uv/medical_nla/bin/activate"
fi
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH="${REPO_DIR}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee /tmp/d10_gpu_model.txt

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
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/${RUN_NAME}"
mkdir -p "${D9A}/approved_pairs" "${ROOT}" "${EVAL}" "${LOG_ROOT}"

for path in "${TRAIN_SCORES}" "${VAL_SCORES}" "${APPROVED}" \
  "${TRAIN_MANIFEST}" "${CF_MANIFEST}" "${VAL_MANIFEST}"; do
  test -s "${path}" || { echo "[error] missing ${path} (bundle extracted?)" >&2; exit 2; }
done

echo "[stage 1/5] rebuild frozen D9a train and validation pairs"
python scripts/make_ddxplus_d9a_supported_pairs.py \
  --train-scores "${TRAIN_SCORES}" \
  --validation-scores "${VAL_SCORES}" \
  --original-manifest "${TRAIN_MANIFEST}" \
  --counterfactual-manifest "${CF_MANIFEST}" \
  --approved-protocol "${APPROVED}" \
  --output-jsonl "${TRAIN_PAIRS}" \
  --protocol-json "${D9A}/approved_pairs/protocol.json" \
  --summary-md "${D9A}/approved_pairs/summary.md"

python scripts/make_ddxplus_d10_validation_pairs.py \
  --validation-scores "${VAL_SCORES}" \
  --validation-manifest "${VAL_MANIFEST}" \
  --approved-protocol "${APPROVED}" \
  --output-jsonl "${VAL_PAIRS}" \
  --report-json "${D9A}/approved_pairs/validation_protocol.json" \
  --summary-md "${D9A}/approved_pairs/validation_summary.md"

for gpus in $(printf '%s\n%s\n' "${GPUS_CONTROL}" "${GPUS_RANKING}" | sort -u); do
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/check_gpu_setup.py \
    --config "${CONFIG}" --require-free-gb 40
done

latest_resume_checkpoint() {
  local adapter="$1"
  local latest_state
  latest_state="$(find "${adapter}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/checkpoint-step*/trainer_state.pt' -print 2>/dev/null | sort | tail -n 1)"
  if [[ -n "${latest_state}" ]]; then
    dirname "${latest_state}"
  fi
}

train_arm() {
  local seed="$1"
  local arm="$2"
  local ranking_weight="$3"
  local gpus="$4"
  local adapter="${ROOT}/${arm}_seed${seed}"
  local train_log="${LOG_ROOT}/${arm}_seed${seed}_train.log"
  local init_args=()
  local resume_args=()

  if [[ -s "${adapter}/best.json" ]]; then
    echo "[resume] completed adapter: ${adapter}"
    return
  fi
  if [[ -n "${INIT_ADAPTER_TEMPLATE}" ]]; then
    local init_adapter="${INIT_ADAPTER_TEMPLATE//\{seed\}/${seed}}"
    test -s "${init_adapter}/best.json" || {
      echo "[error] missing init adapter ${init_adapter}" >&2
      return 2
    }
    init_args=(--init-adapter "${init_adapter}")
  fi
  if [[ -d "${adapter}" ]]; then
    local resume_checkpoint
    resume_checkpoint="$(latest_resume_checkpoint "${adapter}")"
    test -n "${resume_checkpoint}" || {
      echo "[error] partial adapter has no resumable checkpoint: ${adapter}" >&2
      return 2
    }
    resume_args=(--resume-from-checkpoint "${resume_checkpoint}")
    echo "[resume] ${arm} seed${seed} from ${resume_checkpoint}"
  fi

  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/train_ddxplus_d10_1x2.py \
    --config "${CONFIG}" \
    --train-jsonl "${TRAIN_PAIRS}" \
    --out-dir "${adapter}" \
    "${init_args[@]}" \
    "${resume_args[@]}" \
    --max-steps "${MAX_STEPS}" \
    --checkpoint-steps ${CHECKPOINT_STEPS} \
    --grad-accum-steps 4 \
    --ranking-weight "${ranking_weight}" \
    --temperature 1.0 \
    --lr 2e-4 \
    --seed "${seed}" \
    >>"${train_log}" 2>&1
}

run_pair() {
  # Runs control and ranking either in parallel (distinct GPUs) or
  # sequentially (same GPU) without changing anything else.
  local fn="$1"; shift
  local seed="$1"; shift
  if [[ "${GPUS_CONTROL}" == "${GPUS_RANKING}" ]]; then
    "${fn}" "${seed}" original_only "$@" "${GPUS_CONTROL}"
    "${fn}" "${seed}" ranking "$@" "${GPUS_RANKING}"
  else
    "${fn}" "${seed}" original_only "$@" "${GPUS_CONTROL}" &
    local pid_control=$!
    "${fn}" "${seed}" ranking "$@" "${GPUS_RANKING}" &
    local pid_ranking=$!
    local status_control=0 status_ranking=0
    wait "${pid_control}" || status_control=$?
    wait "${pid_ranking}" || status_ranking=$?
    if [[ "${status_control}" -ne 0 || "${status_ranking}" -ne 0 ]]; then
      echo "[error] arm pair failed (control=${status_control} ranking=${status_ranking})" >&2
      return 1
    fi
  fi
}

train_arm_dispatch() {
  local seed="$1" arm="$2" gpus="${!#}"
  case "${arm}" in
    original_only) train_arm "${seed}" original_only 0.0 "${gpus}" ;;
    ranking) train_arm "${seed}" ranking 1.0 "${gpus}" ;;
  esac
}

echo "[stage 2/5] train six runs"
for seed in ${SEEDS}; do
  echo "[launch] seed ${seed}: control GPU(${GPUS_CONTROL}); ranking GPU(${GPUS_RANKING})"
  run_pair train_arm_dispatch "${seed}"
done

evaluate_arm_checkpoint() {
  local seed="$1"
  local arm="$2"
  local step="$3"
  local gpus="${4}"
  local checkpoint="${ROOT}/${arm}_seed${seed}/checkpoint-step$(printf '%06d' "${step}")"
  local step_dir="${EVAL}/step$(printf '%06d' "${step}")"
  local prefix="${step_dir}/${arm}_seed${seed}"
  local eval_log="${LOG_ROOT}/${arm}_seed${seed}_step${step}_specificity.log"
  mkdir -p "${step_dir}"
  test -s "${checkpoint}/adapter_config.json" || {
    echo "[error] missing checkpoint adapter ${checkpoint}" >&2
    return 2
  }
  if [[ -s "${prefix}.json" ]]; then
    echo "[resume] completed evaluation: ${prefix}.json"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/evaluate_ddxplus_d10_specificity.py \
    --config "${CONFIG}" \
    --pairs "${VAL_PAIRS}" \
    --adapter "${checkpoint}" \
    --output-jsonl "${prefix}_private_scores.jsonl" \
    --output-json "${prefix}.json" \
    --summary-md "${prefix}_summary.md" \
    --batch-size 4 \
    --seed "${seed}" \
    >"${eval_log}" 2>&1
}

eval_dispatch() {
  local seed="$1" arm="$2" step="$3" gpus="${4}"
  evaluate_arm_checkpoint "${seed}" "${arm}" "${step}" "${gpus}"
}

echo "[stage 3/5] evaluate fixed dose-response checkpoints"
for seed in ${SEEDS}; do
  for step in ${CHECKPOINT_STEPS}; do
    echo "[launch] seed=${seed} step=${step}"
    if [[ "${GPUS_CONTROL}" == "${GPUS_RANKING}" ]]; then
      eval_dispatch "${seed}" original_only "${step}" "${GPUS_CONTROL}"
      eval_dispatch "${seed}" ranking "${step}" "${GPUS_RANKING}"
    else
      eval_dispatch "${seed}" original_only "${step}" "${GPUS_CONTROL}" &
      pid_control=$!
      eval_dispatch "${seed}" ranking "${step}" "${GPUS_RANKING}" &
      pid_ranking=$!
      status_control=0; status_ranking=0
      wait "${pid_control}" || status_control=$?
      wait "${pid_ranking}" || status_ranking=$?
      if [[ "${status_control}" -ne 0 || "${status_ranking}" -ne 0 ]]; then
        echo "[error] evaluation seed=${seed} step=${step}" >&2
        exit 1
      fi
    fi
  done
done

echo "[stage 4/5] compare matched arms at every checkpoint"
comparison_args=()
for step in ${CHECKPOINT_STEPS}; do
  step_tag="step$(printf '%06d' "${step}")"
  step_dir="${EVAL}/${step_tag}"
  python scripts/summarize_ddxplus_d10_arms.py \
    --arm "17=control=${step_dir}/original_only_seed17_private_scores.jsonl" \
    --arm "17=ranking=${step_dir}/ranking_seed17_private_scores.jsonl" \
    --arm "29=control=${step_dir}/original_only_seed29_private_scores.jsonl" \
    --arm "29=ranking=${step_dir}/ranking_seed29_private_scores.jsonl" \
    --arm "43=control=${step_dir}/original_only_seed43_private_scores.jsonl" \
    --arm "43=ranking=${step_dir}/ranking_seed43_private_scores.jsonl" \
    --output-json "${step_dir}/paired_arm_comparison.json" \
    --summary-md "${step_dir}/paired_arm_comparison_summary.md"
  comparison_args+=(--comparison "${step}=${step_dir}/paired_arm_comparison.json")
done

echo "[stage 5/5] render the preregistered trajectory and final gate"
python scripts/summarize_ddxplus_d10_budget_trajectory.py \
  "${comparison_args[@]}" \
  --output-json "${EVAL}/budget_trajectory.json" \
  --summary-md "${EVAL}/budget_trajectory_summary.md"

cat "${EVAL}/budget_trajectory_summary.md"
echo "[hardware] $(cat /tmp/d10_gpu_model.txt)"
echo "[done] ${RUN_NAME}"
