#!/usr/bin/env bash
set -euo pipefail

# D16 control-first mechanism smoke. The script intentionally stops after the
# preregistered primary comparison; frozen-z and generation diagnostics are a
# separate queue so a failed primary gate does not consume another long run.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
RUN_NAME="${RUN_NAME:-medical_nla_d16_soft_bottleneck_v1}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D16 4-GPU queue is frozen to server 125 (/data1/heejae)" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DATA_ROOT}/restricted/direct/e3/direct_e3_sft_v1"
PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
CF="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
DDX_TRAIN="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
DDX_VAL="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
D9A="${CF}/d9a_selected_changed_cue_v1/approved_pairs/pairs_train.jsonl"
TEACHER_ROOT="${CF}/oof_finding_teacher_hs32_k5_v2"
TEACHER_SCORES="${TEACHER_ROOT}/private_teacher_scores.jsonl"
TEACHER_REPORT="${TEACHER_ROOT}/report.json"
ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
PCA_ROOT="${ROOT}/pca_init"
PCA="${PCA_ROOT}/nla_bottleneck.pt"
LAMBDA="${ROOT}/lambda_protocol.json"
ADAPTERS="${ROOT}/adapters"
EVAL="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_alignment_val_v1"
FLOOR="${EVAL}/effect_floor_protocol.json"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/${RUN_NAME}"
PROMPT="prompt_templates/common_p0_clinical_state_readout.txt"
mkdir -p "${ROOT}" "${ADAPTERS}" "${EVAL}" "${LOG_ROOT}"

for path in \
  "${DIRECT}/sft_train.jsonl" "${DIRECT}/sft_val.jsonl" \
  "${DDX_TRAIN}" "${DDX_VAL}" "${D9A}" \
  "${TEACHER_SCORES}" "${TEACHER_REPORT}" "${PROMPT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
test "$(wc -l < "${DIRECT}/sft_train.jsonl")" -eq 248
test "$(wc -l < "${DIRECT}/sft_val.jsonl")" -eq 50
test "$(wc -l < "${DDX_TRAIN}")" -eq 4655
test "$(wc -l < "${DDX_VAL}")" -eq 10006
test "$(wc -l < "${D9A}")" -eq 3104
test "$(wc -l < "${TEACHER_SCORES}")" -eq 9310

echo "[stage 1/7] source-balanced PCA and validation cosine gate"
if [[ ! -s "${PCA}" ]]; then
  python scripts/fit_medical_nla_bottleneck_pca.py \
    --ddxplus-train "${DDX_TRAIN}" \
    --direct-train "${DIRECT}/sft_train.jsonl" \
    --ddxplus-validation "${DDX_VAL}" \
    --direct-validation "${DIRECT}/sft_val.jsonl" \
    --out-dir "${PCA_ROOT}" \
    --path-map /data/heejae=/data1/heejae
fi
python - "${PCA_ROOT}/protocol.json" <<'PY'
import json, sys
report = json.load(open(sys.argv[1], encoding="utf-8"))
assert report["pca_sanity_gate_passed"] is True
assert report["d_z"] == 256
print("[gate] PCA sanity PASS")
PY

echo "[stage 2/7] one-shot seed-17 gradient parity lambda"
if [[ ! -s "${LAMBDA}" ]]; then
  CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}" python \
    scripts/calibrate_medical_nla_bottleneck_lambda.py \
    --direct-train "${DIRECT}/sft_train.jsonl" \
    --d9a-pairs "${D9A}" \
    --teacher-scores "${TEACHER_SCORES}" \
    --teacher-report "${TEACHER_REPORT}" \
    --pca-artifact "${PCA}" \
    --output "${LAMBDA}" \
    --path-map /data/heejae=/data1/heejae
fi

train_arm() {
  local arm="$1" seed="$2" gpus="$3"
  local out="${ADAPTERS}/${arm}_seed${seed}"
  local log="${LOG_ROOT}/${arm}_seed${seed}_train.log"
  if [[ -s "${out}/best.json" ]]; then
    echo "[skip] complete ${arm} seed ${seed}"
    return
  fi
  args=(
    --direct-train "${DIRECT}/sft_train.jsonl"
    --d9a-pairs "${D9A}"
    --teacher-scores "${TEACHER_SCORES}"
    --teacher-report "${TEACHER_REPORT}"
    --pca-artifact "${PCA}"
    --out-dir "${out}"
    --arm "${arm}"
    --seed "${seed}"
    --actor-prompt-template-file "${PROMPT}"
    --path-map /data/heejae=/data1/heejae
  )
  if [[ "${arm}" == "auxiliary" ]]; then
    args+=(
      --lambda-protocol "${LAMBDA}"
      --floor-protocol "${FLOOR}"
      --matched-control "${ADAPTERS}/control_seed${seed}"
      --aux-head-audit-dir "${ROOT}/training_only_aux_heads"
    )
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/train_medical_nla_soft_bottleneck.py \
    "${args[@]}" > "${log}" 2>&1
}

audit_arm() {
  local arm="$1" seed="$2" gpus="$3"
  local adapter="${ADAPTERS}/${arm}_seed${seed}"
  local prefix="${EVAL}/${arm}_seed${seed}"
  local log="${LOG_ROOT}/${arm}_seed${seed}_alignment.log"
  if [[ -s "${prefix}.json" && -s "${prefix}_private_scores.jsonl" ]]; then
    echo "[skip] complete alignment ${arm} seed ${seed}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/audit_medical_nla_target_alignment.py \
    --manifest "${DIRECT}/sft_val.jsonl" \
    --adapter "${adapter}" \
    --output-jsonl "${prefix}_private_scores.jsonl" \
    --output-json "${prefix}.json" \
    --summary-md "${prefix}_summary.md" \
    --seed "${seed}" \
    --batch-size 4 > "${log}" 2>&1
}

run_two_then_one() {
  local function_name="$1" arm="$2"
  "${function_name}" "${arm}" 17 "${GPU_PAIR_A}" & p1=$!
  "${function_name}" "${arm}" 29 "${GPU_PAIR_B}" & p2=$!
  status1=0
  status2=0
  wait "${p1}" || status1=$?
  wait "${p2}" || status2=$?
  if [[ "${status1}" -ne 0 || "${status2}" -ne 0 ]]; then
    echo "[error] parallel ${function_name} ${arm}: seed17=${status1} seed29=${status2}" >&2
    return 1
  fi
  "${function_name}" "${arm}" 43 "${GPU_PAIR_A}"
}

echo "[stage 3/7] control training: seeds 17/29 in parallel, then 43"
run_two_then_one train_arm control
echo "[stage 4/7] control Direct alignment and immutable floor"
run_two_then_one audit_arm control
if [[ ! -s "${FLOOR}" ]]; then
  python scripts/freeze_medical_nla_bottleneck_effect_floor.py \
    --control-audit "17=${EVAL}/control_seed17.json" \
    --control-audit "29=${EVAL}/control_seed29.json" \
    --control-audit "43=${EVAL}/control_seed43.json" \
    --control-adapter "17=${ADAPTERS}/control_seed17" \
    --control-adapter "29=${ADAPTERS}/control_seed29" \
    --control-adapter "43=${ADAPTERS}/control_seed43" \
    --output "${FLOOR}"
fi

echo "[stage 5/7] proposed training only after floor freeze"
run_two_then_one train_arm auxiliary
echo "[stage 6/7] proposed Direct alignment"
run_two_then_one audit_arm auxiliary

echo "[stage 7/7] seed-matched paired comparison"
python scripts/compare_medical_nla_bottleneck_arms.py \
  --seed-scores "17=${EVAL}/control_seed17_private_scores.jsonl=${EVAL}/auxiliary_seed17_private_scores.jsonl" \
  --seed-scores "29=${EVAL}/control_seed29_private_scores.jsonl=${EVAL}/auxiliary_seed29_private_scores.jsonl" \
  --seed-scores "43=${EVAL}/control_seed43_private_scores.jsonl=${EVAL}/auxiliary_seed43_private_scores.jsonl" \
  --floor-protocol "${FLOOR}" \
  --output-json "${EVAL}/paired_arm_comparison.json" \
  --summary-md "${EVAL}/paired_arm_comparison_summary.md"

cat "${EVAL}/paired_arm_comparison_summary.md"
echo "[done] ${RUN_NAME} primary D16 queue"
echo "[next] run frozen-z and generation diagnostics before any promotion claim"
