#!/usr/bin/env bash
set -euo pipefail

# D20 RunPod queue. This script intentionally refuses to run until the numeric
# gate protocol is committed and human-approved.

DATA_ROOT="/data1/heejae"
REPO_DIR="${REPO_DIR:-/home/eagle0914/medical_nla}"
CONFIG="${CONFIG:-configs/runpod.yaml}"
CONTROL_RUN_NAME="${CONTROL_RUN_NAME:-ddxplus_d10_budget1552_v1}"
RUN_NAME="${RUN_NAME:-ddxplus_d20_specificity_anchor1552_v1}"
GATE_PROTOCOL="${GATE_PROTOCOL:-configs/ddxplus_d20_gate_protocol.json}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-0}"
MAX_STEPS=1552
CHECKPOINT_STEPS="194 388 776 1164 1552"
SEEDS="17 29 43"

cd "${REPO_DIR}"
if [[ -f "${DATA_ROOT}/uv/medical_nla/bin/activate" ]]; then
  source "${DATA_ROOT}/uv/medical_nla/bin/activate"
fi
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH="${REPO_DIR}"

test -s "${CONFIG}" || { echo "[error] missing ${CONFIG}" >&2; exit 2; }
test -s "${GATE_PROTOCOL}" || {
  echo "[error] missing approved D20 gate protocol: ${GATE_PROTOCOL}" >&2; exit 2;
}
python - "${GATE_PROTOCOL}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
assert p.get("human_approved") is True, "D20 protocol is not human-approved"
required = {
    "retained_gap_delta_max",
    "changed_original_nll_relative_increase_max",
    "retained_original_nll_relative_increase_max",
    "mean_claim_relative_drop_max",
}
assert required <= set(p.get("gates") or {}), "D20 protocol lacks numeric gates"
assert all(p["gates"][key] is not None for key in required), "D20 gate is null"
PY

PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
CF_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
E5_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
D9A="${CF_ROOT}/d9a_selected_changed_cue_v1"
TRAIN_MANIFEST="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
CF_MANIFEST="${CF_ROOT}/activations/ddxplus_counterfactual_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
VAL_MANIFEST="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
TRAIN_PAIRS="${D9A}/approved_pairs/pairs_train.jsonl"
VAL_PAIRS="${D9A}/approved_pairs/pairs_validation.jsonl"
ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
EVAL="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_validation_v1"
CONTROL_ROOT="${DATA_ROOT}/restricted/direct/e3/${CONTROL_RUN_NAME}"
CONTROL_EVAL="${DATA_ROOT}/restricted/direct/e4/${CONTROL_RUN_NAME}_validation_v1"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/${RUN_NAME}"
mkdir -p "${ROOT}" "${EVAL}" "${LOG_ROOT}" "${D9A}/approved_pairs"

python - "${GATE_PROTOCOL}" "${CONTROL_EVAL}" <<'PY'
import hashlib, json, pathlib, sys
protocol = json.load(open(sys.argv[1]))
root = pathlib.Path(sys.argv[2]) / "step001552"
for seed in (17, 29, 43):
    path = root / f"original_only_seed{seed}_private_scores.jsonl"
    if not path.is_file():
        raise SystemExit(f"[error] missing frozen control scores: {path}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = protocol["control_score_sha256"][str(seed)]
    if observed != expected:
        raise SystemExit(f"[error] control score hash mismatch for seed {seed}")
print("[gate] approved numeric protocol and control score hashes verified")
PY

for path in \
  "${D9A}/train_audit/private_scores.jsonl" \
  "${D9A}/validation_null_audit/private_scores.jsonl" \
  "${D9A}/cut_selection/protocol_approved.json" \
  "${TRAIN_MANIFEST}" "${CF_MANIFEST}" "${VAL_MANIFEST}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

echo "[stage 1/5] rebuild frozen D9a pairs"
python scripts/make_ddxplus_d9a_supported_pairs.py \
  --train-scores "${D9A}/train_audit/private_scores.jsonl" \
  --validation-scores "${D9A}/validation_null_audit/private_scores.jsonl" \
  --original-manifest "${TRAIN_MANIFEST}" \
  --counterfactual-manifest "${CF_MANIFEST}" \
  --approved-protocol "${D9A}/cut_selection/protocol_approved.json" \
  --output-jsonl "${TRAIN_PAIRS}" \
  --protocol-json "${D9A}/approved_pairs/protocol.json" \
  --summary-md "${D9A}/approved_pairs/summary.md"
python scripts/make_ddxplus_d10_validation_pairs.py \
  --validation-scores "${D9A}/validation_null_audit/private_scores.jsonl" \
  --validation-manifest "${VAL_MANIFEST}" \
  --approved-protocol "${D9A}/cut_selection/protocol_approved.json" \
  --output-jsonl "${VAL_PAIRS}" \
  --report-json "${D9A}/approved_pairs/validation_protocol.json" \
  --summary-md "${D9A}/approved_pairs/validation_summary.md"

latest_resume_checkpoint() {
  local adapter="$1" latest_state
  latest_state="$(find "${adapter}" -mindepth 2 -maxdepth 2 -type f \
    -path '*/checkpoint-step*/trainer_state.pt' -print 2>/dev/null | sort | tail -n 1)"
  [[ -n "${latest_state}" ]] && dirname "${latest_state}"
}

train_seed() {
  local seed="$1" gpu="$2"
  local adapter="${ROOT}/anchored_seed${seed}"
  local log="${LOG_ROOT}/anchored_seed${seed}_train.log"
  local resume_args=()
  if [[ -s "${adapter}/best.json" ]]; then
    echo "[skip] completed anchored seed ${seed}"
    return
  fi
  if [[ -d "${adapter}" ]]; then
    resume="$(latest_resume_checkpoint "${adapter}")"
    test -n "${resume}" || { echo "[error] no resume checkpoint ${adapter}" >&2; return 2; }
    resume_args=(--resume-from-checkpoint "${resume}")
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/train_ddxplus_d10_1x2.py \
    --config "${CONFIG}" --train-jsonl "${TRAIN_PAIRS}" --out-dir "${adapter}" \
    "${resume_args[@]}" \
    --max-steps "${MAX_STEPS}" --checkpoint-steps ${CHECKPOINT_STEPS} \
    --grad-accum-steps 4 --ranking-weight 1.0 --retained-anchor-weight 1.0 \
    --temperature 1.0 --margin 0.0 --lr 2e-4 --seed "${seed}" >"${log}" 2>&1
}

echo "[stage 2/5] train anchored seeds 17/29, then 43"
if [[ "${GPU_A}" == "${GPU_B}" ]]; then
  train_seed 17 "${GPU_A}"
  train_seed 29 "${GPU_A}"
else
  train_seed 17 "${GPU_A}" & p17=$!
  train_seed 29 "${GPU_B}" & p29=$!
  s17=0; s29=0
  wait "${p17}" || s17=$?
  wait "${p29}" || s29=$?
  if [[ "${s17}" -ne 0 || "${s29}" -ne 0 ]]; then
    echo "[error] anchored training seed17=${s17} seed29=${s29}" >&2; exit 1
  fi
fi
train_seed 43 "${GPU_A}"

evaluate_seed_step() {
  local seed="$1" step="$2" gpu="$3"
  local checkpoint="${ROOT}/anchored_seed${seed}/checkpoint-step$(printf '%06d' "${step}")"
  local step_dir="${EVAL}/step$(printf '%06d' "${step}")"
  local prefix="${step_dir}/anchored_seed${seed}"
  mkdir -p "${step_dir}"
  test -s "${checkpoint}/adapter_config.json"
  [[ -s "${prefix}.json" ]] && return
  CUDA_VISIBLE_DEVICES="${gpu}" python scripts/evaluate_ddxplus_d10_specificity.py \
    --config "${CONFIG}" --pairs "${VAL_PAIRS}" --adapter "${checkpoint}" \
    --output-jsonl "${prefix}_private_scores.jsonl" \
    --output-json "${prefix}.json" --summary-md "${prefix}_summary.md" \
    --batch-size 4 --seed "${seed}" \
    >"${LOG_ROOT}/anchored_seed${seed}_step${step}_specificity.log" 2>&1
}

echo "[stage 3/5] evaluate report-only checkpoints and frozen final checkpoint"
for step in ${CHECKPOINT_STEPS}; do
  for seed in ${SEEDS}; do
    control="${CONTROL_EVAL}/step$(printf '%06d' "${step}")/original_only_seed${seed}_private_scores.jsonl"
    test -s "${control}" || { echo "[error] missing control ${control}" >&2; exit 2; }
  done
  if [[ "${GPU_A}" == "${GPU_B}" ]]; then
    evaluate_seed_step 17 "${step}" "${GPU_A}"
    evaluate_seed_step 29 "${step}" "${GPU_A}"
  else
    evaluate_seed_step 17 "${step}" "${GPU_A}" & p17=$!
    evaluate_seed_step 29 "${step}" "${GPU_B}" & p29=$!
    wait "${p17}"; wait "${p29}"
  fi
  evaluate_seed_step 43 "${step}" "${GPU_A}"
done

echo "[stage 4/5] compare every dose without checkpoint selection"
for step in ${CHECKPOINT_STEPS}; do
  tag="step$(printf '%06d' "${step}")"
  step_dir="${EVAL}/${tag}"
  extra=()
  [[ "${step}" -ne 1552 ]] && extra=(--report-only)
  python scripts/summarize_ddxplus_d20_arms.py \
    --arm "17=control=${CONTROL_EVAL}/${tag}/original_only_seed17_private_scores.jsonl" \
    --arm "17=anchored=${step_dir}/anchored_seed17_private_scores.jsonl" \
    --arm "29=control=${CONTROL_EVAL}/${tag}/original_only_seed29_private_scores.jsonl" \
    --arm "29=anchored=${step_dir}/anchored_seed29_private_scores.jsonl" \
    --arm "43=control=${CONTROL_EVAL}/${tag}/original_only_seed43_private_scores.jsonl" \
    --arm "43=anchored=${step_dir}/anchored_seed43_private_scores.jsonl" \
    --gate-protocol "${GATE_PROTOCOL}" "${extra[@]}" \
    --output-json "${step_dir}/paired_comparison.json" \
    --summary-md "${step_dir}/paired_comparison_summary.md"
done

echo "[stage 5/5] frozen teacher-forced decision"
cat "${EVAL}/step001552/paired_comparison_summary.md"
python - "${EVAL}/step001552/paired_comparison.json" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
if p["gate"]["teacher_forced_gate_passed"]:
    print("[next] PASS: run the preregistered generation gate on the fixed pilot")
else:
    print("[stop] FAIL: no generation, extension, checkpoint selection, or sweep")
PY
