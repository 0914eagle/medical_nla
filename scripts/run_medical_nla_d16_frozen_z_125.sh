#!/usr/bin/env bash
set -euo pipefail

# Fresh frozen-z train->validation probes for every D16 control/proposed seed.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-2}"
RUN_NAME="${RUN_NAME:-medical_nla_d16_soft_bottleneck_v1}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D16 frozen-z queue is frozen to server 125" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
DDX_TRAIN="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
DDX_VAL="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
HARD_PAIRS="${E5}/hard_shuffle_pairs_validation.jsonl"
ADAPTERS="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}/adapters"
PRIMARY="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_alignment_val_v1/paired_arm_comparison.json"
OUT="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_frozen_z_val_v1"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/${RUN_NAME}"
mkdir -p "${OUT}" "${LOG_ROOT}"

for path in "${PRIMARY}" "${DDX_TRAIN}" "${DDX_VAL}" "${HARD_PAIRS}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

run_one() {
  local arm="$1" seed="$2" gpu="$3"
  local label="${arm}_seed${seed}"
  local adapter="${ADAPTERS}/${label}"
  local root="${OUT}/${label}"
  local train_z="${root}/train_z"
  local val_z="${root}/validation_z"
  local probe="${root}/probe"
  local eval="${root}/evaluation"
  local log="${LOG_ROOT}/${label}_frozen_z.log"
  test -s "${adapter}/best.json"
  test -s "${adapter}/nla_bottleneck.pt"
  if [[ -s "${eval}/results.json" ]]; then
    echo "[skip] frozen-z ${label}"
    return
  fi
  {
    if [[ ! -s "${train_z}/layer32/last_token/manifest.jsonl" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" python \
        scripts/materialize_medical_nla_bottleneck_latents.py \
        --manifest "${DDX_TRAIN}" \
        --bottleneck-projector "${adapter}/nla_bottleneck.pt" \
        --out-root "${train_z}" \
        --path-map /data/heejae=/data1/heejae
    fi
    if [[ ! -s "${val_z}/layer32/last_token/manifest.jsonl" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" python \
        scripts/materialize_medical_nla_bottleneck_latents.py \
        --manifest "${DDX_VAL}" \
        --bottleneck-projector "${adapter}/nla_bottleneck.pt" \
        --out-root "${val_z}" \
        --path-map /data/heejae=/data1/heejae
    fi
    if [[ ! -s "${probe}/finding_value_hs32.pt" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" python \
        scripts/train_ddxplus_finding_value_probes.py \
        --train-root "${train_z}" \
        --validation-root "${val_z}" \
        --validation-hard-pairs "${HARD_PAIRS}" \
        --out-dir "${probe}" \
        --layers 32 \
        --device cuda
    fi
    CUDA_VISIBLE_DEVICES="${gpu}" python \
      scripts/evaluate_ddxplus_finding_value_probes.py \
      --artifact "${probe}/finding_value_hs32.pt" \
      --manifest "${val_z}/layer32/last_token/manifest.jsonl" \
      --hard-pairs "${HARD_PAIRS}" \
      --out-dir "${eval}" \
      --population-label validation \
      --device cuda
  } > "${log}" 2>&1
}

run_pair() {
  local arm="$1"
  run_one "${arm}" 17 "${GPU_A}" & p1=$!
  run_one "${arm}" 29 "${GPU_B}" & p2=$!
  s1=0; s2=0
  wait "${p1}" || s1=$?
  wait "${p2}" || s2=$?
  if [[ "${s1}" -ne 0 || "${s2}" -ne 0 ]]; then
    echo "[error] frozen-z ${arm}: seed17=${s1} seed29=${s2}" >&2
    return 1
  fi
  run_one "${arm}" 43 "${GPU_A}"
}

run_pair control
run_pair auxiliary

for arm in control auxiliary; do
  for seed in 17 29 43; do
    echo "===== ${arm} seed ${seed} ====="
    cat "${OUT}/${arm}_seed${seed}/evaluation/summary.md"
  done
done
echo "[done] ${OUT}"
