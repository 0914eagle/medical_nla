#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-17}"
EPOCHS="${EPOCHS:-3}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_NAME="${RUN_NAME:-common_medical_nla_pilot_v1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DIRECT:-${DATA_ROOT}/restricted/direct/e3/direct_e3_sft_v1}"
DDX_TRAIN="${DDX_TRAIN:-${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1/activations/ddxplus_probe_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl}"
DDX_VAL="${DDX_VAL:-${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl}"
ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
DATASET="${ROOT}/dataset"
mkdir -p "${ROOT}/adapters" "${DATA_ROOT}/medical_nla/logs"

for path in \
  "${DIRECT}/sft_train.jsonl" \
  "${DIRECT}/sft_val.jsonl" \
  "${DDX_TRAIN}" \
  "${DDX_VAL}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

python scripts/make_common_medical_nla_sft_dataset.py \
  --train "direct=${DIRECT}/sft_train.jsonl" \
  --train "ddxplus=${DDX_TRAIN}" \
  --val "direct=${DIRECT}/sft_val.jsonl" \
  --val "ddxplus=${DDX_VAL}" \
  --out-dir "${DATASET}" \
  --train-per-source 248 \
  --val-per-source 50 \
  --max-cues 12 \
  --seed 17

cat "${DATASET}/summary.md"
test "$(wc -l < "${DATASET}/sft_train.jsonl")" -eq 496
test "$(wc -l < "${DATASET}/sft_val.jsonl")" -eq 100

extra_args=()
if [[ -n "${MAX_STEPS}" ]]; then
  extra_args+=(--max-steps "${MAX_STEPS}")
fi

for seed in ${SEEDS}; do
  out="${ROOT}/adapters/${RUN_NAME}_seed${seed}"
  if [[ -s "${out}/best.json" ]]; then
    echo "[skip] seed ${seed} already complete"
    continue
  fi
  if [[ -e "${out}" ]]; then
    echo "[error] incomplete output already exists: ${out}" >&2
    exit 2
  fi
  echo "[train] seed=${seed} GPUs=${GPUS} epochs=${EPOCHS} max_steps=${MAX_STEPS:-none}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/train_medical_nla_lora.py \
    --config configs/default.yaml \
    --train-jsonl "${DATASET}/sft_train.jsonl" \
    --val-jsonl "${DATASET}/sft_val.jsonl" \
    --out-dir "${out}" \
    --actor-prompt-template-file prompt_templates/common_p0_clinical_state_readout.txt \
    --epochs "${EPOCHS}" \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --max-eval-rows 100 \
    --select-on source_macro_content \
    --seed "${seed}" \
    "${extra_args[@]}"
done

echo "[done] requested seeds: ${SEEDS}"
