#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-17}"
EPOCHS="${EPOCHS:-3}"
EXPECTED_TRAIN="${EXPECTED_TRAIN:-248}"
EXPECTED_VAL="${EXPECTED_VAL:-50}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E1="${DATA_ROOT}/restricted/direct/e1"
SPLIT="${DATA_ROOT}/restricted/direct/splits/direct_patient_pdd_confirmatory_v1"
ACT="${E1}/direct_e1_reindexed_confirmatory_v1/activations"
E3="${DATA_ROOT}/restricted/direct/e3"
DATASET="${E3}/direct_e3_sft_v1"
mkdir -p "${E3}/adapters" "${DATA_ROOT}/medical_nla/logs"

python scripts/make_direct_e3_sft_dataset.py \
  --split-dir "${SPLIT}" \
  --activation-root "${ACT}" \
  --source-answers \
    "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl" \
    "${E1}/direct_e1_test_v1/source_cot_answers.jsonl" \
  --out-dir "${DATASET}" \
  --max-observations 12 \
  --seed 17

cat "${DATASET}/summary.md"

train_rows="$(wc -l < "${DATASET}/sft_train.jsonl")"
val_rows="$(wc -l < "${DATASET}/sft_val.jsonl")"
if [[ "${train_rows}" -ne "${EXPECTED_TRAIN}" || "${val_rows}" -ne "${EXPECTED_VAL}" ]]; then
  echo "[error] unexpected SFT population: train=${train_rows}, val=${val_rows}; " \
    "expected ${EXPECTED_TRAIN}/${EXPECTED_VAL}" >&2
  exit 2
fi
echo "[population] train=${train_rows} val=${val_rows} (expected)"

for seed in ${SEEDS}; do
  out="${E3}/adapters/direct_e3_sft_v1_seed${seed}"
  if [[ -s "${out}/best.json" ]]; then
    echo "[skip] seed ${seed} already complete"
    continue
  fi
  if [[ -e "${out}" ]]; then
    echo "[error] incomplete output already exists: ${out}" >&2
    exit 2
  fi
  echo "[train] seed=${seed} GPUs=${GPUS} epochs=${EPOCHS}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/train_medical_nla_lora.py \
    --config configs/default.yaml \
    --train-jsonl "${DATASET}/sft_train.jsonl" \
    --val-jsonl "${DATASET}/sft_val.jsonl" \
    --out-dir "${out}" \
    --actor-prompt-template-file prompt_templates/direct_p0_evidence_readout.txt \
    --epochs "${EPOCHS}" \
    --batch-size 1 \
    --grad-accum-steps 8 \
    --max-eval-rows 52 \
    --select-on content \
    --seed "${seed}"
done

echo "[done] requested seeds: ${SEEDS}"
