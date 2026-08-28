#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
RUN_NAME="${RUN_NAME:-common_medical_nla_full_sft_v1}"
OUT="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_alignment_val_v1"
MANIFEST="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}/dataset/sft_val.jsonl"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla
mkdir -p "${OUT}" "${LOG_ROOT}"

test -s "${MANIFEST}" || { echo "[error] missing ${MANIFEST}" >&2; exit 2; }
for pair in 0,1 2,3; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

run_seed() {
  local seed="$1"
  local gpus="$2"
  local adapter="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}/adapters/${RUN_NAME}_seed${seed}"
  test -s "${adapter}/best.json" || { echo "[error] incomplete ${adapter}" >&2; return 2; }
  CUDA_VISIBLE_DEVICES="${gpus}" python scripts/audit_medical_nla_target_alignment.py \
    --config configs/default.yaml \
    --manifest "${MANIFEST}" \
    --adapter "${adapter}" \
    --output-jsonl "${OUT}/seed${seed}_private_scores.jsonl" \
    --output-json "${OUT}/seed${seed}.json" \
    --summary-md "${OUT}/seed${seed}_summary.md" \
    --source-dataset direct \
    --batch-size 4 \
    --seed 17 \
    >"${LOG_ROOT}/${RUN_NAME}_alignment_seed${seed}.log" 2>&1
}

echo "[launch] seed17 on GPUs 0,1"
run_seed 17 0,1 &
pid17=$!
echo "[launch] seed29 on GPUs 2,3"
run_seed 29 2,3 &
pid29=$!

status17=0
status29=0
wait "${pid17}" || status17=$?
wait "${pid29}" || status29=$?
echo "[workers] seed17=${status17} seed29=${status29}"
if [[ "${status17}" -ne 0 || "${status29}" -ne 0 ]]; then
  exit 1
fi
cat "${OUT}/seed17_summary.md"
cat "${OUT}/seed29_summary.md"
echo "[done] ${OUT}"
