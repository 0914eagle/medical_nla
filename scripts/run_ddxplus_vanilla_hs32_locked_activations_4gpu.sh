#!/usr/bin/env bash
set -euo pipefail

# Build the complete DDXPlus locked-test HS32 activation manifest on one
# four-GPU host. Two 12B backbone workers each use a two-GPU pair.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
CONFIRMATION="${CONFIRMATION:-}"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
ACTIVATION_ROWS="${ACTIVATION_ROWS:-${E5}/activation_rows_test.jsonl}"
RUN_NAME="${RUN_NAME:-ddxplus_e5_test_cot_p0_hs32_readout_v1}"
WORK_ROOT="${WORK_ROOT:-${E5}/activations/${RUN_NAME}_work}"
OUT_DIR="${OUT_DIR:-${E5}/activations/ddxplus_e5_test_cot_p0_hs32_merged_v1}"
LOG_ROOT="${LOG_ROOT:-${DATA_ROOT}/medical_nla/logs/${RUN_NAME}}"

if [[ "${CONFIRMATION}" != "I_ACCEPT_DDXPLUS_HS32_READOUT_EXTRACTION" ]]; then
  echo "[error] set CONFIRMATION=I_ACCEPT_DDXPLUS_HS32_READOUT_EXTRACTION" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${ACTIVATION_ROWS}" || { echo "[error] missing ${ACTIVATION_ROWS}" >&2; exit 2; }
mkdir -p "${WORK_ROOT}/input_shards" "${LOG_ROOT}" "${OUT_DIR}/provenance"

python scripts/validate_ddxplus_locked_population.py \
  --input "${ACTIVATION_ROWS}" \
  --expected-rows 10028 \
  --expected-variant original=4543 \
  --expected-variant cue_deleted=4543 \
  --expected-variant value_edited=942 \
  --report "${OUT_DIR}/provenance/input_population.json"

if [[ -s "${OUT_DIR}/layer32/last_token/manifest.jsonl" ]]; then
  python scripts/validate_ddxplus_locked_population.py \
    --input "${OUT_DIR}/layer32/last_token/manifest.jsonl" \
    --expected-rows 10028 \
    --expected-variant original=4543 \
    --expected-variant cue_deleted=4543 \
    --expected-variant value_edited=942 \
    --expected-layer 32 --require-activation-files \
    --report "${OUT_DIR}/provenance/output_population.json"
  echo "[skip] complete HS32 locked manifest already exists: ${OUT_DIR}"
  exit 0
fi

python scripts/shard_jsonl_by_key.py \
  --input "${ACTIVATION_ROWS}" \
  --out-dir "${WORK_ROOT}/input_shards" \
  --num-shards 2 \
  --key base_id

for pair in "${GPU_PAIR_A}" "${GPU_PAIR_B}"; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

run_shard() {
  local index="$1"
  local gpus="$2"
  local input
  local output="${WORK_ROOT}/activation_shard${index}"
  printf -v input '%s/input_shards/shard_%03d_of_002.jsonl' "${WORK_ROOT}" "${index}"
  DATA_ROOT="${DATA_ROOT}" GPUS="${gpus}" INPUT_FILE="${input}" \
    OUT_DIR="${output}" RUN_NAME="${RUN_NAME}_shard${index}" \
    CONFIRMATION=I_ACCEPT_DDXPLUS_HS32_READOUT_EXTRACTION \
    bash scripts/run_ddxplus_vanilla_hs32_locked_activation_shard.sh \
    >"${LOG_ROOT}/activation_shard${index}.log" 2>&1
}

run_shard 0 "${GPU_PAIR_A}" &
pid_a=$!
run_shard 1 "${GPU_PAIR_B}" &
pid_b=$!
status_a=0
status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?
if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
  echo "[error] HS32 extraction failed: shard0=${status_a} shard1=${status_b}" >&2
  exit 1
fi

python scripts/merge_activation_shards.py \
  --shard-roots \
    "${WORK_ROOT}/activation_shard0" \
    "${WORK_ROOT}/activation_shard1" \
  --out-dir "${OUT_DIR}" \
  --expected-layers 32

python scripts/validate_ddxplus_locked_population.py \
  --input "${OUT_DIR}/layer32/last_token/manifest.jsonl" \
  --expected-rows 10028 \
  --expected-variant original=4543 \
  --expected-variant cue_deleted=4543 \
  --expected-variant value_edited=942 \
  --expected-layer 32 --require-activation-files \
  --report "${OUT_DIR}/provenance/output_population.json"

git rev-parse HEAD > "${OUT_DIR}/provenance/git_commit.txt"
sha256sum configs/default.yaml "${ACTIVATION_ROWS}" \
  "${OUT_DIR}/layer32/last_token/manifest.jsonl" \
  > "${OUT_DIR}/provenance/input_output_hashes.sha256"
echo "[done] ${OUT_DIR}"
