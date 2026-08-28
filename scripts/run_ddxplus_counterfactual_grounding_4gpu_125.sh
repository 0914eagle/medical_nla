#!/usr/bin/env bash
set -euo pipefail

# End-to-end DDXPlus official-train counterfactual grounding experiment.
# Server 125 exposes four RTX 4090s under /data1/heejae. Two independent
# two-GPU workers extract the derived activations, then train/evaluate two seeds.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
RUN_NAME="${RUN_NAME:-ddxplus_counterfactual_sft_v1}"
SEED_LEFT="${SEED_LEFT:-17}"
SEED_RIGHT="${SEED_RIGHT:-29}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
RUN_GROUNDING="${RUN_GROUNDING:-1}"
GROUNDING_SHARDS="${GROUNDING_SHARDS:-0 1 2 3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this wrapper is frozen for server 125 (/data1/heejae)" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

LOG_ROOT="${DATA_ROOT}/medical_nla/logs"
PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
E5_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
CF_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
SHARDS="${CF_ROOT}/activation_shards_cot_p0_v1"
ACT_ROOT="${CF_ROOT}/activations"
MERGED="${ACT_ROOT}/ddxplus_counterfactual_train_cot_p0_merged_v1"
EXPERIMENT_ROOT="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}"
DATASET="${EXPERIMENT_ROOT}/dataset"
mkdir -p "${LOG_ROOT}" "${CF_ROOT}" "${ACT_ROOT}" "${EXPERIMENT_ROOT}/adapters"

ORIGINAL_SOURCE="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
ORIGINAL_LOCAL="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
VALIDATION_SOURCE="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
VALIDATION_LOCAL="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"

for path in \
  "${PROBE_ROOT}/cases_train.jsonl" \
  "${DATA_ROOT}/ddxplus/release_evidences.json" \
  "${ORIGINAL_SOURCE}" \
  "${VALIDATION_SOURCE}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

echo "[stage 1/6] derive official-train counterfactual families"
if [[ ! -s "${CF_ROOT}/protocol.json" ]]; then
  python scripts/prepare_ddxplus_counterfactual_train.py \
    --cases-train "${PROBE_ROOT}/cases_train.jsonl" \
    --evidences "${DATA_ROOT}/ddxplus/release_evidences.json" \
    --out-dir "${CF_ROOT}" \
    --seed 17
fi
DERIVED_ROWS="$(wc -l < "${CF_ROOT}/activation_rows_counterfactual_train.jsonl")"
BASE_ROWS="$(wc -l < "${PROBE_ROOT}/cases_train.jsonl")"
test "${BASE_ROWS}" -eq 4655
test "${DERIVED_ROWS}" -ge "${BASE_ROWS}"
cat "${CF_ROOT}/summary.md"

echo "[stage 2/6] shard and extract derived CoT-P0/HS32 activations"
python scripts/shard_jsonl_by_key.py \
  --input "${CF_ROOT}/activation_rows_counterfactual_train.jsonl" \
  --out-dir "${SHARDS}" \
  --num-shards 2 \
  --key base_id

for pair in 0,1 2,3; do
  CUDA_VISIBLE_DEVICES="${pair}" python scripts/check_gpu_setup.py \
    --config configs/default.yaml --require-free-gb 20
done

extract_shard() {
  local index="$1"
  local gpus="$2"
  local input
  local name="ddxplus_counterfactual_train_cot_p0_shard${index}_v1"
  local output="${ACT_ROOT}/${name}"
  local log="${LOG_ROOT}/${name}.log"
  printf -v input '%s/shard_%03d_of_002.jsonl' "${SHARDS}" "${index}"
  CUDA_VISIBLE_DEVICES="${gpus}" python -m src.extract_activations \
    --config configs/default.yaml \
    --input "${input}" \
    --run-name "${name}" \
    --output-dir "${output}" \
    --layers 32 \
    --batch-size 1 \
    --resume >"${log}" 2>&1
  test "$(wc -l < "${output}/layer32/last_token/manifest.jsonl")" \
    -eq "$(wc -l < "${input}")"
}

extract_shard 0 0,1 &
pid_extract_0=$!
extract_shard 1 2,3 &
pid_extract_1=$!
status_0=0
status_1=0
wait "${pid_extract_0}" || status_0=$?
wait "${pid_extract_1}" || status_1=$?
echo "[extract workers] shard0=${status_0} shard1=${status_1}"
if [[ "${status_0}" -ne 0 || "${status_1}" -ne 0 ]]; then
  exit 1
fi

python scripts/merge_activation_shards.py \
  --shard-roots \
    "${ACT_ROOT}/ddxplus_counterfactual_train_cot_p0_shard0_v1" \
    "${ACT_ROOT}/ddxplus_counterfactual_train_cot_p0_shard1_v1" \
  --out-dir "${MERGED}" \
  --expected-layers 32
DERIVED_MANIFEST="${MERGED}/layer32/last_token/manifest.jsonl"
test "$(wc -l < "${DERIVED_MANIFEST}")" -eq "${DERIVED_ROWS}"

echo "[stage 3/6] localize existing original/validation manifests"
if [[ ! -s "${ORIGINAL_LOCAL}" ]]; then
  mkdir -p "$(dirname "${ORIGINAL_LOCAL}")"
  python scripts/remap_activation_manifest_paths.py \
    --input "${ORIGINAL_SOURCE}" \
    --output "${ORIGINAL_LOCAL}" \
    --path-map /data/heejae=/data1/heejae \
    --expected-rows 4655
fi
if [[ ! -s "${VALIDATION_LOCAL}" ]]; then
  mkdir -p "$(dirname "${VALIDATION_LOCAL}")"
  python scripts/remap_activation_manifest_paths.py \
    --input "${VALIDATION_SOURCE}" \
    --output "${VALIDATION_LOCAL}" \
    --path-map /data/heejae=/data1/heejae \
    --expected-rows 10006
fi

echo "[stage 4/6] build complete-family diagnosis-free SFT dataset"
if [[ ! -s "${DATASET}/protocol.json" ]]; then
  python scripts/make_ddxplus_counterfactual_sft_dataset.py \
    --train-manifest "${ORIGINAL_LOCAL}" \
    --train-manifest "${DERIVED_MANIFEST}" \
    --validation-manifest "${VALIDATION_LOCAL}" \
    --out-dir "${DATASET}" \
    --max-cues 64
fi
cat "${DATASET}/summary.md"

extra_args=()
if [[ -n "${MAX_STEPS}" ]]; then
  extra_args+=(--max-steps "${MAX_STEPS}")
fi

echo "[stage 5/6] train two independent counterfactual SFT seeds"
train_and_score() {
  local seed="$1"
  local gpus="$2"
  local adapter="${EXPERIMENT_ROOT}/adapters/${RUN_NAME}_seed${seed}"
  local train_log="${LOG_ROOT}/${RUN_NAME}_seed${seed}_train.log"
  local grounding_log="${LOG_ROOT}/${RUN_NAME}_seed${seed}_grounding.log"
  if [[ ! -s "${adapter}/best.json" ]]; then
    if [[ -e "${adapter}" ]]; then
      echo "[error] incomplete adapter exists: ${adapter}" >&2
      return 2
    fi
    CUDA_VISIBLE_DEVICES="${gpus}" python scripts/train_medical_nla_lora.py \
      --config configs/default.yaml \
      --train-jsonl "${DATASET}/sft_train.jsonl" \
      --val-jsonl "${DATASET}/sft_validation.jsonl" \
      --out-dir "${adapter}" \
      --actor-prompt-template-file prompt_templates/common_p0_clinical_state_readout.txt \
      --epochs "${EPOCHS}" \
      --batch-size "${BATCH_SIZE}" \
      --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
      --source-sampling-alpha 1.0 \
      --max-eval-rows 256 \
      --select-on content \
      --seed "${seed}" \
      "${extra_args[@]}" >"${train_log}" 2>&1
  fi

  if [[ "${RUN_GROUNDING}" == "1" ]]; then
    SOURCE_MANIFEST="${VALIDATION_LOCAL}" \
      DATA_ROOT="${DATA_ROOT}" GPUS="${gpus}" \
      COMMON_RUN_NAME="${RUN_NAME}" SEEDS="${seed}" \
      SELECT_SHARDS="${GROUNDING_SHARDS}" \
      MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" BATCH_SIZE=4 \
      GROUNDING_RUN_NAME="${RUN_NAME}_seed${seed}_ddx_grounding_val_v1" \
      bash scripts/run_overnight_common_ddx_grounding.sh >"${grounding_log}" 2>&1
  fi
}

train_and_score "${SEED_LEFT}" 0,1 &
pid_train_left=$!
train_and_score "${SEED_RIGHT}" 2,3 &
pid_train_right=$!
status_left=0
status_right=0
wait "${pid_train_left}" || status_left=$?
wait "${pid_train_right}" || status_right=$?
echo "[train workers] seed${SEED_LEFT}=${status_left} seed${SEED_RIGHT}=${status_right}"
if [[ "${status_left}" -ne 0 || "${status_right}" -ne 0 ]]; then
  exit 1
fi

echo "[stage 6/6] report validation grounding gates"
if [[ "${RUN_GROUNDING}" == "1" ]]; then
  comparison_args=()
  for seed in "${SEED_LEFT}" "${SEED_RIGHT}"; do
    grounding_dir="${E5_ROOT}/${RUN_NAME}_seed${seed}_ddx_grounding_val_v1"
    summary="${grounding_dir}/paired_scores_summary.md"
    test -s "${summary}"
    echo "===== seed ${seed} ====="
    cat "${summary}"
    comparison_args+=(--readout "counterfactual_seed${seed}=${grounding_dir}/medical_nla_seed${seed}.jsonl")
  done
  for seed in 17 29; do
    baseline="${E5_ROOT}/common_medical_nla_full_sft_v1_seed${seed}_ddx_grounding_val_v1/medical_nla_seed${seed}.jsonl"
    if [[ -s "${baseline}" ]]; then
      comparison_args+=(--readout "original_only_seed${seed}=${baseline}")
    fi
  done
  comparison_dir="${E5_ROOT}/${RUN_NAME}_comparison_val_v1"
  mkdir -p "${comparison_dir}"
  python scripts/score_ddxplus_e5_readout_pilot.py \
    "${comparison_args[@]}" \
    --threshold 0.5 \
    --output-json "${comparison_dir}/paired_scores.json" \
    --summary-md "${comparison_dir}/paired_scores_summary.md"
  echo "===== common population comparison ====="
  cat "${comparison_dir}/paired_scores_summary.md"
fi
echo "[done] ${RUN_NAME}"
