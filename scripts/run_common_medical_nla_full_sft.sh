#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
SEEDS="${SEEDS:-17}"
EPOCHS="${EPOCHS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
RUN_NAME="${RUN_NAME:-common_medical_nla_full_sft_v1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
SOURCE_SAMPLING_ALPHA="${SOURCE_SAMPLING_ALPHA:-0.5}"
PREPARE_ONLY="${PREPARE_ONLY:-0}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DIRECT:-${DATA_ROOT}/restricted/direct/e3/direct_e3_sft_v1}"
default_ddx_train="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1/activations/ddxplus_probe_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
default_ddx_val="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"

prepare_server125_manifest() {
  local input="$1"
  local output="$2"
  local expected_rows="$3"
  test -s "${input}" || { echo "[error] missing ${input}" >&2; exit 2; }
  mkdir -p "$(dirname "${output}")"
  python scripts/remap_activation_manifest_paths.py \
    --input "${input}" \
    --output "${output}" \
    --path-map /data/heejae=/data1/heejae \
    --expected-rows "${expected_rows}"
}

if [[ "${DATA_ROOT}" == "/data1/heejae" ]]; then
  server125_ddx_train="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
  server125_ddx_val="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
  if [[ -z "${DDX_TRAIN:-}" ]]; then
    prepare_server125_manifest "${default_ddx_train}" "${server125_ddx_train}" 4655
    default_ddx_train="${server125_ddx_train}"
  fi
  if [[ -z "${DDX_VAL:-}" ]]; then
    prepare_server125_manifest "${default_ddx_val}" "${server125_ddx_val}" 10006
    default_ddx_val="${server125_ddx_val}"
  fi
fi
DDX_TRAIN="${DDX_TRAIN:-${default_ddx_train}}"
DDX_VAL="${DDX_VAL:-${default_ddx_val}}"
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

if [[ ! -s "${DATASET}/protocol.json" ]]; then
  python scripts/make_common_medical_nla_sft_dataset.py \
    --train "direct=${DIRECT}/sft_train.jsonl" \
    --train "ddxplus=${DDX_TRAIN}" \
    --val "direct=${DIRECT}/sft_val.jsonl" \
    --val "ddxplus=${DDX_VAL}" \
    --out-dir "${DATASET}" \
    --use-all-train-rows \
    --val-per-source 50 \
    --max-cues 12 \
    --cue-order source \
    --seed 17
fi

cat "${DATASET}/summary.md"
test "$(wc -l < "${DATASET}/sft_train.jsonl")" -eq 4903
test "$(wc -l < "${DATASET}/sft_val.jsonl")" -eq 100
python - "${DATASET}/protocol.json" <<'PY'
import json
import sys

protocol = json.load(open(sys.argv[1], encoding="utf-8"))
assert protocol["use_all_train_rows"] is True
assert protocol["target_style"].endswith("source_order")
assert protocol["selected"]["train"]["ddxplus"]["rows"] == 4655
assert protocol["selected"]["train"]["direct"]["rows"] == 248
print("[dataset] frozen full-data population verified")
PY

if [[ "${PREPARE_ONLY}" == "1" ]]; then
  echo "[done] prepared ${DATASET}"
  exit 0
fi

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
  echo "[train] seed=${seed} GPUs=${GPUS} epochs=${EPOCHS} batch=${BATCH_SIZE}x${GRAD_ACCUM_STEPS} alpha=${SOURCE_SAMPLING_ALPHA}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/train_medical_nla_lora.py \
    --config configs/default.yaml \
    --train-jsonl "${DATASET}/sft_train.jsonl" \
    --val-jsonl "${DATASET}/sft_val.jsonl" \
    --out-dir "${out}" \
    --actor-prompt-template-file prompt_templates/common_p0_clinical_state_readout.txt \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --grad-accum-steps "${GRAD_ACCUM_STEPS}" \
    --source-sampling-alpha "${SOURCE_SAMPLING_ALPHA}" \
    --max-eval-rows 100 \
    --select-on source_macro_content \
    --seed "${seed}" \
    "${extra_args[@]}"
done

echo "[done] requested seeds: ${SEEDS}"
