#!/usr/bin/env bash
set -euo pipefail

# Unattended validation-only paired grounding diagnostic for common Medical-NLA.
# Four of forty deterministic base-ID shards yield roughly 1,000 rows while
# preserving every selected original/deletion/value-edit family.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae on server 62}"
GPUS="${GPUS:-2,3}"
COMMON_RUN_NAME="${COMMON_RUN_NAME:-common_medical_nla_pilot_v1}"
SEEDS="${SEEDS:-17 43}"
NUM_SHARDS="${NUM_SHARDS:-40}"
SELECT_SHARDS="${SELECT_SHARDS:-0 1 2 3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"
THRESHOLD="${THRESHOLD:-0.5}"
GROUNDING_RUN_NAME="${GROUNDING_RUN_NAME:-${COMMON_RUN_NAME}_ddx_grounding_val_v1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
SOURCE_MANIFEST="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
COMMON_ROOT="${DATA_ROOT}/restricted/direct/e3/${COMMON_RUN_NAME}"
OUT="${E5}/${GROUNDING_RUN_NAME}"
SHARDS="${OUT}/manifest_shards"
PILOT="${OUT}/paired_manifest.jsonl"
PROMPT="prompt_templates/common_p0_clinical_state_readout.txt"
mkdir -p "${OUT}" "${DATA_ROOT}/medical_nla/logs"

test -s "${SOURCE_MANIFEST}" || { echo "[error] missing ${SOURCE_MANIFEST}" >&2; exit 2; }
source_rows="$(wc -l < "${SOURCE_MANIFEST}")"
if [[ "${source_rows}" -ne 10006 ]]; then
  echo "[error] validation manifest has ${source_rows} rows; expected 10006" >&2
  exit 2
fi

python scripts/shard_jsonl_by_key.py \
  --input "${SOURCE_MANIFEST}" \
  --out-dir "${SHARDS}" \
  --num-shards "${NUM_SHARDS}" \
  --key base_id

merge_args=()
expected_rows=0
for shard in ${SELECT_SHARDS}; do
  printf -v path '%s/shard_%03d_of_%03d.jsonl' "${SHARDS}" "${shard}" "${NUM_SHARDS}"
  test -s "${path}" || { echo "[error] missing selected shard ${path}" >&2; exit 2; }
  merge_args+=(--input "${path}")
  rows="$(wc -l < "${path}")"
  expected_rows=$((expected_rows + rows))
done
python scripts/merge_jsonl_files.py \
  "${merge_args[@]}" \
  --output "${PILOT}" \
  --expected-rows "${expected_rows}"

score_args=()
for seed in ${SEEDS}; do
  label="medical_nla_seed${seed}"
  adapter="${COMMON_ROOT}/adapters/${COMMON_RUN_NAME}_seed${seed}"
  output="${OUT}/${label}.jsonl"
  test -s "${adapter}/best.json" || { echo "[error] incomplete ${adapter}" >&2; exit 2; }
  if [[ ! -s "${output}" ]] || [[ "$(wc -l < "${output}")" -ne "${expected_rows}" ]]; then
    echo "[readout] ${label}: ${expected_rows} paired validation rows"
    CUDA_VISIBLE_DEVICES="${GPUS}" python -m src.run_nla \
      --config configs/default.yaml \
      --manifest "${PILOT}" \
      --output "${output}" \
      --actor-prompt-template-file "${PROMPT}" \
      --adapter-id "${adapter}" \
      --max-new-tokens "${MAX_NEW_TOKENS}" \
      --batch-size "${BATCH_SIZE}"
  else
    echo "[skip] ${label}: complete ${expected_rows}-row output"
  fi
  score_args+=(--readout "${label}=${output}")
done

python scripts/score_ddxplus_e5_readout_pilot.py \
  "${score_args[@]}" \
  --threshold "${THRESHOLD}" \
  --output-json "${OUT}/paired_scores.json" \
  --summary-md "${OUT}/paired_scores_summary.md"

echo "[done] ${OUT}"
cat "${OUT}/paired_scores_summary.md"
