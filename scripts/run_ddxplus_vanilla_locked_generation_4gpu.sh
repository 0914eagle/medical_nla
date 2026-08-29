#!/usr/bin/env bash
set -euo pipefail

# Generate and seal all 10,028 DDXPlus locked Vanilla AV readouts without
# semantic mapping or text inspection. A later scorer must reuse this seal.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the server-local data root}"
MANIFEST="${MANIFEST:?Set the server-local locked HS32 manifest}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
RUN_NAME="${RUN_NAME:-ddxplus_vanilla_locked_generation_v1}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/${RUN_NAME}}"
CONFIRMATION="${CONFIRMATION:-}"
OPERATOR_ATTESTATION="${OPERATOR_ATTESTATION:-}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ "${CONFIRMATION}" != "I_GENERATE_SEALED_DDXPLUS_VANILLA" ]]; then
  echo "[error] set CONFIRMATION=I_GENERATE_SEALED_DDXPLUS_VANILLA" >&2
  exit 2
fi
if [[ "${OPERATOR_ATTESTATION}" != "NO_LOCKED_TEXT_INSPECTED" ]]; then
  echo "[error] set OPERATOR_ATTESTATION=NO_LOCKED_TEXT_INSPECTED" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

test -s "${MANIFEST}" || { echo "[error] missing ${MANIFEST}" >&2; exit 2; }
mkdir -p "${OUT}/provenance" "${OUT}/shards"

if [[ -s "${OUT}/generation_seal.json" ]]; then
  python scripts/manage_nla_generation_seal.py verify \
    --receipt "${OUT}/generation_seal.json"
  echo "[skip] sealed locked generation already complete: ${OUT}"
  exit 0
fi

python scripts/validate_ddxplus_locked_population.py \
  --input "${MANIFEST}" \
  --expected-rows 10028 \
  --expected-variant original=4543 \
  --expected-variant cue_deleted=4543 \
  --expected-variant value_edited=942 \
  --expected-layer 32 --require-activation-files \
  --report "${OUT}/provenance/manifest_population.json"

if [[ -s "${OUT}/generation_protocol.json" ]]; then
  python scripts/manage_nla_generation_seal.py verify-protocol \
    --protocol "${OUT}/generation_protocol.json" --require-current-git
else
  if [[ -s "${OUT}/vanilla_shard0.jsonl" || -s "${OUT}/vanilla_shard1.jsonl" ]]; then
    echo "[error] readout shard exists without a frozen generation protocol" >&2
    exit 2
  fi
  python scripts/dump_nla_actor_prompt_protocol.py \
    --config configs/default.yaml \
    --prompt-output "${OUT}/provenance/vanilla_actor_prompt.txt" \
    --metadata-output "${OUT}/provenance/vanilla_model_metadata.json"
  test -s "${OUT}/provenance/vanilla_actor_prompt.txt"

  python scripts/manage_nla_generation_seal.py freeze \
    --manifest "${MANIFEST}" \
    --actor-prompt "${OUT}/provenance/vanilla_actor_prompt.txt" \
    --config configs/default.yaml \
    --model-metadata "${OUT}/provenance/vanilla_model_metadata.json" \
    --population-report "${OUT}/provenance/manifest_population.json" \
    --model-id kitft/nla-gemma3-12b-L32-av \
    --expected-rows 10028 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    --confirmation I_FREEZE_DDXPLUS_VANILLA_GENERATION \
    --output "${OUT}/generation_protocol.json"
fi
model_revision="$(python - "${OUT}/generation_protocol.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["model_snapshot_revision"])
PY
)"

python scripts/shard_jsonl_by_key.py \
  --input "${MANIFEST}" \
  --out-dir "${OUT}/shards" \
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
  local output="${OUT}/vanilla_shard${index}.jsonl"
  local report="${OUT}/provenance/shard${index}_population.json"
  printf -v input '%s/shards/shard_%03d_of_002.jsonl' "${OUT}" "${index}"
  local expected
  expected="$(wc -l < "${input}")"
  if [[ -s "${output}" ]]; then
    if python scripts/validate_nla_readout_population.py \
      --manifest "${input}" --readout "${output}" \
      --expected-rows "${expected}" --expected-max-new-tokens "${MAX_NEW_TOKENS}" \
      --expected-do-sample false \
      --expected-actor-prompt-file "${OUT}/provenance/vanilla_actor_prompt.txt" \
      --expected-model-revision "${model_revision}" --report "${report}"; then
      echo "[skip] complete readout shard ${index}"
      return 0
    fi
    echo "[error] partial or invalid ${output}; preserve it and choose a new OUT" >&2
    return 2
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "${input}" \
    --output "${output}" \
    --actor-prompt-template-file "${OUT}/provenance/vanilla_actor_prompt.txt" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" \
    >"${OUT}/shard${index}.log" 2>&1
  python scripts/validate_nla_readout_population.py \
    --manifest "${input}" --readout "${output}" \
    --expected-rows "${expected}" --expected-max-new-tokens "${MAX_NEW_TOKENS}" \
    --expected-do-sample false \
    --expected-actor-prompt-file "${OUT}/provenance/vanilla_actor_prompt.txt" \
    --expected-model-revision "${model_revision}" --report "${report}"
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
  echo "[error] Vanilla generation failed: shard0=${status_a} shard1=${status_b}" >&2
  exit 1
fi

python scripts/merge_jsonl_files.py \
  --input "${OUT}/vanilla_shard0.jsonl" \
  --input "${OUT}/vanilla_shard1.jsonl" \
  --output "${OUT}/vanilla_locked.jsonl" \
  --expected-rows 10028
python scripts/validate_nla_readout_population.py \
  --manifest "${MANIFEST}" \
  --readout "${OUT}/vanilla_locked.jsonl" \
  --expected-rows 10028 \
  --expected-variant original=4543 \
  --expected-variant cue_deleted=4543 \
  --expected-variant value_edited=942 \
  --expected-max-new-tokens "${MAX_NEW_TOKENS}" \
  --expected-do-sample false \
  --expected-actor-prompt-file "${OUT}/provenance/vanilla_actor_prompt.txt" \
  --expected-model-revision "${model_revision}" \
  --report "${OUT}/population_validation.json"

python scripts/manage_nla_generation_seal.py seal \
  --protocol "${OUT}/generation_protocol.json" \
  --readout "${OUT}/vanilla_locked.jsonl" \
  --population-report "${OUT}/population_validation.json" \
  --operator-attestation "${OPERATOR_ATTESTATION}" \
  --output "${OUT}/generation_seal.json"

nvidia-smi -q > "${OUT}/provenance/nvidia_smi_q.txt"
python -m pip freeze > "${OUT}/provenance/pip_freeze.txt"
git rev-parse HEAD > "${OUT}/provenance/git_commit.txt"
chmod 0440 "${OUT}/vanilla_locked.jsonl" "${OUT}/generation_protocol.json" \
  "${OUT}/generation_seal.json"
echo "[done] sealed generation: ${OUT}"
echo "[do not inspect] wait for the G1-G4 mapper freeze receipt before scoring"
