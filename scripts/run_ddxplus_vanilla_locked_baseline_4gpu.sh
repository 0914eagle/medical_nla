#!/usr/bin/env bash
set -euo pipefail

# Paper-safe DDXPlus Vanilla baseline. Semantic scoring is deliberately a
# required external frozen protocol: the lexical pilot scorer is not accepted.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
MANIFEST="${MANIFEST:?Set the server-local locked-test HS32 manifest}"
SEMANTIC_PROTOCOL="${SEMANTIC_PROTOCOL:?Set the approved semantic mapper protocol JSON}"
SEMANTIC_SCORER="${SEMANTIC_SCORER:?Set the approved semantic scorer Python script}"
EXPECTED_ACTOR_PROMPT_SHA256="${EXPECTED_ACTOR_PROMPT_SHA256:?Set frozen actor prompt SHA256}"
EXPECTED_SEMANTIC_PROTOCOL_SHA256="${EXPECTED_SEMANTIC_PROTOCOL_SHA256:?Set semantic protocol SHA256}"
EXPECTED_SEMANTIC_SCORER_SHA256="${EXPECTED_SEMANTIC_SCORER_SHA256:?Set semantic scorer SHA256}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
RUN_NAME="${RUN_NAME:-ddxplus_vanilla_locked_v1}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/${RUN_NAME}}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

for path in "${MANIFEST}" "${SEMANTIC_PROTOCOL}" "${SEMANTIC_SCORER}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
actual_semantic_hash="$(sha256sum "${SEMANTIC_PROTOCOL}" | awk '{print $1}')"
actual_scorer_hash="$(sha256sum "${SEMANTIC_SCORER}" | awk '{print $1}')"
test "${actual_semantic_hash}" = "${EXPECTED_SEMANTIC_PROTOCOL_SHA256}" || {
  echo "[error] semantic protocol hash mismatch" >&2; exit 2;
}
test "${actual_scorer_hash}" = "${EXPECTED_SEMANTIC_SCORER_SHA256}" || {
  echo "[error] semantic scorer hash mismatch" >&2; exit 2;
}

mkdir -p "${OUT}/provenance" "${OUT}/shards" "${DATA_ROOT}/medical_nla/logs"
python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${MANIFEST}" \
  --output "${OUT}/unused_dump_mode.jsonl" \
  --dump-actor-prompt-template \
  > "${OUT}/provenance/vanilla_actor_prompt.txt"
actual_prompt_hash="$(sha256sum "${OUT}/provenance/vanilla_actor_prompt.txt" | awk '{print $1}')"
test "${actual_prompt_hash}" = "${EXPECTED_ACTOR_PROMPT_SHA256}" || {
  echo "[error] actor prompt hash mismatch: ${actual_prompt_hash}" >&2; exit 2;
}

python scripts/shard_jsonl_by_key.py \
  --input "${MANIFEST}" \
  --out-dir "${OUT}/shards" \
  --num-shards 2 \
  --key base_id

SHARD_A="${OUT}/shards/shard_000_of_002.jsonl"
SHARD_B="${OUT}/shards/shard_001_of_002.jsonl"
READOUT_A="${OUT}/vanilla_shard0.jsonl"
READOUT_B="${OUT}/vanilla_shard1.jsonl"

CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}" python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${SHARD_A}" \
  --output "${READOUT_A}" \
  --max-new-tokens 512 \
  --batch-size 4 \
  > "${OUT}/shard0.log" 2>&1 &
pid_a=$!
CUDA_VISIBLE_DEVICES="${GPU_PAIR_B}" python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${SHARD_B}" \
  --output "${READOUT_B}" \
  --max-new-tokens 512 \
  --batch-size 4 \
  > "${OUT}/shard1.log" 2>&1 &
pid_b=$!

status_a=0
status_b=0
wait "${pid_a}" || status_a=$?
wait "${pid_b}" || status_b=$?
if [[ "${status_a}" -ne 0 || "${status_b}" -ne 0 ]]; then
  echo "[error] generation failed: shard0=${status_a} shard1=${status_b}" >&2
  exit 1
fi

python scripts/merge_jsonl_files.py \
  --input "${READOUT_A}" \
  --input "${READOUT_B}" \
  --output "${OUT}/vanilla_locked.jsonl" \
  --expected-rows 10028
python scripts/validate_nla_readout_population.py \
  --manifest "${MANIFEST}" \
  --readout "${OUT}/vanilla_locked.jsonl" \
  --expected-rows 10028 \
  --expected-variant original=4543 \
  --expected-variant cue_deleted=4543 \
  --expected-variant value_edited=942 \
  --expected-max-new-tokens 512 \
  --expected-do-sample false \
  --report "${OUT}/population_validation.json"

nvidia-smi -q > "${OUT}/provenance/nvidia_smi_q.txt"
python -m pip freeze > "${OUT}/provenance/pip_freeze.txt"
git rev-parse HEAD > "${OUT}/provenance/git_commit.txt"
sha256sum configs/default.yaml "${MANIFEST}" "${SEMANTIC_PROTOCOL}" "${SEMANTIC_SCORER}" \
  > "${OUT}/provenance/input_hashes.sha256"

python "${SEMANTIC_SCORER}" \
  --readouts "${OUT}/vanilla_locked.jsonl" \
  --manifest "${MANIFEST}" \
  --protocol "${SEMANTIC_PROTOCOL}" \
  --out-dir "${OUT}/semantic"

test -s "${OUT}/semantic/results.json" || {
  echo "[error] semantic scorer did not write ${OUT}/semantic/results.json" >&2; exit 2;
}
echo "[done] ${OUT}"
