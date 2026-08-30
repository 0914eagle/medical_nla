#!/usr/bin/env bash
set -euo pipefail

# Resume-safe parallel semantic mapping for the already sealed DDXPlus Vanilla
# locked-test readouts. Each worker owns one judgement file; the final merge
# requires exact equality with the frozen request population.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
WORKERS="${WORKERS:-8}"
PRIMARY_MODEL="${PRIMARY_MODEL:-gpt-5.6-sol}"
CODEX_CMD="${CODEX_CMD:-codex}"
TIMEOUT="${TIMEOUT:-300}"
GEN="${GEN:-${DATA_ROOT}/medical_nla/results/ddxplus_vanilla_locked_generation_v1}"
MAP="${MAP:-${DATA_ROOT}/medical_nla/results/ddxplus_semantic_mapper_validation_v2}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_vanilla_locked_semantic_v1}"
HARD_PAIRS="${HARD_PAIRS:-${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/hard_shuffle_pairs_test.jsonl}"
COMMITTED_RECEIPT="docs/experiments/receipts/ddxplus_semantic_mapper_v2_receipt.json"
SCORER="scripts/score_ddxplus_semantic_readouts.py"

if ! [[ "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "[error] WORKERS must be a positive integer" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

GENERATION_SEAL="${GEN}/generation_seal.json"
MAPPER_RECEIPT="${MAP}/semantic_mapper_freeze_receipt.json"
SEMANTIC_PROTOCOL="${MAP}/frozen/semantic_protocol.json"
READOUT="${GEN}/vanilla_locked.jsonl"
REQUESTS="${OUT}/semantic_requests.jsonl"
PREPARED="${OUT}/prepared_items.jsonl"
SHARD_ROOT="${OUT}/parallel_judge_${WORKERS}way"
REQUEST_SHARDS="${SHARD_ROOT}/requests"
JUDGEMENT_SHARDS="${SHARD_ROOT}/judgements"
LOG_ROOT="${SHARD_ROOT}/logs"
MERGED="${OUT}/semantic_judgements.jsonl"

for path in "${GENERATION_SEAL}" "${MAPPER_RECEIPT}" "${SEMANTIC_PROTOCOL}" \
  "${READOUT}" "${HARD_PAIRS}" "${COMMITTED_RECEIPT}" "${SCORER}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

python scripts/manage_nla_generation_seal.py verify --receipt "${GENERATION_SEAL}"
python - "${MAPPER_RECEIPT}" "${COMMITTED_RECEIPT}" "${SEMANTIC_PROTOCOL}" "${SCORER}" "${PRIMARY_MODEL}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

runtime_path, committed_path, protocol_path, scorer_path = map(Path, sys.argv[1:5])
expected_model = sys.argv[5]
runtime = json.load(runtime_path.open())
committed = json.load(committed_path.open())
if runtime != committed:
    raise SystemExit("[error] runtime mapper receipt differs from committed receipt")
if not runtime.get("all_gates_passed"):
    raise SystemExit("[error] mapper validation gates did not all pass")
if runtime.get("primary_model_id") != expected_model:
    raise SystemExit("[error] PRIMARY_MODEL differs from frozen mapper receipt")

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if digest(protocol_path) != runtime["protocol_sha256"]:
    raise SystemExit("[error] semantic protocol hash mismatch")
if digest(scorer_path) != runtime["scorer"]["sha256"]:
    raise SystemExit("[error] frozen semantic scorer hash mismatch")
print("[gate] committed mapper receipt, protocol, scorer, and model verified")
PY
python scripts/validate_semantic_mapper_freeze_receipt.py \
  --receipt "${MAPPER_RECEIPT}" \
  --expected-protocol-sha256 "$(sha256sum "${SEMANTIC_PROTOCOL}" | awk '{print $1}')"

mkdir -p "${OUT}" "${REQUEST_SHARDS}" "${JUDGEMENT_SHARDS}" "${LOG_ROOT}"
echo "[stage 1/5] prepare frozen locked-test requests"
python "${SCORER}" prepare \
  --readouts "${READOUT}" \
  --protocol "${SEMANTIC_PROTOCOL}" \
  --population locked_test \
  --out-dir "${OUT}"

expected_rows="$(wc -l < "${REQUESTS}" | tr -d ' ')"
test "${expected_rows}" -gt 0 || { echo "[error] no semantic requests" >&2; exit 2; }
echo "[population] semantic requests=${expected_rows} workers=${WORKERS}"

echo "[stage 2/5] deterministically shard requests"
python scripts/shard_jsonl_by_key.py \
  --input "${REQUESTS}" \
  --out-dir "${REQUEST_SHARDS}" \
  --num-shards "${WORKERS}" \
  --key id \
  --prefix request

echo "[stage 3/5] judge ${WORKERS} shards in parallel (resume-safe)"
pids=()
labels=()
for request_shard in "${REQUEST_SHARDS}"/request_*.jsonl; do
  label="$(basename "${request_shard}" .jsonl)"
  judgement="${JUDGEMENT_SHARDS}/${label}.jsonl"
  log="${LOG_ROOT}/${label}.log"
  python scripts/run_judge.py \
    --requests "${request_shard}" \
    --out "${judgement}" \
    --backend codex \
    --model "${PRIMARY_MODEL}" \
    --codex-cmd "${CODEX_CMD}" \
    --timeout "${TIMEOUT}" \
    > "${log}" 2>&1 &
  pid="$!"
  pids+=("${pid}")
  labels+=("${label}")
  echo "[launch] ${label} pid=${pid} log=${log}"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "[worker done] ${labels[$index]}"
  else
    echo "[worker failed] ${labels[$index]}" >&2
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "[error] at least one worker failed; inspect ${LOG_ROOT} and rerun" >&2
  exit 1
fi

echo "[stage 4/5] verify and merge exact judgement population"
merge_args=()
for judgement in "${JUDGEMENT_SHARDS}"/request_*.jsonl; do
  merge_args+=(--judgement "${judgement}")
done
python scripts/merge_semantic_judgement_shards.py \
  --requests "${REQUESTS}" \
  "${merge_args[@]}" \
  --output "${MERGED}" \
  --expected-model "${PRIMARY_MODEL}" \
  --report "${OUT}/semantic_judgement_merge_report.json"

echo "[stage 5/5] frozen finalize"
generation_protocol="$(python - "${GENERATION_SEAL}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["generation_protocol"]["path"])
PY
)"
manifest="$(python - "${generation_protocol}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["manifest"]["path"])
PY
)"
readout_hash="$(python - "${GENERATION_SEAL}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["readout"]["sha256"])
PY
)"
python "${SCORER}" finalize \
  --prepared-items "${PREPARED}" \
  --requests "${REQUESTS}" \
  --judgements "${MERGED}" \
  --manifest "${manifest}" \
  --protocol "${SEMANTIC_PROTOCOL}" \
  --mapper-receipt "${MAPPER_RECEIPT}" \
  --hard-pairs "${HARD_PAIRS}" \
  --readouts-sha256 "${readout_hash}" \
  --population locked_test \
  --out-dir "${OUT}"

sha256sum "${GENERATION_SEAL}" "${MAPPER_RECEIPT}" "${COMMITTED_RECEIPT}" \
  "${SEMANTIC_PROTOCOL}" "${SCORER}" "${REQUESTS}" "${MERGED}" \
  "${OUT}/semantic_decisions.jsonl" "${OUT}/results.json" \
  > "${OUT}/scoring_receipt.sha256"
echo "[done] locked semantic scores: ${OUT}"
cat "${OUT}/summary.md"
