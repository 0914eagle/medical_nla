#!/usr/bin/env bash
set -euo pipefail

# Server-125 entrypoint for the single approved DiReCT locked baseline batch.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU_PAIR="${GPU_PAIR:-0,1}"
JUDGE_GPU="${JUDGE_GPU:-2}"
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-codex}"
RUN_NAME="${RUN_NAME:-direct_locked_baselines_v1}"

test "${DATA_ROOT}" = "/data1/heejae" || {
  echo "[error] this entrypoint is restricted to server 125 (/data1/heejae)" >&2
  exit 2
}

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

D19=configs/decisions/d19_d10_budget1552_fail_v1.json
D21=configs/decisions/d21_d20_specificity_anchor_fail_v1.json
RECIPE=configs/decisions/direct_locked_baseline_only_v1.json
PROBE_CONTROL=configs/direct_locked_probe_control_v1.json
DIRECT="${DATA_ROOT}/restricted/direct"
SPLITS="${DIRECT}/splits/direct_patient_pdd_confirmatory_v1"
ACT="${DIRECT}/e1/direct_e1_reindexed_confirmatory_v1/activations"
PREFLIGHT="${DIRECT}/paper/${RUN_NAME}/preflight"

python scripts/validate_direct_locked_baseline_recipe.py \
  --d19 "${D19}" --d21 "${D21}" --recipe "${RECIPE}"

for path in \
  "${SPLITS}/protocol.json" \
  "${SPLITS}/test_seen.jsonl" \
  "${SPLITS}/test_pdd_heldout.jsonl" \
  "${DIRECT}/e2/direct_e2_probe_pdd_val_v1/canonical_pdd_hs24.pt" \
  "${DIRECT}/e2/direct_e2_probe_category_val_v1/disease_category_hs24.pt" \
  "${ACT}/layer24/last_token/manifest_test_seen.jsonl" \
  "${ACT}/layer32/last_token/manifest_test_seen.jsonl" \
  "${ACT}/layer32/last_token/manifest_test_pdd_heldout.jsonl"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
  echo "[preflight] OK ${path}"
done

mkdir -p "${PREFLIGHT}"
python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${ACT}/layer32/last_token/manifest_test_seen.jsonl" \
  --output "${PREFLIGHT}/unused_dump_mode.jsonl" \
  --dump-actor-prompt-template \
  > "${PREFLIGHT}/vanilla_actor_prompt.txt"

EXPECTED_ACTOR_PROMPT_SHA256="$(sha256sum "${PREFLIGHT}/vanilla_actor_prompt.txt" | awk '{print $1}')"
EXPECTED_SPLIT_PROTOCOL_SHA256="$(sha256sum "${SPLITS}/protocol.json" | awk '{print $1}')"
printf '[preflight] actor prompt sha256=%s\n' "${EXPECTED_ACTOR_PROMPT_SHA256}"
printf '[preflight] split protocol sha256=%s\n' "${EXPECTED_SPLIT_PROTOCOL_SHA256}"

DATA_ROOT="${DATA_ROOT}" \
D10_DECISION="${D19}" \
D20_DECISION="${D21}" \
FINAL_RECIPE="${RECIPE}" \
PROBE_CONTROL_PROTOCOL="${PROBE_CONTROL}" \
EXPECTED_ACTOR_PROMPT_SHA256="${EXPECTED_ACTOR_PROMPT_SHA256}" \
EXPECTED_SPLIT_PROTOCOL_SHA256="${EXPECTED_SPLIT_PROTOCOL_SHA256}" \
GPU_PAIR="${GPU_PAIR}" \
JUDGE_GPU="${JUDGE_GPU}" \
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND}" \
RUN_NAME="${RUN_NAME}" \
bash scripts/run_direct_locked_baseline_batch.sh
