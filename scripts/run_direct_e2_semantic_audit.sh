#!/usr/bin/env bash
set -euo pipefail

# Keep restricted readouts and semantic judgements under the private data root.
DATA_ROOT="${DATA_ROOT:-/data/heejae}"
GPU="${GPU:-2}"
LIMIT="${LIMIT:-}"
RUN="${DATA_ROOT}/restricted/direct/e2/direct_e2_val_v1"
E1="${DATA_ROOT}/restricted/direct/e1"
AUDIT="${RUN}/semantic_audit_v1"
MODEL="${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original"
OFFICIAL="${DATA_ROOT}/restricted/direct/official_repo"

mkdir -p "${AUDIT}"

READOUT_ARGS=()
for prompt in default task_aligned; do
  for layer in 16 24 32; do
    path="${RUN}/vanilla_av_${prompt}_p0_hs${layer}_val.jsonl"
    test -f "${path}"
    READOUT_ARGS+=(--readout "${prompt}_HS${layer}=${path}")
  done
done

python scripts/make_direct_e2_semantic_audit.py \
  "${READOUT_ARGS[@]}" \
  --source-answers \
    "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl" \
    "${E1}/direct_e1_test_v1/source_cot_answers.jsonl" \
  --requests "${AUDIT}/requests.jsonl" \
  --index "${AUDIT}/private_index.jsonl"

LIMIT_ARGS=()
if [[ -n "${LIMIT}" ]]; then
  LIMIT_ARGS=(--limit "${LIMIT}")
fi

CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
  scripts/run_direct_local_llama_judge.py \
  --requests "${AUDIT}/requests.jsonl" \
  --out "${AUDIT}/judgements.jsonl" \
  --official-repo "${OFFICIAL}" \
  --ckpt-dir "${MODEL}" \
  --tokenizer-path "${MODEL}/tokenizer.model" \
  --max-seq-len 8192 \
  --max-batch-size 4 \
  --max-gen-len 192 \
  "${LIMIT_ARGS[@]}"

python scripts/analyze_direct_e2_semantic_audit.py \
  --index "${AUDIT}/private_index.jsonl" \
  --judgements "${AUDIT}/judgements.jsonl" \
  --summary-md "${AUDIT}/summary.md" \
  --audit-jsonl "${AUDIT}/private_audit.jsonl" \
  --manual-primary-jsonl "${AUDIT}/manual_default_hs32.jsonl" \
  --primary-arm default_HS32

cat "${AUDIT}/summary.md"
