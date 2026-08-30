#!/usr/bin/env bash
set -euo pipefail

# Single authorized DiReCT locked-label batch for Tables 1A, 1B, and 2.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
D10_DECISION="${D10_DECISION:-configs/decisions/d19_d10_budget1552_fail_v1.json}"
D20_DECISION="${D20_DECISION:-configs/decisions/d21_d20_specificity_anchor_fail_v1.json}"
FINAL_RECIPE="${FINAL_RECIPE:-configs/decisions/direct_locked_baseline_only_v1.json}"
PROBE_CONTROL_PROTOCOL="${PROBE_CONTROL_PROTOCOL:-configs/direct_locked_probe_control_v1.json}"
EXPECTED_D10_DECISION_SHA256="${EXPECTED_D10_DECISION_SHA256:-e459d5275a80b9493b2792f0fb3f181d717a2f6bba0f9b4c3573572e90438e48}"
EXPECTED_D20_DECISION_SHA256="${EXPECTED_D20_DECISION_SHA256:-1928046f8deeff2dccdd23d904019332979e22f1269dffcb278dafce49e88901}"
EXPECTED_FINAL_RECIPE_SHA256="${EXPECTED_FINAL_RECIPE_SHA256:-35bb8519df014e38d3cea89dcd6fc10969c67cd20e39aeb00e9f45fff19e45c5}"
EXPECTED_PROBE_CONTROL_SHA256="${EXPECTED_PROBE_CONTROL_SHA256:-391ea95fdf775d06ba45c1a6a5096fd23496afa65c10faf460f5800ef4a0115a}"
EXPECTED_ACTOR_PROMPT_SHA256="${EXPECTED_ACTOR_PROMPT_SHA256:?Set vanilla prompt SHA256}"
EXPECTED_SPLIT_PROTOCOL_SHA256="${EXPECTED_SPLIT_PROTOCOL_SHA256:?Set split protocol SHA256}"
GPU_PAIR="${GPU_PAIR:-0,1}"
JUDGE_GPU="${JUDGE_GPU:-2}"
EXTRACTOR_BACKEND="${EXTRACTOR_BACKEND:-codex}"
RUN_NAME="${RUN_NAME:-direct_locked_baselines_v1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DATA_ROOT}/restricted/direct"
SPLITS="${DIRECT}/splits/direct_patient_pdd_confirmatory_v1"
E1="${DIRECT}/e1"
ACT="${E1}/direct_e1_reindexed_confirmatory_v1/activations"
OUT="${DIRECT}/paper/${RUN_NAME}"
OFFICIAL="${DIRECT}/official_repo"
JUDGE="${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original"
MANIFEST_ALL="${DIRECT}/manifests/direct_canonical_v3_private.jsonl"

for path in \
  "${D10_DECISION}" "${D20_DECISION}" "${FINAL_RECIPE}" \
  "${PROBE_CONTROL_PROTOCOL}" \
  "${SPLITS}/protocol.json" "${SPLITS}/test_seen.jsonl" \
  "${SPLITS}/test_pdd_heldout.jsonl" "${MANIFEST_ALL}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
hash_matches() {
  local path="$1"
  local expected="$2"
  local label="$3"
  local actual
  actual="$(sha256sum "${path}" | awk '{print $1}')"
  test "${actual}" = "${expected}" || {
    echo "[error] ${label} hash mismatch: ${actual}" >&2
    exit 2
  }
}
hash_matches "${D10_DECISION}" "${EXPECTED_D10_DECISION_SHA256}" "D10 decision"
hash_matches "${D20_DECISION}" "${EXPECTED_D20_DECISION_SHA256}" "D20 decision"
hash_matches "${FINAL_RECIPE}" "${EXPECTED_FINAL_RECIPE_SHA256}" "final recipe"
hash_matches "${PROBE_CONTROL_PROTOCOL}" "${EXPECTED_PROBE_CONTROL_SHA256}" "probe control"
hash_matches "${SPLITS}/protocol.json" "${EXPECTED_SPLIT_PROTOCOL_SHA256}" "split protocol"
python scripts/validate_direct_locked_baseline_recipe.py \
  --d19 "${D10_DECISION}" \
  --d21 "${D20_DECISION}" \
  --recipe "${FINAL_RECIPE}"

DIRECT_ANSWERS=(
  "${E1}/direct_e1_trainval_direct_v1/source_direct_answers.jsonl"
  "${E1}/direct_e1_test_v1/source_direct_answers.jsonl"
)
COT_ANSWERS=(
  "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl"
  "${E1}/direct_e1_test_v1/source_cot_answers.jsonl"
)
PROBE_DIR="${DIRECT}/e2/direct_e2_probe_pdd_val_v1"
CATEGORY_PROBE_DIR="${DIRECT}/e2/direct_e2_probe_category_val_v1"
for path in \
  "${DIRECT_ANSWERS[@]}" "${COT_ANSWERS[@]}" \
  "${PROBE_DIR}/canonical_pdd_hs24.pt" \
  "${CATEGORY_PROBE_DIR}/disease_category_hs24.pt" \
  "${ACT}/layer24/last_token/manifest_train.jsonl" \
  "${ACT}/layer24/last_token/manifest_test_seen.jsonl" \
  "${ACT}/layer32/last_token/manifest_test_seen.jsonl" \
  "${ACT}/layer32/last_token/manifest_test_pdd_heldout.jsonl"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

mkdir -p "${OUT}/provenance" "${OUT}/manifests" "${OUT}/readouts"
python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${ACT}/layer32/last_token/manifest_test_seen.jsonl" \
  --output "${OUT}/unused_dump_mode.jsonl" \
  --dump-actor-prompt-template \
  > "${OUT}/provenance/vanilla_actor_prompt.txt"
hash_matches "${OUT}/provenance/vanilla_actor_prompt.txt" \
  "${EXPECTED_ACTOR_PROMPT_SHA256}" "vanilla actor prompt"

python scripts/reindex_and_score_direct_locked_source_outputs.py \
  --split-dir "${SPLITS}" \
  --answers "direct=${DIRECT_ANSWERS[0]}" \
  --answers "direct=${DIRECT_ANSWERS[1]}" \
  --answers "cot=${COT_ANSWERS[0]}" \
  --answers "cot=${COT_ANSWERS[1]}" \
  --out-dir "${OUT}/table1a_source" \
  --confirmation I_ACCEPT_DIRECT_LOCKED_SOURCE_REINDEX

CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" python scripts/evaluate_direct_locked_probes.py \
  --artifact "${PROBE_DIR}/canonical_pdd_hs24.pt" \
  --artifact "${CATEGORY_PROBE_DIR}/disease_category_hs24.pt" \
  --train-manifest "${ACT}/layer24/last_token/manifest_train.jsonl" \
  --manifest "${ACT}/layer24/last_token/manifest_test_seen.jsonl" \
  --control-protocol "${PROBE_CONTROL_PROTOCOL}" \
  --out-dir "${OUT}/table1b_probes" \
  --confirmation I_ACCEPT_DIRECT_LOCKED_PROBE_EVALUATION

python scripts/merge_jsonl_files.py \
  --input "${ACT}/layer32/last_token/manifest_test_seen.jsonl" \
  --input "${ACT}/layer32/last_token/manifest_test_pdd_heldout.jsonl" \
  --output "${OUT}/manifests/locked_hs32.jsonl" \
  --expected-rows 178
CUDA_VISIBLE_DEVICES="${GPU_PAIR}" python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "${OUT}/manifests/locked_hs32.jsonl" \
  --output "${OUT}/readouts/vanilla_locked.jsonl" \
  --max-new-tokens 512 \
  --batch-size 4
python scripts/validate_nla_readout_population.py \
  --manifest "${OUT}/manifests/locked_hs32.jsonl" \
  --readout "${OUT}/readouts/vanilla_locked.jsonl" \
  --expected-rows 178 \
  --expected-max-new-tokens 512 \
  --expected-do-sample false \
  --report "${OUT}/readouts/population_validation.json"

for split in test_seen test_pdd_heldout; do
  expected=72
  if [[ "${split}" = "test_pdd_heldout" ]]; then expected=106; fi
  POOL="${OUT}/${split}"
  mkdir -p "${POOL}"
  python scripts/filter_jsonl_by_field.py \
    --input "${OUT}/readouts/vanilla_locked.jsonl" \
    --field split \
    --value "${split}" \
    --expected-rows "${expected}" \
    --output "${POOL}/vanilla.jsonl"
  python scripts/make_direct_e4_claim_requests.py \
    --cohort "${SPLITS}/${split}.jsonl" \
    --case-manifest "${SPLITS}/${split}.jsonl" \
    --candidate-manifest "${MANIFEST_ALL}" \
    --source-answers "${COT_ANSWERS[@]}" \
    --readout "vanilla=${POOL}/vanilla.jsonl" \
    --requests "${POOL}/extraction_requests.jsonl" \
    --private-index "${POOL}/private_index.jsonl" \
    --summary-md "${POOL}/requests_summary.md" \
    --expected-cases "${expected}"
  if [[ "${EXTRACTOR_BACKEND}" = "codex" ]]; then
    python scripts/run_judge.py \
      --requests "${POOL}/extraction_requests.jsonl" \
      --out "${POOL}/extraction_judgements.jsonl" \
      --backend codex \
      --timeout 300
  else
    CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" torchrun --nproc_per_node 1 \
      scripts/run_direct_local_llama_judge.py \
      --requests "${POOL}/extraction_requests.jsonl" \
      --out "${POOL}/extraction_judgements.jsonl" \
      --official-repo "${OFFICIAL}" \
      --ckpt-dir "${JUDGE}" \
      --tokenizer-path "${JUDGE}/tokenizer.model" \
      --max-seq-len 8192 --max-batch-size 1 --max-gen-len 768 \
      --temperature 0 --top-p 1
  fi
  python scripts/apply_direct_e4_claim_extractions.py \
    --private-index "${POOL}/private_index.jsonl" \
    --judgements "${POOL}/extraction_judgements.jsonl" \
    --candidate-manifest "${MANIFEST_ALL}" \
    --prediction-root "${POOL}/predictions" \
    --audit-jsonl "${POOL}/private_extraction_audit.jsonl" \
    --summary-md "${POOL}/extraction_summary.md" \
    --expected-cases "${expected}"
  for method in cot vanilla; do
    CUDA_VISIBLE_DEVICES="${JUDGE_GPU}" torchrun --nproc_per_node 1 \
      scripts/run_direct_official_evaluator.py \
      --official-repo "${OFFICIAL}" \
      --samples-root "${DIRECT}/samples" \
      --prediction-root "${POOL}/predictions/${method}" \
      --eval-root "${POOL}/evaluations/${method}" \
      --ckpt-dir "${JUDGE}" \
      --tokenizer-path "${JUDGE}/tokenizer.model" \
      --max-seq-len 8192 --max-batch-size 4 --temperature 0 --top-p 1 \
      --response-mode official \
      --error-jsonl "${POOL}/private_errors_${method}.jsonl"
    python scripts/score_direct_official_eval.py \
      --prediction-root "${POOL}/predictions/${method}" \
      --eval-root "${POOL}/evaluations/${method}" \
      --output-json "${POOL}/reports/${method}.json" \
      --summary-md "${POOL}/reports/${method}.md"
  done
done

python scripts/summarize_direct_locked_paper_tables.py \
  --root "${OUT}" \
  --output-json "${OUT}/paper_tables_summary.json" \
  --summary-md "${OUT}/paper_tables_summary.md"

nvidia-smi -q > "${OUT}/provenance/nvidia_smi_q.txt"
python -m pip freeze > "${OUT}/provenance/pip_freeze.txt"
git rev-parse HEAD > "${OUT}/provenance/git_commit.txt"
echo "[done] ${OUT}"
