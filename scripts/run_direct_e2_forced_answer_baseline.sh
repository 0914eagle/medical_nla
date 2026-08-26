#!/usr/bin/env bash
set -euo pipefail

# Validation-only DiReCT candidate ranking. The locked test paths do not appear
# in this script, so this job cannot accidentally consume them during method
# selection.
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to /data/heejae or /data1/heejae}"
GPUS="${GPUS:-0,1}"
LABEL_FIELD="${LABEL_FIELD:-canonical_pdd}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

case "${LABEL_FIELD}" in
  canonical_pdd)
    RUN_LABEL="pdd"
    ;;
  disease_category)
    RUN_LABEL="category"
    ;;
  *)
    echo "Unsupported LABEL_FIELD=${LABEL_FIELD}; expected canonical_pdd or disease_category" >&2
    exit 2
    ;;
esac

CANONICAL="${DATA_ROOT}/restricted/direct/manifests/direct_canonical_v3_private.jsonl"
ACT_ROOT="${DATA_ROOT}/restricted/direct/e1/direct_e1_reindexed_confirmatory_v1/activations"
VAL_MANIFEST="${ACT_ROOT}/layer32/last_token/manifest_val_seen.jsonl"
OUT="${DATA_ROOT}/restricted/direct/e2/direct_e2_forced_answer_${RUN_LABEL}_val_v1"
ONTOLOGY="${OUT}/${RUN_LABEL}_ontology.jsonl"

mkdir -p "${OUT}"

python scripts/make_direct_candidate_ontology.py \
  --manifest "${CANONICAL}" \
  --label-field "${LABEL_FIELD}" \
  --output-jsonl "${ONTOLOGY}" \
  --summary-md "${OUT}/ontology_summary.md"

# This is an early forced-answer continuation of the CoT prompt, not the raw
# next-token distribution and not a linear readout of the stored P0 vector.
CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/score_source_diagnosis_logprobs.py \
  --config configs/default.yaml \
  --input "${VAL_MANIFEST}" \
  --candidates-jsonl "${ONTOLOGY}" \
  --prompt-field prompt \
  --diagnosis-id-field "${LABEL_FIELD}" \
  --diagnosis-name-field "${LABEL_FIELD}" \
  --completion-prefix "The answer is" \
  --rank-field logprob_mean \
  --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
  --output-jsonl "${OUT}/scores.jsonl" \
  --summary-md "${OUT}/summary.md"

echo "[done] ${OUT}/summary.md"
