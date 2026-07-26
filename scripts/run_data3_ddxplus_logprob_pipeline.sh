#!/usr/bin/env bash
set -euo pipefail

# End-to-end /data3 pipeline:
# 1) build DDXPlus probe rows
# 2) extract Gemma activations
# 3) filter multi_format manifest
# 4) run vanilla NLA diagnosis logprob probe
#
# Override defaults with environment variables, for example:
#   GPU=1 FORCE=1 bash scripts/run_data3_ddxplus_logprob_pipeline.sh

CODE_ROOT="${CODE_ROOT:-/home/eagle0914/medical_nla}"
DATA_ROOT="${DATA_ROOT:-/data3/heejae}"
VENV_ACTIVATE="${VENV_ACTIVATE:-${DATA_ROOT}/uv/medical_nla/bin/activate}"
CONFIG="${CONFIG:-configs/data3.yaml}"
GPU="${GPU:-0}"
RUN_NAME="${RUN_NAME:-ddxplus_probe_v1}"
EXAMPLES_PER_DIAGNOSIS="${EXAMPLES_PER_DIAGNOSIS:-100}"
MAX_CUES="${MAX_CUES:-3}"
SEED="${SEED:-17}"
CANDIDATE_BATCH_SIZE="${CANDIDATE_BATCH_SIZE:-8}"
FORCE="${FORCE:-0}"

ARTIFACT_ROOT="${DATA_ROOT}/medical_nla"
PATIENTS="${PATIENTS:-${DATA_ROOT}/ddxplus/train.csv}"
EVIDENCES="${EVIDENCES:-${DATA_ROOT}/ddxplus/release_evidences.json}"
CASES_JSONL="${CASES_JSONL:-${ARTIFACT_ROOT}/${RUN_NAME}_cases.jsonl}"
VARIANTS_JSONL="${VARIANTS_JSONL:-${ARTIFACT_ROOT}/${RUN_NAME}_variants.jsonl}"
ACTIVATION_DIR="${ARTIFACT_ROOT}/activations/${RUN_NAME}"
MANIFEST_JSONL="${ACTIVATION_DIR}/manifest.jsonl"
MULTI_FORMAT_MANIFEST="${ACTIVATION_DIR}/manifest_multi_format.jsonl"
RESULT_JSONL="${RESULT_JSONL:-${ARTIFACT_ROOT}/results/ddxplus_nla_multi_format_logprobs_calibrated_v1.jsonl}"
SUMMARY_MD="${SUMMARY_MD:-${ARTIFACT_ROOT}/results/ddxplus_nla_multi_format_logprobs_calibrated_v1_summary.md}"

mkdir -p \
  "${ARTIFACT_ROOT}/activations" \
  "${ARTIFACT_ROOT}/results" \
  "${ARTIFACT_ROOT}/reports" \
  "${ARTIFACT_ROOT}/logs" \
  "${ARTIFACT_ROOT}/probe" \
  "${ARTIFACT_ROOT}/train" \
  "${ARTIFACT_ROOT}/adapters" \
  "${DATA_ROOT}/hf_cache"

cd "${CODE_ROOT}"
source "${VENV_ACTIVATE}"

export PYTHONPATH="${CODE_ROOT}"
export HF_HOME="${HF_HOME:-${DATA_ROOT}/hf_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${DATA_ROOT}/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${DATA_ROOT}/hf_cache/datasets}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "[env] CODE_ROOT=${CODE_ROOT}"
echo "[env] DATA_ROOT=${DATA_ROOT}"
echo "[env] CONFIG=${CONFIG}"
echo "[env] GPU=${GPU}"
echo "[env] RUN_NAME=${RUN_NAME}"
echo "[env] HF_HOME=${HF_HOME}"

if [[ ! -f "${PATIENTS}" ]]; then
  echo "[error] Missing DDXPlus patients file: ${PATIENTS}" >&2
  echo "Download it first with: hf download aai530-group6/ddxplus --repo-type dataset --local-dir ${DATA_ROOT}/ddxplus" >&2
  exit 1
fi
if [[ ! -f "${EVIDENCES}" ]]; then
  echo "[error] Missing DDXPlus evidences file: ${EVIDENCES}" >&2
  exit 1
fi

if [[ "${FORCE}" == "1" || ! -s "${VARIANTS_JSONL}" || ! -s "${CASES_JSONL}" ]]; then
  echo "[stage 1/4] building DDXPlus probe rows"
  python scripts/make_ddxplus_probe_dataset.py \
    --patients "${PATIENTS}" \
    --evidences "${EVIDENCES}" \
    --cases-output "${CASES_JSONL}" \
    --variants-output "${VARIANTS_JSONL}" \
    --examples-per-diagnosis "${EXAMPLES_PER_DIAGNOSIS}" \
    --max-cues "${MAX_CUES}" \
    --seed "${SEED}"
else
  echo "[stage 1/4] reusing existing probe rows: ${VARIANTS_JSONL}"
fi

echo "[check] cases=$(wc -l < "${CASES_JSONL}") variants=$(wc -l < "${VARIANTS_JSONL}")"

if [[ "${FORCE}" == "1" || ! -s "${MANIFEST_JSONL}" ]]; then
  echo "[stage 2/4] extracting activations"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m src.extract_activations \
    --config "${CONFIG}" \
    --input "${VARIANTS_JSONL}" \
    --run-name "${RUN_NAME}"
else
  echo "[stage 2/4] reusing existing activation manifest: ${MANIFEST_JSONL}"
fi

if [[ "${FORCE}" == "1" || ! -s "${MULTI_FORMAT_MANIFEST}" ]]; then
  echo "[stage 3/4] filtering multi_format manifest"
  python - "${MANIFEST_JSONL}" "${MULTI_FORMAT_MANIFEST}" <<'PY'
import json
import sys

src, dst = sys.argv[1], sys.argv[2]
count = 0
with open(src, encoding="utf-8") as f, open(dst, "w", encoding="utf-8") as out:
    for line in f:
        row = json.loads(line)
        if row.get("variant") == "multi_format":
            out.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
print(f"multi_format_rows: {count}")
PY
else
  echo "[stage 3/4] reusing existing multi_format manifest: ${MULTI_FORMAT_MANIFEST}"
fi

echo "[check] manifest=$(wc -l < "${MANIFEST_JSONL}") multi_format=$(wc -l < "${MULTI_FORMAT_MANIFEST}")"

if [[ "${FORCE}" == "1" || ! -s "${RESULT_JSONL}" || ! -s "${SUMMARY_MD}" ]]; then
  echo "[stage 4/4] scoring NLA diagnosis logprobs"
  CUDA_VISIBLE_DEVICES="${GPU}" python -m scripts.score_nla_diagnosis_logprobs \
    --config "${CONFIG}" \
    --manifest "${MULTI_FORMAT_MANIFEST}" \
    --output-jsonl "${RESULT_JSONL}" \
    --summary-md "${SUMMARY_MD}" \
    --candidate-batch-size "${CANDIDATE_BATCH_SIZE}" \
    --rank-field calibrated_logprob_mean \
    --calibrate-with-token-baseline
else
  echo "[stage 4/4] reusing existing logprob result: ${RESULT_JSONL}"
fi

echo "[done] result=${RESULT_JSONL}"
echo "[done] summary=${SUMMARY_MD}"
if [[ -f "${SUMMARY_MD}" ]]; then
  echo "[summary]"
  cat "${SUMMARY_MD}"
fi
