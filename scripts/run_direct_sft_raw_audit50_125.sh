#!/usr/bin/env bash
set -euo pipefail

# Private DiReCT 50-case raw audit. Restricted text is judged only by the
# server-local Llama checkpoint and remains under restricted/direct/e4.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU="${GPU:-0}"
MODE="${MODE:-all}"
OUT="${OUT:-${DATA_ROOT}/restricted/direct/e4/sft_raw_audit50_v2}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this wrapper is frozen for server 125 (/data1/heejae)" >&2
  exit 2
fi
if [[ "${MODE}" != "prepare" && "${MODE}" != "run" && "${MODE}" != "repair" && "${MODE}" != "finalize" && "${MODE}" != "all" ]]; then
  echo "[error] MODE must be prepare, run, repair, finalize, or all" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT="${DATA_ROOT}/restricted/direct"
E1="${DIRECT}/e1"
E3="${DIRECT}/e3"
E4="${DIRECT}/e4"
DIRECT_READOUTS="${DIRECT_READOUTS:-${E4}/validation_readouts_v1}"
DIRECT_SEMANTIC="${DIRECT_SEMANTIC:-${E4}/validation_full_v1}"
FULL_READOUTS="${FULL_READOUTS:-${E4}/common_medical_nla_full_sft_v1_validation_v1}"
FULL_SEMANTIC="${FULL_SEMANTIC:-${E4}/common_medical_nla_full_sft_v1_direct_semantic_val_v1}"
VANILLA_READOUT="${VANILLA_READOUT:-${DIRECT_READOUTS}/vanilla.jsonl}"
VANILLA_SEMANTIC="${VANILLA_SEMANTIC:-${DIRECT_SEMANTIC}}"
VANILLA_SOURCE_FILTER="${VANILLA_SOURCE_FILTER:--}"
JUDGE="${DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original"

# Server 125 originally generated the shared Vanilla arm in the 100-row common
# pilot. Its 50 Direct rows use the same frozen DiReCT validation activations.
# Keep the readout and semantic root paired rather than copying only one file.
if [[ ! -s "${VANILLA_READOUT}" ]]; then
  fallback_readout="${E4}/common_medical_nla_pilot_v1_validation_v1/vanilla.jsonl"
  fallback_semantic="${E4}/common_medical_nla_pilot_v1_direct_semantic_val_v1"
  if [[ -s "${fallback_readout}" && -s "${fallback_semantic}/private_extraction_audit.jsonl" ]]; then
    VANILLA_READOUT="${fallback_readout}"
    VANILLA_SEMANTIC="${fallback_semantic}"
    VANILLA_SOURCE_FILTER=direct
    echo "[fallback] Vanilla Direct rows from ${VANILLA_READOUT}"
  fi
fi

required=(
  "${E3}/direct_e3_sft_v1/sft_val.jsonl"
  "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl"
  "${E1}/direct_e1_test_v1/source_cot_answers.jsonl"
  "${VANILLA_READOUT}"
  "${DIRECT_READOUTS}/medical_nla_seed17.jsonl"
  "${DIRECT_READOUTS}/medical_nla_seed29.jsonl"
  "${DIRECT_READOUTS}/medical_nla_seed43.jsonl"
  "${FULL_READOUTS}/medical_nla_seed17.jsonl"
  "${FULL_READOUTS}/medical_nla_seed29.jsonl"
  "${DIRECT_SEMANTIC}/private_extraction_audit.jsonl"
  "${FULL_SEMANTIC}/private_extraction_audit.jsonl"
)
for path in "${required[@]}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

prepare() {
  python scripts/audit_sft_family_raw_outputs.py prepare-direct \
    --cohort "${E3}/direct_e3_sft_v1/sft_val.jsonl" \
    --source-answers \
      "${E1}/direct_e1_trainval_v1/source_cot_answers.jsonl" \
      "${E1}/direct_e1_test_v1/source_cot_answers.jsonl" \
    --method "source_cot|-|${FULL_SEMANTIC}|cot|-" \
    --method "vanilla|${VANILLA_READOUT}|${VANILLA_SEMANTIC}|vanilla|${VANILLA_SOURCE_FILTER}" \
    --method "direct_only_seed17|${DIRECT_READOUTS}/medical_nla_seed17.jsonl|${DIRECT_SEMANTIC}|medical_nla_seed17|-" \
    --method "direct_only_seed29|${DIRECT_READOUTS}/medical_nla_seed29.jsonl|${DIRECT_SEMANTIC}|medical_nla_seed29|-" \
    --method "direct_only_seed43|${DIRECT_READOUTS}/medical_nla_seed43.jsonl|${DIRECT_SEMANTIC}|medical_nla_seed43|-" \
    --method "full_data_seed17|${FULL_READOUTS}/medical_nla_seed17.jsonl|${FULL_SEMANTIC}|medical_nla_seed17|direct" \
    --method "full_data_seed29|${FULL_READOUTS}/medical_nla_seed29.jsonl|${FULL_SEMANTIC}|medical_nla_seed29|direct" \
    --cases 50 \
    --seed 17 \
    --methods-per-request 2 \
    --out-dir "${OUT}"
}

if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
  echo "[stage 1/3] deterministic intersection preflight and private bundle"
  prepare
fi

if [[ "${MODE}" == "run" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/requests.jsonl" || { echo "[error] prepare first" >&2; exit 2; }
  echo "[stage 2/3] local-only AI checklist; no external API"
  CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
    scripts/run_direct_local_llama_judge.py \
    --requests "${OUT}/requests.jsonl" \
    --out "${OUT}/judgements.jsonl" \
    --official-repo "${DIRECT}/official_repo" \
    --ckpt-dir "${JUDGE}" \
    --tokenizer-path "${JUDGE}/tokenizer.model" \
    --max-seq-len 4096 \
    --max-batch-size 1 \
    --max-gen-len 768 \
    --temperature 0 \
    --top-p 1 \
    --judge-model Meta-Llama-3-8B-Instruct
fi

if [[ "${MODE}" == "repair" ]]; then
  test -s "${OUT}/judgements.jsonl" || { echo "[error] run judge first" >&2; exit 2; }
  mkdir -p "${OUT}/retries"
  current="${OUT}/judgements.jsonl"
  for attempt in 1 2 3; do
    retry_requests="${OUT}/retries/retry_requests_${attempt}.jsonl"
    audit_report="${OUT}/retries/audit_${attempt}.json"
    python scripts/audit_sft_family_raw_outputs.py audit-direct-judgements \
      --private-bundle "${OUT}/private_bundle.jsonl" \
      --requests "${OUT}/requests.jsonl" \
      --judgements "${current}" \
      --retry-requests "${retry_requests}" \
      --report "${audit_report}"
    invalid="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["invalid"])' "${audit_report}")"
    if [[ "${invalid}" -eq 0 ]]; then
      cp "${current}" "${OUT}/judgements_validated.jsonl"
      break
    fi
    echo "[repair] attempt=${attempt} invalid=${invalid}"
    retry_judgements="${OUT}/retries/retry_judgements_${attempt}.jsonl"
    CUDA_VISIBLE_DEVICES="${GPU}" torchrun --nproc_per_node 1 \
      scripts/run_direct_local_llama_judge.py \
      --requests "${retry_requests}" \
      --out "${retry_judgements}" \
      --official-repo "${DIRECT}/official_repo" \
      --ckpt-dir "${JUDGE}" \
      --tokenizer-path "${JUDGE}/tokenizer.model" \
      --max-seq-len 4096 \
      --max-batch-size 1 \
      --max-gen-len 768 \
      --temperature 0 \
      --top-p 1 \
      --judge-model Meta-Llama-3-8B-Instruct
    merged="${OUT}/retries/judgements_merged_${attempt}.jsonl"
    python scripts/merge_semantic_judgement_shards.py \
      --requests "${OUT}/requests.jsonl" \
      --judgement "${current}" \
      --replacement-judgement "${retry_judgements}" \
      --output "${merged}" \
      --expected-model Meta-Llama-3-8B-Instruct \
      --report "${OUT}/retries/merge_${attempt}.json"
    current="${merged}"
  done
  test -s "${OUT}/judgements_validated.jsonl" || {
    echo "[error] invalid local-judge responses remain after three repairs" >&2
    exit 1
  }
fi

if [[ "${MODE}" == "finalize" || "${MODE}" == "all" ]]; then
  judgement_path="${OUT}/judgements_validated.jsonl"
  if [[ ! -s "${judgement_path}" ]]; then
    judgement_path="${OUT}/judgements.jsonl"
  fi
  test -s "${judgement_path}" || { echo "[error] run judge first" >&2; exit 2; }
  echo "[stage 3/3] validate quotes and emit aggregate-only summary"
  python scripts/audit_sft_family_raw_outputs.py finalize-direct \
    --private-bundle "${OUT}/private_bundle.jsonl" \
    --judgements "${judgement_path}" \
    --out-dir "${OUT}"
fi

echo "[done] ${OUT}"
