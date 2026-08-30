#!/usr/bin/env bash
set -euo pipefail

# Public DDXPlus validation raw audit. Deletion and value-edit cohorts are
# selected separately so the deletion audit is not conditioned on value data.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
MODE="${MODE:-all}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-5.4}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_sft_raw_audit50_v1}"

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

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
CONTROL17="${CONTROL17:-${E5}/common_medical_nla_full_sft_v1_seed17_ddx_grounding_val_v1/medical_nla_seed17.jsonl}"
CONTROL29="${CONTROL29:-${E5}/common_medical_nla_full_sft_v1_seed29_ddx_grounding_val_v1/medical_nla_seed29.jsonl}"
CF17="${CF17:-${E5}/ddxplus_counterfactual_sft_v1_seed17_ddx_grounding_val_v1/medical_nla_seed17.jsonl}"
CF29="${CF29:-${E5}/ddxplus_counterfactual_sft_v1_seed29_ddx_grounding_val_v1/medical_nla_seed29.jsonl}"

for path in "${CONTROL17}" "${CONTROL29}" "${CF17}" "${CF29}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
  echo "[stage 1/3] deterministic deletion/value-edit cohort preflight"
  python scripts/audit_sft_family_raw_outputs.py prepare-ddxplus \
    --readout "original_only_seed17=${CONTROL17}" \
    --readout "original_only_seed29=${CONTROL29}" \
    --readout "counterfactual_seed17=${CF17}" \
    --readout "counterfactual_seed29=${CF29}" \
    --cases 50 \
    --seed 17 \
    --out-dir "${OUT}"
fi

if [[ "${MODE}" == "run" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/requests.jsonl" || { echo "[error] prepare first" >&2; exit 2; }
  echo "[stage 2/3] method-blind AI checklist"
  python scripts/run_judge.py \
    --requests "${OUT}/requests.jsonl" \
    --out "${OUT}/judgements.jsonl" \
    --backend codex \
    --model "${JUDGE_MODEL}" \
    --timeout 300
fi

if [[ "${MODE}" == "repair" ]]; then
  test -s "${OUT}/judgements.jsonl" || { echo "[error] run judge first" >&2; exit 2; }
  mkdir -p "${OUT}/retries"
  current="${OUT}/judgements.jsonl"
  for attempt in 1 2 3; do
    retry_requests="${OUT}/retries/retry_requests_${attempt}.jsonl"
    audit_report="${OUT}/retries/audit_${attempt}.json"
    python scripts/audit_sft_family_raw_outputs.py audit-ddxplus-judgements \
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
    python scripts/run_judge.py \
      --requests "${retry_requests}" \
      --out "${retry_judgements}" \
      --backend codex \
      --model "${JUDGE_MODEL}" \
      --timeout 300
    merged="${OUT}/retries/judgements_merged_${attempt}.jsonl"
    python scripts/merge_semantic_judgement_shards.py \
      --requests "${OUT}/requests.jsonl" \
      --judgement "${current}" \
      --replacement-judgement "${retry_judgements}" \
      --output "${merged}" \
      --expected-model "${JUDGE_MODEL}" \
      --report "${OUT}/retries/merge_${attempt}.json"
    current="${merged}"
  done
  if [[ ! -s "${OUT}/judgements_validated.jsonl" ]]; then
    python scripts/audit_sft_family_raw_outputs.py audit-ddxplus-judgements \
      --private-bundle "${OUT}/private_bundle.jsonl" \
      --requests "${OUT}/requests.jsonl" \
      --judgements "${current}" \
      --retry-requests "${OUT}/retries/retry_requests_final.jsonl" \
      --report "${OUT}/retries/audit_final.json"
    final_invalid="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["invalid"])' "${OUT}/retries/audit_final.json")"
    if [[ "${final_invalid}" -eq 0 ]]; then
      cp "${current}" "${OUT}/judgements_validated.jsonl"
    fi
  fi
  test -s "${OUT}/judgements_validated.jsonl" || {
    echo "[error] invalid DDXPlus judge responses remain after three repairs" >&2
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
  python scripts/audit_sft_family_raw_outputs.py finalize-ddxplus \
    --private-bundle "${OUT}/private_bundle.jsonl" \
    --judgements "${judgement_path}" \
    --out-dir "${OUT}"
fi

echo "[done] ${OUT}"
