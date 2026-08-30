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
if [[ "${MODE}" != "prepare" && "${MODE}" != "run" && "${MODE}" != "finalize" && "${MODE}" != "all" ]]; then
  echo "[error] MODE must be prepare, run, finalize, or all" >&2
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

if [[ "${MODE}" == "finalize" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/judgements.jsonl" || { echo "[error] run judge first" >&2; exit 2; }
  echo "[stage 3/3] validate quotes and emit aggregate-only summary"
  python scripts/audit_sft_family_raw_outputs.py finalize-ddxplus \
    --private-bundle "${OUT}/private_bundle.jsonl" \
    --judgements "${OUT}/judgements.jsonl" \
    --out-dir "${OUT}"
fi

echo "[done] ${OUT}"
