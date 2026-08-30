#!/usr/bin/env bash
set -euo pipefail

# Validation-only mapper freeze and G1-G4 audit. It never reads locked outputs.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
PRIMARY_MODEL="${PRIMARY_MODEL:-}"
AUDITOR_MODEL="${AUDITOR_MODEL:-}"
CODEX_CMD="${CODEX_CMD:-codex}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MODE="${MODE:-prepare}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_semantic_mapper_validation_v1}"

[[ "${MODE}" == "prepare" || "${MODE}" == "run" ]] || {
  echo "[error] MODE must be prepare or run" >&2; exit 2;
}

if [[ "${MODE}" == "run" ]]; then
  [[ -n "${PRIMARY_MODEL}" && -n "${AUDITOR_MODEL}" ]] || {
    echo "[error] MODE=run requires PRIMARY_MODEL and AUDITOR_MODEL" >&2; exit 2;
  }
  [[ "${PRIMARY_MODEL}" != "${AUDITOR_MODEL}" ]] || {
    echo "[error] PRIMARY_MODEL and AUDITOR_MODEL must differ" >&2; exit 2;
  }
  if [[ "${PRIMARY_MODEL,,}" == *gemma* || "${AUDITOR_MODEL,,}" == *gemma* ]]; then
    echo "[error] Gemma-family mapper/auditor is forbidden" >&2; exit 2
  fi
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
PROBE="${DATA_ROOT}/medical_nla/results/ddxplus_finding_value_probe_val_v1"
STRUCTURED_PROTOCOL="${STRUCTURED_PROTOCOL:-${PROBE}/structured_reader_hs24_protocol_v1.json}"
READER_READOUTS="${READER_READOUTS:-${DATA_ROOT}/medical_nla/results/ddxplus_structured_reader_validation_v1/readouts.jsonl}"
HARD_PAIRS="${HARD_PAIRS:-${E5}/hard_shuffle_pairs_validation.jsonl}"
OPEN_READOUTS="${OPEN_READOUTS:-${DATA_ROOT}/restricted/direct/e4/common_medical_nla_pilot_v1_validation_v1/vanilla.jsonl}"
EVIDENCES="${EVIDENCES:-${DATA_ROOT}/ddxplus/release_evidences.json}"

for path in "${STRUCTURED_PROTOCOL}" "${READER_READOUTS}" "${HARD_PAIRS}" \
  "${OPEN_READOUTS}" "${EVIDENCES}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
mkdir -p "${OUT}"

echo "[stage 1/8] freeze train/release-derived mapper artifacts"
python scripts/freeze_ddxplus_semantic_mapper.py \
  --structured-protocol "${STRUCTURED_PROTOCOL}" \
  --evidences "${EVIDENCES}" \
  --batch-size "${BATCH_SIZE}" \
  --out-dir "${OUT}/frozen"
PROTOCOL="${OUT}/frozen/semantic_protocol.json"

echo "[stage 2/8] prepare validation-only G1/G2 and open-text G4 pool"
python scripts/audit_ddxplus_semantic_mapper.py prepare \
  --protocol "${PROTOCOL}" \
  --reader-readouts "${READER_READOUTS}" \
  --hard-pairs "${HARD_PAIRS}" \
  --open-readouts "${OPEN_READOUTS}" \
  --open-source-dataset ddxplus \
  --out-dir "${OUT}/audit"
python scripts/run_judge.py \
  --requests "${OUT}/audit/primary_requests.jsonl" \
  --out "${OUT}/audit/primary_judgements.jsonl" \
  --backend dry-run --out-tokens 512
if [[ "${MODE}" == "prepare" ]]; then
  echo "[stop] inspect ${OUT}/audit/dry_run_report.json, then rerun MODE=run"
  exit 0
fi

echo "[stage 3/8] primary blind semantic mapping"
python scripts/run_judge.py \
  --requests "${OUT}/audit/primary_requests.jsonl" \
  --out "${OUT}/audit/primary_judgements.jsonl" \
  --backend codex --model "${PRIMARY_MODEL}" --codex-cmd "${CODEX_CMD}" \
  --timeout 300

echo "[stage 4/8] G1-G3 and frozen G4 sample"
python scripts/audit_ddxplus_semantic_mapper.py apply-primary \
  --protocol "${PROTOCOL}" \
  --prepared-items "${OUT}/audit/prepared_items.jsonl" \
  --requests "${OUT}/audit/primary_requests.jsonl" \
  --judgements "${OUT}/audit/primary_judgements.jsonl" \
  --out-dir "${OUT}/audit"

echo "[stage 5/8] cold duplicate diagnostic with the same primary model"
python scripts/run_judge.py \
  --requests "${OUT}/audit/cold_requests.jsonl" \
  --out "${OUT}/audit/cold_judgements.jsonl" \
  --backend codex --model "${PRIMARY_MODEL}" --codex-cmd "${CODEX_CMD}" \
  --timeout 300
python scripts/audit_ddxplus_semantic_mapper.py apply-cold \
  --protocol "${PROTOCOL}" \
  --primary-report "${OUT}/audit/primary_gate_report.json" \
  --primary-sample "${OUT}/audit/g3_primary_sample.jsonl" \
  --cold-requests "${OUT}/audit/cold_requests.jsonl" \
  --cold-judgements "${OUT}/audit/cold_judgements.jsonl"

echo "[stage 6/8] independent blind G4 remapping"
python scripts/run_judge.py \
  --requests "${OUT}/audit/auditor_requests.jsonl" \
  --out "${OUT}/audit/auditor_judgements.jsonl" \
  --backend codex --model "${AUDITOR_MODEL}" --codex-cmd "${CODEX_CMD}" \
  --timeout 300

echo "[stage 7/8] freeze G1-G4 receipt"
python scripts/audit_ddxplus_semantic_mapper.py finalize \
  --protocol "${PROTOCOL}" \
  --primary-report "${OUT}/audit/primary_gate_report.json" \
  --primary-sample "${OUT}/audit/g4_primary_sample.jsonl" \
  --auditor-requests "${OUT}/audit/auditor_requests.jsonl" \
  --auditor-judgements "${OUT}/audit/auditor_judgements.jsonl" \
  --output "${OUT}/semantic_mapper_freeze_receipt.json"

protocol_hash="$(sha256sum "${PROTOCOL}" | awk '{print $1}')"
echo "[stage 8/8] validate receipt and frozen artifacts"
python scripts/validate_semantic_mapper_freeze_receipt.py \
  --receipt "${OUT}/semantic_mapper_freeze_receipt.json" \
  --expected-protocol-sha256 "${protocol_hash}"
echo "[done] ${OUT}"
