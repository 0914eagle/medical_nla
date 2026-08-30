#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPUS="${GPUS:-0,1}"
MODE="${MODE:-all}"
CASES="${CASES:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PRIMARY_MODEL="${PRIMARY_MODEL:-gpt-5.6-sol}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/ddxplus_d22_patchscope50_v1}"
EXPECTED_MAPPER_SHA="12e4500fa45f90d11c0146ad12e972afd9b5bd80128f49b388b11dea360b506b"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] this validation wrapper is frozen for server 125" >&2
  exit 2
fi
case "${MODE}" in
  prepare|generate|mapper-prepare|mapper-run|finalize|all) ;;
  *) echo "[error] invalid MODE=${MODE}" >&2; exit 2 ;;
esac

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
TRAIN="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
VALIDATION_MANIFEST="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
TRAIN_MANIFEST="${TRAIN}/activations/ddxplus_probe_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
STRUCTURED="${DATA_ROOT}/medical_nla/results/ddxplus_finding_value_probe_val_v1/structured_reader_hs24_protocol_v1.json"
MAPPER_ROOT="${DATA_ROOT}/medical_nla/results/ddxplus_semantic_mapper_validation_v2"
MAPPER_PROTOCOL="${MAPPER_ROOT}/frozen/semantic_protocol.json"
MAPPER_RECEIPT="${MAPPER_ROOT}/semantic_mapper_freeze_receipt.json"
MAP_OUT="${OUT}/semantic_mapping"

for path in "${VALIDATION_MANIFEST}" "${TRAIN_MANIFEST}" "${STRUCTURED}" "${MAPPER_PROTOCOL}" "${MAPPER_RECEIPT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
actual_mapper_sha="$(sha256sum "${MAPPER_PROTOCOL}" | awk '{print $1}')"
if [[ "${actual_mapper_sha}" != "${EXPECTED_MAPPER_SHA}" ]]; then
  echo "[error] frozen mapper SHA mismatch: ${actual_mapper_sha}" >&2
  exit 2
fi
mkdir -p "${OUT}" "${MAP_OUT}"

if [[ "${MODE}" == "prepare" || "${MODE}" == "all" ]]; then
  echo "[stage 1/5] freeze validation population, donors, train mean, and logical cells"
  python scripts/run_ddxplus_d22_patchscope.py prepare \
    --validation-manifest "${VALIDATION_MANIFEST}" \
    --train-manifest "${TRAIN_MANIFEST}" \
    --structured-protocol "${STRUCTURED}" \
    --path-map /data/heejae=/data1/heejae \
    --cases "${CASES}" \
    --out-dir "${OUT}"
fi

if [[ "${MODE}" == "generate" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/population_protocol.json" || { echo "[error] prepare first" >&2; exit 2; }
  echo "[stage 2/5] native-layer Patchscope generation on GPUs ${GPUS}"
  CUDA_VISIBLE_DEVICES="${GPUS}" python scripts/run_ddxplus_d22_patchscope.py generate \
    --config configs/default.yaml \
    --generation-manifest "${OUT}/generation_manifest.jsonl" \
    --population-protocol "${OUT}/population_protocol.json" \
    --output "${OUT}/unique_generations.jsonl" \
    --receipt "${OUT}/generation_receipt.json" \
    --seal "${OUT}/generation_seal.json" \
    --batch-size "${BATCH_SIZE}"
  python scripts/run_ddxplus_d22_patchscope.py materialize-logical \
    --generated "${OUT}/unique_generations.jsonl" \
    --logical-manifest "${OUT}/logical_manifest.jsonl" \
    --output "${OUT}/logical_readouts.jsonl"
fi

if [[ "${MODE}" == "mapper-prepare" || "${MODE}" == "all" ]]; then
  test -s "${OUT}/logical_readouts.jsonl" || { echo "[error] generate first" >&2; exit 2; }
  echo "[stage 3/5] frozen mapper request preparation"
  python scripts/run_ddxplus_d22_patchscope.py mapper-prepare \
    --readouts "${OUT}/logical_readouts.jsonl" \
    --protocol "${MAPPER_PROTOCOL}" \
    --out-dir "${MAP_OUT}"
fi

if [[ "${MODE}" == "mapper-run" || "${MODE}" == "all" ]]; then
  test -f "${MAP_OUT}/semantic_requests.jsonl" || { echo "[error] mapper-prepare first" >&2; exit 2; }
  echo "[stage 4/5] method-blind frozen semantic mapping via ${PRIMARY_MODEL}"
  if [[ -s "${MAP_OUT}/semantic_requests.jsonl" ]]; then
    BASE_JUDGEMENTS="${MAP_OUT}/semantic_judgements_base.jsonl"
    MERGED_JUDGEMENTS="${MAP_OUT}/semantic_judgements.jsonl"
    RETRY_ROOT="${MAP_OUT}/retries"
    mkdir -p "${RETRY_ROOT}"
    python scripts/run_judge.py \
      --requests "${MAP_OUT}/semantic_requests.jsonl" \
      --out "${BASE_JUDGEMENTS}" \
      --backend codex \
      --model "${PRIMARY_MODEL}" \
      --timeout 300
    # A transient Codex failure leaves a request absent; resumable reruns fill it.
    for _ in 1 2; do
      python scripts/run_judge.py \
        --requests "${MAP_OUT}/semantic_requests.jsonl" \
        --out "${BASE_JUDGEMENTS}" \
        --backend codex \
        --model "${PRIMARY_MODEL}" \
        --timeout 300
    done
    python scripts/merge_semantic_judgement_shards.py \
      --requests "${MAP_OUT}/semantic_requests.jsonl" \
      --judgement "${BASE_JUDGEMENTS}" \
      --output "${MERGED_JUDGEMENTS}" \
      --expected-model "${PRIMARY_MODEL}" \
      --report "${MAP_OUT}/semantic_judgement_merge_report.json"
    replacement_args=()
    for attempt in 1 2 3; do
      retry_requests="${RETRY_ROOT}/retry_requests_${attempt}.jsonl"
      audit_report="${RETRY_ROOT}/audit_${attempt}.json"
      python scripts/audit_semantic_judgement_batches.py \
        --prepared "${MAP_OUT}/prepared_items.jsonl" \
        --requests "${MAP_OUT}/semantic_requests.jsonl" \
        --judgements "${MERGED_JUDGEMENTS}" \
        --protocol "${MAPPER_PROTOCOL}" \
        --retry-requests "${retry_requests}" \
        --report "${audit_report}"
      invalid="$(python - "${audit_report}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["invalid"])
PY
)"
      [[ "${invalid}" -eq 0 ]] && break
      retry_judgements="${RETRY_ROOT}/retry_judgements_${attempt}.jsonl"
      python scripts/run_judge.py \
        --requests "${retry_requests}" \
        --out "${retry_judgements}" \
        --backend codex \
        --model "${PRIMARY_MODEL}" \
        --timeout 300
      replacement_args+=(--replacement-judgement "${retry_judgements}")
      python scripts/merge_semantic_judgement_shards.py \
        --requests "${MAP_OUT}/semantic_requests.jsonl" \
        --judgement "${BASE_JUDGEMENTS}" \
        "${replacement_args[@]}" \
        --output "${MERGED_JUDGEMENTS}" \
        --expected-model "${PRIMARY_MODEL}" \
        --report "${MAP_OUT}/semantic_judgement_merge_report.json"
    done
    python scripts/audit_semantic_judgement_batches.py \
      --prepared "${MAP_OUT}/prepared_items.jsonl" \
      --requests "${MAP_OUT}/semantic_requests.jsonl" \
      --judgements "${MERGED_JUDGEMENTS}" \
      --protocol "${MAPPER_PROTOCOL}" \
      --retry-requests "${RETRY_ROOT}/retry_requests_final.jsonl" \
      --report "${RETRY_ROOT}/audit_final.json"
    final_invalid="$(python - "${RETRY_ROOT}/audit_final.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["invalid"])
PY
)"
    if [[ "${final_invalid}" -ne 0 ]]; then
      echo "[error] frozen mapper parser still rejects ${final_invalid} requests" >&2
      exit 1
    fi
  else
    : > "${MAP_OUT}/semantic_judgements.jsonl"
  fi
fi

if [[ "${MODE}" == "finalize" || "${MODE}" == "all" ]]; then
  test -f "${MAP_OUT}/semantic_judgements.jsonl" || { echo "[error] mapper-run first" >&2; exit 2; }
  echo "[stage 5/5] exact-receipt validation and frozen Patchscope gates"
  python scripts/run_ddxplus_d22_patchscope.py mapper-finalize \
    --prepared-items "${MAP_OUT}/prepared_items.jsonl" \
    --requests "${MAP_OUT}/semantic_requests.jsonl" \
    --judgements "${MAP_OUT}/semantic_judgements.jsonl" \
    --logical-manifest "${OUT}/logical_manifest.jsonl" \
    --logical-readouts "${OUT}/logical_readouts.jsonl" \
    --protocol "${MAPPER_PROTOCOL}" \
    --mapper-receipt "${MAPPER_RECEIPT}" \
    --out-dir "${MAP_OUT}"
fi

echo "[done] ${OUT}"
