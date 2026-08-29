#!/usr/bin/env bash
set -euo pipefail

# Score an already sealed DDXPlus Vanilla generation. This wrapper refuses to
# run until the independent AI mapper receipt proves that G1-G4 passed.

DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the server-local data root}"
GENERATION_SEAL="${GENERATION_SEAL:?Set generation_seal.json}"
MAPPER_RECEIPT="${MAPPER_RECEIPT:?Set the validation-only G1-G4 freeze receipt}"
SEMANTIC_PROTOCOL="${SEMANTIC_PROTOCOL:?Set the frozen semantic protocol JSON}"
SEMANTIC_SCORER="${SEMANTIC_SCORER:?Set the frozen semantic scorer script}"
EXPECTED_SEMANTIC_PROTOCOL_SHA256="${EXPECTED_SEMANTIC_PROTOCOL_SHA256:?Set protocol SHA256}"
EXPECTED_SEMANTIC_SCORER_SHA256="${EXPECTED_SEMANTIC_SCORER_SHA256:?Set scorer SHA256}"
OUT="${OUT:?Set a new locked semantic scoring output directory}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

for path in "${GENERATION_SEAL}" "${MAPPER_RECEIPT}" \
  "${SEMANTIC_PROTOCOL}" "${SEMANTIC_SCORER}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done
actual_protocol_hash="$(sha256sum "${SEMANTIC_PROTOCOL}" | awk '{print $1}')"
actual_scorer_hash="$(sha256sum "${SEMANTIC_SCORER}" | awk '{print $1}')"
test "${actual_protocol_hash}" = "${EXPECTED_SEMANTIC_PROTOCOL_SHA256}" || {
  echo "[error] semantic protocol hash mismatch" >&2; exit 2;
}
test "${actual_scorer_hash}" = "${EXPECTED_SEMANTIC_SCORER_SHA256}" || {
  echo "[error] semantic scorer hash mismatch" >&2; exit 2;
}

python scripts/manage_nla_generation_seal.py verify --receipt "${GENERATION_SEAL}"
python scripts/validate_semantic_mapper_freeze_receipt.py \
  --receipt "${MAPPER_RECEIPT}" \
  --expected-protocol-sha256 "${EXPECTED_SEMANTIC_PROTOCOL_SHA256}"

readout="$(python - "${GENERATION_SEAL}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["readout"]["path"])
PY
)"
generation_protocol="$(python - "${GENERATION_SEAL}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["generation_protocol"]["path"])
PY
)"
manifest="$(python - "${generation_protocol}" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["manifest"]["path"])
PY
)"

mkdir -p "${OUT}"
python "${SEMANTIC_SCORER}" \
  --readouts "${readout}" \
  --manifest "${manifest}" \
  --protocol "${SEMANTIC_PROTOCOL}" \
  --mapper-receipt "${MAPPER_RECEIPT}" \
  --out-dir "${OUT}"
test -s "${OUT}/results.json" || {
  echo "[error] scorer did not write ${OUT}/results.json" >&2
  exit 2
}
sha256sum "${GENERATION_SEAL}" "${MAPPER_RECEIPT}" "${SEMANTIC_PROTOCOL}" \
  "${SEMANTIC_SCORER}" "${OUT}/results.json" > "${OUT}/scoring_receipt.sha256"
echo "[done] locked semantic scores: ${OUT}"
