#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATA_ROOT="${DATA_ROOT:-${MEDICAL_NLA_DATA_ROOT:-}}"
if [[ -z "$DATA_ROOT" ]]; then
  echo "Set DATA_ROOT (/data/heejae on server 62; /data1/heejae on server 125)." >&2
  exit 2
fi

DDX_ROOT="${DDX_ROOT:-$DATA_ROOT/ddxplus}"
OUT_DIR="${OUT_DIR:-$DATA_ROOT/medical_nla/data/ddxplus_e5_canonical_v1}"
SEED="${SEED:-17}"
EXAMPLES_PER_DIAGNOSIS="${EXAMPLES_PER_DIAGNOSIS:-100}"
EXPECTED_COMMON_DIAGNOSES="${EXPECTED_COMMON_DIAGNOSES:-}"

EXTRA_ARGS=()
if [[ -n "$EXPECTED_COMMON_DIAGNOSES" ]]; then
  EXTRA_ARGS+=(--expected-common-diagnoses "$EXPECTED_COMMON_DIAGNOSES")
fi

python scripts/prepare_ddxplus_e5.py \
  --split "validation=$DDX_ROOT/validate.csv" \
  --split "test=$DDX_ROOT/test.csv" \
  --evidences "$DDX_ROOT/release_evidences.json" \
  --out-dir "$OUT_DIR" \
  --seed "$SEED" \
  --examples-per-diagnosis "$EXAMPLES_PER_DIAGNOSIS" \
  "${EXTRA_ARGS[@]}"

echo "[done] $OUT_DIR"
