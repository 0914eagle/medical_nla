#!/usr/bin/env bash
set -euo pipefail

# Render canonical Figures 2/3 and materialize their complete numeric summary.
# This queue is intentionally CPU-only and is designed for server 62.

DATA_ROOT="${DATA_ROOT:-/data/heejae}"
RUN_NAME="${RUN_NAME:-paper_cpu_metrics_62_v1}"

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

if [[ "$(hostname)" != "solvook-ml" ]]; then
  echo "[error] this canonical queue must run on server 62 (solvook-ml)" >&2
  exit 2
fi
if [[ "${DATA_ROOT}" != "/data/heejae" ]]; then
  echo "[error] server 62 DATA_ROOT must be /data/heejae" >&2
  exit 2
fi

PDD_DIR="${DATA_ROOT}/restricted/direct/e2/direct_e2_probe_pdd_val_v1"
CATEGORY_DIR="${DATA_ROOT}/medical_nla/imports/server125/direct_e2_probe_category_val_v1"
DDX_RESULT_DIR="${DATA_ROOT}/medical_nla/results/ddxplus_finding_value_probe_val_v1"
DDX_ACTIVATION_DIR="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1/activations"

PDD_RESULTS="${PDD_RESULTS:-${PDD_DIR}/validation_results.json}"
CATEGORY_RESULTS="${CATEGORY_RESULTS:-${CATEGORY_DIR}/validation_results.json}"
DDX_RESULTS="${DDX_RESULTS:-${DDX_RESULT_DIR}/results.json}"
DDX_ARTIFACT="${DDX_ARTIFACT:-${DDX_RESULT_DIR}/finding_value_hs24.pt}"
DDX_RUN="${DDX_ACTIVATION_DIR}/ddxplus_e5_test_cot_p0_hs24_merged_v1"
DDX_MANIFEST="${DDX_MANIFEST:-${DDX_RUN}/layer24/last_token/manifest.jsonl}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/results/${RUN_NAME}}"

require_file() {
  test -s "$1" || { echo "[error] missing $1" >&2; exit 2; }
}
for path in \
  "${PDD_RESULTS}" "${CATEGORY_RESULTS}" "${DDX_RESULTS}" \
  "${DDX_ARTIFACT}" "${DDX_MANIFEST}"; do
  require_file "${path}"
done

manifest_rows="$(wc -l < "${DDX_MANIFEST}")"
if [[ "${manifest_rows}" -ne 10028 ]]; then
  echo "[error] DDXPlus HS24 manifest has ${manifest_rows} rows; expected 10028" >&2
  exit 2
fi

mkdir -p "${OUT}/provenance"
echo "[1/4] fixed paper-cell audit"
python scripts/sync_paper_table_fixed_cells.py --check

echo "[2/4] Figure 2 validation layer sensitivity"
python scripts/make_medical_nla_probe_layer_figure.py \
  --direct-results "${PDD_RESULTS}" \
  --direct-results "${CATEGORY_RESULTS}" \
  --ddxplus-results "${DDX_RESULTS}" \
  --output "${OUT}/figure2_probe_layers.pdf" \
  --values-json "${OUT}/figure2_probe_layers_values.json"

echo "[3/4] Figure 3 locked DDXPlus counterfactual response"
python scripts/make_ddxplus_counterfactual_figure.py \
  --artifact "${DDX_ARTIFACT}" \
  --manifest "${DDX_MANIFEST}" \
  --device cpu \
  --output "${OUT}/figure3_counterfactual.pdf" \
  --values-json "${OUT}/figure3_counterfactual_values.json"

echo "[4/4] canonical numeric summary and provenance"
python scripts/summarize_paper_figure_values.py \
  --figure2-values "${OUT}/figure2_probe_layers_values.json" \
  --figure3-values "${OUT}/figure3_counterfactual_values.json" \
  --output-json "${OUT}/figure2_figure3_results.json" \
  --summary-md "${OUT}/summary.md"

sha256sum \
  "${PDD_RESULTS}" "${CATEGORY_RESULTS}" "${DDX_RESULTS}" \
  "${DDX_ARTIFACT}" "${DDX_MANIFEST}" \
  "${OUT}/figure2_probe_layers_values.json" \
  "${OUT}/figure3_counterfactual_values.json" \
  > "${OUT}/provenance/input_output_hashes.sha256"
git rev-parse HEAD > "${OUT}/provenance/git_commit.txt"
python --version > "${OUT}/provenance/python_version.txt" 2>&1

cat "${OUT}/summary.md"
ls -lh "${OUT}/figure2_probe_layers.pdf" "${OUT}/figure3_counterfactual.pdf"
echo "[done] ${OUT}"
