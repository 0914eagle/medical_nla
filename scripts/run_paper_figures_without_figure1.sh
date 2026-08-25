#!/usr/bin/env bash
set -euo pipefail

# Render every current paper figure except conceptual Figure 1 and the manual
# Appendix case-study panel. Usage:
#
#   bash scripts/run_paper_figures_without_figure1.sh /data1/heejae
#   FORMAT=pdf bash scripts/run_paper_figures_without_figure1.sh /data1/heejae

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ $# -gt 0 ]]; then
  # shellcheck disable=SC1091
  source scripts/env.sh "$1"
else
  # shellcheck disable=SC1091
  source scripts/env.sh
fi

VENV_ACTIVATE="$MEDICAL_NLA_DATA_ROOT/uv/medical_nla/bin/activate"
if [[ -f "$VENV_ACTIVATE" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_ACTIVATE"
fi

FORMAT="${FORMAT:-png}"
OUT_DIR="${OUT_DIR:-$ART/results/paper_figures}"

DDX_DIRECT="${DDX_DIRECT:-$ART/results/ddxplus_hint_answers_v2_rescored.jsonl}"
DDX_NEUTRAL="${DDX_NEUTRAL:-$ART/results/ddxplus_hint_answers_neutral_rescored.jsonl}"
DDX_CORRECT="${DDX_CORRECT:-$ART/results/ddxplus_hint_answers_correct_rescored.jsonl}"
MCR_ANSWERS="${MCR_ANSWERS:-$ART/results/mcr_hint_answers_full_rescored.jsonl}"
TRAJECTORY_DUMP="${TRAJECTORY_DUMP:-$ART/results/trajectory_dump.json}"
DETECTION_VALUES="${DETECTION_VALUES:-configs/figure4_detection_correction_canonical.json}"

DDX_DUMP="${DDX_DUMP:-$ART/results/figure2_ddx_dump.json}"
MCR_DUMP="${MCR_DUMP:-$ART/results/figure2_mcr_dump.json}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "[missing] $1" >&2
    exit 1
  fi
}

for path in \
  "$DDX_DIRECT" "$DDX_NEUTRAL" "$DDX_CORRECT" \
  "$MCR_ANSWERS" "$TRAJECTORY_DUMP" "$DETECTION_VALUES"; do
  require_file "$path"
done

echo "[1/3] rebuilding canonical Figure 2 dumps"
python scripts/analyze_hint_effect.py \
  --answers "$DDX_DIRECT" "$DDX_NEUTRAL" "$DDX_CORRECT" \
  --require-canonical-no-note-correct \
  --dump "$DDX_DUMP"

python scripts/analyze_hint_effect.py \
  --answers "$MCR_ANSWERS" \
  --require-canonical-no-note-correct \
  --dump "$MCR_DUMP"

echo "[2/3] rendering Figures 2--4 and Appendix Figure A1"
python scripts/make_paper_figures.py \
  --ddx-dump "$DDX_DUMP" \
  --mcr-dump "$MCR_DUMP" \
  --trajectory-dump "$TRAJECTORY_DUMP" \
  --detection-values "$DETECTION_VALUES" \
  --out-dir "$OUT_DIR" \
  --format "$FORMAT"

echo "[3/3] outputs"
for name in \
  "figure2_behavior.$FORMAT" \
  "figure3_trajectory.$FORMAT" \
  "figure4_detection_correction.$FORMAT" \
  "appendix_figure_a1_readout_map.$FORMAT"; do
  require_file "$OUT_DIR/$name"
  ls -lh "$OUT_DIR/$name"
done

echo "[done] Figure 1 intentionally excluded; output dir: $OUT_DIR"
