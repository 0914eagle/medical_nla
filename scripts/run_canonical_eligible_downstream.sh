#!/usr/bin/env bash
set -euo pipefail

# Recompute every DDXPlus downstream result that previously inherited the
# matcher-era fixed cohort (1,747 cases). The primary population is defined
# once here: canonically rescored no-note-correct cases (currently 1,729).
#
# Usage:
#   nohup bash scripts/run_canonical_eligible_downstream.sh /data1/heejae \
#     > /data1/heejae/medical_nla/logs/canonical_eligible_downstream.log 2>&1 &

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
export PYTHONPATH="$REPO_ROOT"

RES="$ART/results"
REPORTS="$ART/reports"
FIGURES="$RES/paper_figures"
mkdir -p "$RES" "$REPORTS" "$FIGURES" "$ART/logs"

ANSWERS="${ANSWERS:-$RES/ddxplus_hint_answers_v2_rescored.jsonl}"
CASES="${CASES:-$DATA/ddxplus_hint_cases_v2.jsonl}"
COT="${COT:-$RES/ddxplus_hint_answers_cot_full_rescored.jsonl}"
READOUT="${READOUT:-$RES/readout_hint_final_L32_v2.jsonl}"
FINAL_MANIFEST="${FINAL_MANIFEST:-$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl}"
MONITOR="${MONITOR:-$RES/judge_cot_monitor.jsonl}"

PROBE="$RES/probe_verdicts_canonical_eligible.jsonl"
TRAJECTORY="$RES/trajectory_dump_canonical_eligible.json"
CHANNELS="$RES/channel_scores_canonical_eligible.jsonl"
VALUES="$REPORTS/figure4_detection_correction_canonical_eligible.json"
VALUES_MD="$REPORTS/figure4_detection_correction_canonical_eligible_summary.md"

require_file() {
  if [[ ! -s "$1" ]]; then
    echo "[missing] $1" >&2
    exit 1
  fi
}

for path in "$ANSWERS" "$CASES" "$COT" "$READOUT" "$FINAL_MANIFEST" "$MONITOR"; do
  require_file "$path"
done

TRAJECTORY_MANIFESTS=(
  "$ART"/activations/trajectory_fixed_L32/layer32/*/manifest.jsonl
)
if [[ ! -e "${TRAJECTORY_MANIFESTS[0]}" ]]; then
  echo "[missing] trajectory manifests under $ART/activations/trajectory_fixed_L32" >&2
  exit 1
fi

RUNGS=()
for rung in 3 4 5 6; do
  path="$RES/ddxplus_ladder_r${rung}_rescored.jsonl"
  require_file "$path"
  RUNGS+=("$path")
done

echo "[1/8] cross-fitted final-state probe on canonical-eligible cases"
python scripts/evaluate_probe_disagreement.py \
  --answers "$ANSWERS" \
  --cases "$CASES" \
  --manifests "$FINAL_MANIFEST" \
  --require-canonical-no-note-correct \
  --dump "$PROBE" \
  | tee "$REPORTS/probe_canonical_eligible.log"

echo "[2/8] cross-fitted landmark trajectory on the same cohort"
python scripts/analyze_trajectory.py \
  --answers "$ANSWERS" \
  --cases "$CASES" \
  --manifests "${TRAJECTORY_MANIFESTS[@]}" \
  --require-canonical-no-note-correct \
  --dump "$TRAJECTORY" \
  | tee "$REPORTS/trajectory_canonical_eligible.log"

echo "[3/8] output, CoT, and AV attribution channels"
python scripts/compare_channels_on_attribution.py \
  --answers "$ANSWERS" \
  --cases "$CASES" \
  --cot-answers "$COT" \
  --readouts "$READOUT" \
  --readout-manifests "$FINAL_MANIFEST" \
  --require-canonical-no-note-correct \
  --dump "$CHANNELS" \
  | tee "$REPORTS/channels_canonical_eligible.log"

echo "[4/8] paired monitor-versus-readout bootstrap"
python scripts/bootstrap_channel_gap.py \
  --dump "$CHANNELS" \
  --monitor "$MONITOR" \
  --a "llm monitor over the chain" \
  --b "answer omits the internal conclusion (containment)" \
  | tee "$REPORTS/channel_gap_canonical_eligible.log"

echo "[5/8] correction ladder on exactly the same IDs"
python scripts/analyze_correction_ladder.py \
  --rungs "${RUNGS[@]}" \
  --probe-flags "$PROBE" \
  --eligibility-answers "$ANSWERS" \
  --eligibility-cases "$CASES" \
  --require-canonical-no-note-correct \
  | tee "$REPORTS/correction_ladder_canonical_eligible.log"

echo "[6/8] machine-readable Figure 4 values"
python scripts/build_canonical_figure4_values.py \
  --answers "$ANSWERS" \
  --cases "$CASES" \
  --channel-scores "$CHANNELS" \
  --monitor "$MONITOR" \
  --probe-verdicts "$PROBE" \
  --rungs "${RUNGS[@]}" \
  --output-json "$VALUES" \
  --summary-md "$VALUES_MD"

echo "[7/8] canonical Figures 3 and 4"
python scripts/make_figure_trajectory.py \
  --dump "$TRAJECTORY" \
  --output "$FIGURES/figure3_trajectory_canonical_eligible.png"
python scripts/make_figure_detection_correction.py \
  --values "$VALUES" \
  --output "$FIGURES/figure4_detection_correction_canonical_eligible.png"

echo "[8/8] reader-trust reaggregation when completed judgements are present"
RT_JUDGEMENTS="${RT_JUDGEMENTS:-$RES/judge_reader_trust_deduped.jsonl}"
RT_CASES="${RT_CASES:-$DATA/ddxplus_reader_trust_cases_v2.jsonl}"
if [[ -s "$RT_JUDGEMENTS" && -s "$RT_CASES" ]]; then
  python scripts/analyze_reader_trust.py \
    --judgements "$RT_JUDGEMENTS" \
    --cases "$RT_CASES" \
    --eligibility-answers "$ANSWERS" \
    --eligibility-cases "$CASES" \
    --require-canonical-no-note-correct \
    | tee "$REPORTS/reader_trust_canonical_eligible.log"
else
  echo "[skip] reader-trust inputs not found: $RT_JUDGEMENTS / $RT_CASES"
fi

RT_CONTROL_JUDGEMENTS="${RT_CONTROL_JUDGEMENTS:-$RES/judge_reader_trust_controlled_deduped.jsonl}"
RT_CONTROL_CASES="${RT_CONTROL_CASES:-$DATA/ddxplus_reader_trust_cases_controlled.jsonl}"
if [[ -s "$RT_CONTROL_JUDGEMENTS" && -s "$RT_CONTROL_CASES" ]]; then
  python scripts/analyze_reader_trust.py \
    --judgements "$RT_CONTROL_JUDGEMENTS" \
    --cases "$RT_CONTROL_CASES" \
    --eligibility-answers "$ANSWERS" \
    --eligibility-cases "$CASES" \
    --require-canonical-no-note-correct \
    | tee "$REPORTS/reader_trust_controlled_canonical_eligible.log"
else
  echo "[skip] controlled reader-trust inputs not found"
fi

echo "[done] canonical cohort and all primary downstream artifacts rebuilt"
echo "[summary] $VALUES_MD"
echo "[trajectory] $TRAJECTORY"
echo "[figure 3] $FIGURES/figure3_trajectory_canonical_eligible.png"
echo "[figure 4] $FIGURES/figure4_detection_correction_canonical_eligible.png"
