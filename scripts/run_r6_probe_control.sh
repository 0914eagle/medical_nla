#!/usr/bin/env bash
# Rung 6 -- the probe-content control for the correction ladder (Table 5).
#   CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/run_r6_probe_control.sh > /dev/null 2>&1 &
# Builds the r6 second-pass prompts (CPU; the feedback is the linear probe's
# argmax class and nothing else) and answers them: ~15 min of 1,747 rows.
# r5 minus r6 on the moved population is the pure contribution of the
# readout's natural-language content beyond naming a diagnosis.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/r6_probe_control_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

PREFIX="$DATA/ddxplus_ladder"
if [ ! -s "${PREFIX}_r6.jsonl" ]; then
  say "building r6 prompts (CPU)"
  python scripts/make_correction_ladder_cases.py \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
    --readouts "$ART/results/readout_hint_final_L32_v2.jsonl" \
    --readout-manifests "$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl" \
    --rungs 6 \
    --probe-verdicts "$ART/results/probe_verdicts.jsonl" \
    --output-prefix "$PREFIX" >>"$MAIN" 2>&1 || { say "FAILED r6 build"; exit 1; }
fi

OUT="$ART/results/ddxplus_ladder_r6.jsonl"
if [ -s "$OUT" ]; then say "skip rung 6 (answers exist)"; else
  say "rung 6"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "${PREFIX}_r6.jsonl" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_ladder_r6.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED rung 6"; exit 1; }
fi
say "ALL DONE (r6 probe-content control)"
