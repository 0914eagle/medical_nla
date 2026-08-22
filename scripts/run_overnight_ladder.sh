#!/usr/bin/env bash
# Overnight queue, cards 2,3 -- the correction ladder (Table 5).
#   CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/run_overnight_ladder.sh > /dev/null 2>&1 &
# Builds the second-pass prompts from the finished first pass and readouts
# (CPU), then answers each rung: ~15 min per rung of 1,747 prefilled rows.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/overnight_ladder_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

PREFIX="$DATA/ddxplus_ladder"
if [ ! -s "${PREFIX}_r5.jsonl" ]; then
  say "building ladder prompts (CPU)"
  python scripts/make_correction_ladder_cases.py \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
    --readouts "$ART/results/readout_hint_final_L32_v2.jsonl" \
    --readout-manifests "$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl" \
    --output-prefix "$PREFIX" >>"$MAIN" 2>&1 || { say "FAILED ladder build"; exit 1; }
fi

for R in 3 4 5; do
  OUT="$ART/results/ddxplus_ladder_r${R}.jsonl"
  if [ -s "$OUT" ]; then say "skip rung $R (exists)"; continue; fi
  say "rung $R"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "${PREFIX}_r${R}.jsonl" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_ladder_r${R}.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED rung $R"; exit 1; }
  say "done rung $R"
done
say "ALL DONE (ladder queue)"
