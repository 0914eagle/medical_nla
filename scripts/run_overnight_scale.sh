#!/usr/bin/env bash
# Overnight queue, cards 0,1 -- widen the intervention.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_overnight_scale.sh > /dev/null 2>&1 &
# Order: cheap first. Two wording variants at ~25 min each, then the full
# chain-of-thought expansion, which owns the rest of the night.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/overnight_scale_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

for W in colleague patient; do
  CASES="$DATA/ddxplus_hint_cases_${W}.jsonl"
  OUT="$ART/results/ddxplus_hint_answers_${W}.jsonl"
  if [ -s "$OUT" ]; then say "skip wording $W (exists)"; continue; fi
  say "wording variant: $W -- build cases"
  python scripts/make_hint_injection_cases.py \
    --cases "$DATA/ddxplus_cue_count_cases.jsonl" \
    --answers "$ART/results/ddxplus_source_answers.jsonl" \
    --wording "$W" --arms none wrong \
    --output "$CASES" >>"$MAIN" 2>&1 || { say "FAILED build $W"; exit 1; }
  say "wording variant: $W -- direct answers"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$CASES" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_hint_answers_${W}.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED answers $W"; exit 1; }
  say "done wording $W"
done

COT_OUT="$ART/results/ddxplus_hint_answers_cot_full.jsonl"
if [ -s "$COT_OUT" ]; then
  say "skip full CoT (exists)"
else
  say "full chain-of-thought over 1,747 cases (none+wrong) -- the long one"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --where hint_variant=none,wrong \
    --output-jsonl "$COT_OUT" \
    --summary-json "$ART/reports/ddxplus_hint_answers_cot_full.json" \
    --condition cot --batch-size 4 >>"$MAIN" 2>&1 || { say "FAILED full CoT"; exit 1; }
  say "done full CoT"
fi
say "ALL DONE (scale queue)"
