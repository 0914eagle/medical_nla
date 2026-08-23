#!/usr/bin/env bash
# Explanation-quality judging: readout vs chain vs probe, blinded, same case.
#   CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/run_explanation_judging.sh > /dev/null 2>&1 &
# Builds ~620 blinded judge prompts (all moved cases + 300 kept) on CPU, then
# has the local backbone judge them (~30 min). Same-family caveat is real and
# goes in the paper: the judge shares the backbone with the system under test,
# so this pass establishes the comparison and an API judge can re-score the
# same prompt file later -- the prompts and the analyzer are judge-agnostic.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/explanation_judging_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

CASES="$DATA/ddxplus_judge_quality_cases.jsonl"
if [ ! -s "$CASES" ]; then
  say "building judge prompts (CPU)"
  python scripts/make_explanation_judging_cases.py \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
    --cot-answers "$ART/results/ddxplus_hint_answers_cot_full.jsonl" \
    --readouts "$ART/results/readout_hint_final_L32_v2.jsonl" \
    --readout-manifests "$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl" \
    --probe-verdicts "$ART/results/probe_verdicts.jsonl" \
    --output "$CASES" >>"$MAIN" 2>&1 || { say "FAILED judge build"; exit 1; }
fi

OUT="$ART/results/ddxplus_judge_quality.jsonl"
if [ -s "$OUT" ]; then say "skip judging (answers exist)"; else
  say "judging"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$CASES" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_judge_quality.json" \
    --condition direct --no-prefill --no-force-answer \
    --max-new-tokens 256 --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED judging"; exit 1; }
fi
say "ALL DONE (explanation judging)"
