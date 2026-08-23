#!/usr/bin/env bash
# The reader-trust task, replacing the rating experiment that failed.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_reader_trust.sh > /dev/null 2>&1 &
#
# The rating run gave the chain of thought exactly 5.000 on 624 of 624 cases
# and picked it as most useful every time. That is a length effect, not a
# judgement: the three channels differ in size by fifty to one and the
# question had no right answer. Here each channel is judged alone against a
# hidden ground truth, so extra words only help if they carry the signal.
#
# ~724 cases x 3 channels; the judge writes a small JSON object, so the
# generation budget is short.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/reports"
MAIN="$LOGS/reader_trust_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

CASES="$DATA/ddxplus_reader_trust_cases.jsonl"
OUT="$ART/results/ddxplus_reader_trust.jsonl"

if [ ! -s "$CASES" ]; then
  say "building trust prompts (CPU)"
  python scripts/make_reader_trust_cases.py \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
    --cot-answers "$ART/results/ddxplus_hint_answers_cot_full.jsonl" \
    --readouts "$ART/results/readout_hint_final_L32_v2.jsonl" \
    --readout-manifests "$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl" \
    --probe-verdicts "$ART/results/probe_verdicts.jsonl" \
    --output "$CASES" >>"$MAIN" 2>&1 || { say "FAILED build"; exit 1; }
fi
say "cases: $(wc -l < "$CASES") rows"

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

if [ -s "$OUT" ]; then say "skip (answers exist)"; else
  say "judging"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$CASES" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_reader_trust.json" \
    --condition direct --no-prefill --no-force-answer \
    --max-new-tokens 96 --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi
say "ALL DONE (reader trust)"
say "analyse with: python scripts/analyze_reader_trust.py --judgements $OUT"
