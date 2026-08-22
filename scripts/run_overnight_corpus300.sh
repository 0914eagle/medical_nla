#!/usr/bin/env bash
# Tomorrow's queue -- triple the testbed: 300 cases per diagnosis.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_overnight_corpus300.sh > /dev/null 2>&1 &
# 100/diagnosis was our sampling choice, not the corpus's size (the release
# holds ~1.3M patients). Same seed, same builder, larger cap; every downstream
# file gets a _300 suffix so nothing already published moves.
# Stages: rebuild cases (CPU, scans train.csv) -> direct answers on 14,700
# (~2h) -> hint cases (all four arms) -> direct answers on the ~5,200-case
# testbed x4 arms (~2.5h) -> extraction (L32 final + hint positions) ->
# conclusion readout with the v2 adapter (~1h). Morning: analyze_hint_effect
# and compare_channels_on_attribution over the _300 files.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/overnight_corpus300_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }
run() { say "$1"; shift; "$@" >>"$MAIN" 2>&1 || { say "FAILED: $1"; exit 1; }; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

CASES="$DATA/ddxplus_cue_count_cases_300.jsonl"
[ -s "$CASES" ] && say "skip case build (exists)" || \
run "build 300/diagnosis cases (CPU scan of train.csv)" \
  python scripts/make_ddxplus_cue_count_cases.py \
    --patients "$RAW/ddxplus/train.csv" \
    --evidences "$RAW/ddxplus/release_evidences.json" \
    --output "$CASES" \
    --examples-per-diagnosis 300 --cue-counts all --seed 17 \
    --no-prefer-symptoms --stop-when-full

SRC="$ART/results/ddxplus_source_answers_300.jsonl"
[ -s "$SRC" ] && say "skip source answers (exists)" || \
run "direct answers on 14,700 cases" \
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$CASES" \
    --output-jsonl "$SRC" \
    --summary-json "$ART/reports/ddxplus_source_answers_300.json" \
    --condition direct --batch-size 8

HINT="$DATA/ddxplus_hint_cases_300.jsonl"
[ -s "$HINT" ] && say "skip hint build (exists)" || \
run "hint cases (four arms) over the correct pool" \
  python scripts/make_hint_injection_cases.py \
    --cases "$CASES" --answers "$SRC" --output "$HINT"

ANS="$ART/results/ddxplus_hint_answers_300.jsonl"
[ -s "$ANS" ] && say "skip hint answers (exists)" || \
run "direct answers on all four arms" \
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$HINT" \
    --output-jsonl "$ANS" \
    --summary-json "$ART/reports/ddxplus_hint_answers_300.json" \
    --condition direct --batch-size 8

ROWS="$DATA/ddxplus_hint_position_rows_300.jsonl"
[ -s "$ROWS" ] && say "skip position rows (exists)" || \
run "position rows (final + hint)" \
  python scripts/make_hint_position_rows.py --cases "$HINT" --output "$ROWS"

run "extract L32 activations" \
  python -m src.extract_activations --config configs/default.yaml \
    --input "$ROWS" \
    --run-name hint_positions_300_L32 --layers 32 --strategies last_subtoken --batch-size 8

READ="$ART/results/readout_hint_final_300_L32_v2.jsonl"
[ -s "$READ" ] && say "skip readout (exists)" || \
run "conclusion readout (v2 adapter)" \
  python -m src.run_nla --config configs/default.yaml \
    --manifest "$ART/activations/hint_positions_300_L32/layer32/last_token/manifest.jsonl" \
    --output "$READ" \
    --adapter-id "$ART/adapters/medical_nla_ddxplus_v2_alpha_lora" \
    --actor-prompt-template-file prompt_templates/medical_nla_v2_readout.txt \
    --batch-size 16

say "ALL DONE (corpus-300 queue). Morning:"
say "  python scripts/analyze_hint_effect.py --answers $ANS"
say "  python scripts/compare_channels_on_attribution.py --answers $ANS --readouts $READ"
