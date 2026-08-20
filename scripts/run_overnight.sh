#!/usr/bin/env bash
# The whole GPU queue, unattended and sequential.
#
#   nohup bash scripts/run_overnight.sh > /dev/null 2>&1 &
#
# Everything runs on whichever cards the session pinned -- `scripts/env.sh`
# sets CUDA_VISIBLE_DEVICES=0,1 -- one stage at a time. No device is named
# here: to run this on the other pair, export CUDA_VISIBLE_DEVICES before
# launching and nothing else changes, because `max_memory` in the config
# indexes the *visible* devices.
#
# Each stage writes its own log under $ART/logs and a failure is reported
# rather than cascading, so a late stage failing does not hide an early
# stage's result.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh

# Scale knobs, so a disk that cannot hold the full sweep does not mean editing
# the script at 3am. Sizes below are measured per corpus, float32, d_model 3840.
#
#   LAYERS       13 hidden-state indices; halving them halves the reductions
#   SPAN_LAYERS  full token spans, the expensive part -- empty disables them
#                (about 8GB on DDXPlus and 28GB on MedCaseReasoning)
#   STRATEGIES   last_subtoken alone halves the reductions again
#   RUN_MCR      0 to stop after the DDXPlus sweep
#
# Full:  DDXPlus 17GB + MedCaseReasoning 47GB + a 25GB checkpoint.
# No spans: about 5GB + 19GB.
LAYERS="${LAYERS:-0 4 8 12 16 20 24 28 32 36 40 44 47}"
SPAN_LAYERS="${SPAN_LAYERS-16 24 32}"
STRATEGIES="${STRATEGIES:-last_subtoken span_mean}"
RUN_MCR="${RUN_MCR:-1}"
NEED_GB="${NEED_GB:-90}"

span_args() {
  # An empty SPAN_LAYERS must produce no flag at all: `--span-layers` with no
  # values is an argparse error, not "none".
  [ -n "$SPAN_LAYERS" ] && printf -- '--span-layers %s' "$SPAN_LAYERS"
}

LOGS="$ART/logs"
RESULTS="$ART/results"
mkdir -p "$LOGS" "$RESULTS"

STAMP=$(date +%Y%m%d_%H%M%S)
MAIN="$LOGS/overnight_${STAMP}.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "cards:      CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
say "layers:     $LAYERS"
say "span:       ${SPAN_LAYERS:-<none>}"
say "strategies: $STRATEGIES"
say "run mcr:    $RUN_MCR"

# Activations are the bulk of this and there is no partial credit for filling
# a disk: the DDXPlus sweep is about 17GB and MedCaseReasoning about 47GB,
# most of it the full spans kept at three layers, plus a 25GB checkpoint.
AVAIL_GB=$(df -BG --output=avail "$MEDICAL_NLA_DATA_ROOT" | tail -1 | tr -dc '0-9')
say "free space on $MEDICAL_NLA_DATA_ROOT: ${AVAIL_GB}GB (need about ${NEED_GB}GB)"
if [ "${AVAIL_GB:-0}" -lt "$NEED_GB" ]; then
  say "not enough disk; refusing to start rather than dying halfway"
  exit 1
fi

say "0/6 warming the model cache"
python - >>"$MAIN" 2>&1 <<'PY'
from huggingface_hub import snapshot_download
print(f"[cache] {snapshot_download('google/gemma-3-12b-it')}")
PY
if [ $? -ne 0 ]; then
  say "model download failed -- check the Hugging Face login and the Gemma licence"
  exit 1
fi

# The answers come first and are quick. DDXPlus should reproduce 0.3724 at a
# parse rate of 1.000; anything else means the move changed something, and it
# is worth knowing that before hours of extraction.
say "1/6 DDXPlus answers -> $LOGS/ddxplus_answers.log"
python scripts/run_source_answers.py \
  --config configs/default.yaml \
  --cases "$DATA/ddxplus_cue_count_cases.jsonl" \
  --output-jsonl "$RESULTS/ddxplus_source_answers.jsonl" \
  --summary-json "$RESULTS/ddxplus_source_answers_summary.json" \
  --condition direct --batch-size 8 \
  >"$LOGS/ddxplus_answers.log" 2>&1 \
  && say "1/6 done" || say "1/6 FAILED -- see $LOGS/ddxplus_answers.log"

say "2/6 MedCaseReasoning answers -> $LOGS/mcr_answers.log"
python scripts/run_source_answers.py \
  --config configs/default.yaml \
  --cases "$DATA/mcr_cases_test.jsonl" \
  --output-jsonl "$RESULTS/mcr_source_answers_test.jsonl" \
  --summary-json "$RESULTS/mcr_source_answers_test_summary.json" \
  --condition direct --batch-size 8 \
  >"$LOGS/mcr_answers.log" 2>&1 \
  && say "2/6 done" || say "2/6 FAILED -- see $LOGS/mcr_answers.log"

say "3/6 answer summaries -> $LOGS/answer_summaries.log"
{
  python scripts/summarize_source_answers.py \
    --answers "$RESULTS/ddxplus_source_answers.jsonl" \
    --summary-json "$RESULTS/ddxplus_source_answers_by_diagnosis.json" \
    --min-minority-rate 0.10
  echo
  python scripts/audit_answer_matching.py \
    --answers "$RESULTS/mcr_source_answers_test.jsonl" --show 60
} >"$LOGS/answer_summaries.log" 2>&1 \
  && say "3/6 done" || say "3/6 FAILED -- see $LOGS/answer_summaries.log"

# Eight prompts costs a minute and catches an unresolvable cue or a bad layer
# index before the long jobs commit to them.
say "4/6 extraction smoke test -> $LOGS/extract_smoke.log"
python -m src.extract_activations \
  --config configs/default.yaml \
  --input "$DATA/ddxplus_cuepos_rows.jsonl" \
  --run-name smoke --limit-prompts 8 --no-resume \
  --layers 0 24 32 --strategies $STRATEGIES \
  >"$LOGS/extract_smoke.log" 2>&1
if [ $? -ne 0 ]; then
  say "4/6 smoke FAILED -- see $LOGS/extract_smoke.log; not starting the sweeps"
  exit 1
fi
say "4/6 done"

# --resume is on by default and keyed on manifest ids, so re-running this
# script after a crash picks up where it stopped.
say "5/6 DDXPlus sweep -> $LOGS/ddxplus_sweep.log"
python -m src.extract_activations \
  --config configs/default.yaml \
  --input "$DATA/ddxplus_cuepos_rows.jsonl" "$DATA/ddxplus_format_rows.jsonl" \
  --run-name ddxplus_sweep_v1 \
  --layers $LAYERS $(span_args) \
  --strategies $STRATEGIES \
  --batch-size 8 \
  >>"$LOGS/ddxplus_sweep.log" 2>&1 \
  && say "5/6 done" || say "5/6 FAILED -- see $LOGS/ddxplus_sweep.log"

if [ "$RUN_MCR" = "1" ]; then
  say "6/6 MedCaseReasoning sweep -> $LOGS/mcr_sweep.log"
  python -m src.extract_activations \
    --config configs/default.yaml \
    --input "$DATA/mcr_cuepos_train.jsonl" \
    --run-name mcr_sweep_v1 \
    --layers $LAYERS $(span_args) \
    --strategies $STRATEGIES \
    --batch-size 4 \
    >>"$LOGS/mcr_sweep.log" 2>&1 \
    && say "6/6 done" || say "6/6 FAILED -- see $LOGS/mcr_sweep.log"
else
  say "6/6 skipped (RUN_MCR=0)"
fi

# ------------------------------------------------------------------ report
{
  echo
  echo "==================== morning summary ===================="
  echo "DDXPlus answers, expected accuracy 0.3724 / parse 1.000:"
  [ -s "$RESULTS/ddxplus_source_answers_summary.json" ] \
    && cat "$RESULTS/ddxplus_source_answers_summary.json" || echo "  (missing)"
  echo
  echo "MedCaseReasoning answers, expected accuracy 0.1340 / parse 1.000:"
  [ -s "$RESULTS/mcr_source_answers_test_summary.json" ] \
    && cat "$RESULTS/mcr_source_answers_test_summary.json" || echo "  (missing)"
  echo
  for run in ddxplus_sweep_v1 mcr_sweep_v1; do
    echo "extraction $run:"
    if [ -s "$ART/activations/$run/run.json" ]; then
      python -c "
import json
d = json.load(open('$ART/activations/$run/run.json'))
for k in ('n_rows','n_prompts','layers','span_layers','tensors_written','prompt_token_count_max'):
    print(f'  {k}: {d.get(k)}')
"
    else
      echo "  (no run.json -- did not finish)"
    fi
    du -sh "$ART/activations/$run" 2>/dev/null | sed 's/^/  on disk: /'
  done
  echo
  echo "logs: $LOGS"
  echo "========================================================"
} | tee -a "$MAIN"

say "all done"
