#!/usr/bin/env bash
# The whole GPU queue, unattended.
#
#   nohup bash scripts/run_overnight.sh > /dev/null 2>&1 &
#
# Two device pairs run at once: cards 0,1 take the answer labels and then the
# DDXPlus sweep; cards 2,3 take the MedCaseReasoning sweep. Each stage writes
# its own log under $ART/logs, and a stage that fails stops its own chain
# without killing the other.
#
# The model is downloaded once, before either chain starts. Two processes
# reaching a cold Hugging Face cache at the same time contend on the same lock
# files, and a 25GB checkpoint is not something to discover racing at 3am.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh

LOGS="$ART/logs"
RESULTS="$ART/results"
mkdir -p "$LOGS" "$RESULTS"

STAMP=$(date +%Y%m%d_%H%M%S)
MAIN="$LOGS/overnight_${STAMP}.log"

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

# Activations are the bulk of this and there is no partial credit for filling
# a disk: the DDXPlus sweep is about 17GB and MedCaseReasoning about 47GB,
# most of it the full spans kept at three layers.
NEED_GB=90
AVAIL_GB=$(df -BG --output=avail "$MEDICAL_NLA_DATA_ROOT" | tail -1 | tr -dc '0-9')
say "free space on $MEDICAL_NLA_DATA_ROOT: ${AVAIL_GB}GB (need about ${NEED_GB}GB)"
if [ "${AVAIL_GB:-0}" -lt "$NEED_GB" ]; then
  say "not enough disk; refusing to start rather than dying halfway"
  exit 1
fi

say "warming the model cache (once, so the two chains do not race it)"
python - >>"$MAIN" 2>&1 <<'PY'
from huggingface_hub import snapshot_download
path = snapshot_download("google/gemma-3-12b-it")
print(f"[cache] {path}")
PY
if [ $? -ne 0 ]; then
  say "model download failed -- check the Hugging Face login and the Gemma licence"
  exit 1
fi

# ---------------------------------------------------------------- cards 2,3
say "launching MedCaseReasoning extraction on cards 2,3 -> $LOGS/mcr_sweep.log"
CUDA_VISIBLE_DEVICES=2,3 nohup python -m src.extract_activations \
  --config configs/default.yaml \
  --input "$DATA/mcr_cuepos_train.jsonl" \
  --run-name mcr_sweep_v1 \
  --layers 0 4 8 12 16 20 24 28 32 36 40 44 47 \
  --span-layers 16 24 32 \
  --strategies last_subtoken span_mean \
  --batch-size 4 \
  >"$LOGS/mcr_sweep.log" 2>&1 &
MCR_PID=$!

# ---------------------------------------------------------------- cards 0,1
export CUDA_VISIBLE_DEVICES=0,1

# The answers come first and are quick. DDXPlus should reproduce 0.3724 with a
# parse rate of 1.000; anything else means the move changed something, and it
# is worth knowing that before nine hours of extraction.
say "1/4 DDXPlus answers -> $LOGS/ddxplus_answers.log"
python scripts/run_source_answers.py \
  --config configs/default.yaml \
  --cases "$DATA/ddxplus_cue_count_cases.jsonl" \
  --output-jsonl "$RESULTS/ddxplus_source_answers.jsonl" \
  --summary-json "$RESULTS/ddxplus_source_answers_summary.json" \
  --condition direct --batch-size 8 \
  >"$LOGS/ddxplus_answers.log" 2>&1 \
  && say "1/4 done" || { say "1/4 FAILED -- see $LOGS/ddxplus_answers.log"; }

say "2/4 MedCaseReasoning answers -> $LOGS/mcr_answers.log"
python scripts/run_source_answers.py \
  --config configs/default.yaml \
  --cases "$DATA/mcr_cases_test.jsonl" \
  --output-jsonl "$RESULTS/mcr_source_answers_test.jsonl" \
  --summary-json "$RESULTS/mcr_source_answers_test_summary.json" \
  --condition direct --batch-size 8 \
  >"$LOGS/mcr_answers.log" 2>&1 \
  && say "2/4 done" || { say "2/4 FAILED -- see $LOGS/mcr_answers.log"; }

say "3/4 answer summaries"
{
  python scripts/summarize_source_answers.py \
    --answers "$RESULTS/ddxplus_source_answers.jsonl" \
    --summary-json "$RESULTS/ddxplus_source_answers_by_diagnosis.json" \
    --min-minority-rate 0.10
  echo
  python scripts/audit_answer_matching.py \
    --answers "$RESULTS/mcr_source_answers_test.jsonl" --show 60
} >"$LOGS/answer_summaries.log" 2>&1 \
  && say "3/4 done" || say "3/4 FAILED -- see $LOGS/answer_summaries.log"

# A smoke run first: eight prompts costs a minute and catches an unresolvable
# cue or a bad layer index before the long job commits to them.
say "4/4 DDXPlus extraction, smoke first -> $LOGS/ddxplus_sweep.log"
python -m src.extract_activations \
  --config configs/default.yaml \
  --input "$DATA/ddxplus_cuepos_rows.jsonl" \
  --run-name smoke --limit-prompts 8 --no-resume \
  --layers 0 24 32 --strategies last_subtoken span_mean \
  >"$LOGS/ddxplus_smoke.log" 2>&1
if [ $? -ne 0 ]; then
  say "4/4 smoke FAILED -- see $LOGS/ddxplus_smoke.log; not starting the sweep"
else
  python -m src.extract_activations \
    --config configs/default.yaml \
    --input "$DATA/ddxplus_cuepos_rows.jsonl" "$DATA/ddxplus_format_rows.jsonl" \
    --run-name ddxplus_sweep_v1 \
    --layers 0 4 8 12 16 20 24 28 32 36 40 44 47 \
    --span-layers 16 24 32 \
    --strategies last_subtoken span_mean \
    --batch-size 8 \
    >>"$LOGS/ddxplus_sweep.log" 2>&1 \
    && say "4/4 done" || say "4/4 FAILED -- see $LOGS/ddxplus_sweep.log"
fi

say "cards 0,1 finished; waiting on MedCaseReasoning (pid $MCR_PID)"
wait "$MCR_PID" && say "MedCaseReasoning extraction done" \
                || say "MedCaseReasoning extraction FAILED -- see $LOGS/mcr_sweep.log"

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
    [ -s "$ART/activations/$run/run.json" ] \
      && python -c "
import json, sys
d = json.load(open('$ART/activations/$run/run.json'))
for k in ('n_rows','n_prompts','layers','span_layers','tensors_written','prompt_token_count_max'):
    print(f'  {k}: {d.get(k)}')
" || echo "  (no run.json -- did not finish)"
    du -sh "$ART/activations/$run" 2>/dev/null | sed 's/^/  on disk: /'
  done
  echo
  echo "logs: $LOGS"
  echo "========================================================"
} | tee -a "$MAIN"

say "all done"
