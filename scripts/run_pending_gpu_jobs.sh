#!/usr/bin/env bash
# The two GPU jobs still owed to the paper, queued behind whatever is running.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_pending_gpu_jobs.sh > /dev/null 2>&1 &
#
# Launched while the MCR conclusion readout still had two hours to go. Without
# a queue the cards sit idle from the moment it finishes until somebody
# notices, which on a Sunday night is the whole night.
#
# WHAT IS IN THE QUEUE, AND WHY EACH IS WORTH CARDS
#
# 1. **r7 -- the model's own chain fed back.** Table 4 claims "feed back what
#    is inside", and its comparison so far is against re-showing the input.
#    The obvious competitor -- hand the model its own reasoning -- has never
#    been measured, and until it is, the claim is not established against the
#    thing a reader will immediately propose. No adapter needed; it reuses the
#    finished DDXPlus CoT run.
#
#    r7's population is smaller by construction: cases whose CoT answer
#    differs from the direct first answer are dropped, because a rung that
#    starts from a different first answer is not on the same ladder. The
#    builder prints the surviving id count, and the comparison must be against
#    r3-r6 restricted to those ids -- never against their full columns.
#
# 2. **The DDXPlus correct arm.** Table 2's fourth cell was filled from
#    corpus-300's correct arm over all 4,995 unfiltered rows -- a different run
#    and a different population -- and is currently blank. It carries the claim
#    that the cost of insertion does not depend on the suggestion's direction:
#    a note naming the RIGHT diagnosis still costs accuracy. The placebo cannot
#    make that separation, because the placebo has no opinion at all.
#
# The queue waits rather than refusing, which is the opposite of every other
# launcher here. Those refuse because a job started on a busy card dies in
# nine seconds and takes the queue with it; this one exists precisely to be
# started while a card is busy.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

WAIT_MINUTES="${WAIT_MINUTES:-240}"
POLL_SECONDS="${POLL_SECONDS:-300}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results" "$ART/reports"
MAIN="$LOGS/pending_gpu_jobs_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "queue: r7 ladder, then the DDXPlus correct arm"
say "cards CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

# Wait for the cards, up to WAIT_MINUTES. A queue that gave up immediately
# would be the launcher we already have.
DEADLINE=$(( $(date +%s) + WAIT_MINUTES * 60 ))
while ! python scripts/check_gpu_setup.py --config configs/default.yaml \
        --require-free-gb 20 >>"$MAIN" 2>&1; do
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then
    say "GAVE UP: cards still busy after ${WAIT_MINUTES} min"
    exit 1
  fi
  say "cards busy -- waiting ${POLL_SECONDS}s"
  sleep "$POLL_SECONDS"
done
say "cards free -- starting"

# ---------------------------------------------------------------- job 1: r7
PREFIX="$DATA/ddxplus_ladder"
COT="$ART/results/ddxplus_hint_answers_cot_full.jsonl"
R7_OUT="$ART/results/ddxplus_ladder_r7.jsonl"

if [ ! -s "$COT" ]; then
  say "SKIP r7: no CoT answers at $COT"
elif [ -s "$R7_OUT" ]; then
  say "skip r7 (exists: $(grep -c . "$R7_OUT") rows)"
else
  if [ ! -s "${PREFIX}_r7.jsonl" ]; then
    say "building r7 prompts (CPU)"
    python scripts/make_correction_ladder_cases.py \
      --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
      --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
      --cot-answers "$COT" \
      --rungs 7 \
      --output-prefix "$PREFIX" 2>&1 | tee -a "$MAIN" || { say "FAILED r7 build"; exit 1; }
  fi
  say "r7: answering $(grep -c . "${PREFIX}_r7.jsonl") rows"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "${PREFIX}_r7.jsonl" \
    --output-jsonl "$R7_OUT" \
    --summary-json "$ART/reports/ddxplus_ladder_r7.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED r7"; exit 1; }
  say "done r7 -> $R7_OUT"
fi

# ------------------------------------------------- job 2: DDXPlus correct arm
CORRECT_OUT="$ART/results/ddxplus_hint_answers_correct.jsonl"
if [ -s "$CORRECT_OUT" ]; then
  say "skip correct arm (exists: $(grep -c . "$CORRECT_OUT") rows)"
else
  say "correct arm"
  bash scripts/run_ddxplus_correct_arm.sh >>"$MAIN" 2>&1 \
    && say "done correct arm" || say "FAILED correct arm"
fi

say ""
say "ALL DONE. Next, on CPU:"
say "  python scripts/analyze_correction_ladder.py --ladder $ART/results/ddxplus_ladder_r*.jsonl \\"
say "      --answers $ART/results/ddxplus_hint_answers_v2.jsonl --cases $DATA/ddxplus_hint_cases_v2.jsonl"
say "  python scripts/analyze_hint_effect.py --answers $ART/results/ddxplus_hint_answers_v2.jsonl \\"
say "      $CORRECT_OUT --cases $DATA/ddxplus_hint_cases_v2.jsonl"
say ""
say "r7 is comparable only to r3-r6 RESTRICTED to r7's surviving ids -- the"
say "build log above prints how many that is. Comparing it to their full"
say "columns compares two different case sets."
