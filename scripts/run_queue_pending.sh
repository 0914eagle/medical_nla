#!/usr/bin/env bash
# The three GPU jobs the paper is still waiting on, run one after another on
# one pair of cards.
#   nohup bash scripts/run_queue_pending.sh > /dev/null 2>&1 &
#   nohup bash scripts/run_queue_pending.sh adapter > /dev/null 2>&1 &   # just one
#
# Order is deliberate. The first two CONFIRM numbers already written into the
# draft; the third OPENS cells that are still empty. If the queue dies partway
# through the night, the confirmations are the ones worth having.
#
#   1. trajectory   re-extract with the fixed last_cue anchor. Decides whether
#                   the headline 84.8% stands at its upper value or falls back
#                   toward 74.1% -- the interval in the thesis closes here.
#   2. vanilla      untuned checkpoint at the ANSWER position. Fills the one
#                   Table 1 row that answers "is the rift an artefact of our
#                   adapter". Independent of job 1: the last_cue fix moves the
#                   last-finding landmark, not the final-token rows the tuned
#                   readout narrated, so the existing manifest still applies.
#   3. adapter      MCR conclusion readout. Opens Table 3's ᵈ cell and the MCR
#                   r5 rung -- the open-vocabulary column.
#
# Each job is skipped if its output already exists, so re-running after a
# crash resumes rather than repeats.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

JOBS=("$@")
if [ "${#JOBS[@]}" -eq 0 ]; then JOBS=(trajectory vanilla adapter); fi

LAYER="${LAYER:-32}"
CASES="${CASES:-$DATA/ddxplus_hint_cases_v2.jsonl}"
TRAJ_ROWS="${TRAJ_ROWS:-$DATA/trajectory_rows_fixed.jsonl}"
TRAJ_RUN="${TRAJ_RUN:-trajectory_fixed_L${LAYER}}"

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/queue_pending_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

say "queue: ${JOBS[*]}   cards=${CUDA_VISIBLE_DEVICES:-<all>}   log=$MAIN"

run_trajectory() {
  say "=== 1/3 trajectory re-extraction (fixed last_cue anchor) ==="
  if [ ! -s "$CASES" ]; then say "no case file at $CASES"; return 1; fi
  if [ -s "$TRAJ_ROWS" ]; then
    say "rows exist: $(wc -l < "$TRAJ_ROWS")"
  else
    # Both arms: the none arm is the counterfactual baseline the dose-response
    # curve is measured against, and it is also where make_trajectory_rows
    # reads each case's real findings from before the hinted arms overwrite
    # cue_targets with the note sentence -- the bug this re-run exists to fix.
    python scripts/make_trajectory_rows.py \
      --cases "$CASES" --arms none wrong --output "$TRAJ_ROWS" \
      >>"$MAIN" 2>&1 || { say "FAILED (rows)"; return 1; }
    say "rows built: $(wc -l < "$TRAJ_ROWS")"
  fi

  # Always run it: extraction resumes by default, writing only the rows missing
  # from the manifest. Skipping on "a manifest exists" was wrong -- a run killed
  # at 98% leaves a manifest that looks finished, and the gap would have been
  # silently carried into the analysis. A completed run costs one model load.
  python -m src.extract_activations \
    --config configs/default.yaml \
    --input "$TRAJ_ROWS" \
    --layers "$LAYER" \
    --run-name "$TRAJ_RUN" \
    --resume \
    >>"$MAIN" 2>&1 || { say "FAILED (extract)"; return 1; }
  say "trajectory done. Analysis is CPU-only and needs the wrong/none answers:"
  say "  python scripts/analyze_trajectory.py --cases $CASES \\"
  say "    --answers <first-pass answers for the none and wrong arms> \\"
  say "    --manifests $(find "$ART/activations/$TRAJ_RUN" -name manifest.jsonl 2>/dev/null | tr '\n' ' ')"
}

run_vanilla() {
  say "=== 2/3 vanilla readout at the answer position ==="
  bash scripts/run_vanilla_final_position.sh >>"$MAIN" 2>&1 \
    || { say "FAILED (vanilla) -- see $MAIN"; return 1; }
  say "vanilla done"
}

run_adapter() {
  say "=== 3/3 MCR conclusion adapter ==="
  bash scripts/run_mcr_conclusion_adapter.sh >>"$MAIN" 2>&1 \
    || { say "FAILED (adapter) -- see $MAIN"; return 1; }
  say "adapter done"
}

FAILED=()
for job in "${JOBS[@]}"; do
  case "$job" in
    trajectory) run_trajectory || FAILED+=("$job") ;;
    vanilla)    run_vanilla    || FAILED+=("$job") ;;
    adapter)    run_adapter    || FAILED+=("$job") ;;
    *) say "unknown job '$job' (trajectory|vanilla|adapter)"; FAILED+=("$job") ;;
  esac
done

# One job failing must not eat the other two: they are independent, and a queue
# that stops at the first error wastes a night of cards.
say ""
if [ "${#FAILED[@]}" -eq 0 ]; then
  say "QUEUE DONE -- all ${#JOBS[@]} jobs finished"
else
  say "QUEUE DONE with failures: ${FAILED[*]}"
  say "  each job logs into $MAIN; re-run just one with:  bash $0 <job>"
  exit 1
fi
