#!/usr/bin/env bash
# The referring-note intervention on real case reports (MCR), four arms.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_intervention.sh > /dev/null 2>&1 &
#
# DDXPlus is synthetic and closes its diagnoses at 49. The reviewer question
# that follows -- does any of this survive real clinical prose and an open
# label space -- is answered here and nowhere else, so this run is not a
# robustness footnote but the corpus half of the argument.
#
# Both splits feed the pool. MCR's split exists to fine-tune a reasoner and
# nothing here trains on it; the hygiene rule that does apply (train the
# readout adapter on train, read out on test) is untouched by widening an
# observational pool from 113 cases to 1,543.
#
# The wrong arm's suspicion has two provenances, and the rows carry which:
# the model's own confusion for that gold where the corpus supplies one, a
# cue-similar neighbour's diagnosis otherwise. Those are different
# interventions -- the fallback proposed a skin disease for a brain lesion --
# so the analysis splits on suggestion_source rather than averaging them.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/reports"
MAIN="$LOGS/mcr_intervention_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

CASES="$DATA/mcr_hint_cases_full.jsonl"
OUT="$ART/results/mcr_hint_answers_full.jsonl"

if [ ! -s "$CASES" ]; then
  say "building MCR hint cases (CPU, both splits)"
  python scripts/make_mcr_hint_cases.py \
    --cases "$DATA/mcr_cases_train.jsonl" "$DATA/mcr_cases_test.jsonl" \
    --answers "$ART/results/mcr_source_answers_train.jsonl" \
              "$ART/results/mcr_source_answers_test.jsonl" \
    --output "$CASES" >>"$MAIN" 2>&1 || { say "FAILED build"; exit 1; }
fi
say "cases: $(wc -l < "$CASES") rows"

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

if [ -s "$OUT" ]; then say "skip (answers exist)"; else
  say "answering four arms (case reports are long; allow ~1-2h)"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$CASES" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/mcr_hint_answers_full.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi

say "ALL DONE (MCR intervention)"
say "analyse with:"
say "  python scripts/analyze_hint_effect.py --answers $OUT --cases $CASES"
say "  (reports the arms split by suggestion_source; read the confusion"
say "   subset as the main number and the neighbour subset separately)"
