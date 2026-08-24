#!/usr/bin/env bash
# The one arm the main DDXPlus run never had.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_ddxplus_correct_arm.sh > /dev/null 2>&1 &
#
# Table 2's DDXPlus row reads as one population, and three of its four cells
# are that population: the 1,747-case run, clean subset n = 1,220, giving
# .991 / .934 / .760 for none / neutral / wrong. The fourth cell, the correct
# note at .932, came from somewhere else -- an arm scan on 08-24 found no
# `correct` rows in any 1,747-case answers file, and .9313 is corpus-300's
# correct arm over ALL 4,995 cases, unfiltered. So that cell is off by both a
# run and a population filter.
#
# This fills it properly: the correct arm of the same case file the other
# three arms came from. One arm, 1,747 direct answers with prefill, so it is
# short. Nothing else in the paper moves -- the mechanism, trajectory and
# ladder analyses all live on the wrong arm.
#
# Why the cell is worth an hour of cards rather than a footnote: it carries
# the claim that the cost of insertion is not about the suggestion's
# direction. A note that names the RIGHT diagnosis still costs accuracy, which
# separates "the model is being pushed by a suggestion" from "the model is
# disturbed by an extra sentence" -- and the placebo alone cannot make that
# separation, because the placebo has no opinion at all.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

CASES="${CASES:-$DATA/ddxplus_hint_cases_v2.jsonl}"
OUT="${OUT:-$ART/results/ddxplus_hint_answers_correct.jsonl}"
BATCH="${BATCH:-16}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results"
MAIN="$LOGS/ddxplus_correct_arm_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if [ ! -s "$CASES" ]; then say "no case file at $CASES"; exit 1; fi

# Fail before loading the model rather than after: a case file built before
# the four-arm builder has nothing to run here.
N=$(python - "$CASES" <<'PY'
import json, sys
print(sum(1 for line in open(sys.argv[1])
          if json.loads(line).get("hint_variant") == "correct"))
PY
)
if [ "${N:-0}" -eq 0 ]; then
  say "no correct-arm rows in $CASES -- rebuild the cases with make_hint_injection_cases.py"
  exit 1
fi
say "correct-arm rows in cases: $N"

if [ -s "$OUT" ]; then say "skip (exists: $(wc -l < "$OUT") rows)"; else
  if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
    say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
  fi
  say "answering the correct arm -> $OUT"
  python scripts/run_source_answers.py \
    --config configs/default.yaml \
    --cases "$CASES" \
    --where hint_variant=correct \
    --condition direct \
    --batch-size "$BATCH" \
    --output-jsonl "$OUT" \
    >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi

say "ALL DONE. Re-dump Table 2 / Figure 3 with all four arms (CPU):"
say "  python scripts/analyze_hint_effect.py \\"
say "    --answers $ART/results/ddxplus_hint_answers_v2.jsonl \\"
say "              $ART/results/ddxplus_hint_answers_neutral.jsonl \\"
say "              $OUT \\"
say "    --dump $ART/results/fig3_ddx.json"
say "  python scripts/make_figure_intervention.py \\"
say "    --dumps $ART/results/fig3_ddx.json $ART/results/fig3_mcr.json \\"
say "    --labels DDXPlus MedCaseReasoning \\"
say "    --output $ART/results/figure3_intervention.pdf"
say "The [warn] about a missing correct arm should be gone."
