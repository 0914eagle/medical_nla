#!/usr/bin/env bash
# The reader-trust task, run on the external judge instead of the local model.
#   nohup bash scripts/run_reader_trust_judge.sh > /dev/null 2>&1 &
#
# Two reasons this exists beside run_reader_trust.sh, which asks gemma.
#
# 1. **It needs no GPU.** The cards are the bottleneck and this is the one
#    experiment where the natural-language readout can win on its own terms,
#    so it should not queue behind an adapter.
#
# 2. **The reader should not be the model under study.** The question is what
#    an account gives a reader who did not produce it. Asking the same
#    checkpoint to read its own readout confounds the two.
#
# What it asks: given one channel's account of what was going on inside the AI
# -- and nothing else -- does the reader have reason to doubt the answer? The
# ground truth is whether the referring note actually moved it, which the
# reader cannot see. One channel per row, so length helps a channel only if
# the extra words carry the signal.
#
# This is the experiment that decides one word in the thesis. Detection and
# correction are both won by the probe, which returns a class label; a label
# is not something a clinician can act on. If the readout wins here, "we
# describe it in a sentence a clinician can read" is earned. If it does not,
# the sentence stays "we describe it in natural language" and that is the
# honest claim.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

CASES="${CASES:-$DATA/ddxplus_reader_trust_cases.jsonl}"
OUT="${OUT:-$ART/results/judge_reader_trust.jsonl}"
BACKEND="${BACKEND:-codex}"
MODEL="${MODEL:-}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/reports"
MAIN="$LOGS/reader_trust_judge_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if [ ! -s "$CASES" ]; then
  say "building trust prompts (CPU)"
  python scripts/make_reader_trust_cases.py \
    --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
    --answers "$ART/results/ddxplus_hint_answers_v2.jsonl" \
    --cot-answers "$ART/results/ddxplus_hint_answers_cot_full.jsonl" \
    --readouts "$ART/results/readout_hint_final_L32_v2.jsonl" \
    --readout-manifests "$ART/activations/hint_positions_L32/layer32/last_token/manifest.jsonl" \
    --probe-verdicts "$ART/results/probe_verdicts.jsonl" \
    --output "$CASES" 2>&1 | tee -a "$MAIN" || { say "FAILED build"; exit 1; }
fi
say "cases: $(wc -l < "$CASES") rows (one per case per channel)"

# Priced before spending. The rows carry {id, prompt} already, which is the
# judge's input schema -- no conversion step to get wrong.
python scripts/run_judge.py --requests "$CASES" --out /dev/null --backend dry-run \
  --in-price 1.25 --out-price 10 --out-tokens 40 2>&1 | tee -a "$MAIN"

# Three fields of strict JSON, so 40 output tokens is the right estimate --
# the 64 default would overprice this run by half.
say "judging with $BACKEND ${MODEL:-(codex default)}"
python scripts/run_judge.py \
  --requests "$CASES" \
  --out "$OUT" \
  --backend "$BACKEND" \
  ${MODEL:+--model "$MODEL"} \
  >>"$MAIN" 2>&1 || { say "FAILED (judging)"; exit 1; }

say "judged: $(wc -l < "$OUT") rows -> $OUT"
say "analyse with:"
say "  python scripts/analyze_reader_trust.py --judgements $OUT"
