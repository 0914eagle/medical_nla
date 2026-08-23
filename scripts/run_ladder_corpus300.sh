#!/usr/bin/env bash
# The correction ladder on corpus-300 -- powering the two thinnest numbers.
#   CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/run_ladder_corpus300.sh > /dev/null 2>&1 &
#
# Two results in the paper rest on small cells, and both are cells of this
# experiment rather than of the corpus:
#
#   - the natural-language advantage when the fed-back content is wrong
#     (0.40 vs 0.24) rests on eight discordant pairs;
#   - the adoption rate among cases pulled onto the suspicion (41%) rests on
#     95 cases, a +-10pp interval.
#
# corpus-300 holds 4,995 direct-correct cases against the main run's 1,747, so
# a rerun multiplies both by ~2.9: the both-wrong cell goes 50 -> ~140 (8 -> ~23
# discordant pairs) and the moved population 324 -> ~900. Nothing about the
# design changes, which is the point -- this is the same ladder on more cases,
# not a new experiment whose agreement would prove less.
#
# The first-pass answers, the readouts and the activations already exist; only
# the probe verdicts (r6's content, and the hybrid's selector) and the three
# rungs are computed here.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/reports"
MAIN="$LOGS/ladder_corpus300_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

CASES="$DATA/ddxplus_hint_cases_300.jsonl"
ANSWERS="$ART/results/ddxplus_hint_answers_300.jsonl"
READOUTS="$ART/results/readout_hint_final_300_L32_v2.jsonl"
MANIFEST="$ART/activations/hint_positions_300_L32/layer32/last_token/manifest.jsonl"
VERDICTS="$ART/results/probe_verdicts_300.jsonl"
PREFIX="$DATA/ddxplus_ladder_300"

for f in "$CASES" "$ANSWERS" "$READOUTS" "$MANIFEST"; do
  [ -s "$f" ] || { say "missing input: $f"; exit 1; }
done

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

# The probe is refit on corpus-300's own cases, cross-fit within diagnosis as
# before. Reusing the 1,747-case probe would leak: its training cases are a
# subset of these.
if [ ! -s "$VERDICTS" ]; then
  say "probe verdicts (corpus-300)"
  python scripts/evaluate_probe_disagreement.py \
    --answers "$ANSWERS" \
    --cases "$CASES" \
    --manifests "$MANIFEST" \
    --dump "$VERDICTS" >>"$MAIN" 2>&1 || { say "FAILED probe"; exit 1; }
fi

if [ ! -s "${PREFIX}_r6.jsonl" ]; then
  say "building ladder prompts (CPU)"
  python scripts/make_correction_ladder_cases.py \
    --cases "$CASES" \
    --answers "$ANSWERS" \
    --readouts "$READOUTS" \
    --readout-manifests "$MANIFEST" \
    --rungs 4 5 6 \
    --probe-verdicts "$VERDICTS" \
    --output-prefix "$PREFIX" >>"$MAIN" 2>&1 || { say "FAILED build"; exit 1; }
fi

for R in 4 5 6; do
  OUT="$ART/results/ddxplus_ladder_300_r${R}.jsonl"
  if [ -s "$OUT" ]; then say "skip rung $R (exists)"; continue; fi
  say "rung $R"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "${PREFIX}_r${R}.jsonl" \
    --output-jsonl "$OUT" \
    --summary-json "$ART/reports/ddxplus_ladder_300_r${R}.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED rung $R"; exit 1; }
  say "done rung $R"
done

say "ALL DONE (ladder on corpus-300)"
say "analyse with:"
say "  python scripts/analyze_correction_ladder.py \\"
say "    --rungs \$ART/results/ddxplus_ladder_300_r{4,5,6}.jsonl \\"
say "    --probe-flags $VERDICTS"
