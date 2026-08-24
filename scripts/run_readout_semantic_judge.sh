#!/usr/bin/env bash
# Judge job #3 -- fill Table 1's blank scorer cell with an external judge.
#
# Table 1's semantic row (.340 / .731 / .557 for L16, L24, v4) came from a hand
# pass. This re-judges the same pairs with a model that is not the backbone and
# prints both scorers side by side.
#
# Cheaper than the plan said. judge_jobs put this at n=1,314 -- 438 rows times
# three layers. DDXPlus renders its cues from a fixed questionnaire, so those
# rows collapse to 92 + 72 + 74 = 238 distinct (gold, readout) pairs, a 5.5x
# cut. Judging pairs also makes the result self-consistent: the same pair
# cannot come back A in one row and B in another.
#
# No GPU. The inputs are in results_snapshot/, so this runs anywhere codex does.
#
#   source scripts/env.sh && bash scripts/run_readout_semantic_judge.sh
#   DRY=1 bash scripts/run_readout_semantic_judge.sh     # price it first
set -euo pipefail

: "${ART:?run 'source scripts/env.sh' first}"
: "${DATA:?run 'source scripts/env.sh' first}"

BACKEND="${BACKEND:-codex}"
MODEL="${MODEL:-}"
LAYERS="${LAYERS:-v4 L16_v5 L24_v5}"
WORK="$DATA/judge_readout_semantic"
RES="$ART/results"
mkdir -p "$WORK" "$RES" "$ART/logs"

if [ -n "${DRY:-}" ]; then
  BACKEND="dry-run"
fi

say() { echo "[$(date +%H:%M:%S)] $*"; }

for L in $LAYERS; do
  POOL="results_snapshot/${L}_test_heldout_cue_scored_compact.jsonl"
  HAND="results_snapshot/${L}_heldout_pairs_hand_labeled.jsonl"
  if [ ! -s "$POOL" ]; then
    say "SKIP $L -- no pool at $POOL"
    continue
  fi

  say "$L -- build requests"
  python scripts/make_readout_judge_requests.py \
    --readouts "$POOL" \
    --out-dir "$WORK/$L" \
    --requests "$WORK/req_${L}.jsonl"

  OUT="$RES/judge_readout_semantic_${L}.jsonl"
  say "$L -- judge ($BACKEND)"
  # Resumable and locked: a second copy of this script cannot interleave
  # writes into the same file, which has cost us a judged file twice.
  python scripts/run_judge.py \
    --requests "$WORK/req_${L}.jsonl" \
    --out "$OUT" \
    --backend "$BACKEND" \
    ${MODEL:+--model "$MODEL"} \
    --out-tokens 4 \
    --in-price 1.25 --out-price 10

  if [ "$BACKEND" = "dry-run" ]; then
    continue
  fi

  say "$L -- compare against the hand pass"
  python scripts/analyze_readout_semantic_judgements.py \
    --index "$WORK/$L/judge_index.jsonl" \
    --judged "$OUT" \
    ${HAND:+--hand "$HAND"} \
    --label "$L"
done

if [ "$BACKEND" = "dry-run" ]; then
  say "dry run only -- no verdicts written. Drop DRY=1 to judge."
  exit 0
fi

cat <<'EOF'

Next: put the judge row into Table 1 with the judge's model id and date, and
leave the hand row in the audit record. The two rates and their kappa are the
provenance -- a single number with no scorer named is what the blank cell was
avoiding in the first place.
EOF
