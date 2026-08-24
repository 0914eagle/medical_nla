#!/usr/bin/env bash
# Read MCR answer-position states with the conclusion adapter, on held-out rows.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_conclusion_readout.sh > /dev/null 2>&1 &
#
# The adapter finished on 08-24 (best_epoch 1 of 3, selected on content loss
# 1.767 against a scaffold loss of 0.038 -- the XML is learned immediately and
# the diagnosis is not, which is why selecting on content was the fix that let
# it save at all).
#
# TWO THINGS THIS SCRIPT REFUSES TO GUESS.
#
# 1. **Which split.** The extraction covers the whole corpus -- 12,620 rows =
#    train 10,663 + val 1,136 + test 821 -- because the adapter was trained
#    from it. Reading out train rows and reporting a description rate would
#    measure what the adapter memorised, which is the one thing a held-out
#    design exists to prevent. The manifest is filtered to the test split by
#    base_id before anything is generated.
#
# 2. **Which question the prompts can answer.** Note-free states answer the
#    instrument question -- "does the readout describe real clinical prose, not
#    just DDXPlus's templated sentences?" -- which fills Table 1's missing MCR
#    row. Only note-bearing states answer the attribution question (Table 3b's
#    MCR cell, the MCR r5 rung), because that claim is about what the note did.
#    The two are not interchangeable, so the manifest is inspected and the
#    finding reported rather than assumed.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LAYER="${LAYER:-32}"
RUN_NAME="${RUN_NAME:-mcr_answerpos_L${LAYER}}"
ADAPTER="${ADAPTER:-$ART/train/adapters/mcr_conclusion_L${LAYER}_s17}"
SPLIT_DIR="${SPLIT_DIR:-$ART/train/mcr_conclusion_conclusion_L${LAYER}}"
TEMPLATE="${TEMPLATE:-prompt_templates/medical_nla_v2_readout.txt}"
BATCH="${BATCH:-8}"
# Set from the targets this adapter was trained on, not from the config
# default. MCR conclusion targets average 764 characters and pass 2,500 at the
# top; the config's 256 tokens cut 444 of the first run's 821 readouts off
# mid-sentence, which looked like the adapter degenerating and was our budget.
MAX_NEW="${MAX_NEW:-768}"
OUT="${OUT:-$ART/results/readout_mcr_conclusion_L${LAYER}.jsonl}"
TEST_MANIFEST="${TEST_MANIFEST:-$DATA/mcr_conclusion_test_manifest.jsonl}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results"
MAIN="$LOGS/mcr_conclusion_readout_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

[ -s "$ADAPTER/best.json" ] || { say "no trained adapter at $ADAPTER"; exit 1; }
[ -s "$SPLIT_DIR/sft_test.jsonl" ] || { say "no test split at $SPLIT_DIR"; exit 1; }

mapfile -t MANIFESTS < <(find "$ART/activations/$RUN_NAME" -name manifest.jsonl -size +0 2>/dev/null | sort)
[ "${#MANIFESTS[@]}" -gt 0 ] || { say "no manifest under $ART/activations/$RUN_NAME"; exit 1; }
FULL="${MANIFESTS[0]}"
say "adapter  $ADAPTER"
say "manifest $FULL ($(wc -l < "$FULL") rows, all splits)"

python scripts/filter_manifest_to_split.py \
  --manifest "$FULL" --split "$SPLIT_DIR/sft_test.jsonl" --output "$TEST_MANIFEST" \
  2>&1 | tee -a "$MAIN" || { say "FAILED (split filter)"; exit 1; }

# An existing file is only reusable if it was generated at the budget we now
# ask for. Skipping on existence alone is what would silently keep the
# truncated run in place, and every rate computed from it reads as finished
# output.
PRIOR=""
[ -s "$OUT" ] && PRIOR=$(head -1 "$OUT" | python -c \
  'import json,sys; print(json.load(sys.stdin).get("gen_config",{}).get("max_new_tokens",""))' 2>/dev/null)
if [ -s "$OUT" ] && [ -n "$PRIOR" ] && [ "$PRIOR" -lt "$MAX_NEW" ]; then
  say "REDO: $OUT was generated at max_new_tokens=$PRIOR, below $MAX_NEW"
  mv "$OUT" "$OUT.mnt$PRIOR.bak"
  say "     moved aside -> $OUT.mnt$PRIOR.bak"
fi

if [ -s "$OUT" ]; then say "skip (exists: $(wc -l < "$OUT") rows)"; else
  if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
    say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
  fi
  say "reading out -> $OUT"
  python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "$TEST_MANIFEST" \
    --output "$OUT" \
    --adapter-id "$ADAPTER" \
    --actor-prompt-template-file "$TEMPLATE" \
    --batch-size "$BATCH" \
    --max-new-tokens "$MAX_NEW" \
    >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi

# The budget is only right if the outputs stopped on their own.
CUT=$(python - "$OUT" <<'PY'
import json, sys
n = sum(1 for l in open(sys.argv[1])
        if "<supporting_cues>" in (json.loads(l).get("nla_output") or "")
        and "</supporting_cues>" not in (json.loads(l).get("nla_output") or ""))
print(n)
PY
)
say "readouts still cut off mid-cue: $CUT (raise MAX_NEW if this is not ~0)"

say "ALL DONE -> $OUT"
say ""
say "Read ten before trusting any rate. Content loss is 1.77 on an open"
say "vocabulary, so whether the adapter reads MCR states or writes plausible"
say "diagnoses is a question no rate can answer on its own:"
say "  head -3 $OUT | python -m json.tool"
