#!/usr/bin/env bash
# Read MCR answer-position states with the conclusion adapter, on held-out rows.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_conclusion_readout.sh > /dev/null 2>&1 &
#
# The adapter was retrained on 08-24 after the source-correct filter landed:
# 1,298 rows where the model actually reached the gold, against the 10,663 of
# the first build whose targets were 88% states that had concluded something
# else. best_epoch 1 of 3, content loss 1.8209 -- epochs 2 and 3 were worse, so
# this is where it peaks and more epochs are not the missing ingredient. The
# figure is not comparable to the first build's 1.767: the validation set
# changed from 1,136 mostly-wrong targets to 132 correct ones.
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

# An existing file is reusable only if it was produced by THIS adapter, at
# THIS budget, over EVERY row. Existence alone is evidence of none of the
# three, and each has already been wrong once:
#
#   budget    the first run took the config default of 256 and truncated 54%
#   adapter   a killed run left 128 rows written by the pre-filter adapter,
#             and the adapter path is unchanged because the new one replaced
#             it -- only the file's age against the adapter's separates them
#   coverage  those 128 rows of 821 were then skipped over as finished, and
#             the script printed ALL DONE over a run that never happened
REDO=""
if [ -s "$OUT" ]; then
  PRIOR=$(head -1 "$OUT" | python -c \
    'import json,sys; print(json.load(sys.stdin).get("gen_config",{}).get("max_new_tokens",""))' 2>/dev/null)
  HAVE=$(grep -c . "$OUT" 2>/dev/null || echo 0)
  WANT=$(grep -c . "$TEST_MANIFEST" 2>/dev/null || echo 0)
  if [ -n "$PRIOR" ] && [ "$PRIOR" -lt "$MAX_NEW" ]; then
    REDO="generated at max_new_tokens=$PRIOR, below $MAX_NEW"
  fi
  if [ "$WANT" -gt 0 ] && [ "$HAVE" -lt "$WANT" ]; then
    REDO="incomplete: $HAVE of $WANT rows"
  fi
  if [ -f "$ADAPTER/adapter_model.safetensors" ] && \
     [ "$ADAPTER/adapter_model.safetensors" -nt "$OUT" ]; then
    REDO="older than the adapter that should have written it"
  fi
fi
if [ -n "$REDO" ]; then
  say "REDO: $OUT is $REDO"
  mv "$OUT" "$OUT.stale.$(date +%H%M%S).bak"
  say "     moved aside -> $OUT.stale.*.bak"
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
say "Read ten before trusting any rate. Content loss is 1.82 on an open"
say "vocabulary, so whether the adapter reads MCR states or writes plausible"
say "diagnoses is a question no rate can answer on its own:"
say "  head -3 $OUT | python -m json.tool"
