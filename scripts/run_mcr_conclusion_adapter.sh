#!/usr/bin/env bash
# MCR conclusion adapter: build the SFT splits from the finished answer-position
# extraction, then train.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_conclusion_adapter.sh > /dev/null 2>&1 &
#
# This is the adapter that fills the open-vocabulary column. The v2 adapter was
# taught DDXPlus's 49-name vocabulary at cue positions; MCR is 6,934 diagnoses
# written as published case narrative, and reading its answer position in words
# is the one thing v2 cannot do. Nothing here needs a judge: the target is
# assembled from `diagnosis_name` and `cue_targets`, exactly as the DDXPlus
# targets were.
#
# Prerequisite (done): the answer-position extraction
#   python -m src.extract_activations --config configs/default.yaml \
#     --input $DATA/mcr_answerpos_rows_{train,test}.jsonl \
#     --layers 32 --run-name mcr_answerpos_L32
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LAYER="${LAYER:-32}"
RUN_NAME="${RUN_NAME:-mcr_answerpos_L${LAYER}}"
# An sft_train.jsonl the trainer already accepts. Its key set -- not this
# script's opinion -- defines the output schema.
TEMPLATE="${TEMPLATE:-$ART/train/ddxplus_cuepos_L${LAYER}/sft_train.jsonl}"
SPLIT_DIR="$ART/train/mcr_conclusion_conclusion_L${LAYER}"
# The gold is only a legitimate readout target where the model reached it.
# The first build omitted these and trained on states that had concluded
# something else; see make_mcr_conclusion_split.py's docstring.
if [ -n "${ANSWERS:-}" ]; then
  read -r -a ANSWER_FILES <<< "$ANSWERS"
else
  ANSWER_FILES=("$ART/results/mcr_source_answers_train.jsonl"
                "$ART/results/mcr_source_answers_test.jsonl")
fi

LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/mcr_conclusion_adapter_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

# The extraction writes one manifest per (layer, selection); take whatever it
# actually produced rather than assuming last_subtoken.
mapfile -t MANIFESTS < <(find "$ART/activations/$RUN_NAME" -name manifest.jsonl -size +0 2>/dev/null | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  say "no manifest under $ART/activations/$RUN_NAME"
  say "  the answer-position extraction has not landed. Check:"
  say "    ls -R $ART/activations/$RUN_NAME | head"
  exit 1
fi
say "manifests: ${MANIFESTS[*]}"

if [ ! -s "$TEMPLATE" ]; then
  say "no schema template at $TEMPLATE"
  say "  point TEMPLATE= at any sft_train.jsonl the trainer accepts:"
  say "    ls \$ART/train/*/sft_train.jsonl"
  exit 1
fi

for f in "${ANSWER_FILES[@]}"; do
  [ -s "$f" ] || { say "no source answers at $f"; exit 1; }
done
say "answers: ${ANSWER_FILES[*]}"

# A split built before the source-correct filter existed carries no
# source_correct key, and reusing it on existence alone is what would keep the
# wrong training set in place -- the adapter trained from it looked finished.
if [ -s "$SPLIT_DIR/sft_train.jsonl" ] && \
   ! head -1 "$SPLIT_DIR/sft_train.jsonl" | grep -q '"source_correct"'; then
  say "REBUILD: $SPLIT_DIR was built without the source-correct filter"
  mv "$SPLIT_DIR" "$SPLIT_DIR.unfiltered.bak"
  say "     moved aside -> $SPLIT_DIR.unfiltered.bak"
fi

if [ -s "$SPLIT_DIR/sft_train.jsonl" ]; then
  say "skip split build (exists: $(wc -l < "$SPLIT_DIR/sft_train.jsonl") train rows)"
else
  say "building splits -> $SPLIT_DIR"
  python scripts/make_mcr_conclusion_split.py \
    --cases "$DATA/mcr_cases_train.jsonl" "$DATA/mcr_cases_test.jsonl" \
    --split-name train test \
    --manifest "${MANIFESTS[@]}" \
    --template "$TEMPLATE" \
    --answers "${ANSWER_FILES[@]}" \
    --output-dir "$SPLIT_DIR" \
    >>"$MAIN" 2>&1 || { say "FAILED (split build)"; exit 1; }
  say "built: $(wc -l < "$SPLIT_DIR/sft_train.jsonl") train rows"
fi

# The adapter trained from the unfiltered split is not a checkpoint to resume
# from; it learned a different target.
OLD_ADAPTER="$ART/train/adapters/mcr_conclusion_L${LAYER}_s17"
if [ -d "$OLD_ADAPTER" ] && [ -d "$SPLIT_DIR.unfiltered.bak" ]; then
  mv "$OLD_ADAPTER" "$OLD_ADAPTER.unfiltered.bak"
  say "moved the unfiltered adapter aside -> $OLD_ADAPTER.unfiltered.bak"
fi

# One seed, not the usual three: this adapter exists to fill a column, and the
# seed spread is a question for the DDXPlus instrument, which already has it.
SEEDS="${SEEDS:-17}"
say "training mcr_conclusion L${LAYER} seeds ${SEEDS}"
LAYERS="$LAYER" SEEDS="$SEEDS" bash scripts/run_readout_training.sh mcr_conclusion \
  >>"$MAIN" 2>&1 || { say "FAILED (training)"; exit 1; }
say "ALL DONE -- adapter under $ART/train/adapters/mcr_conclusion_L${LAYER}_s*"
