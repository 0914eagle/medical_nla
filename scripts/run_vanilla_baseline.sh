#!/usr/bin/env bash
# Vanilla AV readouts: the same vectors through the released checkpoint, with
# no adapter.
#
#   bash scripts/run_vanilla_baseline.sh
#   LAYERS="24" bash scripts/run_vanilla_baseline.sh
#
# Kill the whole thing with `pkill -f run_vanilla_baseline`. That is the reason
# this file exists: a `for` loop typed at a prompt cannot be killed by name, so
# killing the python it is running just advances it to the next layer with
# whatever code was current when the loop was typed. That happened three times
# in one evening, twice leaving a stale non-batched run holding a card pair.
#
# Two questions, one job:
#
#   Did the adapter add reading, or only format? The vanilla output on the
#   first heldout rows named the finding correctly and then wrote 1,600
#   characters about which token might come next, so this is measured rather
#   than assumed -- and it must be the same layer as the adapter it is compared
#   with, which is why the loop covers all three.
#
#   How much does the AV's own layer-32 training favour L32? The released
#   checkpoint is nla-gemma3-12b-L32-av. If L32 wins among the adapters, that
#   gap is a property of the tool unless this baseline shows the same shape.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh

LAYERS="${LAYERS:-16 24 32}"
CORPUS="${CORPUS:-ddxplus}"
POOL="${POOL:-heldout}"
# Vanilla emits no closing tag, so it runs to the full token budget on every
# row and its KV cache is several times the adapter's. Eight rather than the
# generation default of sixteen.
BATCH="${BATCH:-8}"
TEMPLATE="${TEMPLATE:-prompt_templates/cue_position_readout.txt}"
# With SIDECAR_PROMPT=1 the checkpoint's own AV prompt is used instead of ours,
# which measures what asking for our schema costs a model never trained on it.
SIDECAR_PROMPT="${SIDECAR_PROMPT:-0}"

LOGS="$ART/logs"
mkdir -p "$LOGS" "$ART/results"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN="$LOGS/vanilla_${CORPUS}_${STAMP}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: not enough free GPU memory on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
  say "  nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  echo "Refused -- see $MAIN" >&2
  exit 1
fi

say "layers $LAYERS | pool $POOL | batch $BATCH | sidecar_prompt $SIDECAR_PROMPT"
say "cards CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

for L in $LAYERS; do
  MANIFEST="$ART/train/${CORPUS}_cuepos_L${L}/manifest_test_${POOL}_cue.jsonl"
  if [ ! -s "$MANIFEST" ]; then
    say "no manifest at $MANIFEST -- skipping layer $L"
    continue
  fi
  SUFFIX=""
  PROMPT_ARGS=(--actor-prompt-template-file "$TEMPLATE")
  if [ "$SIDECAR_PROMPT" = "1" ]; then
    SUFFIX="_sidecarprompt"
    PROMPT_ARGS=()
  fi
  OUT="$ART/results/readout_vanilla_L${L}_${POOL}${SUFFIX}.jsonl"
  if [ -s "$OUT" ]; then
    say "skip L${L} ($(wc -l < "$OUT") rows already at $(basename "$OUT"))"
    continue
  fi
  LOG="$LOGS/vanilla_L${L}_${POOL}${SUFFIX}.log"
  say "vanilla L${L} ${POOL}${SUFFIX} -> $LOG"
  python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "$MANIFEST" \
    --output "$OUT" \
    --batch-size "$BATCH" \
    "${PROMPT_ARGS[@]}" \
    >"$LOG" 2>&1 \
    && say "  done ($(wc -l < "$OUT") rows)" || say "  FAILED -- see $LOG"
done

say "all done"
