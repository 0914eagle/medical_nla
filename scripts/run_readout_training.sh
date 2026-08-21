#!/usr/bin/env bash
# Train the cue-position readout across layers and seeds.
#
#   nohup bash scripts/run_readout_training.sh ddxplus > /dev/null 2>&1 &
#   LAYERS="24" SEEDS="17" nohup bash scripts/run_readout_training.sh mcr > /dev/null 2>&1 &
#
# Two operational rules the audit left open are enforced here rather than
# remembered. Every layer gets the same epoch count -- the pilot ran 3 at L32
# and 2 at L16/L24 while claiming one recipe, so its layer trajectory mixed a
# layer effect with an epoch one. And every configuration gets several seeds,
# because one run is a point estimate and the trajectory is read as if the
# differences between layers were larger than the noise within one.
#
# A run whose best.json already exists is skipped, so an interrupted queue can
# be relaunched without repeating what finished.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh

CORPUS="${1:-ddxplus}"
case "$CORPUS" in
  ddxplus|mcr) ;;
  *) echo "usage: $0 [ddxplus|mcr]" >&2; exit 2 ;;
esac

LAYERS="${LAYERS:-16 24 32}"
SEEDS="${SEEDS:-17 18 19}"
EPOCHS="${EPOCHS:-3}"
# Effective batch is BATCH x GRAD_ACCUM, and 4x2 is the same optimizer step as
# 1x8 or 8x1 -- labels are padded with -100 and padding is masked out of
# attention, so the split changes memory and speed, not the objective. Four per
# forward rather than eight because activations for the backward pass are what
# ran a 24GB card out of memory even with the weights split across two.
BATCH="${BATCH:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
# Validation is now a random sample reused across epochs, so a larger number
# buys precision rather than a longer look at the same corner of the corpus.
MAX_EVAL_ROWS="${MAX_EVAL_ROWS:-512}"
# One training budget for both corpora. MedCaseReasoning has 32,724 training
# rows against DDXPlus's 10,195 -- 2.4x the cases at the same cues per case --
# so left uncapped a difference between the two could be the prose or could be
# the extra data, and the cross-corpus comparison is the point of having both.
# It also cuts the queue's wall clock by more than half.
MAX_TRAIN_ROWS="${MAX_TRAIN_ROWS:-10195}"

LOGS="$ART/logs"
ADAPTERS="$ART/train/adapters"
mkdir -p "$LOGS" "$ADAPTERS"
STAMP=$(date +%Y%m%d_%H%M%S)
MAIN="$LOGS/readout_${CORPUS}_${STAMP}.log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

# Weights alone are 11.8 GB a card once the model is split across two, before
# any activation memory, so a card carrying somebody else's job has nothing
# like enough. Checked once here rather than discovered eighteen times: without
# it every run dies in the same nine seconds, the loop dutifully continues, and
# the summary table is eighteen rows of "(did not finish)".
#
# After the log exists, not before. The queue is meant to be launched with
# nohup into /dev/null -- the header says so -- and a refusal printed to stdout
# then leaves nothing behind but "Exit 1" and no file to look in.
if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: not enough free GPU memory on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"
  say "  the per-card report is above; another job is most likely holding a card:"
  say "    nvidia-smi --query-compute-apps=pid,used_memory --format=csv"
  say "  to use the other pair:  export CUDA_VISIBLE_DEVICES=2,3 && source scripts/env.sh"
  echo "Refused -- see $MAIN" >&2
  exit 1
fi

say "corpus $CORPUS | layers $LAYERS | seeds $SEEDS | epochs $EPOCHS | batch $BATCH x $GRAD_ACCUM"
say "train rows capped at $MAX_TRAIN_ROWS (same budget for both corpora)"
say "cards CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}"

for L in $LAYERS; do
  SPLIT_DIR="$ART/train/${CORPUS}_cuepos_L${L}"
  if [ ! -s "$SPLIT_DIR/sft_train.jsonl" ]; then
    say "no splits at $SPLIT_DIR -- skipping layer $L"
    continue
  fi
  for SEED in $SEEDS; do
    OUT="$ADAPTERS/${CORPUS}_L${L}_s${SEED}"
    if [ -s "$OUT/best.json" ]; then
      # Skipping finished work is what lets an interrupted queue be relaunched,
      # but only when the finished work was produced the same way. An adapter
      # whose best.json predates the content/scaffold split had its best epoch
      # chosen on a loss that is mostly constant XML, and reusing it would put
      # two selection rules inside one layer's three seeds.
      if grep -q '"selected_on"' "$OUT/best.json"; then
        say "skip ${CORPUS} L${L} seed ${SEED} (already trained)"
        continue
      fi
      say "retrain ${CORPUS} L${L} seed ${SEED} (best.json predates --select-on)"
      rm -rf "$OUT"
    fi
    LOG="$LOGS/train_${CORPUS}_L${L}_s${SEED}.log"
    say "train ${CORPUS} L${L} seed ${SEED} -> $LOG"
    python scripts/train_medical_nla_lora.py \
      --config configs/default.yaml \
      --train-jsonl "$SPLIT_DIR/sft_train.jsonl" \
      --val-jsonl "$SPLIT_DIR/sft_val.jsonl" \
      --out-dir "$OUT" \
      --epochs "$EPOCHS" --seed "$SEED" \
      --batch-size "$BATCH" --grad-accum-steps "$GRAD_ACCUM" \
      --max-eval-rows "$MAX_EVAL_ROWS" \
      --max-train-rows "$MAX_TRAIN_ROWS" \
      >"$LOG" 2>&1 \
      && say "  done" || say "  FAILED -- see $LOG"
  done
done

{
  echo
  echo "============== $CORPUS readout training =============="
  printf "%-8s %-6s %-12s %s\n" layer seed best_epoch best_val_loss
  for L in $LAYERS; do
    for SEED in $SEEDS; do
      f="$ADAPTERS/${CORPUS}_L${L}_s${SEED}/best.json"
      if [ -s "$f" ]; then
        python -c "
import json
d = json.load(open('$f'))
print(f\"{'L$L':<8} {'$SEED':<6} {d['best_epoch']:<12} {d['best_val_loss']:.4f}\")
"
      else
        printf "%-8s %-6s %-12s %s\n" "L$L" "$SEED" "-" "(did not finish)"
      fi
    done
  done
  echo
  echo "A layer's numbers are only comparable to another layer's if both rows"
  echo "show the same best_epoch range and the seeds agree with each other."
  echo "logs: $LOGS"
  echo "====================================================="
} | tee -a "$MAIN"

say "all done"
