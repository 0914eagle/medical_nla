#!/usr/bin/env bash
# Read MCR answer-position states with the conclusion adapter.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_conclusion_readout.sh > /dev/null 2>&1 &
#
# The adapter finished on 08-24 (best_epoch 1 of 3, selected on content loss
# 1.767 against a scaffold loss of 0.038 -- the XML is learned immediately and
# the diagnosis is not, which is why selecting on content was the fix that let
# it save at all).
#
# WHAT THIS OPENS DEPENDS ON WHICH PROMPTS WERE EXTRACTED, and the difference
# decides two different table cells:
#
#   * Plain MCR cases (no referring note) -> the instrument question. "Does the
#     readout describe real clinical prose, not just DDXPlus's templated
#     sentences?" That fills Table 1's missing MCR row and it is what the
#     mcr_answerpos extraction was built for.
#   * Wrong-note arm prompts -> the attribution question. Table 3b's ᵈ cell and
#     the MCR r5 rung both need the readout of a state that was read under the
#     note, because the claim is about what the note did.
#
# The two are not interchangeable and a manifest cannot be silently repurposed,
# so this script reports which it found and says what that unlocks rather than
# guessing. If the attribution cell is what you need and the manifest holds
# note-free prompts, a new extraction over the MCR hint cases comes first.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LAYER="${LAYER:-32}"
RUN_NAME="${RUN_NAME:-mcr_answerpos_L${LAYER}}"
ADAPTER="${ADAPTER:-$ART/train/adapters/mcr_conclusion_L${LAYER}_s17}"
TEMPLATE="${TEMPLATE:-prompt_templates/medical_nla_v2_readout.txt}"
BATCH="${BATCH:-8}"
OUT="${OUT:-$ART/results/readout_mcr_conclusion_L${LAYER}.jsonl}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results"
MAIN="$LOGS/mcr_conclusion_readout_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if [ ! -s "$ADAPTER/best.json" ]; then
  say "no trained adapter at $ADAPTER (best.json missing)"; exit 1
fi
say "adapter: $(python -c "
import json,sys; d=json.load(open('$ADAPTER/best.json'))
print(f\"epoch {d['best_epoch']}/{d['epochs_run']}, content {d['best_val_content_loss']:.3f}, on {d['selected_on']}\")")"

mapfile -t MANIFESTS < <(find "$ART/activations/$RUN_NAME" -name manifest.jsonl -size +0 2>/dev/null | sort)
if [ "${#MANIFESTS[@]}" -eq 0 ]; then
  say "no manifest under $ART/activations/$RUN_NAME"; exit 1
fi
MANIFEST="${MANIFESTS[0]}"
say "manifest: $MANIFEST ($(wc -l < "$MANIFEST") rows)"

# Which question this manifest can answer. Reported, never assumed.
python - "$MANIFEST" <<'PY' | tee -a "$MAIN"
import json, sys
from collections import Counter
arms, noted, n = Counter(), 0, 0
for line in open(sys.argv[1]):
    row = json.loads(line)
    n += 1
    arms[str(row.get("hint_variant") or "-")] += 1
    if "referring note" in str(row.get("prompt") or "").lower():
        noted += 1
print(f"[manifest] {n:,} rows   arms={dict(arms)}   prompts naming a referring note: {noted:,}")
if noted == 0:
    print("[manifest] NOTE-FREE prompts -> instrument question only.")
    print("           Unlocks: Table 1's MCR description-rate row (judge job #8).")
    print("           Does NOT unlock Table 3b's MCR readout cell or the MCR r5")
    print("           rung -- those need answer-position activations extracted")
    print("           over the MCR hint cases' wrong arm, which is a new run.")
else:
    print("[manifest] note-bearing prompts -> attribution is available too.")
PY

if [ -s "$OUT" ]; then say "skip (exists: $(wc -l < "$OUT") rows)"; else
  if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
    say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
  fi
  say "reading out -> $OUT"
  python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "$MANIFEST" \
    --output "$OUT" \
    --adapter-id "$ADAPTER" \
    --actor-prompt-template-file "$TEMPLATE" \
    --batch-size "$BATCH" \
    >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi

say "ALL DONE -> $OUT"
say ""
say "Look at ten of them before trusting any rate -- the adapter's content loss"
say "is 1.77 on an open vocabulary, so whether it reads MCR states at all is an"
say "open question that a number cannot answer on its own:"
say "  head -3 $OUT | python -m json.tool"
