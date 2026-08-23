#!/usr/bin/env bash
# Vanilla (no-adapter) readouts at the ANSWER position, on the same vectors
# the tuned readout narrated.
#   CUDA_VISIBLE_DEVICES=2,3 nohup bash scripts/run_vanilla_final_position.sh > /dev/null 2>&1 &
#
# The existing vanilla control (Addendum 2, 08-17) covers cue positions only,
# and it settled what the adapter does there: the checkpoint already carries
# the cue content, wrapped in confabulated frames, and the LoRA distills a
# noisy meta-narrator into a precise reader. That answers "did the LoRA create
# the reading" for cues.
#
# It does not touch the claim the paper now rests on, which lives at the final
# token: the state's conclusion is the gold while the emitted answer is not.
# If the untuned checkpoint, handed the same vector and asked for the same two
# fields, also names the gold on lost cases, the rift is a property of the
# activation and not of our adapter. If it names nothing usable, we learn the
# adapter is required for legibility -- and the mechanism claim keeps resting
# where it already rests, on the instrument-neutral probe trajectory.
#
# Same template as the tuned run, deliberately: asking the checkpoint for our
# schema is part of what is being measured. Set SIDECAR_PROMPT=1 to use the
# checkpoint's own AV prompt instead and score by containment only.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

LAYER="${LAYER:-32}"
BATCH="${BATCH:-8}"
TEMPLATE="${TEMPLATE:-prompt_templates/medical_nla_v2_readout.txt}"
# The readout manifest is whatever make_trajectory_readout_manifest.py wrote
# for the tuned run -- the same rows, so the two channels are compared on
# identical vectors. Override MANIFEST= if it lives elsewhere.
MANIFEST="${MANIFEST:-$DATA/ddxplus_trajectory_readout_manifest.jsonl}"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results"
MAIN="$LOGS/vanilla_final_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

if [ ! -s "$MANIFEST" ]; then
  say "no manifest at $MANIFEST"
  say "  point MANIFEST= at the manifest the TUNED trajectory readout run used."
  say "  find it with:  ls -t \$DATA/*trajectory*manifest*.jsonl \$ART/**/manifest*.jsonl 2>/dev/null | head"
  exit 1
fi

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
fi

OUT="$ART/results/readout_vanilla_trajectory_L${LAYER}.jsonl"
if [ -s "$OUT" ]; then say "skip (exists: $(wc -l < "$OUT") rows)"; else
  say "vanilla L${LAYER} on $(wc -l < "$MANIFEST") trajectory rows -> $OUT"
  python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "$MANIFEST" \
    --output "$OUT" \
    --batch-size "$BATCH" \
    --actor-prompt-template-file "$TEMPLATE" \
    >>"$MAIN" 2>&1 || { say "FAILED"; exit 1; }
fi
say "ALL DONE (vanilla at final position)"
say ""
say "Score BOTH channels through the judge, not the regex -- the layer sweep"
say "already showed rule-based scoring of vanilla undercounts (0.04-0.14) and"
say "overcounts (0.56-0.66) the same outputs:"
say "  python scripts/make_readout_extraction_cases.py --readouts $OUT \\"
say "     --channel vanilla --output \$DATA/extract_vanilla.jsonl"
say "  python scripts/make_readout_extraction_cases.py --readouts <tuned> \\"
say "     --channel tuned --output \$DATA/extract_tuned.jsonl"
say "  (run both through run_source_answers --no-prefill --max-new-tokens 32)"
say "  python scripts/analyze_readout_extraction.py --extractions <both> --ladder ..."
