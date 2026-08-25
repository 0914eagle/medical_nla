#!/usr/bin/env bash
# The MCR internal branch, now that the derangement control opened its gate.
#
#   nohup bash scripts/run_mcr_internal_chain.sh > /dev/null 2>&1 &
#
# The answer field reads this case: .2643 against a .0049 deranged control on
# all 821 rows, and .2127 against .0042 on the 710 where the model never
# reached the gold. That is what the extraction below was waiting on.
#
# Two stages, cheap first:
#
#   r3/r4  the rungs that need no new activations. If the chain dies later,
#          Table 4d still gains half its rows.
#   r5     needs wrong-note activations at MCR's final prompt token, which is
#          the multi-day part: position rows -> extract -> read.
#
# r6 is not here. It feeds back a probe's class name, and MCR's diagnoses are
# mostly singletons, so the fixed-class probe that rung depends on does not
# transfer. Saying that is better than shipping a rung built on a probe the
# corpus cannot define.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

RES="$ART/results"
LOGS="$ART/logs"; mkdir -p "$LOGS" "$RES"
MAIN="$LOGS/mcr_internal_$(date +%Y%m%d_%H%M%S).log"
LAYER="${LAYER:-32}"
ADAPTER="${ADAPTER:-$ART/train/adapters/mcr_conclusion_L${LAYER}_s17}"
TEMPLATE="${TEMPLATE:-prompt_templates/medical_nla_v2_readout.txt}"
CASES="$DATA/mcr_hint_cases_full.jsonl"
ROWS="$DATA/mcr_hint_position_rows.jsonl"
RUN_NAME="mcr_hint_positions_L${LAYER}"
MANIFEST="$ART/activations/$RUN_NAME/layer${LAYER}/last_token/manifest.jsonl"
READ_OUT="$RES/readout_mcr_hint_final_L${LAYER}.jsonl"

STEPS="${STEPS:-1 2}"
wants() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }
step() {
  local what="$1"; shift
  say "-> $what"
  if "$@" >>"$MAIN" 2>&1; then say "   ok"; else say "   FAILED: $what"; return 1; fi
}

# One copy at a time. Two of these write the same ladder outputs and put two
# 12B models on the same two cards; the second one does not fail loudly, it
# interleaves. Started as exactly that mistake.
LOCK="$ART/logs/mcr_internal_chain.lock"
if ! ( set -o noclobber; echo "$$" > "$LOCK" ) 2>/dev/null; then
  echo "REFUSED: another chain holds $LOCK (pid $(cat "$LOCK" 2>/dev/null))."
  echo "  If no process is alive, remove the lock and rerun."
  exit 1
fi
trap 'rm -f "$LOCK"' EXIT

say "log: $MAIN"
say "steps: $STEPS   rungs: ${RUNGS:-3 4}"

if ! python scripts/check_gpu_setup.py --config configs/default.yaml \
     --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: no free card (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"
  say "Cards 2 and 3 belong to another user's run and are never ours to take."
  exit 1
fi

# ------------------------------------------------------------- stage 1: ladder
if wants 1; then
say "STAGE 1 -- MCR ladder, rungs ${RUNGS:-3 4}"
# Default to 3 and 4 here, NOT run_mcr_ladder's own "3 4 7". Rung 7 feeds the
# model its own chain, which first needs a CoT pass over every wrong-arm case;
# MCR case reports are long enough that this ran for six hours and blocked the
# extraction behind it. DDXPlus already answered what r7 asks -- self-chain
# feedback recovers .1236 of moved cases -- so it does not go in front of the
# work the derangement gate just cleared. Ask for it explicitly with RUNGS.
if RUNGS="${RUNGS:-3 4}" bash scripts/run_mcr_ladder.sh >>"$MAIN" 2>&1; then
  say "STAGE 1 ok"
else
  say "STAGE 1 FAILED -- see $MAIN. Continuing to the extraction, which does"
  say "  not depend on it."
fi
else
  say "STAGE 1 skipped (STEPS=$STEPS)"
fi

# --------------------------------------------------- stage 2: wrong-note r5
if ! wants 2; then
  say "STAGE 2 skipped (STEPS=$STEPS)"
  say "ALL DONE"
  exit 0
fi
say "STAGE 2 -- wrong-note activations for r5"
[ -s "$CASES" ] || { say "missing $CASES"; exit 1; }

if [ -s "$ROWS" ]; then
  say "-> position rows exist ($(wc -l <"$ROWS") rows)"
else
  step "position rows (final + hint)" \
    python scripts/make_hint_position_rows.py --cases "$CASES" --output "$ROWS" \
    || exit 1
fi

if [ -s "$MANIFEST" ]; then
  say "-> activations exist ($(wc -l <"$MANIFEST") rows in manifest)"
else
  step "extract L${LAYER} activations (hours)" \
    python -m src.extract_activations --config configs/default.yaml \
      --input "$ROWS" --run-name "$RUN_NAME" \
      --layers "$LAYER" --strategies last_subtoken --batch-size 8 \
    || exit 1
fi

[ -s "$ADAPTER/best.json" ] || { say "no trained adapter at $ADAPTER"; exit 1; }

if [ -s "$READ_OUT" ]; then
  say "-> readout exists ($(wc -l <"$READ_OUT") rows); delete it to redo"
else
  # 768, not the config default of 256. MCR conclusion targets average 764
  # characters and the default truncated 54% of the first run mid-sentence,
  # which then read as a readout that trails off rather than a budget that
  # ran out.
  step "conclusion readout at the wrong-note final token" \
    python -m src.run_nla --config configs/default.yaml \
      --manifest "$MANIFEST" --output "$READ_OUT" \
      --adapter-id "$ADAPTER" \
      --actor-prompt-template-file "$TEMPLATE" \
      --max-new-tokens 768 --batch-size 16 \
    || exit 1
fi

say "STAGE 2 ok"
cat <<EOF | tee -a "$MAIN"

Next, and in this order -- the control comes before the claim:

  python scripts/score_readout_against_model.py \\
    --readouts $READ_OUT \\
    --answers \$ART/results/mcr_source_answers_{train,test}_rescored.jsonl

  python scripts/analyze_readout_grounding.py --readouts $READ_OUT

Only then the ladder rung and the Table 3b MCR cell. A wrong-note readout whose
deranged control is not far below its raw rate is measuring MCR's register, not
this case's state, and the conclusion-task gate does not transfer to it -- the
positions are different tokens.
EOF
say "ALL DONE"
