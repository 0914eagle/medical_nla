#!/usr/bin/env bash
# The correction ladder on real case reports.
#   CUDA_VISIBLE_DEVICES=0,1 nohup bash scripts/run_mcr_ladder.sh > /dev/null 2>&1 &
#
# §4.4 is DDXPlus-only, and the asymmetry between the two corpora is not a gap
# to apologise for -- it is the section's closing argument. Which rungs exist
# on MedCaseReasoning is decided by the corpus, not by what we got around to:
#
#   r3  reconsider only .............. runnable now
#   r4  findings re-shown (control) ... runnable now
#   r7  the model's own chain ......... needs an MCR CoT run (this script does it)
#   r5  readout conclusion & grounds .. needs the conclusion adapter (training)
#   r6  probe class label ............. CANNOT EXIST -- 6,934 diagnoses, most
#                                       appearing once, so there is no class
#                                       set to feed back
#
# r6's impossibility is the point. On DDXPlus the probe wins the correction
# comparison, and a reader is entitled to conclude that the natural-language
# channel is redundant. The corpus where that channel is the only one that
# exists is the answer, and it has to be measured rather than asserted.
#
# RUNGS= overrides the default. Set READOUTS= once the conclusion adapter has
# produced MCR readouts, and add 5.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

RUNGS="${RUNGS:-3 4 7}"
CASES="${CASES:-$DATA/mcr_hint_cases_full.jsonl}"
ANSWERS="${ANSWERS:-$ART/results/mcr_hint_answers_full.jsonl}"
READOUTS="${READOUTS:-}"
PREFIX="$DATA/mcr_ladder"

LOGS="$ART/logs"; mkdir -p "$LOGS" "$ART/results" "$ART/reports"
MAIN="$LOGS/mcr_ladder_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

for f in "$CASES" "$ANSWERS"; do
  [ -s "$f" ] || { say "missing $f -- run scripts/run_mcr_intervention.sh first"; exit 1; }
done
say "cases $(wc -l < "$CASES") | first-pass answers $(wc -l < "$ANSWERS") | rungs $RUNGS"

case " $RUNGS " in *" 6 "*)
  say "REFUSED: rung 6 cannot exist on MedCaseReasoning -- no class set to feed"
  say "  back. That impossibility is a result; do not fabricate a label space."
  exit 2 ;;
esac
case " $RUNGS " in *" 5 "*)
  [ -n "$READOUTS" ] || { say "rung 5 needs READOUTS= (MCR conclusion adapter output)"; exit 1; } ;;
esac

# r7 feeds back the model's own chain, so the chains have to exist. The wrong
# arm only: a chain written under a different note is not this case's reasoning.
COT="$ART/results/mcr_hint_answers_cot.jsonl"
case " $RUNGS " in *" 7 "*)
  if [ -s "$COT" ]; then say "skip CoT run (exists: $(wc -l < "$COT") rows)"; else
    if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
      say "REFUSED: cards busy (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>})"; exit 1
    fi
    say "CoT pass over the wrong arm (case reports are long; allow ~1-2h)"
    python scripts/run_source_answers.py --config configs/default.yaml \
      --cases "$CASES" \
      --output-jsonl "$COT" \
      --summary-json "$ART/reports/mcr_hint_answers_cot.json" \
      --condition cot --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED (CoT)"; exit 1; }
  fi ;;
esac

say "building rungs"
BUILD=(python scripts/make_correction_ladder_cases.py
  --cases "$CASES" --answers "$ANSWERS"
  --rungs $RUNGS --output-prefix "$PREFIX")
[ -n "$READOUTS" ] && BUILD+=(--readouts $READOUTS)
case " $RUNGS " in *" 7 "*) BUILD+=(--cot-answers "$COT") ;; esac
"${BUILD[@]}" >>"$MAIN" 2>&1 || { say "FAILED (build)"; exit 1; }
grep -E "^(rung|chains|skipped)" "$MAIN" | tail -12 | while read -r l; do say "  $l"; done

if ! python scripts/check_gpu_setup.py --config configs/default.yaml --require-free-gb 20 >>"$MAIN" 2>&1; then
  say "REFUSED: cards busy"; exit 1
fi

for R in $RUNGS; do
  IN="${PREFIX}_r${R}.jsonl"; OUT="$ART/results/mcr_ladder_r${R}.jsonl"
  [ -s "$IN" ] || { say "no rows for rung $R"; continue; }
  if [ -s "$OUT" ]; then say "skip rung $R (exists)"; continue; fi
  say "answering rung $R ($(wc -l < "$IN") rows)"
  python scripts/run_source_answers.py --config configs/default.yaml \
    --cases "$IN" --output-jsonl "$OUT" \
    --summary-json "$ART/reports/mcr_ladder_r${R}.json" \
    --condition direct --batch-size 8 >>"$MAIN" 2>&1 || { say "FAILED rung $R"; exit 1; }
done

say "ALL DONE (MCR ladder, rungs $RUNGS)"
say "analyse with:"
say "  python scripts/analyze_correction_ladder.py \\"
say "    --ladder $ART/results/mcr_ladder_r*.jsonl"
say ""
say "Read r7 against r3/r4 restricted to r7's ids -- the CoT pass drops cases"
say "whose chain reached a different answer, so the populations differ."
