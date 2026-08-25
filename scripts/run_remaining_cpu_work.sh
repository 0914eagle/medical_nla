#!/usr/bin/env bash
# The remaining work that needs no GPU and no judge, in dependency order.
#
# Five of the eight open items are re-scoring and re-counting on files that
# already exist. They were left open because nobody had run them, not because
# anything was missing, and they gate the GPU item: the MCR derangement control
# decides whether the MCR readout is reading the case at all, and committing
# days of extraction before that answer arrives risks spending them on a
# channel that turns out not to be case-specific.
#
# Each step skips itself when its inputs are absent and says which ones, so a
# partial box still gets everything it can do. Nothing here overwrites a source
# file: rescoring writes _rescored beside the original.
#
#   source scripts/env.sh && bash scripts/run_remaining_cpu_work.sh
#   STEPS="1 2" bash scripts/run_remaining_cpu_work.sh     # just those
set -uo pipefail

: "${ART:?run 'source scripts/env.sh' first}"
: "${DATA:?run 'source scripts/env.sh' first}"
RES="$ART/results"
REPORTS="$ART/reports"
mkdir -p "$RES" "$REPORTS" "$ART/logs"

STEPS="${STEPS:-1 2 3 4 5}"
LOG="$ART/logs/remaining_cpu_$(date +%Y%m%d_%H%M%S).log"

say()  { echo "" | tee -a "$LOG"; echo "=== $* ===" | tee -a "$LOG"; }
note() { echo "    $*" | tee -a "$LOG"; }
run()  { echo "\$ $*" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; return "${PIPESTATUS[0]}"; }
wants() { case " $STEPS " in *" $1 "*) return 0;; *) return 1;; esac; }

# have FILE... -- true when every one is a non-empty file, else names the gaps.
have() {
  local ok=0 f
  for f in "$@"; do
    if [ ! -s "$f" ]; then note "missing: $f"; ok=1; fi
  done
  return $ok
}

# rescore IN OUT -- canonical matcher, only when the output is absent or older
# than the matcher itself. Rescoring is cheap; rescoring twice hides whether
# the file on disk is the one the numbers came from.
rescore() {
  local in="$1" out="$2"
  if [ ! -s "$in" ]; then note "missing: $in"; return 1; fi
  if [ -s "$out" ] && [ "$out" -nt src/answer_matching.py ]; then
    note "reusing $out (newer than the matcher)"
    return 0
  fi
  run python scripts/rescore_source_correct.py --answers "$in" --output "$out"
}

echo "log: $LOG"

# --------------------------------------------------------------------------
if wants 1; then
say "1. corpus-300 -- provenance, then the non-overlap subset"
C300_RAW="$RES/ddxplus_hint_answers_300.jsonl"
C300_RES="$RES/ddxplus_hint_answers_300_rescored.jsonl"
MAIN_RES="$RES/ddxplus_hint_answers_v2_rescored.jsonl"
if rescore "$C300_RAW" "$C300_RES"; then
  note "full corpus-300 (this is the .9800/.9306/.7670/.9180 row)"
  run python scripts/analyze_hint_effect.py --answers "$C300_RES" \
      --dump "$REPORTS/hint_effect_c300.json"
  note "full corpus-300, canonical no-note-eligible primary sensitivity"
  run python scripts/analyze_hint_effect.py --answers "$C300_RES" \
      --require-canonical-no-note-correct \
      --dump "$REPORTS/hint_effect_c300_canonical_eligible.json"
  if have "$MAIN_RES"; then
    note "non-overlap only -- corpus-300 holds 1,676 of the main run's 1,747"
    note "cases, so only the remainder is a second look at anything."
    run python scripts/analyze_hint_effect.py --answers "$C300_RES" \
        --exclude-from "$MAIN_RES" \
        --dump "$REPORTS/hint_effect_c300_nonoverlap.json"
    note "non-overlap only, canonical no-note-eligible primary replication"
    run python scripts/analyze_hint_effect.py --answers "$C300_RES" \
        --exclude-from "$MAIN_RES" \
        --require-canonical-no-note-correct \
        --dump "$REPORTS/hint_effect_c300_nonoverlap_canonical_eligible.json"
  fi
fi
# The ladder half of the same question. The archived non-overlap run found the
# both-wrong cell failing to replicate (11:4, p=0.118) and demoted it to
# exploratory -- under the old matcher. That cell is defined by is_correct, so
# the fix can move rows across it, and the demotion has to be re-earned.
LADDERS=""
for R in 3 4 5 6; do
  F="$RES/ddxplus_ladder_300_r${R}.jsonl"
  [ -s "$F" ] && LADDERS="$LADDERS $F"
done
if [ -n "$LADDERS" ] && have "$MAIN_RES"; then
  note "corpus-300 ladder, non-overlap"
  run python scripts/analyze_correction_ladder.py --rungs $LADDERS \
      --exclude-from "$MAIN_RES"
fi
fi

# --------------------------------------------------------------------------
if wants 2; then
say "2. MCR readout derangement control"
note "Does the MCR conclusion readout read THIS case, or the corpus average?"
note "It scored .2643 against a .0122 floor on the answer field and only"
note "+.025 on the cues. Until a deranged pairing scores lower, neither"
note "number is evidence of anything case-specific -- and the MCR internal"
note "branch, including the GPU extraction, rests on it."
MCR_READ="$RES/readout_mcr_conclusion_L32.jsonl"
if have "$MCR_READ"; then
  note "(a) cue block -- does the supporting_cues text describe THIS case?"
  run python scripts/analyze_readout_grounding.py --readouts "$MCR_READ"

  # The cue control does not touch the answer field, and the answer field is
  # what the .2643 was. Its answers must come from the CONCLUSION task the
  # readout was trained on -- mcr_source_answers_* -- not from the hint
  # intervention, whose base_ids belong to a different experiment and join
  # almost nothing.
  note "(b) answer field -- the gate. Scored against the model, with a"
  note "    deranged pairing as the control."
  MCR_SRC=""
  for F in "$RES/mcr_source_answers_test_rescored.jsonl" \
           "$RES/mcr_source_answers_train_rescored.jsonl" \
           "$RES/mcr_source_answers_test.jsonl" \
           "$RES/mcr_source_answers_train.jsonl"; do
    [ -s "$F" ] && MCR_SRC="$MCR_SRC $F"
  done
  if [ -z "$MCR_SRC" ]; then
    note "missing: \$ART/results/mcr_source_answers_{train,test}.jsonl"
    note "NOT mcr_hint_answers_full -- that is the intervention run, and its"
    note "base_ids do not address the conclusion readout's rows."
  else
    for F in "$RES/mcr_source_answers_train.jsonl" "$RES/mcr_source_answers_test.jsonl"; do
      [ -s "$F" ] && rescore "$F" "${F%.jsonl}_rescored.jsonl"
    done
    MCR_SRC=""
    for F in "$RES/mcr_source_answers_train_rescored.jsonl" \
             "$RES/mcr_source_answers_test_rescored.jsonl"; do
      [ -s "$F" ] && MCR_SRC="$MCR_SRC $F"
    done
    run python scripts/score_readout_against_model.py \
        --readouts "$MCR_READ" --answers $MCR_SRC
  fi
fi
fi

# --------------------------------------------------------------------------
if wants 3; then
say "3. wording variants and CoT -- canonical rescore"
note "These rows still carry the generation-time matcher, which matched one"
note "diagnosis inside another (pe inside pericarditis, stable inside"
note "unstable angina). Every rate built on them is unverified."
for W in v2 colleague patient realistic; do
  IN="$RES/ddxplus_hint_answers_${W}.jsonl"
  OUT="$RES/ddxplus_hint_answers_${W}_rescored.jsonl"
  [ -s "$IN" ] || { note "skip wording $W (no $IN)"; continue; }
  if rescore "$IN" "$OUT"; then
    note "wording: $W"
    run python scripts/analyze_hint_effect.py --answers "$OUT" \
        --dump "$REPORTS/hint_effect_${W}.json"
  fi
done
COT_IN="$RES/ddxplus_hint_answers_cot_full.jsonl"
COT_OUT="$RES/ddxplus_hint_answers_cot_full_rescored.jsonl"
if rescore "$COT_IN" "$COT_OUT"; then
  note "CoT duality -- direct against chain, same cases"
  run python scripts/analyze_hint_effect.py --answers "$COT_OUT" \
      --dump "$REPORTS/hint_effect_cot.json"
  if have "$RES/ddxplus_hint_answers_v2_rescored.jsonl"; then
    run python scripts/compare_direct_vs_cot.py \
        --direct "$RES/ddxplus_hint_answers_v2_rescored.jsonl" \
        --cot "$COT_OUT"
  fi
fi
fi

# --------------------------------------------------------------------------
if wants 4; then
say "4. Figure 5 -- recount the 64.1% under the canonical matcher"
note "The analyzer already calls the canonical matcher; what was stale is the"
note "ladder file it groups by. A rung without took_the_hint falls back to"
note "answer==suggestion, which counts cases whose no-note arm already"
note "answered that suspicion -- cases the note had nothing to move."
LADDER=""
for R in 5 4 3; do
  [ -s "$RES/ddxplus_ladder_r${R}_rescored.jsonl" ] && \
    LADDER="$RES/ddxplus_ladder_r${R}_rescored.jsonl" && break
done
READOUTS=$(ls "$RES"/readout_traj_*.jsonl 2>/dev/null || true)
[ -n "$READOUTS" ] || READOUTS=$(ls "$RES"/readout_hint_final_L32_v2.jsonl 2>/dev/null || true)
if [ -z "$LADDER" ]; then
  note "missing: any \$ART/results/ddxplus_ladder_r{3,4,5}_rescored.jsonl"
elif [ -z "$READOUTS" ]; then
  note "missing: \$ART/results/readout_traj_*.jsonl"
else
  note "ladder: $LADDER"
  run python scripts/analyze_trajectory_readouts.py --readouts $READOUTS \
      --ladder "$LADDER"
fi
fi

# --------------------------------------------------------------------------
if wants 5; then
say "5. reader-trust -- dedupe, score, and build the shuffled control"
RT="$RES/judge_reader_trust.jsonl"
# Whichever case file the judgements actually came from. Pointing the analyzer
# at the wrong generation is silent: the rows that fail to join land in a '?'
# channel and the no_account arm disappears, which reads as "this run has no
# baseline" when in fact it has one under a different filename.
RT_CASES=""
RT_BEST=0
for F in "$DATA/ddxplus_reader_trust_cases_controlled.jsonl" \
         "$DATA/ddxplus_reader_trust_cases_v2.jsonl" \
         "$DATA/ddxplus_reader_trust_cases.jsonl"; do
  if [ -s "$F" ] && [ -s "$RT" ]; then
    hits=$(python - "$RT" "$F" <<'PY'
import json, sys
judged = set()
for line in open(sys.argv[1]):
    line = line.strip()
    if line:
        try: judged.add(str(json.loads(line).get("id")))
        except Exception: pass
n = 0
for line in open(sys.argv[2]):
    line = line.strip()
    if line:
        try:
            if str(json.loads(line).get("id")) in judged: n += 1
        except Exception: pass
print(n)
PY
)
    note "candidate $(basename "$F"): $hits judged ids join"
    if [ "$hits" -gt "$RT_BEST" ]; then
      RT_CASES="$F"; RT_BEST="$hits"
    fi
  fi
done
[ -n "$RT_CASES" ] && note "using $(basename "$RT_CASES") ($RT_BEST joins)"
if [ -n "$RT_CASES" ] && have "$RT" "$RT_CASES"; then
  run python scripts/dedupe_judgements.py --judgements "$RT" \
      --output "$RES/judge_reader_trust_deduped.jsonl"
  run python scripts/analyze_reader_trust.py \
      --judgements "$RES/judge_reader_trust_deduped.jsonl" \
      --cases "$RT_CASES"
fi
SHUF="$DATA/ddxplus_reader_trust_cases_shuffled.jsonl"
if [ -s "$SHUF" ]; then
  note "shuffled control cases already built: $SHUF"
else
  note "building the shuffled control -- an account from another case. If it"
  note "helps readers as much as the real one, the readout's content is not"
  note "what is helping."
  if have "$DATA/ddxplus_hint_cases_v2.jsonl"; then
    run python scripts/make_reader_trust_cases.py \
        --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
        --answers "$RES/ddxplus_hint_answers_v2.jsonl" \
        --cot-answers "$RES/ddxplus_hint_answers_cot_full.jsonl" \
        --readouts "$RES/readout_hint_final_L32_v2.jsonl" \
        --probe-verdicts "$RES/probe_verdicts.jsonl" \
        --controls shuffled \
        --output "$SHUF"
  fi
fi
note "judging the shuffled arm needs the judge: run_reader_trust_judge.sh"
fi

# --------------------------------------------------------------------------
say "done"
cat <<EOF | tee -a "$LOG"
Still outstanding after this, and why:

  judge     438-row semantic rescore   run_readout_semantic_judge.sh  (~\$0.09)
  judge     reader-trust shuffled arm  run_reader_trust_judge.sh
  judge     no-CoT arm                 make_cot_monitor_requests.py --no-cot
  GPU       MCR wrong-note extraction  gated on step 2's derangement result
  reading   Related Work re-check      no script can do this one

Every number this produced goes into
docs/experiments/RESULTS_CANONICAL_2026-08-24.md with its script and input
file BEFORE it goes into a table. Full log: $LOG
EOF
