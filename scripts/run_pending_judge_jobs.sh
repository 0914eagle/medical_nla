#!/usr/bin/env bash
# The shuffled reader-trust control: wait for the v2 run, then judge.
#
#   nohup bash scripts/run_pending_judge_jobs.sh > /dev/null 2>&1 &
#
# Only this one needs nohup. The 238-pair semantic rescore is minutes and runs
# as a plain command (scripts/run_readout_semantic_judge.sh); it writes its own
# file and run_judge locks per output, so it does not care what else is going.
#
# The wait here is not for the lock, it is for the data: this run is seeded
# from the finished reader-trust judgements. The controlled case file holds
# seven arms, four of which carry ids already judged in the v2 run
# ({base_id}__trust_readout and friends are the same string in both files).
# Copying those verdicts in first means run_judge resumes past them and pays
# only for the three shuffled arms -- roughly 2,100 requests instead of 5,000.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

RES="$ART/results"
LOGS="$ART/logs"; mkdir -p "$LOGS"
MAIN="$LOGS/pending_judge_$(date +%Y%m%d_%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$MAIN"; }

RT="$RES/judge_reader_trust.jsonl"
RT_LOCK="$RT.lock"
CONTROLLED="$DATA/ddxplus_reader_trust_cases_controlled.jsonl"
SHUF_OUT="$RES/judge_reader_trust_controlled.jsonl"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-720}"

say "log: $MAIN"

# ------------------------------------------------------- wait for v2 to end
say "1/2 waiting for the reader-trust run to finish (max ${MAX_WAIT_MIN} min)"
waited=0
while [ -e "$RT_LOCK" ]; do
  if [ "$waited" -ge "$((MAX_WAIT_MIN * 60))" ]; then
    say "1/2 GIVING UP: $RT_LOCK still held after ${MAX_WAIT_MIN} min."
    say "    If no run_judge is alive, the lock is stale -- delete it and rerun."
    exit 1
  fi
  sleep 120
  waited=$((waited + 120))
done
say "1/2 lock clear after $((waited / 60)) min; $(wc -l <"$RT" 2>/dev/null || echo 0) rows judged"

# ------------------------------------------------------------- shuffled arm
if [ ! -s "$CONTROLLED" ]; then
  say "2/2 SKIP: no $CONTROLLED"
  exit 1
fi

# Seed, once. Re-copying over a partially judged shuffled run would rewind it.
if [ ! -s "$SHUF_OUT" ]; then
  cp "$RT" "$SHUF_OUT"
  say "2/2 seeded $SHUF_OUT with $(wc -l <"$SHUF_OUT") existing verdicts"
else
  say "2/2 $SHUF_OUT exists ($(wc -l <"$SHUF_OUT") rows) -- resuming, not reseeding"
fi

say "2/2 judging the shuffled arms"
CASES="$CONTROLLED" OUT="$SHUF_OUT" bash scripts/run_reader_trust_judge.sh \
  >>"$MAIN" 2>&1 || { say "2/2 FAILED -- see $MAIN"; exit 1; }

say "2/2 done -- scoring"
python scripts/dedupe_judgements.py --judgements "$SHUF_OUT" \
  --output "$RES/judge_reader_trust_controlled_deduped.jsonl" >>"$MAIN" 2>&1
python scripts/analyze_reader_trust.py \
  --judgements "$RES/judge_reader_trust_controlled_deduped.jsonl" \
  --cases "$CONTROLLED" 2>&1 | tee -a "$MAIN"

cat <<'EOF' | tee -a "$MAIN"

Read the shuffled arms against their real counterparts, not against
no_account. shuffled_readout scoring like readout would mean the readout's
CONTENT is not what moved the reader -- the same conclusion the derangement
control draws for the vector, asked of the prose.
EOF
say "ALL DONE"
