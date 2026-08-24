#!/usr/bin/env bash
# Does every input the remaining work needs actually exist?
#
# The eight remaining items were scheduled from documents, not from disk. A
# document says corpus-300 was rescored; only the file can say so. Run this
# first and the schedule stops being a guess: each step prints READY or the
# exact paths it is missing, and the corpus-300 provenance question -- the one
# open contradiction between the summary and the canonical ledger -- is
# answered by comparing mtimes against the matcher that would have produced it.
#
# Read-only. Nothing here writes, trains, or scores.
#
#   source scripts/env.sh && bash scripts/preflight_remaining_work.sh
set -uo pipefail

: "${ART:?run 'source scripts/env.sh' first}"
: "${DATA:?run 'source scripts/env.sh' first}"
RES="$ART/results"
ACT="$ART/activations"
MATCHER="src/answer_matching.py"

missing_total=0
step_missing=0

hdr() {
  printf '\n\033[1m%s\033[0m\n' "$*"
}

# check LABEL PATH -- present/absent, with size and age so a stale file is
# visible as a stale file rather than a present one.
check() {
  local label="$1" path="$2"
  if [ -s "$path" ]; then
    local n when
    n=$(wc -l <"$path" 2>/dev/null | tr -d ' ')
    when=$(stat -c '%y' "$path" 2>/dev/null | cut -c1-16)
    printf '  \033[32mok  \033[0m %-46s %8s rows  %s\n' "$label" "$n" "$when"
  else
    if [ -e "$path" ]; then
      printf '  \033[31mEMPTY\033[0m %-46s %s\n' "$label" "$path"
    else
      printf '  \033[31mMISS\033[0m %-46s %s\n' "$label" "$path"
    fi
    step_missing=$((step_missing + 1))
    missing_total=$((missing_total + 1))
  fi
}

# Same as check, but a missing file is expected and is itself the finding.
optional() {
  local label="$1" path="$2"
  if [ -s "$path" ]; then
    local n when
    n=$(wc -l <"$path" 2>/dev/null | tr -d ' ')
    when=$(stat -c '%y' "$path" 2>/dev/null | cut -c1-16)
    printf '  \033[32mok  \033[0m %-46s %8s rows  %s\n' "$label" "$n" "$when"
  else
    printf '  \033[33m--  \033[0m %-46s (absent: %s)\n' "$label" "$path"
  fi
}

check_dir() {
  local label="$1" path="$2"
  if [ -d "$path" ]; then
    local n
    n=$(find "$path" -type f 2>/dev/null | wc -l | tr -d ' ')
    printf '  \033[32mok  \033[0m %-46s %8s files\n' "$label" "$n"
  else
    printf '  \033[31mMISS\033[0m %-46s %s\n' "$label" "$path"
    step_missing=$((step_missing + 1))
    missing_total=$((missing_total + 1))
  fi
}

start_step() { step_missing=0; hdr "$*"; }

end_step() {
  if [ "$step_missing" -eq 0 ]; then
    printf '  \033[1;32m=> READY\033[0m\n'
  else
    printf '  \033[1;31m=> BLOCKED (%d missing)\033[0m\n' "$step_missing"
  fi
}

# --------------------------------------------------------------------------
start_step "1. corpus-300 provenance + non-overlap 3,319"
check "corpus-300 hint cases"        "$DATA/ddxplus_hint_cases_300.jsonl"
check "corpus-300 answers (raw)"     "$RES/ddxplus_hint_answers_300.jsonl"
optional "corpus-300 answers (_rescored)" "$RES/ddxplus_hint_answers_300_rescored.jsonl"
check "main run answers (_rescored)" "$RES/ddxplus_hint_answers_v2_rescored.jsonl"
for R in 3 4 5 6; do
  optional "corpus-300 ladder r$R"       "$RES/ddxplus_ladder_300_r${R}.jsonl"
done
for R in 3 4 5 6 7; do
  check "main ladder r$R (_rescored)"    "$RES/ddxplus_ladder_r${R}_rescored.jsonl"
done
end_step

# The provenance question in one comparison. The ledger claims every canonical
# number comes from a _rescored file, but its input table has no corpus-300
# row, while the corpus-300 values moved on an unchanged n=3,343. Either a
# rescored file exists and is newer than the matcher fix -- provenance found,
# record it -- or it does not, and the values must be regenerated before they
# can be cited.
hdr "1-b. corpus-300 provenance verdict"
c300_raw="$RES/ddxplus_hint_answers_300.jsonl"
c300_res="$RES/ddxplus_hint_answers_300_rescored.jsonl"
if [ ! -f "$MATCHER" ]; then
  printf '  \033[33m--  \033[0m matcher not found at %s (run from repo root)\n' "$MATCHER"
elif [ -s "$c300_res" ]; then
  if [ "$c300_res" -nt "$MATCHER" ]; then
    printf '  \033[1;32m=> PROVENANCE FOUND\033[0m rescored file is newer than %s\n' "$MATCHER"
    printf '     Record it in RESULTS_CANONICAL input table, clear the 확인 필요 row.\n'
  else
    printf '  \033[1;31m=> STALE\033[0m rescored file PREDATES the matcher fix -- rescore again.\n'
  fi
elif [ -s "$c300_raw" ]; then
  printf '  \033[1;31m=> NO RESCORED FILE\033[0m the .9800/.9306/.7670/.9180 row has no\n'
  printf '     canonical provenance. Rescore before citing:\n'
  printf '       python scripts/rescore_source_correct.py \\\n'
  printf '         --answers %s \\\n' "$c300_raw"
  printf '         --output  %s\n' "$c300_res"
else
  printf '  \033[1;31m=> corpus-300 answers absent entirely\033[0m\n'
fi

# --------------------------------------------------------------------------
start_step "2. MCR readout derangement control"
check "MCR conclusion readout"       "$RES/readout_mcr_conclusion_L32.jsonl"
check "MCR conclusion test manifest" "$DATA/mcr_conclusion_test_manifest.jsonl"
end_step

# --------------------------------------------------------------------------
start_step "3. MCR ladder r3/r4 (does not wait on step 2)"
check "MCR hint cases"               "$DATA/mcr_hint_cases_full.jsonl"
check "MCR answers (_rescored)"      "$RES/mcr_hint_answers_full_rescored.jsonl"
optional "MCR CoT answers (r7 only)"    "$RES/mcr_hint_answers_cot.jsonl"
for R in 3 4; do
  optional "MCR ladder r$R (output)"     "$RES/mcr_ladder_r${R}.jsonl"
done
end_step

# --------------------------------------------------------------------------
start_step "4. MCR wrong-note activation extraction (GPU, long pole)"
check "MCR hint cases"               "$DATA/mcr_hint_cases_full.jsonl"
optional "MCR wrong-note rows"          "$DATA/mcr_hint_position_rows.jsonl"
check_dir "MCR conclusion activations"  "$ACT/mcr_conclusion_L32"
optional "MCR wrong-note activations"   "$ACT/mcr_hint_positions_L32"
end_step

# --------------------------------------------------------------------------
start_step "5. Figure 5 -- 64.1% canonical recount"
check "trajectory readout manifest"  "$DATA/trajectory_readout_manifest.jsonl"
check "trajectory rows"              "$DATA/trajectory_rows_fixed.jsonl"
check "main answers (_rescored)"     "$RES/ddxplus_hint_answers_v2_rescored.jsonl"
check "hint cases v2"                "$DATA/ddxplus_hint_cases_v2.jsonl"
printf '  note: readout files are per-landmark; list them with\n'
printf '    ls %s/readout_traj_*.jsonl %s/readout_hint_final_L32_v2.jsonl\n' "$RES" "$RES"
end_step

# --------------------------------------------------------------------------
start_step "6. wording (4) + CoT canonical rescore"
check "wording: note (base) answers" "$RES/ddxplus_hint_answers_v2.jsonl"
for W in colleague patient; do
  check "wording: $W answers"          "$RES/ddxplus_hint_answers_${W}.jsonl"
done
optional "wording: realistic answers"   "$RES/ddxplus_hint_answers_realistic.jsonl"
optional "wording: realistic cases"     "$DATA/ddxplus_hint_realistic.jsonl"
check "CoT full answers (raw)"       "$RES/ddxplus_hint_answers_cot_full.jsonl"
optional "CoT full answers (_rescored)" "$RES/ddxplus_hint_answers_cot_full_rescored.jsonl"
end_step

# --------------------------------------------------------------------------
start_step "7. reader-trust completion + shuffled control"
check "reader-trust cases"           "$DATA/ddxplus_reader_trust_cases.jsonl"
check "reader-trust judgements"      "$RES/judge_reader_trust.jsonl"
optional "reader-trust shuffled cases"  "$DATA/ddxplus_reader_trust_cases_shuffled.jsonl"
optional "reader-trust shuffled judged" "$RES/judge_reader_trust_shuffled.jsonl"
end_step

# --------------------------------------------------------------------------
# The 438-row semantic rescore is the one remaining item whose inputs live in
# the repository rather than on the GPU box: three layers x 438 held-out cue
# rows = the 1,314 of judge job #3. It needs no GPU and no $ART at all.
start_step "8. judge #3 (438-row semantic rescore) -- inputs are repo-local"
for L in v4 L16_v5 L24_v5; do
  check "held-out cue scored: $L"      "results_snapshot/${L}_test_heldout_cue_scored_compact.jsonl"
  check "untuned control: $L"          "results_snapshot/${L}_heldout_vanilla_compact.jsonl"
done
optional "judge #3 requests"            "$DATA/judge_readout_semantic.jsonl"
optional "judge #3 verdicts"            "$RES/judge_readout_semantic.jsonl"
end_step

# --------------------------------------------------------------------------
start_step "9. no-CoT arm (separates CoT's own contribution)"
check "hint cases v2"                "$DATA/ddxplus_hint_cases_v2.jsonl"
check "CoT monitor requests"         "$DATA/judge_cot_monitor.jsonl"
check "CoT monitor labels"           "$DATA/judge_cot_monitor_labels.jsonl"
optional "no-CoT arm requests"          "$DATA/judge_cot_monitor_nocot.jsonl"
optional "no-CoT arm verdicts"          "$RES/judge_cot_monitor_nocot.jsonl"
end_step

# --------------------------------------------------------------------------
hdr "reader-trust progress (step 7)"
rt="$RES/judge_reader_trust.jsonl"
if [ -s "$rt" ]; then
  done_n=$(wc -l <"$rt" | tr -d ' ')
  uniq_n=$(python - "$rt" <<'PY' 2>/dev/null || echo "?"
import json, sys
ids = set()
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    try:
        ids.add(json.loads(line).get("id"))
    except Exception:
        pass
print(len(ids))
PY
)
  printf '  %s rows, %s distinct ids, of 2,896 expected\n' "$done_n" "$uniq_n"
  if [ "$uniq_n" != "?" ] && [ "$done_n" != "$uniq_n" ]; then
    printf '  \033[33m  duplicate ids present -- run scripts/dedupe_judgements.py first\033[0m\n'
  fi
else
  printf '  no judgement file yet\n'
fi

hdr "running jobs (do not collide)"
pgrep -af 'run_judge|run_source_answers|train_medical_nla_lora|run_nla' 2>/dev/null | head -10 || true
printf '\n'
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader 2>/dev/null || printf '  (no nvidia-smi)\n'

# --------------------------------------------------------------------------
hdr "summary"
if [ "$missing_total" -eq 0 ]; then
  printf '  \033[1;32mevery required input present\033[0m\n'
else
  printf '  \033[1;31m%d required input(s) missing -- see BLOCKED steps above\033[0m\n' "$missing_total"
fi
printf '  optional (--) rows are outputs that do not exist yet by design.\n'
