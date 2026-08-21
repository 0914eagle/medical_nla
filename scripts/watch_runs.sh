#!/usr/bin/env bash
# What is running, how far along, and how much longer.
#
#   bash scripts/watch_runs.sh            # one look
#   bash scripts/watch_runs.sh -f         # keep looking, every 30s
#   bash scripts/watch_runs.sh -f 60      # ... every 60s
#
# Exists because a generation run prints its setup and then nothing for half an
# hour, and that silence has been read as a hang and investigated twice -- once
# with ss, once with du. Output files grow a line per row, so counting lines
# twice a few seconds apart is the progress bar the runs do not have.
set -uo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source scripts/env.sh >/dev/null

FOLLOW=0
INTERVAL=30
if [ "${1:-}" = "-f" ]; then
  FOLLOW=1
  [ -n "${2:-}" ] && INTERVAL="$2"
fi

# Files touched in the last day, which is what "this session's runs" means in
# practice and avoids listing every result the project has ever produced.
recent() {
  find "$ART/results" "$ART/train/adapters" -maxdepth 2 \
       \( -name '*.jsonl' -o -name 'best.json' \) -mmin -1440 2>/dev/null | sort
}

snapshot() {
  echo "=== $(date +%H:%M:%S) ==="

  local procs
  procs=$(pgrep -af "src\.run_nla|train_medical_nla_lora|run_source_answers|score_source_diagnosis_logprobs|score_nla_diagnosis_logprobs|train_ddxplus_linear_probe" 2>/dev/null)
  if [ -n "$procs" ]; then
    echo "running:"
    # The command line is long; the script name and its output path are what
    # identify a run.
    printf '%s\n' "$procs" | sed 's/--config [^ ]*//; s/\(.\{150\}\).*/\1.../' | sed 's/^/  /'
  else
    echo "running: nothing"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "gpu:"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader | sed 's/^/  /'
  fi

  local files
  files=$(recent)
  if [ -z "$files" ]; then
    echo "no output files touched in the last day"
    return
  fi

  echo "progress:"
  local before=() names=()
  for f in $files; do
    names+=("$f")
    before+=("$(wc -l < "$f" 2>/dev/null || echo 0)")
  done
  sleep 10
  local i=0
  for f in "${names[@]}"; do
    local now delta rate name
    now=$(wc -l < "$f" 2>/dev/null || echo 0)
    delta=$(( now - ${before[$i]} ))
    name=$(basename "$f")
    if [ "$delta" -gt 0 ]; then
      # Lines per 10s into a per-row time, and a guess at what is left if the
      # run is one of the 770-row readout pools.
      rate=$(awk -v d="$delta" 'BEGIN{printf "%.1fs/row", 10/d}')
      if [ "$now" -lt 770 ]; then
        local eta
        eta=$(awk -v n="$now" -v d="$delta" 'BEGIN{printf "~%d min to 770", (770-n)/d*10/60}')
        printf '  %-52s %6d  %-10s %s\n' "$name" "$now" "$rate" "$eta"
      else
        printf '  %-52s %6d  %-10s\n' "$name" "$now" "$rate"
      fi
    else
      printf '  %-52s %6d  (idle)\n' "$name" "$now"
    fi
    i=$((i + 1))
  done
}

if [ "$FOLLOW" -eq 1 ]; then
  while true; do
    snapshot
    echo
    sleep "$INTERVAL"
  done
else
  snapshot
fi
