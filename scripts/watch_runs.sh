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
# Long enough that a slow run is not reported as stopped. A vanilla AV readout
# emits no terminating tag and runs to the full token budget, which is three to
# four times the adapter's seconds per row.
SAMPLE_WINDOW="${SAMPLE_WINDOW:-20}"
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
  sleep "$SAMPLE_WINDOW"
  local i=0
  for f in "${names[@]}"; do
    local now delta rate name
    now=$(wc -l < "$f" 2>/dev/null || echo 0)
    delta=$(( now - ${before[$i]} ))
    # Adapter directories all hold a best.json and a metrics.jsonl, so the
    # basename alone lists the same two names several times over.
    case "$f" in
      */adapters/*) name="$(basename "$(dirname "$f")")/$(basename "$f")" ;;
      *) name=$(basename "$f") ;;
    esac
    if [ "$delta" -gt 0 ]; then
      # A readout pool is 770 rows and a one-epoch metrics.jsonl is 1,274
      # optimizer steps plus an eval line. Applying the pool's target to both
      # printed "~15 min to 770" against a training log that was going to 1,275,
      # which is worse than printing no estimate.
      local target unit
      case "$f" in
        */metrics.jsonl) target=1275; unit="step" ;;
        */results/*.jsonl) target=770; unit="row" ;;
        *) target=0; unit="line" ;;
      esac
      rate=$(awk -v d="$delta" -v w="$SAMPLE_WINDOW" -v u="$unit" 'BEGIN{printf "%.1fs/%s", w/d, u}')
      if [ "$target" -gt 0 ] && [ "$now" -lt "$target" ]; then
        local eta
        eta=$(awk -v n="$now" -v d="$delta" -v w="$SAMPLE_WINDOW" -v t="$target" \
              'BEGIN{printf "~%d min to %d", (t-n)/d*w/60, t}')
        printf '  %-52s %6d  %-12s %s\n' "$name" "$now" "$rate" "$eta"
      else
        printf '  %-52s %6d  %-12s\n' "$name" "$now" "$rate"
      fi
    else
      printf '  %-52s %6d  (no new rows in %ss)\n' "$name" "$now" "$SAMPLE_WINDOW"
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
