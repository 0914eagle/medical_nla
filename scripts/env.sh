# Source this at the start of every session:  source scripts/env.sh
#
# Exists because the alternative is retyping four exports per shell, and a shell
# that missed one does not fail -- it writes to a path built from an empty
# variable. `$MEDICAL_NLA_DATA_ROOT/ddxplus/x.json` with the variable unset is
# `/ddxplus/x.json`, which is a permission error at best and a file nobody finds
# at worst.
#
# The data root is found from the uv virtualenv, which is created once per
# machine underneath it, so nothing has to be remembered or edited when moving
# between hosts with different disks. Override by passing it:
#
#   source scripts/env.sh                  # find it
#   source scripts/env.sh /data9/somebody  # say it
#
# Passed as an argument rather than as `VAR=... source scripts/env.sh`, which
# looks right and is not: `source` is a builtin, so a prefixed assignment is
# temporary and the value is gone by the time the next command runs.

# The root differs per machine (/data/heejae here, /data1/heejae there), and
# hard-coding either one is wrong on the other. The uv virtualenv is created
# once per machine and lives under the root, so it identifies the root without
# anyone having to remember which disk this host mounts.
_MEDICAL_NLA_FOUND=""
# Only an absolute path: sourced with no arguments, $1 is whatever the calling
# shell had, which is not a data root.
case "${1:-}" in
  /*) MEDICAL_NLA_DATA_ROOT="$1"; _MEDICAL_NLA_FOUND="given as an argument" ;;
esac
if [ -z "${MEDICAL_NLA_DATA_ROOT:-}" ]; then
  _MEDICAL_NLA_CANDS=$(ls -d /data*/*/uv/medical_nla /data*/uv/medical_nla 2>/dev/null \
                       | sed 's|/uv/medical_nla$||')
  _MEDICAL_NLA_N=$(printf '%s\n' "$_MEDICAL_NLA_CANDS" | grep -c . || true)
  if [ "${_MEDICAL_NLA_N:-0}" -ge 1 ]; then
    MEDICAL_NLA_DATA_ROOT=$(printf '%s\n' "$_MEDICAL_NLA_CANDS" | head -1)
    _MEDICAL_NLA_FOUND="found via $MEDICAL_NLA_DATA_ROOT/uv/medical_nla"
    if [ "$_MEDICAL_NLA_N" -gt 1 ]; then
      # Picking one silently would send a run to the wrong disk, so say so.
      echo "[env] several candidate roots on this host:"
      printf '[env]   %s\n' $_MEDICAL_NLA_CANDS
      echo "[env] using the first; set MEDICAL_NLA_DATA_ROOT to override"
    fi
  else
    MEDICAL_NLA_DATA_ROOT=/data1/heejae
    _MEDICAL_NLA_FOUND="default (no uv/medical_nla found on this host)"
  fi
else
  _MEDICAL_NLA_FOUND="from the environment"
fi
export MEDICAL_NLA_DATA_ROOT
unset _MEDICAL_NLA_CANDS _MEDICAL_NLA_N

if [ -n "${BASH_SOURCE[0]:-}" ]; then
  _MEDICAL_NLA_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  _MEDICAL_NLA_HERE="$PWD"
fi
export MEDICAL_NLA_CODE_ROOT="${MEDICAL_NLA_CODE_ROOT:-$_MEDICAL_NLA_HERE}"
unset _MEDICAL_NLA_HERE

export PYTHONPATH="$MEDICAL_NLA_CODE_ROOT"
export HF_HOME="$MEDICAL_NLA_DATA_ROOT/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Two of the four cards. The backbone is 24.4GB in bfloat16 and needs two 24GB
# cards; naming them here rather than in the config keeps the config free of
# device identity, so a second job on the other pair is
#   export CUDA_VISIBLE_DEVICES=2,3
#   source scripts/env.sh
# with no edit anywhere -- max_memory keys index the *visible* devices, so
# {0: 22GiB, 1: 22GiB} means the visible pair, whichever pair that is.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Shorthands the commands in EXPERIMENTS.md are written against.
export RAW="$MEDICAL_NLA_DATA_ROOT"
export ART="$MEDICAL_NLA_DATA_ROOT/medical_nla"
export DATA="$ART/data"

# Printed, not assumed: the failure this file prevents is a shell that looks
# configured and is not, so the values are put on screen every time.
echo "[env] host      $(hostname)"
echo "[env] data root $MEDICAL_NLA_DATA_ROOT $([ -d "$MEDICAL_NLA_DATA_ROOT" ] || echo '(MISSING -- wrong machine?)')"
echo "[env]           $_MEDICAL_NLA_FOUND"
echo "[env] code root $MEDICAL_NLA_CODE_ROOT"
echo "[env] hf cache  $HF_HOME"
echo "[env] gpus      CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
unset _MEDICAL_NLA_FOUND
