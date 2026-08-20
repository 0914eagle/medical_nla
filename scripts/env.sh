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
# between hosts with different disks. Override when that guess is wrong:
#   MEDICAL_NLA_DATA_ROOT=/data9/somebody source scripts/env.sh

# The root differs per machine (/data/heejae here, /data1/heejae there), and
# hard-coding either one is wrong on the other. The uv virtualenv is created
# once per machine and lives under the root, so it identifies the root without
# anyone having to remember which disk this host mounts.
_MEDICAL_NLA_FOUND=""
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
unset _MEDICAL_NLA_FOUND
