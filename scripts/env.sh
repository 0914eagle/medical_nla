# Source this at the start of every session:  source scripts/env.sh
#
# Exists because the alternative is retyping four exports per shell, and a shell
# that missed one does not fail -- it writes to a path built from an empty
# variable. `$MEDICAL_NLA_DATA_ROOT/ddxplus/x.json` with the variable unset is
# `/ddxplus/x.json`, which is a permission error at best and a file nobody finds
# at worst.
#
# Override the root before sourcing, or edit the default here on a new machine:
#   MEDICAL_NLA_DATA_ROOT=/data9/somebody source scripts/env.sh

export MEDICAL_NLA_DATA_ROOT="${MEDICAL_NLA_DATA_ROOT:-/data1/heejae}"

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
echo "[env] code root $MEDICAL_NLA_CODE_ROOT"
echo "[env] hf cache  $HF_HOME"
