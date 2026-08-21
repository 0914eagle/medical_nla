#!/usr/bin/env bash
# Bring a fresh machine to the state the experiments start from.
#
# Everything downstream is regenerated from the two raw corpora rather than
# copied, because the processed files have changed under us repeatedly -- gold
# labels, accepted aliases, the direct instruction -- and a copy carries
# whichever version happened to be on the old disk. Regenerating takes about
# fifteen minutes and is the only way to know which version is present.
#
# Paths come from $MEDICAL_NLA_DATA_ROOT (default /data1/heejae), the single
# variable the configs are written against.
#
#   bash scripts/bootstrap_server.sh              # fetch, build, audit
#   bash scripts/bootstrap_server.sh --skip-fetch # rebuild from raw already here
set -euo pipefail

# Same defaults as scripts/env.sh, which a session sources; repeated here so
# the bootstrap works in a shell that has not sourced anything.
DATA_ROOT="${MEDICAL_NLA_DATA_ROOT:-/data1/heejae}"
if [[ ! -d "$DATA_ROOT" ]]; then
  echo "[bootstrap] $DATA_ROOT does not exist on $(hostname)."
  echo "[bootstrap] Either this is the wrong machine, or the disk is mounted"
  echo "[bootstrap] elsewhere -- set MEDICAL_NLA_DATA_ROOT and re-run."
  exit 1
fi
export MEDICAL_NLA_DATA_ROOT="$DATA_ROOT"
CODE_ROOT="${MEDICAL_NLA_CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export MEDICAL_NLA_CODE_ROOT="$CODE_ROOT"
export PYTHONPATH="$CODE_ROOT"

# HF_HOME keeps a 25GB checkpoint off the home partition, which is where a
# "no space left on device" halfway through a download otherwise comes from.
# Only this one: TRANSFORMERS_CACHE names the hub directory itself while
# HF_HOME names its parent, so setting both to one path gives two caches and
# two downloads of every checkpoint.
unset TRANSFORMERS_CACHE
export HF_HOME="$DATA_ROOT/hf_cache"

SKIP_FETCH=0
[[ "${1:-}" == "--skip-fetch" ]] && SKIP_FETCH=1

RAW="$DATA_ROOT"
ART="$DATA_ROOT/medical_nla"
DATA="$ART/data"
REPORTS="$ART/reports"

echo "[bootstrap] data root : $DATA_ROOT"
echo "[bootstrap] code root : $CODE_ROOT"
mkdir -p "$RAW/ddxplus" "$DATA" "$REPORTS" "$ART/results" "$ART/activations" \
         "$ART/logs" "$ART/train" "$HF_HOME"

cd "$CODE_ROOT"

if [[ "$SKIP_FETCH" -eq 0 ]]; then
  echo "[bootstrap] fetching DDXPlus"
  # The original release CSVs live on a disk this machine may not have, so the
  # corpus comes from the Hugging Face mirror of the same patient rows and
  # evidence dictionary.
  python - <<PY
import json, os
from datasets import load_dataset

raw = os.environ["MEDICAL_NLA_DATA_ROOT"] + "/ddxplus"
ds = load_dataset("aai530-group6/ddxplus")
print(ds)
ds["train"].to_csv(f"{raw}/train.csv")
PY

  if [[ ! -s "$RAW/ddxplus/release_evidences.json" ]]; then
    echo "[bootstrap] !! $RAW/ddxplus/release_evidences.json is missing."
    echo "[bootstrap]    It is the questionnaire dictionary every cue is rendered from"
    echo "[bootstrap]    and is not in the Hugging Face mirror. Copy it from the old"
    echo "[bootstrap]    machine or download it from the DDXPlus release, then re-run"
    echo "[bootstrap]    with --skip-fetch."
    exit 1
  fi

  echo "[bootstrap] fetching MedCaseReasoning"
  python - <<PY
import os
from datasets import load_dataset

data = os.environ["MEDICAL_NLA_DATA_ROOT"] + "/medical_nla/data"
ds = load_dataset("zou-lab/MedCaseReasoning")
print(ds)
for split in ds:
    ds[split].to_json(f"{data}/mcr_{split}.jsonl")
PY
fi

echo "[bootstrap] building MedCaseReasoning cases"
for split in train test; do
  python scripts/make_clinical_span_cases.py \
    --input "$DATA/mcr_${split}.jsonl" \
    --output "$DATA/mcr_cases_${split}.jsonl" \
    --report "$REPORTS/mcr_ingest_${split}.json" \
    --dump-relabelled "$REPORTS/mcr_relabelled_${split}.tsv" \
    --min-cues 3 --min-words 4 --max-words 14
  python scripts/audit_clinical_span_cases.py --cases "$DATA/mcr_cases_${split}.jsonl"
done

echo "[bootstrap] building DDXPlus cases"
python scripts/make_ddxplus_cue_count_cases.py \
  --patients "$RAW/ddxplus/train.csv" \
  --evidences "$RAW/ddxplus/release_evidences.json" \
  --output "$DATA/ddxplus_cue_count_cases.jsonl" \
  --examples-per-diagnosis 100 --cue-counts all --seed 17 \
  --no-prefer-symptoms --stop-when-full

python scripts/audit_ddxplus_cue_rendering.py \
  --evidences "$RAW/ddxplus/release_evidences.json" \
  --patients "$RAW/ddxplus/train.csv" \
  --cases "$DATA/ddxplus_cue_count_cases.jsonl" \
  --dump "$REPORTS/ddxplus_cue_vocabulary.tsv"

echo "[bootstrap] building extraction rows"
python scripts/make_ddxplus_cue_position_rows.py \
  --input "$DATA/ddxplus_cue_count_cases.jsonl" \
  --output "$DATA/ddxplus_cuepos_rows.jsonl" \
  --variants cue_count_all --max-cues-per-case 4 --seed 17

python scripts/make_format_position_rows.py \
  --input "$DATA/ddxplus_cue_count_cases.jsonl" \
  --output "$DATA/ddxplus_format_rows.jsonl" \
  --variants cue_count_all

python scripts/make_ddxplus_cue_position_rows.py \
  --input "$DATA/mcr_cases_train.jsonl" \
  --output "$DATA/mcr_cuepos_train.jsonl" \
  --variants cue_count_all --max-cues-per-case 4 --seed 17

echo "[bootstrap] checking GPU placement"
python scripts/check_gpu_setup.py --config configs/default.yaml || true

cat <<'EOF'

[bootstrap] done. Expected, for comparison with the audits above:
  DDXPlus            4,900 cases / 18,646 cue rows / 4,900 format rows
  MedCaseReasoning   11,799 train + 821 test cases / 46,768 train cue rows
  both audits        "hard violations: 0"

Start every later session with, from the repo:
  source scripts/env.sh

If the model load reports anything on cpu/meta, it will refuse and print the
free memory per card; add a max_memory block to configs/default.yaml as the
comment there shows.
EOF
