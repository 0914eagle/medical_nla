#!/usr/bin/env bash
set -euo pipefail

# Validation-only D16 free-generation diagnostics. This is intentionally a
# separate long queue: six adapters x the frozen ~952-row paired DDXPlus pilot.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
GPU_PAIR_A="${GPU_PAIR_A:-0,1}"
GPU_PAIR_B="${GPU_PAIR_B:-2,3}"
JUDGE_GPU="${JUDGE_GPU:-0}"
RUN_NAME="${RUN_NAME:-medical_nla_d16_soft_bottleneck_v1}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ "${DATA_ROOT}" != "/data1/heejae" ]]; then
  echo "[error] D16 generation queue is frozen to server 125" >&2
  exit 2
fi

cd /home/eagle0914/medical_nla
source "${DATA_ROOT}/uv/medical_nla/bin/activate"
unset MEDICAL_NLA_DATA_ROOT HF_HOME TRANSFORMERS_CACHE
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH=/home/eagle0914/medical_nla

DIRECT_VAL="${DATA_ROOT}/restricted/direct/e3/direct_e3_sft_v1/sft_val.jsonl"
E5="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
DDX_VAL="${E5}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
ADAPTERS="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}/adapters"
PRIMARY="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_alignment_val_v1/paired_arm_comparison.json"
DIRECT_OUT="${DATA_ROOT}/restricted/direct/e4/${RUN_NAME}_generation_val_v1"
DDX_OUT="${E5}/${RUN_NAME}_ddx_grounding_val_v1"
SHARDS="${DDX_OUT}/manifest_shards"
PAIRED="${DDX_OUT}/paired_manifest.jsonl"
LOG_ROOT="${DATA_ROOT}/medical_nla/logs/${RUN_NAME}"
PROMPT="prompt_templates/common_p0_clinical_state_readout.txt"
mkdir -p "${DIRECT_OUT}" "${DDX_OUT}" "${LOG_ROOT}"

for path in "${PRIMARY}" "${DIRECT_VAL}" "${DDX_VAL}" "${PROMPT}"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

if [[ ! -s "${PAIRED}" ]]; then
  python scripts/shard_jsonl_by_key.py \
    --input "${DDX_VAL}" --out-dir "${SHARDS}" --num-shards 40 --key base_id
  merge_args=()
  expected=0
  for shard in 0 1 2 3; do
    printf -v path '%s/shard_%03d_of_040.jsonl' "${SHARDS}" "${shard}"
    merge_args+=(--input "${path}")
    expected=$((expected + $(wc -l < "${path}")))
  done
  python scripts/merge_jsonl_files.py "${merge_args[@]}" \
    --output "${PAIRED}" --expected-rows "${expected}"
fi
paired_rows="$(wc -l < "${PAIRED}")"
echo "[population] frozen DDXPlus paired validation rows=${paired_rows}"

run_readout() {
  local population="$1" arm="$2" seed="$3" gpus="$4"
  local label="${arm}_seed${seed}"
  local adapter="${ADAPTERS}/${label}"
  local manifest output log expected
  if [[ "${population}" == "direct" ]]; then
    manifest="${DIRECT_VAL}"
    output="${DIRECT_OUT}/${label}.jsonl"
    log="${LOG_ROOT}/${label}_direct_generation.log"
    expected=50
  else
    manifest="${PAIRED}"
    output="${DDX_OUT}/${label}.jsonl"
    log="${LOG_ROOT}/${label}_ddx_generation.log"
    expected="${paired_rows}"
  fi
  test -s "${adapter}/best.json"
  test -s "${adapter}/nla_bottleneck.pt"
  if [[ -s "${output}" && "$(wc -l < "${output}")" -eq "${expected}" ]]; then
    echo "[skip] ${population} ${label}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${gpus}" python -m src.run_nla \
    --config configs/default.yaml \
    --manifest "${manifest}" \
    --output "${output}" \
    --adapter-id "${adapter}" \
    --actor-prompt-template-file "${PROMPT}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --batch-size "${BATCH_SIZE}" > "${log}" 2>&1
  test "$(wc -l < "${output}")" -eq "${expected}"
}

run_six() {
  local population="$1"
  for arm in control auxiliary; do
    run_readout "${population}" "${arm}" 17 "${GPU_PAIR_A}" & p1=$!
    run_readout "${population}" "${arm}" 29 "${GPU_PAIR_B}" & p2=$!
    s1=0; s2=0
    wait "${p1}" || s1=$?
    wait "${p2}" || s2=$?
    if [[ "${s1}" -ne 0 || "${s2}" -ne 0 ]]; then
      echo "[error] ${population} ${arm}: seed17=${s1} seed29=${s2}" >&2
      return 1
    fi
    run_readout "${population}" "${arm}" 43 "${GPU_PAIR_A}"
  done
}

echo "[stage 1/4] Direct validation generation"
run_six direct

echo "[stage 2/4] auxiliary-head removal byte-identity check"
VERIFY="${DIRECT_OUT}/head_removal_verification"
if [[ ! -s "${VERIFY}/summary.json" ]]; then
  FULL="${ADAPTERS}/auxiliary_seed17"
  HEAD="${DATA_ROOT}/restricted/direct/e3/${RUN_NAME}/training_only_aux_heads/auxiliary_seed17.pt"
  COPY="${VERIFY}/adapter_without_head"
  mkdir -p "${VERIFY}"
  test -s "${HEAD}"
  test ! -e "${FULL}/auxiliary_head_training_only.pt" || {
    echo "[error] training-only head leaked into inference adapter" >&2
    exit 2
  }
  test ! -e "${COPY}" || { echo "[error] incomplete verification copy exists: ${COPY}" >&2; exit 2; }
  rsync -a --exclude='auxiliary_head_training_only.pt' "${FULL}/" "${COPY}/"
  CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}" python -m src.run_nla \
    --config configs/default.yaml --manifest "${DIRECT_VAL}" \
    --output "${VERIFY}/with_head_file.jsonl" --adapter-id "${FULL}" \
    --audit-disconnected-aux-head "${HEAD}" \
    --actor-prompt-template-file "${PROMPT}" --limit 2 --sample-seed 17 \
    --max-new-tokens "${MAX_NEW_TOKENS}" --batch-size 2
  CUDA_VISIBLE_DEVICES="${GPU_PAIR_A}" python -m src.run_nla \
    --config configs/default.yaml --manifest "${DIRECT_VAL}" \
    --output "${VERIFY}/without_head_file.jsonl" --adapter-id "${COPY}" \
    --actor-prompt-template-file "${PROMPT}" --limit 2 --sample-seed 17 \
    --max-new-tokens "${MAX_NEW_TOKENS}" --batch-size 2
  python - "${VERIFY}" "${FULL}" "${COPY}" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
full = pathlib.Path(sys.argv[2])
copy = pathlib.Path(sys.argv[3])
def rows(name):
    return [json.loads(x) for x in (root / name).open() if x.strip()]
def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
a = rows("with_head_file.jsonl")
b = rows("without_head_file.jsonl")
assert [x["base_id"] for x in a] == [x["base_id"] for x in b]
identical = [x["generated_token_ids"] for x in a] == [x["generated_token_ids"] for x in b]
if not identical:
    raise SystemExit("auxiliary-head removal changed generated token IDs")
checkpoint_hashes = {}
for name in ("adapter_config.json", "adapter_model.safetensors", "nla_bottleneck.pt"):
    if not (full / name).is_file():
        continue
    before = digest(full / name)
    after = digest(copy / name)
    if before != after:
        raise SystemExit(f"checkpoint changed across head removal: {name}")
    checkpoint_hashes[name] = before
(root / "summary.json").write_text(json.dumps({
    "rows": len(a),
    "generated_outputs_byte_identical": True,
    "auxiliary_head_connected_to_inference": False,
    "projector_decoder_hashes_identical": True,
    "checkpoint_hashes": checkpoint_hashes,
}, indent=2) + "\n")
print("[gate] auxiliary-head removal generated outputs byte-identical")
PY
fi

echo "[stage 3/4] Direct quote-constrained official semantic evaluation"
DATA_ROOT="${DATA_ROOT}" GPU="${JUDGE_GPU}" \
RUN_NAME="${RUN_NAME}_direct_semantic_val_v1" \
READOUTS_DIR="${DIRECT_OUT}" \
READOUT_METHODS="control_seed17 control_seed29 control_seed43 auxiliary_seed17 auxiliary_seed29 auxiliary_seed43" \
EXTRACTOR_BACKEND=codex \
bash scripts/run_direct_e4_validation_evaluator.sh

echo "[stage 4/4] frozen paired DDXPlus generation and lexical grounding"
run_six ddxplus
score_args=()
for arm in control auxiliary; do
  for seed in 17 29 43; do
    label="${arm}_seed${seed}"
    score_args+=(--readout "${label}=${DDX_OUT}/${label}.jsonl")
  done
done
python scripts/score_ddxplus_e5_readout_pilot.py \
  "${score_args[@]}" \
  --threshold 0.5 \
  --output-json "${DDX_OUT}/paired_scores.json" \
  --summary-md "${DDX_OUT}/paired_scores_summary.md"

cat "${DDX_OUT}/paired_scores_summary.md"
echo "[done] ${RUN_NAME} generation diagnostics"
