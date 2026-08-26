#!/usr/bin/env bash
set -euo pipefail

# Bootstrap a second authorized compute server for the DiReCT experiments.
# Run this script on the destination server after cloning medical_nla.
# Restricted clinical files remain under DEST_DATA_ROOT/restricted and must
# never be committed or copied to an unapproved machine.

SOURCE_HOST="${SOURCE_HOST:-eagle0914@165.132.76.125}"
SOURCE_DATA_ROOT="${SOURCE_DATA_ROOT:-/data1/heejae}"
DEST_DATA_ROOT="${DEST_DATA_ROOT:-/data/heejae}"
CODE_ROOT="${CODE_ROOT:-/home/eagle0914/medical_nla}"
COPY_HF_MODELS="${COPY_HF_MODELS:-1}"
COPY_DDXPLUS="${COPY_DDXPLUS:-0}"
INSTALL_ENV="${INSTALL_ENV:-1}"
TORCH_VERSION="${TORCH_VERSION:-2.5.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

RSYNC_RSH="${RSYNC_RSH:-ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath=/tmp/medical-nla-ssh-%C}"
RSYNC=(rsync -a --partial --append-verify --info=progress2 -e "${RSYNC_RSH}")

for command in git rsync ssh python3; do
  if ! command -v "${command}" >/dev/null 2>&1; then
    echo "[error] required command is unavailable: ${command}" >&2
    exit 1
  fi
done
if [[ "${INSTALL_ENV}" == "1" ]] && ! command -v uv >/dev/null 2>&1; then
  echo "[error] uv is required when INSTALL_ENV=1" >&2
  exit 1
fi

if [[ ! -d "${CODE_ROOT}/.git" ]]; then
  echo "[error] clone the repository at ${CODE_ROOT} before running this script" >&2
  exit 1
fi

mkdir -p \
  "${DEST_DATA_ROOT}/uv" \
  "${DEST_DATA_ROOT}/hf_cache/hub" \
  "${DEST_DATA_ROOT}/medical_nla"/{activations,adapters,data,logs,probe,reports,results,train} \
  "${DEST_DATA_ROOT}/models" \
  "${DEST_DATA_ROOT}/restricted"
chmod 700 "${DEST_DATA_ROOT}/restricted"

echo "[sync 1/2] restricted DiReCT release and private derived manifests"
"${RSYNC[@]}" \
  "${SOURCE_HOST}:${SOURCE_DATA_ROOT}/restricted/direct/" \
  "${DEST_DATA_ROOT}/restricted/direct/"
chmod -R go-rwx "${DEST_DATA_ROOT}/restricted/direct"

echo "[sync 2/2] official native Llama-3 judge checkpoint"
"${RSYNC[@]}" \
  "${SOURCE_HOST}:${SOURCE_DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/" \
  "${DEST_DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/"

if [[ "${COPY_HF_MODELS}" == "1" ]]; then
  echo "[sync optional] Gemma and vanilla NLA Hugging Face cache entries"
  for model_dir in \
    models--google--gemma-3-12b-it \
    models--kitft--nla-gemma3-12b-L32-av \
    models--kitft--nla-gemma3-12b-L32-ar
  do
    if ssh ${RSYNC_RSH#ssh } "${SOURCE_HOST}" \
      "test -d '${SOURCE_DATA_ROOT}/hf_cache/hub/${model_dir}'"
    then
      "${RSYNC[@]}" \
        "${SOURCE_HOST}:${SOURCE_DATA_ROOT}/hf_cache/hub/${model_dir}/" \
        "${DEST_DATA_ROOT}/hf_cache/hub/${model_dir}/"
    else
      echo "[warn] source cache entry missing; download later: ${model_dir}" >&2
    fi
  done
fi

if [[ "${COPY_DDXPLUS}" == "1" ]]; then
  echo "[sync optional] DDXPlus public dataset"
  "${RSYNC[@]}" \
    "${SOURCE_HOST}:${SOURCE_DATA_ROOT}/ddxplus/" \
    "${DEST_DATA_ROOT}/ddxplus/"
fi

if [[ "${INSTALL_ENV}" == "1" ]]; then
  echo "[env] creating Python 3.11 environment"
  uv venv "${DEST_DATA_ROOT}/uv/medical_nla" --python 3.11
  # shellcheck disable=SC1091
  source "${DEST_DATA_ROOT}/uv/medical_nla/bin/activate"
  uv pip install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX_URL}"
  cd "${CODE_ROOT}"
  uv pip install -e '.[dev,direct-eval]'
  uv pip install orjson
fi

echo "[verify] required private data and judge files"
test "$(find "${DEST_DATA_ROOT}/restricted/direct/samples" -type f -name '*.json' | wc -l)" -eq 511
test "$(find "${DEST_DATA_ROOT}/restricted/direct/diagnostic_kg" -type f -name '*.json' | wc -l)" -eq 24
for filename in consolidated.00.pth params.json tokenizer.model; do
  test -s "${DEST_DATA_ROOT}/models/Meta-Llama-3-8B-Instruct/original/${filename}"
done

echo "[done] destination server is staged at ${DEST_DATA_ROOT}"
echo "[next] cd ${CODE_ROOT}"
echo "[next] source ${DEST_DATA_ROOT}/uv/medical_nla/bin/activate"
echo "[next] source scripts/env.sh ${DEST_DATA_ROOT}"
echo "[next] DATA_ROOT=${DEST_DATA_ROOT} GPU=0 LIMIT=10 FORCE=1 bash scripts/run_direct_official_smoke.sh"
