#!/usr/bin/env bash
set -euo pipefail

# One-shot RunPod setup for the D10 budget calibration, for a pod image that
# already ships torch (>=2.3, e.g. the PyTorch 2.4 template). Installs the
# remaining dependencies with pip, logs into HF, extracts the DDXPlus bundle
# at /data1/heejae, and pre-downloads both models.
#
# Usage on the pod:
#   export HF_TOKEN=hf_...        # a token that has accepted the Gemma license
#   bash scripts/setup_runpod_d10.sh /root/d10_runpod_bundle.tar.gz

BUNDLE="${1:?Usage: setup_runpod_d10.sh /path/to/d10_runpod_bundle.tar.gz}"
DATA_ROOT="/data1/heejae"
REPO_DIR="${REPO_DIR:-/home/eagle0914/medical_nla}"
REPO_URL="${REPO_URL:-https://github.com/0914eagle/medical_nla.git}"

test -s "${BUNDLE}" || { echo "[error] bundle not found: ${BUNDLE}" >&2; exit 2; }
test -n "${HF_TOKEN:-}" || { echo "[error] export HF_TOKEN first (Gemma license accepted)" >&2; exit 2; }

echo "[1/6] repo"
if [[ ! -d "${REPO_DIR}/.git" ]]; then
  mkdir -p "$(dirname "${REPO_DIR}")"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"
git pull --ff-only origin main

echo "[2/6] python deps (keeps the image torch)"
python -c 'import torch; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 3), torch.__version__; print("torch", torch.__version__, "cuda", torch.cuda.is_available())'
pip install -e . 2>&1 | tail -2

echo "[3/6] hf login"
python - <<'EOF'
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"], add_to_git_credential=False)
print("[hf] login ok")
EOF

echo "[4/6] extract DDXPlus bundle at ${DATA_ROOT}"
mkdir -p "${DATA_ROOT}"
tar -C "${DATA_ROOT}" -xzf "${BUNDLE}"
find "${DATA_ROOT}/medical_nla/data" -name manifest.jsonl | while read -r m; do
  echo "  manifest: ${m} ($(wc -l < "${m}") rows)"
done

echo "[5/6] pre-download models into the config cache"
export MEDICAL_NLA_DATA_ROOT="${DATA_ROOT}"
CACHE="${DATA_ROOT}/hf_cache/hub"
mkdir -p "${CACHE}"
python - <<EOF
from huggingface_hub import snapshot_download
for repo in ("google/gemma-3-12b-it", "kitft/nla-gemma3-12b-L32-av"):
    path = snapshot_download(repo, cache_dir="${CACHE}")
    print(f"[model] {repo} -> {path}")
EOF

echo "[6/7] generate pod config (default.yaml assumes two 24GB cards)"
python - <<'EOF'
import yaml
import torch

with open("configs/default.yaml", encoding="utf-8") as handle:
    config = yaml.safe_load(handle)

count = torch.cuda.device_count()
budgets = {}
for index in range(count):
    total_gib = torch.cuda.get_device_properties(index).total_memory / 2**30
    budgets[index] = f"{int(total_gib) - 4}GiB"
for section in ("source_model", "nla_model"):
    config[section]["max_memory"] = dict(budgets)

with open("configs/runpod.yaml", "w", encoding="utf-8") as handle:
    yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
print(f"[config] configs/runpod.yaml written for {count} GPU(s): {budgets}")
EOF

echo "[7/7] gpu check"
source scripts/env.sh "${DATA_ROOT}"
export PYTHONPATH="${REPO_DIR}"
CUDA_VISIBLE_DEVICES=0 python scripts/check_gpu_setup.py \
  --config configs/runpod.yaml --require-free-gb 40

echo "[done] next: bash scripts/run_ddxplus_d10_budget_runpod.sh"
