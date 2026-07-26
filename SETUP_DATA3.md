# New A6000 Server Setup (`/data3/heejae`)

Use git for code and `/data3/heejae` for environments, Hugging Face cache,
datasets, activations, logs, adapters, and results.

## 0. Clone Code

```bash
mkdir -p /data3/heejae/{uv,hf_cache,medical_nla,ddxplus}
mkdir -p /data3/heejae/medical_nla/{activations,results,reports,logs,probe,train,adapters}

cd /home/eagle0914
git clone https://github.com/0914eagle/medical_nla.git
cd /home/eagle0914/medical_nla
```

If the repo already exists:

```bash
cd /home/eagle0914/medical_nla
git pull
```

## 1. Create uv Environment Under `/data3`

```bash
cd /home/eagle0914/medical_nla
uv venv /data3/heejae/uv/medical_nla --python 3.11
source /data3/heejae/uv/medical_nla/bin/activate
uv pip install -e ".[dev]"
uv pip install orjson
```

`orjson` is needed by `kitft/nla-inference` if AR reconstruction scoring is used.

## 2. Environment Variables

Minimum variables for this repo:

```bash
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data3/heejae/hf_cache
export TRANSFORMERS_CACHE=/data3/heejae/hf_cache
export HF_DATASETS_CACHE=/data3/heejae/hf_cache/datasets
```

Usually also needed:

```bash
export HF_TOKEN=hf_...
```

Use `HF_TOKEN` or `hf auth login` because Gemma and NLA checkpoints may be gated
or rate-limited.

Optional CUDA memory fragmentation guard:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For one A6000, keep the config as `device_map: cuda` and choose a visible GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

## 3. Hugging Face Login and DDXPlus Download

```bash
source /data3/heejae/uv/medical_nla/bin/activate
export HF_HOME=/data3/heejae/hf_cache
hf auth login

hf download aai530-group6/ddxplus \
  --repo-type dataset \
  --local-dir /data3/heejae/ddxplus
```

Expected files:

```bash
find /data3/heejae/ddxplus -maxdepth 2 -type f | head
```

You should see:

```text
train.csv
test.csv
validate.csv
release_evidences.json
release_conditions.json
README.md
```

## 4. Sanity Checks

```bash
cd /home/eagle0914/medical_nla
source /data3/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data3/heejae/hf_cache
export TRANSFORMERS_CACHE=/data3/heejae/hf_cache
export HF_DATASETS_CACHE=/data3/heejae/hf_cache/datasets

python -m pytest

python scripts/inspect_ddxplus_schema.py \
  --patients /data3/heejae/ddxplus/train.csv \
  --evidences /data3/heejae/ddxplus/release_evidences.json \
  --limit 3
```

## 5. Rebuild DDXPlus Probe Inputs

If you want to run the whole pipeline before sleeping, skip to
[`One-command overnight run`](#one-command-overnight-run). Otherwise run the
steps below one by one.

```bash
python scripts/make_ddxplus_probe_dataset.py \
  --patients /data3/heejae/ddxplus/train.csv \
  --evidences /data3/heejae/ddxplus/release_evidences.json \
  --cases-output /data3/heejae/medical_nla/ddxplus_probe_v1_cases.jsonl \
  --variants-output /data3/heejae/medical_nla/ddxplus_probe_v1_variants.jsonl \
  --examples-per-diagnosis 100 \
  --max-cues 3 \
  --seed 17
```

Expected:

```bash
wc -l /data3/heejae/medical_nla/ddxplus_probe_v1_cases.jsonl
# 4900
wc -l /data3/heejae/medical_nla/ddxplus_probe_v1_variants.jsonl
# 29400
```

## 6. Extract Activations

This recreates the 29,400 DDXPlus probe activations on the new server.

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data3/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data3/heejae/hf_cache
export TRANSFORMERS_CACHE=/data3/heejae/hf_cache
export HF_DATASETS_CACHE=/data3/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m src.extract_activations \
  --config configs/data3.yaml \
  --input /data3/heejae/medical_nla/ddxplus_probe_v1_variants.jsonl \
  --run-name ddxplus_probe_v1
' > /data3/heejae/medical_nla/logs/ddxplus_probe_v1_extract.log 2>&1 &
```

Check:

```bash
tail -f /data3/heejae/medical_nla/logs/ddxplus_probe_v1_extract.log
wc -l /data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest.jsonl
```

## 7. Make `multi_format` Manifest

```bash
python - <<'PY'
import json
src="/data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest.jsonl"
dst="/data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl"
with open(src) as f, open(dst, "w") as out:
    for line in f:
        r=json.loads(line)
        if r.get("variant") == "multi_format":
            out.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
PY

wc -l /data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl
# 4900
```

## 8. First Priority Experiment: NLA Diagnosis Logprob

Smoke test:

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.score_nla_diagnosis_logprobs \
  --config configs/data3.yaml \
  --manifest /data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data3/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_calibrated_smoke.jsonl \
  --summary-md /data3/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_calibrated_smoke_summary.md \
  --limit 20 \
  --candidate-batch-size 8 \
  --rank-field calibrated_logprob_mean \
  --calibrate-with-token-baseline
```

Full run:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data3/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data3/heejae/hf_cache
export TRANSFORMERS_CACHE=/data3/heejae/hf_cache
export HF_DATASETS_CACHE=/data3/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m scripts.score_nla_diagnosis_logprobs \
  --config configs/data3.yaml \
  --manifest /data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data3/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_calibrated_v1.jsonl \
  --summary-md /data3/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_calibrated_v1_summary.md \
  --candidate-batch-size 8 \
  --rank-field calibrated_logprob_mean \
  --calibrate-with-token-baseline
' > /data3/heejae/medical_nla/logs/ddxplus_nla_multi_format_logprobs_calibrated_v1.log 2>&1 &
```

Check:

```bash
tail -f /data3/heejae/medical_nla/logs/ddxplus_nla_multi_format_logprobs_calibrated_v1.log
cat /data3/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_calibrated_v1_summary.md
```

## 8b. Source Model Calibrated Diagnosis Logprob Baseline

This checks whether the DDXPlus diagnosis candidate surface forms are usable
for the base Gemma model itself. The neutral calibration prompt removes
candidate-name priors, similar to the NLA calibrated logprob run.

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data3/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data3/heejae/hf_cache
export TRANSFORMERS_CACHE=/data3/heejae/hf_cache
export HF_DATASETS_CACHE=/data3/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m scripts.score_source_diagnosis_logprobs \
  --config configs/data3.yaml \
  --input /data3/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data3/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_calibrated_v1.jsonl \
  --summary-md /data3/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_calibrated_v1_summary.md \
  --candidate-batch-size 8 \
  --rank-field calibrated_logprob_mean \
  --calibration-prompt "A patient presents with symptoms. What diagnosis is most likely?"
' > /data3/heejae/medical_nla/logs/ddxplus_source_multi_format_logprobs_calibrated_v1.log 2>&1 &
```

Check:

```bash
tail -f /data3/heejae/medical_nla/logs/ddxplus_source_multi_format_logprobs_calibrated_v1.log
cat /data3/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_calibrated_v1_summary.md
```

## One-command Overnight Run

This performs:

1. DDXPlus probe row generation
2. Gemma activation extraction
3. `multi_format` manifest filtering
4. vanilla NLA diagnosis logprob scoring

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
git pull
GPU=0 bash scripts/run_data3_ddxplus_logprob_pipeline.sh
' > /data3/heejae/medical_nla/logs/data3_ddxplus_logprob_pipeline.log 2>&1 &
```

Check progress:

```bash
tail -f /data3/heejae/medical_nla/logs/data3_ddxplus_logprob_pipeline.log
```

If a partial artifact exists and you want to force regeneration:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
git pull
FORCE=1 GPU=0 bash scripts/run_data3_ddxplus_logprob_pipeline.sh
' > /data3/heejae/medical_nla/logs/data3_ddxplus_logprob_pipeline_force.log 2>&1 &
```

## 9. Optional: Copy Existing Activations Instead of Rebuilding

If the old server is reachable and paths are accessible, copy artifacts with
`rsync` instead of re-extracting:

```bash
rsync -avP OLD_USER@OLD_HOST:/data1/heejae/medical_nla/activations/ddxplus_probe_v1/ \
  /data3/heejae/medical_nla/activations/ddxplus_probe_v1/

rsync -avP OLD_USER@OLD_HOST:/data1/heejae/medical_nla/results/ \
  /data3/heejae/medical_nla/results/
```

Do not put activations, checkpoints, caches, uv envs, or results into git.
