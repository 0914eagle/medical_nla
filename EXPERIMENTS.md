# Medical-NLA Next Experiments

This runbook assumes the GPU server layout:

```bash
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
```

## 1. AV Diagnosis Logprob Probe

Tests whether vanilla NLA assigns high probability to the correct diagnosis
even when free generation does not verbalize it.

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets

CUDA_VISIBLE_DEVICES=9 python -m scripts.score_nla_diagnosis_logprobs \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_v1_summary.md \
  --candidate-batch-size 8
' > /data1/heejae/medical_nla/logs/ddxplus_nla_multi_format_logprobs_v1.log 2>&1 &
```

## 2. Source Model Diagnosis Logprob Baseline

No-NLA confidence baseline for the same DDXPlus prompts.

First generate source answers and lexical correctness labels:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
unset HF_TOKEN
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python scripts/run_source_model_qa.py \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_answers_raw_b8_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_answers_raw_b8_v1_summary.md \
  --prompt-mode raw \
  --max-new-tokens 256 \
  --batch-size 8
' > /data1/heejae/medical_nla/logs/ddxplus_source_answers_raw_b8_v1.log 2>&1 &
```

Use the instructed sanity-check prompt separately:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
unset HF_TOKEN
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python scripts/run_source_model_qa.py \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_answers_diagnosis_first_b8_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_answers_diagnosis_first_b8_v1_summary.md \
  --prompt-mode diagnosis_first \
  --max-new-tokens 256 \
  --batch-size 8
' > /data1/heejae/medical_nla/logs/ddxplus_source_answers_diagnosis_first_b8_v1.log 2>&1 &
```

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets

CUDA_VISIBLE_DEVICES=8 python -m scripts.score_source_diagnosis_logprobs \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_v1_summary.md \
  --candidate-batch-size 8
' > /data1/heejae/medical_nla/logs/ddxplus_source_multi_format_logprobs_v1.log 2>&1 &
```

## 3. Probe With Row-Level Predictions

Adds saved probe weights and per-row predictions for error-prediction tables.

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla

CUDA_VISIBLE_DEVICES=9 python scripts/train_ddxplus_linear_probe.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest.jsonl \
  --out-dir /data1/heejae/medical_nla/probe/ddxplus_linear_probe_with_predictions_v1 \
  --epochs 80 \
  --batch-size 512 \
  --lr 1e-3 \
  --weight-decay 1e-2 \
  --write-predictions
' > /data1/heejae/medical_nla/logs/ddxplus_linear_probe_with_predictions_v1.log 2>&1 &
```

## 4. Medical-NLA SFT Dataset

Build target texts for diagnosis-preserving AV fine-tuning.

```bash
python scripts/make_medical_nla_sft_dataset.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1.jsonl \
  --variants multi_format \
  --style diagnosis_first
```

Inspect examples before training:

```bash
head -3 /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1.jsonl
```

## 5. Medical-NLA LoRA SFT

Trains a LoRA adapter on top of the released AV checkpoint.

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets

CUDA_VISIBLE_DEVICES=9 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-4 \
  --lora-r 16 \
  --lora-alpha 32
' > /data1/heejae/medical_nla/logs/medical_nla_ddxplus_lora_v1.log 2>&1 &
```

Evaluate the adapter with the same free-generation and logprob scripts:

```bash
CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_v1.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1

CUDA_VISIBLE_DEVICES=9 python -m scripts.score_nla_diagnosis_logprobs \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_logprobs_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_logprobs_v1_summary.md \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1
```

## 6. Error-Prediction Feature Table

Merges source correctness and NLA/probe features. All inputs must share the same
`base_id` namespace; do not mix the 50-case specificity-v2 files with DDXPlus
4900-case files.

```bash
python scripts/make_error_prediction_table.py \
  --source-answers /data1/heejae/medical_nla/results/ddxplus_source_answers_v1.jsonl \
  --source-logprobs /data1/heejae/medical_nla/results/ddxplus_source_multi_format_logprobs_v1.jsonl \
  --nla-scored /data1/heejae/medical_nla/results/ddxplus_nla_multi_format_v1_scored.jsonl \
  --nla-logprobs /data1/heejae/medical_nla/results/ddxplus_nla_multi_format_logprobs_v1.jsonl \
  --probe-predictions /data1/heejae/medical_nla/probe/ddxplus_linear_probe_with_predictions_v1/multi_format.predictions.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/error_prediction_features_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/error_prediction_features_v1_summary.md
```
