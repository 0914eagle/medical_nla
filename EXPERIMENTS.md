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

### Source Multiple-Choice Baseline

This checks whether the source model can choose the canonical DDXPlus label when
the closed 49-label ontology is shown explicitly. It is the right control after
free generation proved to be mostly clinically nearby but not label-exact.

Run two shards in parallel on two A6000s:

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

CUDA_VISIBLE_DEVICES=0 python scripts/run_source_model_mc.py \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard0.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard0_summary.md \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 0 \
  --max-new-tokens 64 \
  --batch-size 4
' > /data1/heejae/medical_nla/logs/ddxplus_source_mc_shuffled_v1_shard0.log 2>&1 &
```

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

CUDA_VISIBLE_DEVICES=1 python scripts/run_source_model_mc.py \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard1_summary.md \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 1 \
  --max-new-tokens 64 \
  --batch-size 4
' > /data1/heejae/medical_nla/logs/ddxplus_source_mc_shuffled_v1_shard1.log 2>&1 &
```

After both shards finish:

```bash
python scripts/summarize_source_model_mc.py \
  --inputs \
    /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard0.jsonl \
    /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_shard1.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_source_mc_shuffled_v1_summary.md
```

### NLA Multiple-Choice Baseline

This mirrors the source-model MC baseline, but the closed diagnosis-label list
is shown to the injected AV. This tests whether vanilla NLA can use the same
constraint that lifted the source model from free generation to MC.

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

CUDA_VISIBLE_DEVICES=0 python scripts/run_nla_diagnosis_mc.py \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard0.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard0_summary.md \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 0 \
  --max-new-tokens 64
' > /data1/heejae/medical_nla/logs/ddxplus_nla_mc_shuffled_v1_shard0.log 2>&1 &
```

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

CUDA_VISIBLE_DEVICES=1 python scripts/run_nla_diagnosis_mc.py \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard1_summary.md \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 1 \
  --max-new-tokens 64
' > /data1/heejae/medical_nla/logs/ddxplus_nla_mc_shuffled_v1_shard1.log 2>&1 &
```

After both shards finish:

```bash
python scripts/summarize_nla_diagnosis_mc.py \
  --inputs \
    /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard0.jsonl \
    /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_shard1.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_nla_mc_shuffled_v1_summary.md
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

## 4. Medical-NLA v2-alpha Structured Readout

v1 trained the AV to emit a diagnosis sentence. v2-alpha keeps the same
DDXPlus multi-format activations and split discipline, but changes the target
to a structured readout:

```xml
<readout>
  <task_type>diagnosis</task_type>
  <answer>...</answer>
  <supporting_cues>...</supporting_cues>
</readout>
```

Create the leakage-safe v2-alpha SFT split:

```bash
python scripts/make_medical_nla_v2_sft_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest_multi_format.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_alpha \
  --variants multi_format \
  --max-cues 3 \
  --seed 17
```

Train the LoRA adapter:

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

CUDA_VISIBLE_DEVICES=0 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_alpha/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_alpha/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_v2_alpha_lora \
  --actor-prompt-template-file prompt_templates/medical_nla_v2_readout.txt \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-4 \
  --weight-decay 0.0 \
  --max-eval-rows 128
' > /data1/heejae/medical_nla/logs/medical_nla_ddxplus_v2_alpha_lora.log 2>&1 &
```

Generate structured readouts on the held-out test split:

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

CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_alpha/manifest_test.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_alpha_readouts_test.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_v2_alpha_lora \
  --actor-prompt-template-file prompt_templates/medical_nla_v2_readout.txt
' > /data1/heejae/medical_nla/logs/ddxplus_medical_nla_v2_alpha_readouts_test.log 2>&1 &
```

Score answer and supporting-cue readout quality:

```bash
python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_alpha_readouts_test.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_alpha_readouts_test_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_alpha_readouts_test_summary.md
```

### v2-beta: Clean DDXPlus Cue Rendering

v2-alpha exposed noisy DDXPlus cue rendering such as `... N`, `nowhere`, and
generic pain metadata. v2-beta regenerates the DDXPlus prompts with
`--clean-cues` enabled, then reruns activation extraction and the same
structured readout SFT.

Generate cleaned DDXPlus probe rows:

```bash
python scripts/make_ddxplus_probe_dataset.py \
  --patients /data1/heejae/ddxplus/train.csv \
  --evidences /data1/heejae/ddxplus/release_evidences.json \
  --cases-output /data1/heejae/medical_nla/ddxplus_probe_v2_cases.jsonl \
  --variants-output /data1/heejae/medical_nla/ddxplus_probe_v2_variants.jsonl \
  --examples-per-diagnosis 100 \
  --max-cues 3 \
  --seed 17 \
  --clean-cues
```

Extract cleaned activations:

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

CUDA_VISIBLE_DEVICES=0 python -m src.extract_activations \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/ddxplus_probe_v2_variants.jsonl \
  --run-name ddxplus_probe_v2
' > /data1/heejae/medical_nla/logs/ddxplus_probe_v2_extract.log 2>&1 &
```

Create the v2-beta structured SFT split:

```bash
python scripts/make_medical_nla_v2_sft_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v2/manifest.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_beta \
  --variants multi_format \
  --max-cues 3 \
  --seed 17
```

Train v2-beta:

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

CUDA_VISIBLE_DEVICES=0 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_beta/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_beta/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_v2_beta_lora \
  --actor-prompt-template-file prompt_templates/medical_nla_v2_readout.txt \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 8 \
  --lr 2e-4 \
  --weight-decay 0.0 \
  --max-eval-rows 128
' > /data1/heejae/medical_nla/logs/medical_nla_ddxplus_v2_beta_lora.log 2>&1 &
```

Generate and score v2-beta test readouts:

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

CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_v2_beta/manifest_test.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_beta_readouts_test.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_v2_beta_lora \
  --actor-prompt-template-file prompt_templates/medical_nla_v2_readout.txt
' > /data1/heejae/medical_nla/logs/ddxplus_medical_nla_v2_beta_readouts_test.log 2>&1 &
```

```bash
python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_beta_readouts_test.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_beta_readouts_test_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_v2_beta_readouts_test_summary.md
```

## 5. Medical-NLA SFT Splits

Build leakage-safe train/val/test files for diagnosis-preserving AV fine-tuning.
The split is grouped by `base_id` and stratified by `diagnosis_id`.

```bash
python scripts/make_medical_nla_sft_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_probe_v1/manifest.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1 \
  --variants multi_format \
  --style diagnosis_first \
  --train-frac 0.70 \
  --val-frac 0.15 \
  --seed 17
```

Inspect examples before training:

```bash
cat /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/summary.md
head -3 /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/sft_train.jsonl
```

## 6. Medical-NLA LoRA SFT

Trains a LoRA adapter on top of the released AV checkpoint.

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

CUDA_VISIBLE_DEVICES=0 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/sft_val.jsonl \
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
CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/manifest_test.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_test_v1.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1

CUDA_VISIBLE_DEVICES=0 python -m scripts.score_nla_diagnosis_logprobs \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/manifest_test.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_logprobs_test_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_multi_format_logprobs_test_v1_summary.md \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1
```

Evaluate Medical-NLA with the same MC constraint on the held-out test split:

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

CUDA_VISIBLE_DEVICES=0 python scripts/run_nla_diagnosis_mc.py \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/manifest_test.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard0.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard0_summary.md \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1 \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 0 \
  --max-new-tokens 64
' > /data1/heejae/medical_nla/logs/ddxplus_medical_nla_mc_shuffled_test_v1_shard0.log 2>&1 &
```

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

CUDA_VISIBLE_DEVICES=1 python scripts/run_nla_diagnosis_mc.py \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_sft_ddxplus_multi_format_v1/manifest_test.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard1_summary.md \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_ddxplus_lora_v1 \
  --shuffle-options \
  --seed 17 \
  --num-shards 2 \
  --shard-index 1 \
  --max-new-tokens 64
' > /data1/heejae/medical_nla/logs/ddxplus_medical_nla_mc_shuffled_test_v1_shard1.log 2>&1 &
```

```bash
python scripts/summarize_nla_diagnosis_mc.py \
  --inputs \
    /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard0.jsonl \
    /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_shard1.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_mc_shuffled_test_v1_summary.md
```

## 7. Error-Prediction Feature Table

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

## 8. Diagnosis-Heldout (True OOD) Medical-AV

Splits at the diagnosis-class level: train/heldout classes are disjoint, so
the adapter never sees heldout diagnosis names during SFT. `test_seen` keeps
in-distribution rows under the same adapter for a direct seen-vs-unseen
comparison. Use the same all-cue manifest and source MC answers that fed the
source-aligned v2 splits (substitute the actual all-cue source answers file).

```bash
python scripts/make_medical_nla_diagnosis_heldout_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_all_cue_format_v1/manifest.jsonl \
  --source-answers /data1/heejae/medical_nla/results/ddxplus_source_mc_all_cue_v1.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1 \
  --variants cue_count_all \
  --heldout-frac 0.30 \
  --seed 17
```

Train the LoRA adapter on train-class rows only:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_diagnosis_heldout_v1_lora_e3 \
  --epochs 3 \
  --batch-size 2
' > /data1/heejae/medical_nla/logs/medical_nla_diagnosis_heldout_v1_lora_e3.log 2>&1 &
```

Run readouts on both test manifests with the identical adapter:

```bash
for POOL in test_seen test_heldout; do
CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1/manifest_${POOL}.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_${POOL}.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_diagnosis_heldout_v1_lora_e3

python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_${POOL}.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_${POOL}_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_${POOL}_summary.md
done
```

Summarize seen-vs-heldout plus the classifier-collapse check:

```bash
python scripts/summarize_diagnosis_heldout_readouts.py \
  --heldout-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_test_heldout_scored.jsonl \
  --seen-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_test_seen_scored.jsonl \
  --split-dir /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1 \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_analysis.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_analysis_summary.md
```

Reading the result: heldout `answer_hit` low with `cue_recall` high means the
model reads cue semantics but does not generalize diagnosis names; low/low
suggests a seen-class classifier; high/high is a strong OOD readout. A high
`answer_in_train_vocab_rate` is direct evidence of classifier collapse.

## 9. Probe Disagreement Control for Error Prediction

Tests whether the source/Medical-AV disagreement signal (AUROC 0.9427) beats a
source/linear-probe disagreement built from the same activations. Requires
probe predictions on the same rows: retrain the all-cue probe with
`--write-predictions` if `*.predictions.jsonl` is missing. The feature table
now emits `source_probe_answer_agree`, and the evaluator reports
`source_probe_disagree`, `probe_low_top1_prob`, and a paired NLA-vs-probe
AUROC comparison restricted to rows where both signals exist.

```bash
python scripts/make_error_prediction_table.py \
  --source-answers /data1/heejae/medical_nla/results/ddxplus_source_mc_all_cue_v1.jsonl \
  --nla-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_all_cue_source_aligned_v2_readouts_test_e3_b2_scored.jsonl \
  --probe-predictions /data1/heejae/medical_nla/probe/ddxplus_all_cue_format_linear_probe_v1/cue_count_all.predictions.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/error_prediction_features_probe_control_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/error_prediction_features_probe_control_v1_summary.md

python scripts/evaluate_error_prediction.py \
  --input /data1/heejae/medical_nla/results/error_prediction_features_probe_control_v1.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/error_prediction_probe_control_v1.jsonl \
  --summary-md /data1/heejae/medical_nla/results/error_prediction_probe_control_v1_summary.md
```

If `nla_minus_probe_auroc` is near zero, the natural-language readout adds no
error-detection value over a linear probe; the case for NLA must then rest on
explanation quality (cues, trajectories), not on detection AUROC.

## 10. Medical-NLA v3 Cue-First OOD Run

Purpose: test whether the AV can read case-specific clinical content from the
activation, not classify diagnoses. Reuses the diagnosis-heldout split and its
activations unchanged; only the SFT target switches to cue-first, so results
are directly comparable to the v1 run (heldout answer_hit 0%, cue_recall 0.31
memorization level).

Default v3 targets contain no diagnosis text at all — `<observed>` cue list
only. A diagnosis-naming assessment sentence would reopen the label shortcut
(`--include-assessment` exists for later variants). Cue combinations are much
higher-entropy than diagnosis labels, but not memorization-proof: emitting the
nearest seen class's typical cues is the remaining escape, which the
precision/mismatched/counterfactual gates exist to catch.

Gate (all four needed for a "reads the activation" claim; 3-4 only built if
1-2 beat the memorization level):

```text
1. heldout cue recall        (reading quantity)
2. heldout cue precision     (anti cue-spraying)
3. mismatched-activation score drop, same-diagnosis hard negatives included
4. cue-removal counterfactual: removed cue disappears AND retained cues stay
   (diagnosis change excluded from the primary judgment)
```

Generate v3 targets from the existing split (CPU, seconds):

```bash
python scripts/make_medical_nla_v3_cue_first_targets.py \
  --split-dir /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1 \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v3_cue_first \
  --seed 17
```

Train (GPU, hours):

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v3_cue_first/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v3_cue_first/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_diagnosis_heldout_v3_cue_first_lora_e3 \
  --epochs 3 \
  --batch-size 2
' > /data1/heejae/medical_nla/logs/medical_nla_diagnosis_heldout_v3_cue_first_lora_e3.log 2>&1 &
```

Readout + scoring + seen-vs-heldout summary (GPU then CPU):

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for POOL in test_seen test_heldout; do
CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v3_cue_first/manifest_${POOL}.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_${POOL}.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_diagnosis_heldout_v3_cue_first_lora_e3

python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_${POOL}.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_${POOL}_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_${POOL}_summary.md
done

python scripts/summarize_diagnosis_heldout_readouts.py \
  --heldout-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_test_heldout_scored.jsonl \
  --seen-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_test_seen_scored.jsonl \
  --split-dir /data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v3_cue_first \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_analysis.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v3_analysis_summary.md
' > /data1/heejae/medical_nla/logs/medical_nla_diagnosis_heldout_v3_readouts.log 2>&1 &
```

Reading the result: v1 memorization baselines are heldout output_cue_recall
0.3066 and answer_in_train_vocab_rate 0.9875. v3 passes gates 1-2 when heldout
cue recall clearly exceeds ~0.31 with precision holding (not cue-spraying);
then build gates 3-4 (mismatched AR ranking, cue-removal counterfactuals).
