# medical_nla

Natural Language Autoencoder (NLA) two-pass inference pipeline for diagnosing how released Gemma-3 NLAs behave on medical prompts.

The code is intended to live at:

```bash
/home/eagle0914/medical_nla
```

All non-code artifacts are configured to live under:

```bash
/data1/heejae
```

That includes the uv environment, Hugging Face cache, downloaded models, extracted activations, logs, reports, and JSONL outputs.

## Server Setup

```bash
cd /home/eagle0914/medical_nla
uv venv /data1/heejae/uv/medical_nla --python 3.11
source /data1/heejae/uv/medical_nla/bin/activate
uv pip install -e ".[dev]"
```

Install the CUDA-compatible PyTorch build required by the server before or after this step if the default resolver does not select the right wheel.

## NLA Checkpoint

This repo is configured for the released Gemma-3-12B NLA checkpoints from
[kitft/natural_language_autoencoders](https://github.com/kitft/natural_language_autoencoders):

- AV: `kitft/nla-gemma3-12b-L32-av`
- AR: `kitft/nla-gemma3-12b-L32-ar`
- source model: `google/gemma-3-12b-it`
- extraction layer: `32`
- d_model: `3840`

The AV prompt template, injection token ids, and injection scale are loaded from the checkpoint's `nla_meta.yaml` sidecar at runtime. Do not hardcode or locally edit these values unless debugging a sidecar mismatch.

## Run

Pass 1 extracts Gemma activations and writes `.pt` files plus a manifest:

```bash
python -m src.extract_activations \
  --config configs/default.yaml \
  --input data/prompts_general.jsonl \
  --run-name pilot_general
```

Pass 2 injects saved activations into the AV prompt via `inputs_embeds`.
The code downloads `nla_meta.yaml`, uses the sidecar prompt exactly, tokenizes it with one-step `apply_chat_template(..., tokenize=True)`, rescales the activation to the sidecar `injection_scale`, verifies injection-token neighbors, and parses `<explanation>...</explanation>`:

```bash
python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/pilot_general/manifest.jsonl \
  --output /data1/heejae/medical_nla/results/pilot_general.jsonl
```

Generate a readable markdown report:

```bash
python -m src.report \
  --input /data1/heejae/medical_nla/results/pilot_general.jsonl \
  --output /data1/heejae/medical_nla/reports/pilot_general.md
```

Run tests:

```bash
pytest
```

## AR Reconstruction Scoring

Install the lightweight AR helper on the GPU server:

```bash
source /data1/heejae/uv/medical_nla/bin/activate
cd /data1/heejae
git clone https://github.com/kitft/nla-inference
cd nla-inference
pip install -e .
```

If editable install is unavailable, the scoring script can still import the standalone file with `--nla-inference-path /data1/heejae/nla-inference`.

Then score AV outputs against the original activation vectors:

```bash
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate

CUDA_VISIBLE_DEVICES=9 python score_reconstruction_mse.py \
  --inputs /data1/heejae/medical_nla/results/pilot_medical_v3.jsonl \
           /data1/heejae/medical_nla/results/pilot_general_v3.jsonl \
  --ar kitft/nla-gemma3-12b-L32-ar \
  --out /data1/heejae/medical_nla/scored/v3 \
  --nla-inference-path /data1/heejae/nla-inference \
  --high-norm-threshold 120000
```

Outputs:

- `scored.jsonl`
- `mse_vs_confab.png`
- `summary.md`

The original `12000` high-norm cutoff is a conservative warning threshold from non-Gemma examples. Gemma-3-12B L32 activations are often around the sidecar `injection_scale=80000`, so use a Gemma-appropriate threshold such as `120000` for summary statistics unless a norm histogram says otherwise.

## Prompt-Sensitivity Probe

To test whether AV outputs are format-driven by the default prompt, rerun verbalization against the same saved activations with a medical-content-focused actor prompt suffix. First inspect the sidecar default prompt:

```bash
python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/pilot_medical_v3/manifest.jsonl \
  --output /tmp/unused.jsonl \
  --dump-actor-prompt-template
```

The safest probe preserves the sidecar default prompt and appends a medical instruction, so the `{injection_char}` neighborhood from `nla_meta.yaml` is unchanged:

```bash
CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/pilot_medical_v3/manifest.jsonl \
  --output /data1/heejae/medical_nla/results/pilot_medical_v3_medprompt.jsonl \
  --actor-prompt-suffix-file prompts/medical_actor_prompt_suffix.txt
```

A full replacement template is also supported with `--actor-prompt-template-file`, but it must preserve `{injection_char}` in the same token neighborhood expected by `nla_meta.yaml`; otherwise the neighbor check fails loudly.

Compare `pilot_medical_v3.jsonl` with `pilot_medical_v3_medprompt.jsonl` for the same `id`s. This does not require rerunning Gemma activation extraction.

## Entity-Position Probe

The v3 run used `position_mode=last_token`, which often captures the assistant-answer boundary or response-format state. To test content-bearing positions, use `data/prompts_medical_entities.jsonl`. Each row has a `target_text` substring such as `warfarin`, `ST elevations`, or `beta-lactam antibiotics`.

Substring mapping rule:

- find the target substring in the chat-templated prompt text, case-insensitive
- use tokenizer `offset_mapping` to find all subword tokens overlapping that character span
- default `target_text_strategy=last_subtoken`
- alternatives: `first_subtoken`, `span`, `span_mean`

Run extraction and AV on entity positions:

```bash
python scripts/make_entity_position_variants.py \
  --input data/prompts_medical_entities.jsonl \
  --output data/prompts_medical_position_variants.jsonl

CUDA_VISIBLE_DEVICES=8 python -m src.extract_activations \
  --config configs/default.yaml \
  --input data/prompts_medical_position_variants.jsonl \
  --run-name pilot_medical_position_variants_v1

CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/pilot_medical_position_variants_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/results/pilot_medical_position_variants_v1.jsonl
```

This creates 200 rows: 50 format last-token controls plus 50 each for `first_subtoken`, `last_subtoken`, and `span_mean`. Compare by `base_id`: format outputs should be format-driven; entity-position outputs should be more content-driven if the NLA can read the relevant medical activation.

## Non-Diagnostic Position Baseline

To control for oracle entity tagging, run the same `target_text` extraction on non-diagnostic tokens such as `patient`, `woman`, `Describe`, or `Explain`. This tests whether medical content appears because the selected entity truly carries content, or because any token in the prompt already carries the full diagnosis.

```bash
CUDA_VISIBLE_DEVICES=8 python -m src.extract_activations \
  --config configs/default.yaml \
  --input data/prompts_medical_nondiagnostic_entities.jsonl \
  --run-name pilot_medical_nondiagnostic_v1

CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/pilot_medical_nondiagnostic_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/results/pilot_medical_nondiagnostic_v1.jsonl
```

Expected control: diagnostic `entity_span_mean` should recover target medical content far more often than this non-diagnostic baseline.

## Notes

SGLang and vLLM are intentionally not used by default. The pipeline uses pure `transformers`, extracts Gemma hidden states in a first pass, unloads Gemma, then loads the NLA AV model for generation. This avoids the SGLang Gemma-3 `input_embeds` wrapper/radix-cache pitfalls documented upstream, at the cost of lower throughput.

If outputs are mostly CJK text or describe the injection marker itself, treat that as a likely injection failure and check the sidecar, injection scale, prompt template, and token position first.
