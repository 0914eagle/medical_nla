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

## Notes

SGLang and vLLM are intentionally not used by default. The pipeline uses pure `transformers`, extracts Gemma hidden states in a first pass, unloads Gemma, then loads the NLA AV model for generation. This avoids the SGLang Gemma-3 `input_embeds` wrapper/radix-cache pitfalls documented upstream, at the cost of lower throughput.

If outputs are mostly CJK text or describe the injection marker itself, treat that as a likely injection failure and check the sidecar, injection scale, prompt template, and token position first.
