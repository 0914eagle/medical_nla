# medical_nla

OpenNLA-style two-pass inference pipeline for diagnosing how a general-domain NLA behaves on medical prompts.

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

## Required Manual Step

Fill these fields in `configs/default.yaml` after selecting the public OpenNLA checkpoint/repo:

- `nla_model.model_id`
- `nla_model.adapter_id`, if the checkpoint is a LoRA adapter
- `nla_model.placeholder_token`
- `nla_model.query_template`
- any normalization/projection behavior documented by the OpenNLA repo

Record the source repo findings in `NOTES.md` before running full experiments.

## Run

Pass 1 extracts Gemma activations and writes `.pt` files plus a manifest:

```bash
python -m src.extract_activations \
  --config configs/default.yaml \
  --input data/prompts_general.jsonl \
  --run-name pilot_general
```

Pass 2 injects saved activations into the NLA prompt via `inputs_embeds`:

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

SGLang and vLLM are intentionally not used. The pipeline uses pure `transformers`, extracts hidden states in a first pass, unloads Gemma, then loads the NLA model for generation.
