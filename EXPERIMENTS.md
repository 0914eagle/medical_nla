# Medical-NLA Next Experiments

## Server setup

Two server layouts are in use. Pick the matching config and env block; the
commands below are written against the older `/data1` layout, so on the
`/data` server substitute the config name (`configs/default.yaml` ->
`configs/data.yaml`, `configs/layer24.yaml` -> `configs/data_layer24.yaml`)
and the paths.

### Current server (`/data/heejae`, configs `data*.yaml`)

First time only:

```bash
cd ~ && git clone https://github.com/0914eagle/medical_nla.git
mkdir -p /data/heejae/{uv,hf_cache/datasets,medical_nla/{activations,results,reports,data}}
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv venv /data/heejae/uv/medical_nla --python 3.11
source /data/heejae/uv/medical_nla/bin/activate
cd ~/medical_nla && uv pip install -e ".[dev]"
```

Hugging Face login (first time on a new `HF_HOME`): the backbone is a gated
repo, and the credential lives in `$HF_HOME/token`, so moving `HF_HOME` to a
new disk means logging in again. Use the account that already accepted the
Gemma license.

```bash
export HF_HOME=/data/heejae/hf_cache
huggingface-cli login          # newer clients: hf auth login
huggingface-cli whoami         # verify
```

Every session:

```bash
cd /home/eagle0914/medical_nla
source /data/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data/heejae/hf_cache
export TRANSFORMERS_CACHE=/data/heejae/hf_cache
export HF_DATASETS_CACHE=/data/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Note the older blocks below `unset HF_TOKEN` so the stored token file is used
rather than a stale environment variable; keep that unless you deliberately
authenticate through `HF_TOKEN`.

The backbone spans both GPUs on this server (`device_map: auto` in the
`data*.yaml` configs), so extraction and readout runs take
`CUDA_VISIBLE_DEVICES=2,3` rather than a single device.

### Moving servers: pull the corpora instead of rebuilding them

Rebuilding DDXPlus rescans a million-row CSV and replays every cue-rendering
decision; MedCaseReasoning re-segments 12k case reports. Neither is expensive in
compute, but both are expensive in decisions, and a move should not be a chance
to silently pick different ones. Publish once, pull thereafter.

```bash
python scripts/push_datasets_to_hub.py \
  --repo-id <account>/medical-nla-cases \
  --files /data/heejae/medical_nla/data/ddxplus_cue_count_cases.jsonl \
          /data/heejae/medical_nla/data/mcr_cases_train.jsonl \
          /data/heejae/medical_nla/data/mcr_cases_test.jsonl
```

Private unless `--no-private` is passed: publishing redistributes derivatives of
DDXPlus and MedCaseReasoning, so read the attribution in the generated card
first. `--card-only` prints that card without uploading. The card is built from
the artifacts, not the command line — the generator flags that shaped the corpus
(`clean_cues`, `negative_cues`, `prefer_symptoms`) are recorded on every case
row and reported from there, so a downloaded corpus says what it is.

On the new machine:

```bash
export HF_HOME=/data/heejae/hf_cache
python -c "
from huggingface_hub import snapshot_download
snapshot_download('<account>/medical-nla-cases', repo_type='dataset',
                  local_dir='/data/heejae/medical_nla/data')
"
```

Activations are deliberately not published. They are a function of the prompt,
the backbone and the layer, so they are only worth freezing once the prompt is,
and being derived from Gemma they carry that model's terms in a way these text
artifacts do not.

### Previous server (`/data1/heejae`, configs `default.yaml` / `layer*.yaml`)

```bash
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
```

## 0. MedCaseReasoning ingestion

Real case-report text, so the same cue-position pipeline applies to
non-synthetic clinical prose. Download lands in `$HF_HOME`.

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('zou-lab/MedCaseReasoning')
print(ds)
for split in ds:
    ds[split].to_json(f'/data/heejae/medical_nla/data/mcr_{split}.jsonl')
"
```

### Why cues are clause spans, not clinician quotes

The first ingestion (`scripts/make_medcasereasoning_cases.py`) treated the
quoted spans inside `diagnostic_reasoning` as evidence cues. Measurement
refuted that assumption: `quote_match_rate` came back **0.0164**, and
`scripts/inspect_medcasereasoning_quotes.py` localized why — 1.7% of quotes
resolve in `case_prompt`, 42.7% resolve only in the full article (they quote
the discussion, not the presentation), and 56.9% resolve nowhere, being the
reasoning's own paraphrase. MedCaseReasoning does not annotate which part of
the presentation is evidence.

It does not need to. Readout scoring compares the readout against the text
of the span whose activation was injected, so a cue only has to be
well-defined clinical text — not a human judgement of diagnostic relevance.
So cues are cut out of `case_prompt` itself, which makes every span exact by
construction (`unresolved_spans: 0`). The quote-based script is kept only for
the diagnostic it supports; do not use it to build cases.

```bash
python scripts/make_clinical_span_cases.py \
  --input /data/heejae/medical_nla/data/mcr_train.jsonl \
  --output /data/heejae/medical_nla/data/mcr_cases_train.jsonl \
  --report /data/heejae/medical_nla/reports/mcr_ingest_train.json \
  --min-cues 3 --min-words 4 --max-words 14
```

`--max-words 14` is not cosmetic: at the default 25 the mean cue ran 13.89
words, far longer than DDXPlus cues (~5-10), which would have confounded any
DDXPlus-vs-MCR comparison with cue length. At 14 the mean is 8.97 words.

Then the cue-position generator applies unchanged, while prose counterfactuals
need span substitution rather than prompt reconstruction:

```bash
python scripts/make_ddxplus_cue_position_rows.py \
  --input /data/heejae/medical_nla/data/mcr_cases_train.jsonl \
  --output /data/heejae/medical_nla/data/mcr_cuepos_train.jsonl \
  --variants cue_count_all --max-cues-per-case 4

python scripts/make_span_counterfactual_rows.py \
  --cases /data/heejae/medical_nla/data/mcr_cases_test.jsonl \
  --output /data/heejae/medical_nla/data/mcr_cf_test.jsonl \
  --num-cases 500 --swap-slots 2 \
  --report /data/heejae/medical_nla/reports/mcr_cf.json
```

### Splits: unseen cues come for free here

DDXPlus draws cues from a fixed questionnaire, so an unseen-cue pool has to
be manufactured by holding cue strings out of training. Case-report prose is
the opposite — splitting by case already leaves most evaluation cues unseen,
so the split is measured rather than imposed:

```bash
python scripts/make_prose_cue_position_splits.py \
  --manifest /data/heejae/medical_nla/activations/mcr_cuepos_train_L24/manifest.jsonl \
  --out-dir /data/heejae/medical_nla/train/mcr_cue_position_L24 \
  --seed 17
```

Measured on the 15,864-row L24 train extraction (4,000 cases): train 11,108 /
val 1,589 / test_seen_cue 129 / test_heldout_cue 3,038, i.e.
`unseen_cue_rate_in_test = 0.9593`. Two properties of this corpus carry
argument weight and should be quoted in the paper: 96% of test cues are
naturally unseen, and the 12,766 cases carry 6,934 distinct diagnosis labels
— which is why the 26-way likelihood and the linear probe, both defined over
a closed label set, have no counterpart here.

## 0b. DDXPlus rebuild on the new server

The `/data1` disk holding the original DDXPlus CSVs is not mounted on the
current server, so the corpus is re-downloaded from Hugging Face rather than
copied. `aai530-group6/ddxplus` carries the same patient rows and evidence
dictionary the pilot used.

```bash
python -c "
from datasets import load_dataset
ds = load_dataset('aai530-group6/ddxplus')
ds['train'].to_csv('/data/heejae/ddxplus/train.csv')
"
```

Case files, both experiments' inputs, at 100 cases per diagnosis:

```bash
python scripts/make_ddxplus_probe_dataset.py \
  --patients /data/heejae/ddxplus/train.csv \
  --evidences /data/heejae/ddxplus/release_evidences.json \
  --cases-output /data/heejae/medical_nla/data/ddxplus_probe_cases.jsonl \
  --variants-output /data/heejae/medical_nla/data/ddxplus_probe_variants.jsonl \
  --examples-per-diagnosis 100 --max-cues 3 --seed 17 --clean-cues

python scripts/make_ddxplus_cue_count_cases.py \
  --patients /data/heejae/ddxplus/train.csv \
  --evidences /data/heejae/ddxplus/release_evidences.json \
  --output /data/heejae/medical_nla/data/ddxplus_cue_count_cases.jsonl \
  --examples-per-diagnosis 100 --cue-counts all --seed 17 \
  --negative-cues --no-prefer-symptoms --stop-when-full
```

Two flags decide what reaches the prompt, and both are recorded per case so the
choice is never silent:

- `--negative-cues` keeps negatively-answered items, rendered by negating the
  question's auxiliary. Without it prompts carry positive findings only, which
  drops about 10.6% of evidence entries — roughly two per patient.
- `--no-prefer-symptoms` keeps antecedents. The default drops every antecedent
  whenever the case has any symptom, and antecedents are where most negative
  answers live (travel, family history, smoking, prior diagnoses), so leaving it
  on reduced negative cues to 0.35% of the corpus even with `--negative-cues` set.

`--stop-when-full` ends the scan once every diagnosis has its quota, which in
practice happens around row 400,000 of the million-row CSV; the remaining rows
are rendered and discarded. Output is identical whenever the corpus holds
exactly `--max-diagnoses` diagnoses, which DDXPlus does (49).

Both reproduce the pilot's 4,900 cases (49 diagnoses × 100). Note that the
cue-count generator's input is the *patient* CSV; the probe experiment's
`ddxplus_variants.jsonl` is a different artifact and has no `cue_count_all`
variant, so it cannot stand in here.

### Cue rendering must be audited before anything is extracted

DDXPlus stores `(question id, answer value)`, so the English cue is built here,
and building it wrong is silent. Two rounds of measurement found that most
renderings were malformed and that negatives were being *inverted* rather than
dropped, which put false statements into pilot prompts. Sampling prompts does
not catch this. Two exhaustive passes do, and they are what
`scripts/audit_ddxplus_cue_rendering.py` runs:

- the questionnaire is a finite vocabulary, so every `(question, value)` pair is
  rendered and checked, which covers every prompt at the cue level;
- every case is checked for what only assembly can break — a cue that is not
  verbatim in its own prompt, a duplicate, a runaway cue count.

```bash
python scripts/audit_ddxplus_cue_rendering.py \
  --evidences /data/heejae/ddxplus/release_evidences.json \
  --cases /data/heejae/medical_nla/data/ddxplus_cue_count_cases.jsonl \
  --dump /data/heejae/medical_nla/reports/ddxplus_cue_vocabulary.tsv \
  --show-longest 3
```

It exits non-zero on a hard violation, so it can gate a rebuild. The TSV dump
is the full rendered vocabulary and is the source for the paper's data-processing
appendix. Soft flags (leftover question marks, surviving second person) are
reported for judgement, not failed; questions the rules cannot reach go in
`CUE_PHRASE_OVERRIDES` with both polarities written out.

Cue-position rows come from the cue-count case file:

```bash
python scripts/make_ddxplus_cue_position_rows.py \
  --input /data/heejae/medical_nla/data/ddxplus_cue_count_cases.jsonl \
  --output /data/heejae/medical_nla/data/ddxplus_cuepos_rows.jsonl \
  --variants cue_count_all --max-cues-per-case 4 --seed 17
```

This yields **16,410 rows from 4,900 cases, 0 cues unresolved** — against the
pilot's 12,800. The difference is coverage, not construction: the pilot
exploded a 3,200-case subset (the earlier all-cue format manifest), while
this run covers the full 4,900. Same per-case rule (up to 4 cues, seed 17,
`last_subtoken`), so the pilot's rows are a subset of this distribution and
the two remain comparable; the extra cases just make the heldout-cue pool
larger. The shortfall from 4,900 × 4 = 19,600 is cases carrying fewer than
four usable cues after cue cleaning.

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

## 11. v4 Cue-Position Positive Control

Purpose: decide whether the v3 failure was positional (detail compressed
away at the format position) or mechanistic (single-vector NLA readout
cannot carry case-specific detail from anywhere). One activation per
(case, cue) at the cue's own token span, where its information is
guaranteed present. The OOD unit is the cue STRING (cues are shared
across diagnoses, so diagnosis-heldout would not keep cue text unseen).

Generate per-cue extraction rows from the existing all-cue manifest
(prompt + cue_targets are carried in it):

```bash
python scripts/make_ddxplus_cue_position_rows.py \
  --input /data1/heejae/medical_nla/activations/ddxplus_all_cue_format_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/activations/ddxplus_cue_position_v1_rows.jsonl \
  --max-cues-per-case 4 \
  --seed 17
```

Extract activations at cue spans (GPU pass 1; layer comes from the config):

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0 python -m src.extract_activations \
  --config configs/default.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_cue_position_v1_rows.jsonl \
  --run-name ddxplus_cue_position_v1
' > /data1/heejae/medical_nla/logs/ddxplus_cue_position_v1_extract.log 2>&1 &
```

Cue-heldout splits (25% of cue strings never supervised) + single-cue targets:

```bash
python scripts/make_medical_nla_v4_cue_position_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_cue_position_v1/manifest.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_cue_position_v4 \
  --heldout-cue-frac 0.25 \
  --seed 17
```

Train (same trainer; the train pool is larger than v3, expect a longer run):

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
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_cue_position_v4/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_cue_position_v4/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_cue_position_v4_lora_e3 \
  --epochs 3 \
  --batch-size 2
' > /data1/heejae/medical_nla/logs/medical_nla_cue_position_v4_lora_e3.log 2>&1 &
```

Readout + scoring + summary:

```bash
nohup bash -lc '
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for POOL in test_seen_cue test_heldout_cue; do
CUDA_VISIBLE_DEVICES=0 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/train/medical_nla_cue_position_v4/manifest_${POOL}.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_${POOL}.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_cue_position_v4_lora_e3

python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_${POOL}.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_${POOL}_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_${POOL}_summary.md
done

python scripts/summarize_cue_position_readouts.py \
  --seen-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_test_seen_cue_scored.jsonl \
  --heldout-scored /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_test_heldout_cue_scored.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_analysis.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_medical_nla_cue_position_v4_analysis_summary.md
' > /data1/heejae/medical_nla/logs/medical_nla_cue_position_v4_readouts.log 2>&1 &
```

Reading the result (test_heldout_cue read_rate is the decision number):

```text
high (>~0.6): the AV can verbalize unseen case-specific detail from a
  position that has it -> the v3 failure was positional; proceed to the
  positional/layer-wise map (which cue positions/layers keep detail).
low while seen-cue is high: seen-cue reading was memorization; the
  single-vector readout mechanism is the bottleneck -> rethink injection
  (multi-token/span) before more position hunting.
```

## 12. Layer Sweep at Cue Positions (v5)

Purpose: with the cue-position reader validated at layer 32 (semantic read
55.7% on unseen cues), map where cue detail lives across depth. Same
recipe per layer; only the extraction layer changes
(`configs/layer{8,16,24}.yaml`). The NLA checkpoint stays the L32-AV —
each layer gets its own LoRA, realizing the shared-decoder +
per-layer-adapter architecture; whether LoRA absorbs the cross-layer
subspace shift is part of the question. Norms are handled by the sidecar
injection-scale rescaling in run_nla.

Recommended order: L16 first (max contrast with 32), then L24, then L8.
Same rows file and seed 17 keep the cue/case splits identical across
layers, so results are directly comparable.

Per layer L (substitute L=16 etc.; each stage mirrors section 11):

```bash
# 1) extraction at layer L (GPU)
CUDA_VISIBLE_DEVICES=9 python -m src.extract_activations \
  --config configs/layer16.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_cue_position_v1_rows.jsonl \
  --run-name ddxplus_cue_position_L16_v1

# 2) splits (CPU; identical assignment to v4 by construction)
python scripts/make_medical_nla_v4_cue_position_splits.py \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_cue_position_L16_v1/manifest.jsonl \
  --out-dir /data1/heejae/medical_nla/train/medical_nla_cue_position_L16_v5 \
  --heldout-cue-frac 0.25 \
  --seed 17

# 3) train per-layer LoRA (GPU)
CUDA_VISIBLE_DEVICES=9 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl /data1/heejae/medical_nla/train/medical_nla_cue_position_L16_v5/sft_train.jsonl \
  --val-jsonl /data1/heejae/medical_nla/train/medical_nla_cue_position_L16_v5/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_cue_position_L16_v5_lora_e2 \
  --epochs 2 \
  --batch-size 2

# 4) readout + score + summarize (GPU then CPU) — as in section 11 with
#    L16_v5 paths and the L16 adapter; wrap stages in the usual nohup env.
```

Cross-layer reading: compare test_heldout_cue strict/soft/hand-labeled
read rates per layer against the L32 baseline (0.178 strict / 0.557
hand-labeled A+B). Earlier layers winning on detail supports
"pre-integration layers preserve evidence"; earlier layers failing
entirely bounds where the LoRA can bridge the subspace shift.

## 13. Cue Counterfactual Faithfulness (L24 reader)

The last gate for "the readout follows the activation": swap a cue and
the swapped span's readout must track the NEW content (still emitting the
old cue = reading case context, not the span); remove a cue and retained
spans must stay correct with no phantom mention of the removed cue.
Prompts are rebuilt from the cue list and verified against the stored
prompt, so counterfactuals are construction-exact. Runs on the L24
reader (the best operating point).

```bash
nohup bash -lc '
set -e
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/make_cue_counterfactual_rows.py \
  --cases /data1/heejae/medical_nla/activations/ddxplus_all_cue_format_v1/manifest.jsonl \
  --split-dir /data1/heejae/medical_nla/train/medical_nla_cue_position_L24_v5 \
  --output /data1/heejae/medical_nla/activations/ddxplus_cue_counterfactual_v1_rows.jsonl \
  --num-cases 150 \
  --seed 17

CUDA_VISIBLE_DEVICES=9 python -m src.extract_activations \
  --config configs/layer24.yaml \
  --input /data1/heejae/medical_nla/activations/ddxplus_cue_counterfactual_v1_rows.jsonl \
  --run-name ddxplus_cue_counterfactual_L24_v1

CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest /data1/heejae/medical_nla/activations/ddxplus_cue_counterfactual_L24_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_cue_position_L24_v5_lora_e2

python scripts/score_medical_nla_v2_readouts.py \
  --input /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1_scored.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1_scored_summary.md

python scripts/evaluate_cue_counterfactuals.py \
  --scored /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1_scored.jsonl \
  --output-jsonl /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1_eval.jsonl \
  --summary-md /data1/heejae/medical_nla/results/ddxplus_cue_counterfactual_L24_v1_eval_summary.md
' > /data1/heejae/medical_nla/logs/ddxplus_cue_counterfactual_L24_v1.log 2>&1 &
```

Faithful-reader signature:

```text
swap_reads_replacement        high  (readout tracks the span content)
swap_still_reads_original     ~0    (no case-context memorization)
retained read rates           stable across orig/swap/removed
phantom_rate_removed_cue      ~0
```

Note: the swap replacement cue is drawn from the whole corpus vocabulary;
rows record cf_original_cue/cf_replacement_cue so the soft-matching
family-overlap caveat can be checked by hand on the examples table.

## 14. Format-Position Layer Sweep (trajectory: where the conclusion forms)

The other half of the trajectory map. The cue-position sweep (section 11)
showed individual cue detail peaks at L24 and fades by L32. This runs the
SAME v3 diagnosis-heldout / cue-first recipe at the FORMAT position (the
prompt's final token, the integrated answer state) across layers, so the
two curves overlay: as depth increases, cue detail should fall while the
format position's integrated content behaves differently — localizing
where evidence folds into the conclusion. L32 format is already the v3
result (heldout cue_recall 0.19); this adds L16 and L24.

Step 0 — format-position rows (once):

```bash
python scripts/make_format_position_rows.py \
  --input /data1/heejae/medical_nla/activations/ddxplus_all_cue_format_v1/manifest.jsonl \
  --output /data1/heejae/medical_nla/activations/ddxplus_format_position_rows.jsonl \
  --variants cue_count_all
```

Per layer L (run L16 on GPU 9, L24 on GPU 8 in parallel; substitute
16/L16 accordingly). Uses the same source answers the v3 split used:

```bash
nohup bash -lc '
set -e
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=/data1/heejae/hf_cache
export TRANSFORMERS_CACHE=/data1/heejae/hf_cache
export HF_DATASETS_CACHE=/data1/heejae/hf_cache/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

A=/data1/heejae/medical_nla/activations
T=/data1/heejae/medical_nla/train
R=/data1/heejae/medical_nla/results

CUDA_VISIBLE_DEVICES=9 python -m src.extract_activations \
  --config configs/layer16.yaml \
  --input $A/ddxplus_format_position_rows.jsonl \
  --run-name ddxplus_format_position_L16

python scripts/make_medical_nla_diagnosis_heldout_splits.py \
  --manifest $A/ddxplus_format_position_L16/manifest.jsonl \
  --source-answers /data1/heejae/medical_nla/results/ddxplus_source_mc_cue_count_v1.jsonl \
  --out-dir $T/medical_nla_format_position_L16_heldout \
  --variants cue_count_all --heldout-frac 0.30 --seed 17

python scripts/make_medical_nla_v3_cue_first_targets.py \
  --split-dir $T/medical_nla_format_position_L16_heldout \
  --out-dir $T/medical_nla_format_position_L16_v3 --seed 17

CUDA_VISIBLE_DEVICES=9 python scripts/train_medical_nla_lora.py \
  --config configs/default.yaml \
  --train-jsonl $T/medical_nla_format_position_L16_v3/sft_train.jsonl \
  --val-jsonl $T/medical_nla_format_position_L16_v3/sft_val.jsonl \
  --out-dir /data1/heejae/medical_nla/adapters/medical_nla_format_position_L16_v3_lora_e3 \
  --epochs 3 --batch-size 2

for POOL in test_seen test_heldout; do
CUDA_VISIBLE_DEVICES=9 python -m src.run_nla \
  --config configs/default.yaml \
  --manifest $T/medical_nla_format_position_L16_v3/manifest_${POOL}.jsonl \
  --output $R/ddxplus_format_position_L16_v3_${POOL}.jsonl \
  --adapter-id /data1/heejae/medical_nla/adapters/medical_nla_format_position_L16_v3_lora_e3
python scripts/score_medical_nla_v2_readouts.py \
  --input $R/ddxplus_format_position_L16_v3_${POOL}.jsonl \
  --output-jsonl $R/ddxplus_format_position_L16_v3_${POOL}_scored.jsonl \
  --summary-md $R/ddxplus_format_position_L16_v3_${POOL}_summary.md
done

python scripts/summarize_diagnosis_heldout_readouts.py \
  --heldout-scored $R/ddxplus_format_position_L16_v3_test_heldout_scored.jsonl \
  --seen-scored $R/ddxplus_format_position_L16_v3_test_seen_scored.jsonl \
  --split-dir $T/medical_nla_format_position_L16_v3 \
  --output-jsonl $R/ddxplus_format_position_L16_v3_analysis.jsonl \
  --summary-md $R/ddxplus_format_position_L16_v3_analysis_summary.md
' > /data1/heejae/medical_nla/logs/format_position_L16_v3.log 2>&1 &
```

Overlay reading: for each layer, format-position heldout cue_recall vs the
cue-position curve (L16 0.34 / L24 0.73 / L32 0.56) and the v3 L32 format
baseline (0.19). If format-position content is low at all layers, the
conclusion position never verbalizes individual cues (folding is complete
by the format token at every depth). If it rises at earlier layers, the
format token still carries cue detail early and loses it with depth —
the fold happens between those layers.

Optional complement (cleaner "where the diagnosis becomes decodable"):
train a linear diagnosis probe on the same per-layer format activations
(`scripts/train_ddxplus_linear_probe.py`) and overlay probe accuracy vs
layer. Probe is the right instrument for the conclusion-decodability axis
(NLA hits the memorization wall on diagnosis names); the L32 format probe
is already 0.9917, so this adds L8/L16/L24 for the rising-conclusion curve.
