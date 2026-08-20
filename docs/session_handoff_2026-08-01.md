# Medical NLA Project Handoff

Date: 2026-08-01  
Primary repo: `/Users/heejae/Developer/medical_nla`  
Remote repo: `https://github.com/0914eagle/medical_nla.git`  
Primary server: `eagle0914@165.132.77.33`  
Main server code path: `/home/eagle0914/medical_nla`  
Main artifact root: `$ART`  
Main uv env: `/data1/heejae/uv/medical_nla`  
Main HF cache: `$HF_HOME`  
Main GPU setting used recently: A6000, usually `CUDA_VISIBLE_DEVICES=0` or `0/1`

This document is a handoff note. It is meant to let another Codex session, another AI model, or a human collaborator continue the project without reading the entire conversation history.

## 1. Project Goal

The original high-level goal was:

> Fine-tune/adapt NLA for the medical domain and see whether it can help with medical LLM interpretability, error diagnosis, and eventually correction.

Professor feedback framed the project around three axes:

1. **Explanation**
   - Can we explain why the backbone was right or wrong?
   - Can the explanation expose what the model was internally considering?

2. **Diagnosis**
   - Can we predict whether the backbone answer is likely wrong?
   - Can activation readouts reveal a conflict between internal diagnostic signal and final answer?

3. **Solution / correction**
   - Can the readout be used to improve the backbone answer?
   - Possible directions: reconsideration prompt, verifier, error-note memory, dataset valuation, text/activation patching.

Current status:

> We have not built a full faithful Medical-NLA yet. We have built and evaluated an AV-only, LoRA-adapted medical activation readout. This readout is useful as a post-hoc second opinion / verifier. The strongest result is that disagreement between the source answer and Medical-AV readout strongly predicts source error.

## 2. Key Terminology

### Backbone / source model

The source/backbone model is:

- `google/gemma-3-12b-it`
- 48 transformer layers
- hidden size / d_model = 3840

Most OpenNLA experiments here used:

- layer 32
- checkpoint family: `kitft/nla-gemma3-12b-L32-*`

### OpenNLA

NLA means Natural Language Autoencoder. It has two sides:

1. **AV: Activation Verbalizer**
   - Input: activation vector `h`
   - Output: natural language description `z`

2. **AR: Activation Reconstructor**
   - Input: natural language description `z`
   - Output: reconstructed activation vector `h'`

Original NLA training/evaluation loop:

```text
activation h
  -> AV
natural language z
  -> AR
reconstructed activation h'

loss / evaluation:
  distance(h, h')
```

MSE is computed between the original activation and reconstructed activation:

```text
MSE(h, h')
```

It is **not** MSE between text and activation.

### Medical-AV vs full Medical-NLA

What we trained so far:

```text
pretrained OpenNLA AV
  + LoRA SFT
  + medical structured targets
```

This should be called:

- Medical-AV
- Medical activation readout
- AV-only medical readout

It should **not** be called full faithful Medical-NLA without qualification.

Reason:

- We fine-tuned the AV side only.
- We did not jointly train AV + AR with reconstruction.
- We did not prove that generated text is fully faithful to the activation.

### Probe

A linear probe is:

```text
activation h -> linear classifier -> diagnosis class
```

It is not a natural language explanation. It is a supervised diagnostic classifier on top of hidden states. Probe results are used to ask:

> Is diagnosis information present in the activation at all?

### Source confidence baseline

These are error-prediction signals that use only the source model's option probabilities/logprobs, with no Medical-AV:

- low top-1 option probability
- low top1-top2 probability margin
- low logprob score margin
- high entropy

These ask:

> Does the source model look uncertain?

They are compared against:

> Does the source answer disagree with the Medical-AV readout?

## 3. Environment Setup

### Local workspace

```bash
cd /Users/heejae/Developer/medical_nla
```

The repo is clean as of this handoff note.

### Server workspace

```bash
ssh eagle0914@165.132.77.33
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

For Gemma gated access:

```bash
hf auth whoami
hf auth login --force
```

If the token is invalid, Gemma loading fails with 401 Unauthorized. This happened once on `/data1`; the fix was to refresh HF auth.

### Alternate server attempt: `/data3/heejae`

We tried moving to `/data3/heejae`, but PyTorch installed as `2.13.0+cu130` while the NVIDIA driver was CUDA 12.4-era:

```text
RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
torch: 2.13.0+cu130
torch cuda: 13.0
cuda available: False
```

Resolution at the time:

- We moved back to `/data1`.
- If using `/data3`, install a PyTorch build compatible with the driver, e.g. CUDA 12.1/12.4-compatible.

## 4. Important Repo Scripts

Core extraction / NLA:

- `src.extract_activations`
- `src.run_nla`
- `src.nla`
- `src.modeling`

Dataset generation:

- `scripts/make_ddxplus_probe_dataset.py`
- `scripts/make_ddxplus_cue_count_cases.py`
- `scripts/make_ddxplus_cue_count_activation_rows.py`
- `scripts/make_medical_nla_v2_sft_splits.py`
- `scripts/make_medical_nla_v2_source_aligned_splits.py`

Source model evaluation:

- `scripts/run_source_model_mc.py`
- `scripts/summarize_source_model_mc.py`
- `scripts/run_source_model_qa.py`
- `scripts/run_source_model_answers.py`
- `scripts/score_source_mc_option_logprobs.py`

Medical-AV training/evaluation:

- `scripts/train_medical_nla_lora.py`
- `scripts/score_medical_nla_v2_readouts.py`
- `scripts/run_nla_diagnosis_mc.py`

Probe:

- `scripts/train_ddxplus_linear_probe.py`

Error prediction:

- `scripts/make_error_prediction_table.py`
- `scripts/evaluate_error_prediction.py`
- `scripts/evaluate_error_prediction_combined.py`

Sampling stability:

- `scripts/run_nla_sampling_stability.py`
- `scripts/evaluate_nla_sampling_stability.py`

AR/MSE scoring:

- `score_reconstruction_mse.py`
- `src.reconstruction_scoring`

## 5. OpenNLA / Gemma Setup Details

Important OpenNLA checkpoint references:

```text
AV: kitft/nla-gemma3-12b-L32-av
AR: kitft/nla-gemma3-12b-L32-ar
Layer: 32
d_model: 3840
Gemma total layers: 48
```

Important NLA injection details learned early:

- Use `nla_meta.yaml` sidecar. Do not hardcode prompt template / injection char / injection scale if avoidable.
- Gemma NLA injection char/token was around `㈜`, token id 246566 in prior notes.
- Gemma L32 injection scale was around 80000.
- Initial concern about activation norms being 70k to 90k was resolved: this is normal for Gemma L32 scaled residual stream, not necessarily a bug.
- Using model embedding forward avoids the Gemma scaled embedding trap.
- One-step chat template tokenization matters. Splitting chat template and encode can introduce BOS/position mismatch.

## 6. DDXPlus Data Processing

Dataset was downloaded from Hugging Face:

```bash
hf download aai530-group6/ddxplus \
  --repo-type dataset \
  --local-dir $RAW/ddxplus
```

Files:

```text
$RAW/ddxplus/train.csv
$RAW/ddxplus/test.csv
$RAW/ddxplus/validate.csv
$RAW/ddxplus/release_evidences.json
$RAW/ddxplus/release_conditions.json
```

DDXPlus row structure:

- `PATHOLOGY`: canonical diagnosis label
- `EVIDENCES`: present evidence IDs
- `release_evidences.json`: maps evidence IDs to natural language cue text and metadata

Pipeline:

```text
DDXPlus train.csv patient rows
  -> sample cases by diagnosis
  -> parse positive/meaningful evidence IDs
  -> map evidence IDs to natural language cues via release_evidences.json
  -> make prompts such as:
     "A patient presents with {cue1}, {cue2}, ... What diagnosis is most likely?"
  -> make activation rows by target position
  -> extract Gemma layer 32 activations
```

The DDXPlus processing is rule-based. Evidence IDs such as `E_56_@_4` are mapped to natural-language strings from `release_evidences.json`.

## 7. Early Experiments and Lessons

### 7.1 Initial medical/general NLA pilot

Initial `pilot_medical_v3` and `pilot_general_v3` outputs showed that vanilla NLA often described:

```text
structured medical Q&A format
answer-requesting structure
```

rather than the actual diagnosis.

Initial suspicion:

> Vanilla NLA cannot handle medical domain.

Corrected interpretation:

> We were often extracting the wrong token position. Format/answer position activations genuinely encode the task/format state, not necessarily the clinical entity content.

### 7.2 Position variants

We compared:

- `format_last`: question end / answer-requesting state
- diagnostic entity positions: e.g. `ST elevations`, `warfarin`, `aphasia`
- non-diagnostic tokens: `patient`, `man`, `woman`, `Explain`

Main conclusion:

> Medical meaning is position-sensitive. Diagnostic entity positions carry local clinical meaning. Format positions carry answer/task state. Non-diagnostic tokens do not recover target diagnosis.

Important correction:

- Initially broad keyword matching overestimated content in non-diagnostic tokens.
- Diagnosis-target recall showed non-diagnostic baseline full recall was effectively 0/50 in that pilot.
- This argued against the simple "oracle tagging confound" claim.

### 7.3 AR reconstruction MSE

AR MSE was added to AV outputs:

- Format activations had low MSE.
- Content activations had low MSE.
- Non-diagnostic activations also had low MSE.

Conclusion:

> Reconstruction MSE tests whether text can reconstruct the activation, not whether the text is clinically useful or contains the correct diagnosis.

MSE is not enough for medical explanation quality.

## 8. Specificity Experiments

Purpose:

> Test whether vanilla NLA can verbalize integrated multi-cue diagnosis, and whether Gemma itself can answer the cases.

### Key v2 specificity numbers

Strict source baseline:

```text
Gemma source answer: 49/50
```

Diagnosis-only NLA scoring:

```text
specific cue positions: 98/150 diagnosis hit
format positions: 3/100 diagnosis hit
```

Interpretation:

> Gemma can diagnose the vignettes. Some specific cue activations contain diagnosis-related signal. Vanilla NLA almost never verbalizes integrated diagnosis from format/answer positions.

This suggested:

> Activation may contain the information, but vanilla NLA readout fails.

## 9. Probe Experiments

Purpose:

> Determine whether diagnosis information exists in the activation, independently of natural language readout.

### 9.1 Three-cue DDXPlus probe

Dataset:

```text
49 diagnoses
100 examples per diagnosis
4900 cases
6 variants per case:
  single_cue
  single_format
  multi_cue_1
  multi_cue_2
  multi_cue_3
  multi_format
29400 activation rows
```

Linear probe results:

```text
chance_acc1 = 0.0204

single_cue      acc1 = 0.4218
single_format   acc1 = 0.4122
multi_cue_1     acc1 = 0.4313
multi_cue_2     acc1 = 0.6912
multi_cue_3     acc1 = 0.8272
multi_cue_all   acc1 = 0.6649
multi_format    acc1 = 0.8354
```

Key number:

```text
3-cue multi_format probe acc1 = 83.54%
```

Meaning:

> Layer 32 multi-format activation contains diagnosis information. This is a supervised probe readout, not natural language NLA output.

### 9.2 All-cue linear probe

After moving to all-cue format activation:

Manifest:

```text
$ART/activations/ddxplus_all_cue_format_v1/manifest.jsonl
```

Variant name was `cue_count_all`, not `cues_all`.

Probe result:

```text
all-cue format linear probe:
  test_acc1 = 0.9917
  test_acc5 = 1.0000
```

Meaning:

> In this all-cue selected diagnosis setting, layer-32 format activation contains DDXPlus diagnosis class information almost perfectly linearly.

Caveat:

> This is in-distribution over seen diagnosis classes. It is not yet true OOD / diagnosis-heldout Medical-AV evaluation.

## 10. Source Model Baselines

### 10.1 Cue count source MC

Question:

> Is 3 cues enough for source Gemma to diagnose DDXPlus cases?

Result:

```text
cue_count_3:
  n = 3200
  hit = 1185
  acc = 0.3703

cue_count_5:
  n = 3200
  hit = 1478
  acc = 0.4619

cue_count_all:
  n = 3200
  hit = 1825
  acc = 0.5703
```

Interpretation:

> 3-cue setting is a partial-evidence pilot. DDXPlus often requires more than three cues. We moved to all-cue setting for a fairer source-aligned experiment.

### 10.2 Source MC on all-cue source-aligned test subset

Source-aligned test subset:

```text
n = 1058
source_correct = 273
source_wrong = 785
```

This test subset is source-error enriched because train/val were source-correct only.

Important caveat:

> Source accuracy on this subset is not the overall DDXPlus all-cue accuracy.

## 11. Medical-AV SFT Experiments

### 11.1 What we trained

Input:

```text
Gemma layer 32 activation at selected position
```

Output target:

```xml
<explanation>
<readout>
  <task_type>diagnosis</task_type>
  <answer>...</answer>
  <supporting_cues>...</supporting_cues>
</readout>
</explanation>
```

Loss:

```text
next-token cross-entropy over target text
```

This is SFT loss, not reconstruction loss.

### 11.2 LoRA loss values

| Run | Epochs | First train loss | Final train loss | Train mean | Val loss |
|---|---:|---:|---:|---:|---:|
| `medical_nla_ddxplus_lora_v1` | 1 | 9.3374 | 0.2484 | 0.3524 | 0.1658 |
| `medical_nla_ddxplus_v2_alpha_lora` | 1 | 3.7589 | 0.1450 | 0.2177 | 0.1173 |
| `medical_nla_ddxplus_v2_beta_lora` | 1 | 4.0550 | 0.0878 | 0.1722 | 0.0840 |
| `medical_nla_all_cue_source_aligned_v2_lora_e3_b2` | 3 | 4.4343 | 0.0449 | epoch 3 mean 0.0421 | 0.0362 |

All-cue source-aligned v2 epoch losses:

```text
epoch 1:
  train_mean_loss = 0.1714
  val_loss = 0.0454

epoch 2:
  train_mean_loss = 0.0539
  val_loss = 0.0436

epoch 3:
  train_mean_loss = 0.0421
  val_loss = 0.0362
```

Interpretation:

> The readout format is learned well. Loss alone does not imply diagnostic accuracy.

### 11.3 Three-cue Medical-AV

```text
n = 735
answer_hit = 412/735 = 56.05%
```

Comparison:

```text
source MC = 251/735 = 34.15%
probe reference = 83.54%
```

### 11.4 All-cue source-aligned Medical-AV

Why source-aligned?

> If source Gemma is wrong, pairing its activation with gold diagnosis may be misaligned. For train/val, we therefore used only source-correct cases.

Split:

```text
selected diagnoses = 26
selected rows = 2600
train_rows = 1270, source-correct only
val_rows = 272, source-correct only
test_rows = 1058, source-correct + source-wrong
```

Test distribution:

```text
source_correct = 273
source_wrong = 785
```

Readout result:

```text
n = 1058
parsed_readout = 1058/1058
answer_hit = 920/1058 = 86.96%
mean_cue_recall = 0.7994
```

Source correctness breakdown:

```text
source_correct cases:
  Medical-AV correct = 265/273 = 97.07%

source_wrong cases:
  Medical-AV correct = 655/785 = 83.44%
```

Interpretation:

> Even when Gemma's final MC answer is wrong, the layer-32 all-cue format activation often still contains gold/canonical diagnostic signal that Medical-AV can read out.

Caveat:

> This is not a proof that Medical-AV faithfully narrates the source model's final reasoning. It is better described as an activation-based diagnostic second opinion.

## 12. Mean Cue Recall

`mean_cue_recall` measures:

> Of the DDXPlus positive evidence cues for a row, what fraction did Medical-AV output in `<supporting_cues>`?

Example:

```text
gold cues:
  fever
  productive cough
  shortness of breath
  lobar consolidation

Medical-AV supporting_cues:
  fever
  productive cough
  lobar consolidation

cue_recall = 3/4 = 0.75
```

All-cue result:

```text
mean_cue_recall = 0.7994
```

Meaning:

> Medical-AV recovered about 80% of positive evidence cues on average.

## 13. Error Prediction via Source / Medical-AV Disagreement

Core rule:

```text
if source answer != Medical-AV readout:
    predict source error
```

All-cue source-aligned v2 test:

| Source answer vs Medical-AV readout | n | Source correct | Source accuracy | Source error rate | Medical-AV correct | Medical-AV accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Agree | 332 | 265 | 79.82% | 20.18% | 265 | 79.82% |
| Disagree | 726 | 8 | 1.10% | 98.90% | 655 | 90.22% |

Key facts:

```text
disagreement cases = 726
Medical-AV correct among disagreement = 655
P(Medical-AV correct | disagreement) = 655/726 = 90.22%
```

Confusion-style breakdown:

```text
source wrong / Medical-AV wrong  = 130
source wrong / Medical-AV right  = 655
source right / Medical-AV wrong  = 8
source right / Medical-AV right  = 265
```

Interpretation:

> When source and Medical-AV disagree, source is almost always wrong, and Medical-AV is usually correct.

## 14. Error Prediction Metrics

| Signal | What it uses | AUROC | AP |
|---|---|---:|---:|
| `source_nla_disagree` | Source answer and Medical-AV readout mismatch | 0.9427 | 0.9708 |
| `source low top1 option prob` | Source-selected option probability is low | 0.6935 | 0.8458 |
| `source low top1-top2 prob margin` | Gap between top-1 and top-2 option probabilities is small | 0.7010 | 0.8415 |
| `source low score margin` | Gap between top-1 and top-2 raw logprob scores is small | 0.6955 | 0.8392 |
| `source entropy` | Source option distribution entropy is high | 0.6699 | 0.8414 |

Definitions:

- AUROC: how well the signal ranks source-wrong above source-correct. 0.5 is random, 1.0 is perfect.
- AP: average precision over source-error class. AP is influenced by error base rate.

Source confidence baselines:

> AUROC around 0.67 to 0.70.

Medical-AV disagreement:

> AUROC 0.9427.

Conclusion:

> Medical-AV disagreement is much stronger than source confidence baselines.

## 15. Combined Error Prediction

| Evaluation | Source confidence only AUROC | Medical-AV disagree AUROC | Combined AUROC |
|---|---:|---:|---:|
| Stratified combiner check | 0.6968 | 0.9427 | 0.9592 |
| Diagnosis-heldout combiner check | 0.6658 | 0.9427 | 0.9262 |

Important correction:

> This diagnosis-heldout result is not a full OOD Medical-AV experiment. It is only a combiner-level sanity check over existing features. We have not yet trained/evaluated Medical-AV on unseen diagnosis classes.

Correct interpretation:

```text
Stratified:
  feature combiner trained/evaluated with mixed diagnoses.

Diagnosis-heldout combiner:
  feature combiner checked with diagnosis-level split.
  This does not prove Medical-AV generalizes to unseen diagnoses.
```

## 16. NLA-Only Sampling Stability

Question:

> Can Medical-AV alone, without source answer comparison, predict source error by being unstable across samples?

Method:

- Sample Medical-AV multiple times per activation.
- Compute answer entropy, top-answer fraction, unique answer count, cue recall stability.

Summary:

```text
n = 1058
errors = 785/1058 = 74.2%
mean_top_answer_fraction = 0.9470
mean_answer_entropy_norm = 0.0548
mean_unique_answer_count = 1.23
```

Signals:

| NLA-only signal | AUROC | AP |
|---|---:|---:|
| answer entropy | 0.5657 | 0.7754 |
| low top-answer fraction | 0.5651 | 0.7713 |
| unique answer fraction | 0.5663 | 0.7746 |
| low parsed answer rate | 0.5000 | 0.6850 |
| low mean cue recall | 0.5418 | 0.7876 |

Conclusion:

> NLA-only sampling instability is weak. Medical-AV is usually stable, even on source-wrong cases. The strongest current signal is source answer vs Medical-AV readout disagreement.

## 17. Important Conceptual Corrections

### 17.1 We did not train AR

We trained AV only:

```text
activation -> structured text
```

AR was used only in reconstruction scoring experiments, not as part of the Medical-AV SFT loop.

### 17.2 Current model is not full faithful NLA

Current model is:

> supervised medical activation readout initialized from OpenNLA AV.

It is useful but not fully faithful by default.

### 17.3 Disagreement is post-hoc

Current strong error signal requires both:

- source answer
- Medical-AV readout

Therefore:

> It is a post-hoc verifier / second opinion, not yet a pre-answer predictor.

### 17.4 Source wrong but Medical-AV correct is plausible

It sounds surprising that:

```text
source wrong, Medical-AV correct = 655/785
```

But this is plausible because:

- Linear probe shows diagnosis information is in activation.
- Final MC decoding/selection can fail even if intermediate representation contains correct signal.
- Medical-AV was trained to read gold/canonical diagnostic signal, not to imitate source final answer.

### 17.5 Probe and source generation are not the same task

Probe:

```text
activation -> supervised diagnosis class
```

Source MC:

```text
prompt text + options -> generated/selected answer
```

Do not compare probe accuracy and source accuracy as if they are the same task. Use probe to show information exists in activation.

### 17.6 True OOD has not been done

We have not yet done:

```text
train on some diagnosis classes
test on unseen diagnosis classes
```

This is an important future experiment.

## 18. What We Can Claim Now

Safe claim:

> Gemma layer-32 all-cue format activation contains strong diagnostic information, as shown by a linear probe. A LoRA-adapted Medical-AV can extract much of this signal as structured natural-language readout. Disagreement between source answer and Medical-AV readout is a strong post-hoc predictor of source error.

Stronger but still acceptable with caveats:

> The final source answer can be wrong even when the intermediate activation contains gold diagnostic signal. Medical-AV can serve as an activation-based second opinion.

Do not claim yet:

- We built full faithful Medical-NLA.
- Medical-AV explains why the model is wrong in a causal sense.
- Medical-AV works on unseen diagnosis classes.
- Medical-AV improves backbone performance.
- NLA-only readout predicts errors before source output.

## 19. Current Best Research Framing

This is the best framing as of 2026-08-01:

> A single-layer Medical-AV readout already reveals a diagnostic signal that can diverge from the model's final answer and strongly predict source errors. This motivates a layer-wise Medical-NLA framework to trace when and how medical reasoning trajectories diverge from the correct diagnosis.

Korean version:

> 한 layer의 Medical-AV만으로도 source answer와 다른 diagnostic signal을 읽어 오답을 강하게 탐지할 수 있었다. 따라서 이를 layer-wise full Medical-NLA로 확장하면, 모델이 어느 단계에서 어떤 잘못된 진단 방향으로 drift하는지 추적할 수 있을 것이다.

## 20. Why Layer-Wise Full Medical-NLA?

The intended next research direction is not simply "make AV predict diagnosis better." It is:

> Build a layer-wise medical activation readout system that explains diagnostic trajectory, predicts error risk, and eventually enables correction.

What layer-wise readout could show:

```text
early layers:
  symptom/cue encoding

middle layers:
  candidate diagnoses and differential diagnosis

late layers:
  leading diagnosis / answer-state drift
```

Wrong-case taxonomy examples:

```text
1. Missing cue:
   model never encodes a key symptom strongly

2. Distractor overweighting:
   wrong candidate grows despite correct cue signal

3. Late drift:
   middle layer has gold signal, late layer shifts to wrong answer

4. Decoding mismatch:
   activation readout remains gold, final output selects wrong option
```

This connects to professor's three axes:

1. Explanation:
   - layer-wise trajectory explains internal state changes.

2. Diagnosis:
   - trajectory conflict can predict source error.

3. Solution:
   - error subtype can guide correction strategy.

## 21. Full NLA Tuning Ideas

### 21.1 Why not reconstruction-only?

If we use original NLA reconstruction-only objective:

```text
activation -> explanation -> reconstructed activation
```

the model may again produce explanations like:

```text
structured medical Q&A format
```

These can reconstruct activation but are not clinically useful.

Therefore Medical-NLA needs both:

- faithfulness / reconstruction
- medically useful semantic readout

### 21.2 Why not diagnosis-only SFT?

If target is only:

```xml
<answer>Pneumonia</answer>
```

the model can become a supervised classifier:

```text
activation cluster -> diagnosis label
```

This is useful but not full NLA.

### 21.3 Better target style

Do not force the current XML forever. Use a more natural, lightly structured output:

```text
Observed activation content:
- fever
- productive respiratory symptoms
- focal consolidation

Clinical interpretation:
- lower respiratory infection pattern
- pneumonia favored

Uncertainty/conflict:
- bronchitis remains a nearby alternative
```

The key is to separate:

- observed content in activation
- inferred/candidate diagnosis
- uncertainty/conflict

### 21.4 Layer-conditioned Medical-NLA

A unified layer-wise model may be possible because NLA outputs natural language, a shared semantic space. Unlike SAE, which is usually layer/hook-specific, NLA may share a decoder if given layer/hook metadata.

Possible architecture:

```text
activation h_l
+ layer id
+ hook point
+ position type
  -> layer-conditioned projection / adapter
  -> shared language decoder
  -> explanation z
```

AR side:

```text
explanation z
+ layer id
+ hook point
  -> reconstructed activation h'_l
```

Loss:

```text
MSE(h_l, h'_l)
+ semantic/readout loss
```

### 21.5 One model or layer-specific models?

SAEs are usually layer/hook-specific because activation distributions differ by layer/hook.

NLA may be more flexible because output is natural language. Three options:

1. **Separate NLA per layer**
   - most stable
   - expensive and hard to manage

2. **Shared NLA + layer token**
   - simplest unified version
   - may suffer from distribution mismatch

3. **Shared decoder + layer-specific projection/adapters**
   - likely best research direction
   - layer distribution handled by adapters
   - language readout shared

## 22. Natural Language Input and MSE

It is okay for NLA input to contain natural-language instruction/metadata.

MSE is not computed between text and activation. It is computed between:

```text
original activation h
reconstructed activation h'
```

For layer-wise Medical-NLA:

```text
AV input:
  instruction + layer metadata + injected activation h_l

AV output:
  natural language explanation z

AR input:
  z + layer metadata

AR output:
  reconstructed activation h'_l

loss:
  MSE(h_l, h'_l)
```

Allowed natural-language metadata:

- task instruction
- layer id
- hook point
- position type

Dangerous metadata:

- gold diagnosis
- gold cues
- source/gold labels

Do not put the answer into the prompt if the goal is to test whether the activation contains it.

## 23. Related Research Positioning

Known areas:

1. **Hidden-state probes / logit lens / tuned lens**
   - Many existing works read intermediate layers using probes or lenses.
   - These usually output class/logit scores, not natural-language explanations.

2. **SAE**
   - Usually layer/hook-specific.
   - Gives sparse features but not necessarily natural-language medical trajectories.

3. **OpenNLA**
   - Activation to natural language explanation to reconstructed activation.
   - Natural-language bottleneck.

Our project sits between them:

> Use NLA-style natural-language activation readout, but adapt it for medical diagnostic trajectories and error detection.

Novelty is not simply "read a middle layer." The potential contribution is:

> NLA-style natural-language readout of medical activation trajectories, with error detection and eventually correction utility.

## 24. Immediate Future Work

### 24.1 Layer-wise pilot

Goal:

> See where diagnosis information appears and where wrong-case drift happens.

Suggested pilot:

```text
Backbone: Gemma-3-12B-IT
Dataset: DDXPlus all-cue source-aligned test
Layers: 8, 16, 24, 32, 40, maybe 48
Positions:
  format position
  selected cue positions if available

Groups:
  source correct / Medical-AV correct
  source wrong / Medical-AV correct
  source wrong / Medical-AV wrong
```

First do linear probes layer-wise:

```text
layer activation -> diagnosis class
```

Then do AV/NLA readouts on informative layers.

### 24.2 True diagnosis-heldout Medical-AV

Goal:

> Distinguish seen-class classifier behavior from semantic activation verbalization.

Design:

```text
train diagnoses:
  subset of diagnosis classes

test diagnoses:
  completely unseen diagnosis classes
```

Metrics:

- answer_hit
- cue_recall
- natural language judge score if available

Interpretation:

```text
answer_hit low + cue_recall high:
  model reads medical cue semantics but does not generalize diagnosis names.

answer_hit low + cue_recall low:
  likely seen-class classifier.

answer_hit high + cue_recall high:
  strong OOD readout.
```

### 24.3 Source-state vs gold-readout separation

Current Medical-AV is trained toward gold diagnosis. It is not trained to narrate the source final answer.

Future direction:

Train/compare:

1. **Gold readout**
   - activation -> gold diagnosis/cues

2. **Source-state readout**
   - activation -> source-selected diagnosis/state

This can help answer:

> Is the activation carrying both gold and wrong-source signals? Where do they diverge?

### 24.4 Full NLA / AR faithfulness

Possible stages:

1. Post-hoc AR consistency:

```text
Medical-AV output z -> AR -> h'
compare h and h'
```

2. Reranking:

```text
generate multiple explanations
score = semantic utility + alpha * reconstruction consistency
```

3. Joint objective:

```text
SFT readout loss + lambda * reconstruction loss
```

Joint training is harder due to discrete text. Start with post-hoc AR and reranking.

### 24.5 Correction / backbone improvement

Do this after readout nature is clearer.

Possible experiment:

```text
source initial answer = A
Medical-NLA readout = B

If A != B:
  ask source to reconsider with readout B and observed cues
```

Baselines:

- source-only
- generic reconsider: "Think carefully again"
- Medical-NLA readout reconsider
- Medical-NLA answer-only

Success criterion:

> Medical-NLA readout improves correction beyond generic reconsideration.

## 25. Useful Commands

### Pull latest repo on server

```bash
cd /home/eagle0914/medical_nla
git pull
```

### Standard env

```bash
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
export HF_HOME=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME/datasets
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Tail logs

```bash
tail -f $ART/logs/<log_name>.log
```

### Stop nohup process

Find:

```bash
ps -ef | grep python | grep medical_nla
```

Kill:

```bash
kill <PID>
```

Only use `kill -9` if regular kill does not stop.

### SCP from server

```bash
scp eagle0914@165.132.77.33:$ART/results/<file> .
scp eagle0914@165.132.77.33:$ART/logs/<file> .
scp eagle0914@165.132.77.33:$ART/probe/<dir>/summary.md .
```

For custom port server, use:

```bash
scp -P <PORT> eagle0914@<HOST>:/path/to/file .
```

## 26. Files to Look For on Server

Important result locations:

```text
$ART/results/
$ART/logs/
$ART/probe/
$ART/adapters/
$ART/activations/
$ART/sft/
```

Most important current artifacts:

```text
$ART/adapters/medical_nla_all_cue_source_aligned_v2_lora_e3_b2

$ART/results/ddxplus_medical_nla_all_cue_source_aligned_v2_readouts_test_e3_b2_scored.jsonl
$ART/results/ddxplus_medical_nla_all_cue_source_aligned_v2_readouts_test_e3_b2_summary.md

$ART/results/ddxplus_error_prediction_all_cue_source_aligned_v2_e3_b2_filtered.jsonl
$ART/results/ddxplus_error_prediction_all_cue_source_aligned_v2_e3_b2_filtered_summary.md

$ART/probe/ddxplus_all_cue_format_linear_probe_v1/summary.md
$ART/probe/ddxplus_all_cue_format_linear_probe_v1/results.json
```

## 27. Suggested Professor Meeting Narrative

Use this concise story:

1. We started with vanilla OpenNLA on medical prompts.
2. We found apparent failures, but corrected the interpretation:
   - token position matters.
   - format positions encode task/answer state.
   - diagnostic entity positions encode local clinical meaning.
3. We then asked whether integrated diagnosis information exists in the activation.
   - Linear probe says yes.
   - all-cue layer-32 format probe acc1 = 99.17%.
4. We adapted the OpenNLA AV into a Medical-AV readout.
   - all-cue source-aligned test answer_hit = 86.96%.
   - mean_cue_recall = 79.94%.
5. The key utility signal:
   - if source and Medical-AV disagree, source error rate = 98.90%.
   - Medical-AV correct among disagreement = 90.22%.
   - disagreement AUROC = 0.9427 vs source confidence 0.67 to 0.70.
6. This is not yet full faithful NLA.
   - It is a strong AV-only diagnostic readout pilot.
7. Next step:
   - build layer-wise Medical-NLA/readout to trace diagnostic trajectories.
   - add AR/reconstruction faithfulness.
   - later use trajectory/readout for correction.

One-sentence close:

> This pilot suggests that medical LLM errors are not always due to absence of diagnostic signal in the activation. The signal can exist internally but fail to become the final answer. A layer-wise Medical-NLA could expose when that divergence happens and eventually guide correction.

