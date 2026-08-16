# Results 2026-08-13: Probe Control and Diagnosis-Heldout OOD

Two Tier-1 control experiments for the headline source/Medical-AV
disagreement result (AUROC 0.9427), plus a vanilla-AV control. All runs on
the 33 server, layer 32, all-cue format-position activations, DDXPlus.

## TL;DR

Both controls fell against the current Medical-AV recipe:

1. A linear probe replicates (and exceeds) the error-detection signal, so
   error detection is not an NLA-specific contribution.
2. The Medical-AV readout does not generalize to unseen diagnosis classes
   at all; it is a seen-class classifier with a natural-language surface.
3. Vanilla (non-LoRA) AV on the same heldout activations narrates prompt
   format, not clinical content.

What survives: layer-32 activation contains near-linear diagnosis
information (probe 99.17%), and source errors are largely decoding
failures rather than missing information. What must change: the SFT
recipe, which trains a closed-vocabulary classifier instead of an
open-vocabulary readout.

## Experiment A: Probe Disagreement Control

Setup: source-aligned v2 test set (n=1058, error rate 74.2%). Probe
predictions from `ddxplus_all_cue_format_linear_probe_v1`
(`cue_count_all.predictions.jsonl`); overlap with the test set is the
probe's own held-back test rows.

| signal | n | AUROC | AP |
|---|---:|---:|---:|
| source_nla_disagree | 1058 | 0.9427 | 0.9708 |
| source_probe_disagree | 152 | 1.0000 | 1.0000 |
| probe_low_top1_prob | 152 | 0.6102 | 0.8231 |
| source confidence baselines | 1058 | 0.67-0.70 | ~0.84 |

Paired comparison (152 rows where both signals exist):

- nla_disagree_auroc: 0.9282
- probe_disagree_auroc: 1.0000
- nla_minus_probe_auroc: -0.0718

Probe binary rule on paired rows: 118/0/34/0 (tp/fp/tn/fn) — perfect.

Interpretation: with a 99%-accurate probe, "source disagrees with probe"
is nearly identical to "source is wrong", so near-perfect detection is
structural. In-distribution error detection is available to any strong
reader of the activation; it cannot carry the case for natural-language
readout.

Caveats: paired n is only 152 (probe predictions exist only for the
probe's own test split). A rigorous version would retrain the probe on
the v2 train rows only and predict the full 1058-row test manifest;
given probe accuracy, the direction is unlikely to change.

## Experiment B: Diagnosis-Heldout OOD Medical-AV

Split `medical_nla_diagnosis_heldout_v1`: 18 train / 8 heldout classes
(seed 17), train/val source-correct only (884/189 rows), test_seen 727
rows, test_heldout 800 rows. Adapter
`medical_nla_diagnosis_heldout_v1_lora_e3` (3 epochs, val_loss 0.0396;
training healthy).

| pool | n | answer_hit | mean_cue_recall | mean_output_cue_recall |
|---|---:|---:|---:|---:|
| test_seen | 727 | 0.9037 | 0.7690 | 0.7690 |
| test_heldout | 800 | 0.0000 | 0.3066 | 0.3066 |

Classifier-collapse check: 790/800 (98.75%) of heldout answers are
train-class names. 0/800 heldout answers name the gold diagnosis.
Verdict: seen-class classifier.

Per-class structure is not uniform: the collapse maps to the nearest
seen class with varying sense — `urti -> Bronchitis` (98%, cue_recall
0.71, clinically adjacent) down to `sle -> Scombroid food poisoning`
(96%, nonsensical) and `pulmonary_neoplasm -> Anemia` (100%). The
activation-space neighborhood is partially preserved, but there is no
open-vocabulary semantic readout. The residual heldout cue_recall
(0.31) is best read as memorized class-typical cue text that happens to
overlap lexically, not as reading.

Operational note: the first test_seen readout run was duplicated by a
concurrent manual run (1419 rows, 727 unique); deduplicated before the
numbers above. test_heldout was clean (800/800 unique).

## Experiment C: Vanilla AV Control on Heldout

Same 800 heldout activations, same prompt path, no adapter. All metrics
0.0000 (answer, output-level answer, tag and output-level cue recall);
cjk_fraction 0.0, outputs are clean English.

Sampled outputs narrate the prompt format and answer state ("Structured
medical Q&A format signals a clinical diagnosis response ..."),
occasionally echoing shallow cue lists, never attempting a diagnosis.
This matches the 2026-07 specificity pilot (3/100 diagnosis mentions at
format positions). Caveat: exact-string cue matching underestimates the
shallow echoes (e.g. "chest pain, fever, GI symptoms" not matching the
DDXPlus cue phrasings), so treat the 0.0 as "no diagnosis integration",
not "zero content awareness".

## Combined Reading

```text
Same heldout activations (layer 32, all-cue format position):
  linear probe (seen classes):  99% — information is linearly present
  vanilla AV:                   narrates format, no clinical integration
  LoRA AV:                      collapses to memorized train-class labels
```

Information exists in the activation, but neither the vanilla nor the
LoRA-SFT natural-language readout extracts it open-vocabulary. The LoRA
recipe (closed 18-label `<answer>` targets) actively destroyed the
open-vocabulary behavior vanilla had, replacing it with a classifier
head.

Claims that survive today:

- Layer-32 all-cue format activation carries strong linear diagnostic
  signal.
- Source final answers can be wrong while the activation contains the
  gold signal (decoding failure).
- Source-vs-reader disagreement detects errors — but a probe does this
  at least as well; it is not an NLA contribution.

Claims now dead without a recipe change:

- "Medical-AV semantically reads the activation" (heldout: 0%).
- "The 86.96% answer_hit demonstrates readout" (it demonstrates
  in-distribution classification).

## Next Steps (decision pending)

1. Retarget the SFT (handoff §21.3): open-vocabulary, lightly structured
   targets separating observed content / interpretation / uncertainty;
   consider mixing vanilla-NLA-style data to preserve open-vocabulary
   generation. Re-run the heldout evaluation as the acceptance test.
2. Position/layer exploration: entity positions carried local clinical
   meaning in earlier pilots; format position may be the wrong place to
   read integrated content in natural language even though a probe can
   classify from it.
3. Correction experiment (handoff §24.5) remains viable with the current
   in-distribution classifier-readout and is the cheapest path to the
   "solution" axis.
4. Interventional faithfulness (cue-removal counterfactuals, activation
   patching) as the NLA-specific evidence probes cannot mimic — applies
   to whichever readout the retargeted recipe produces.

## Addendum 2026-08-16: v3 Cue-First OOD Result

Same heldout split and activations; target switched to cue-first
(`<observed>` list only, no diagnosis text; adapter
`medical_nla_diagnosis_heldout_v3_cue_first_lora_e3`, epoch-3 val_loss
0.1233).

| pool | n | cue_recall | cue_precision |
|---|---:|---:|---:|
| test_seen | 727 | 0.6251 | 0.6962 |
| test_heldout | 800 | 0.1876 | 0.2437 |

Gate 1 failed: heldout cue recall is below even the v1 memorization level
(0.3066), and precision 0.2437 means ~76% of emitted heldout cues are not
in the case's gold set — consistent with emitting the nearest train
class's typical cue list. The per-class pattern matches v1's adjacency
structure (urti 0.5713 where train has bronchitis/influenza; most others
0.02-0.34). Answer metrics are vacuous by design (no diagnosis text in
targets); judgment is cue-based only.

Combined with the vanilla control and the v1 label-SFT collapse, three
readout attempts now converge on the same conclusion for this position:

> Layer-32 last-token (format-position) activation carries strong
> diagnosis-class signal (probe 99.17%) but does not preserve
> case-specific individual cue information in a form the AV can decode
> into natural language. Evidence appears already compressed into a
> conclusion at this point.

This is the pre-registered failure outcome, and it sets the next track:
run the cue-first recipe at cue-token (entity) positions where local
clinical content was previously observed, and at earlier layers (16/24)
before integration — i.e., the start of the layer-wise pilot.

Sample inspection (heldout acute_pulmonary_edema rows) refines the
conclusion: outputs are not random spray. They stay in the correct cue
family (swelling/dyspnea themes, occasional exact hits like ankle(R))
but get case-specific detail wrong (locations, laterality, enumerating
many location variants), and one output reproduces the textbook cue set
of the nearest train class (pulmonary embolism: pleuritic pain, pain on
movement, cough, calf swelling). So at this position the decodable
content is theme-level summary plus nearest-seen-cluster templates;
case-level detail is what fails — strengthening the case for reading
cue-token (entity) positions, where that detail should live.
