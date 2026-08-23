# Results 2026-08-17: Cue Counterfactual Faithfulness (L24 reader)

The interventional gate for "the readout follows the activation, not the
case context." 150 test-pool cases, construction-exact counterfactual
prompts (rebuilt from the cue list, verified against the stored prompt):
per case an orig/swap pair at one swapped slot plus two retained slots
probed under orig / swap / removed variants. L24 reader
(`medical_nla_cue_position_L24_v5_lora_e2`), soft token-recall threshold
0.5.

## Results — faithful-reader signature met

| signal | value | target |
|---|---:|---|
| swap_reads_replacement (tracks new content) | 0.887 | high |
| swap_still_reads_original (context memorization) | **0.040** | ~0 |
| retained read rate: orig / swap / removed | 0.973 / 0.967 / 0.967 | stable |
| retained degraded under swap | 0.007 | ~0 |
| retained degraded under removal | 0.000 | ~0 |
| phantom (removed cue reappears) | 0.053 | ~0 |

Reading: swapping one cue in an otherwise identical prompt moves the
swapped-span readout to the new content 88.7% of the time, while the
reader keeps emitting the old cue only 4.0% of the time. Retained slots
are unmoved by the perturbation (degradation ~0), and a removed cue
almost never reappears as a phantom (5.3%). The readout keys on the
span's vector, not the case — this is causal (intervention-based)
faithfulness, which probes and SAEs cannot provide.

## The 0.887 is a floor, not a ceiling

Many `tracks?=N` example rows are scoring artifacts, not memorization —
the readout followed the new cue's THEME but missed a fine attribute, so
exact-token matching scored it a miss:

- swap -> "recently had stools that were black (like coal)"; readout
  "light red blood or blood clots in their stool (hematuria)" — tracked
  to GI bleeding (the new cue), wrong on color/type.
- swap -> "where is the swelling located sole(R)"; readout "iliac
  fossa(R)" — moved off the original entirely but landed on a wrong body
  region.

This is the same attribute-resolution limit seen in the layer sweep
(ankle->calf, black->light red). The honest memorization number is the
4.0% old-cue persistence; most residual N is detail error on a correctly
tracked theme, so true swap sensitivity is above 0.887.

## Standing: grounded-readout evidence is now four-fold

1. Diagnosis-heldout OOD (v4): 55.7% semantic read of unseen cue strings
   at L32 cue positions (hand-labeled).
2. Anti-memorization: outputs sit closer to never-seen gold strings than
   to any train cue (L24 63% vs 36%); layer contrast rules out a
   training-set explanation.
3. Vanilla control: content pre-exists but is buried in confabulated
   meta-narration; the LoRA distills it into a precise reader.
4. Counterfactual (this doc): swap sensitivity 0.887 with 4.0%
   context-memorization and near-zero phantom/degradation.

Remaining honest limits: attribute resolution (color, exact body
part/laterality) is the dominant error mode across all four; single
rater on the qualitative labels; L24 chosen as the operating point but
the fold between L24 and L32 is not yet localized.

## Next

1. Format-position layer sweep (the other half of the trajectory map:
   where does the conclusion form?), to overlay on the cue-position
   inverted U.
2. With a validated grounded reader, the applied axes open: error notes
   ("what evidence was internally present when the model was wrong"),
   and the correction experiment.
3. Optional: attribute-resolution probe (does the vector even separate
   ankle vs calf, black vs red?) to attribute the residual error to
   representation vs readout.

## Addendum: Full Manual Re-Scoring of All 150 Swap Pairs

Every swapped-slot swap row read by hand (labels in
`results_snapshot/cf_L24_v1_swap_hand_labeled.jsonl`). Auto soft-matching
(0.887) counted attribute-detail misses as failures; the manual pass
separates "did the readout leave the original and move to the new cue"
(the actual faithfulness question) from "did it get every fine attribute
right."

| manual class | n | rate |
|---|---:|---:|
| T full track (new cue read correctly / paraphrase) | 106 | 0.707 |
| D tracked new cue's family, wrong fine attribute | 43 | 0.287 |
| O other / wrong cue-type | 1 | 0.007 |
| X reads the ORIGINAL cue (memorization) | **0** | **0.000** |

- **Responded to the swap (T+D): 149/150 = 0.993.** The readout moved off
  the original and onto the new cue's content in all but one row.
- **Genuine context-memorization: 0/150.** Not a single row kept reading
  the original cue. (The auto metric's 4.0% "swap_still_reads_original"
  is token-overlap noise between original and replacement, not real
  persistence.)
- The 43 detail errors are dominated by body-LOCATION cues: 28 of 43
  collapse to a default "iliac fossa" regardless of the true region
  (sole, buttock, thigh, flank, labia majora, side of neck all map
  there); the rest are adjacency (foot->ankle, thigh->calf), laterality
  flips, and the black->light-red stool color error. Non-location content
  (cheek, biceps, thyroid cartilage, epigastric, symptom descriptions)
  tracks near-perfectly.

Phantom re-check (removed variant, 300 retained rows): 16 auto-flagged,
but 15 are template-overlap false positives — the removed and retained
cues share a frame ("where is the swelling located ___", "what color is
the rash ___"), so the correctly-read retained cue trips the removed-cue
token test. Exactly 1 is a true phantom. **True phantom rate ~0.3%, not
5.3%.**

Revised verdict: faithfulness is stronger than the automatic summary
showed — swap sensitivity ~99% at the theme level with zero
memorization and near-zero phantoms. The one persistent weakness is
localized precisely: body-location attribute resolution, with an
"iliac fossa" default attractor. This is the concrete target for the
attribute-resolution follow-up (representation vs readout).
