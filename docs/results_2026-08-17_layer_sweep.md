# Results 2026-08-17: Cue-Position Layer Sweep (v5)

Same v4 recipe (per-cue activations, cue-string heldout, single-cue
targets, seed 17 — identical splits across layers) run at layers 16 and
24, compared against the layer-32 baseline. L16/L24 trained 2 epochs
(adapters `medical_nla_cue_position_L{16,24}_v5_lora_e2`), L32 3 epochs.
All layers read through the same L32-AV checkpoint with a per-layer LoRA
(shared decoder + per-layer adapter).

## Headline: unseen-cue readability is an inverted U peaking at layer 24

Hand-labeled semantic classification of all 438 heldout-cue rows per
layer (same rater and rubric as the L32 analysis; labels in
`results_snapshot/L{16,24}_v5_heldout_pairs_hand_labeled.jsonl`):

| layer | A exact | B paraphrase | **A+B semantic read** | C family/detail-wrong | D wrong |
|---|---:|---:|---:|---:|---:|
| 16 | 10.7% | 23.3% | **34.0%** | 52.1% | 13.9% |
| 24 | 16.7% | 56.4% | **73.1%** | 26.0% | **0.9%** |
| 32 | 17.8% | 37.9% | **55.7%** | 35.8% | 8.4% |

Automatic metrics agree on the ordering (heldout strict / soft@0.5 /
mean token recall): L16 0.107/0.607/0.510, L24 0.167/0.813/0.658, L32
0.178/0.699/0.589. Seen-cue pools are ~0.97-0.99 at every layer —
in-distribution reading is easy everywhere; layers differ in
open-vocabulary generalization.

L24 wins despite one FEWER training epoch than L32.

## Qualitative structure per layer

**L24 (peak).** Paraphrases are precise and often minimal-edit: "how
severe is the itching" -> "how bad/intense is the itching"; "chest pain
even at rest" -> "chest pain at rest"; "bottom lip(R)" -> "lower
lip(R)"; bloated abdomen -> "swelling of the abdomen (this is called
ascites)" (adds a correct clinical interpretation). Wrong-class D is
nearly absent (4/438) and the L32 polarity flips do not occur ("out of
breath with minimal effort" reads correctly). Remaining C errors are
adjacent-location confusions (ankle->calf, dorsal foot->sole) and the
black-stool color error.

**L32.** Good but blunter: more template-family drift, polarity flips
appear ("no shortness of breath...", rest/exertion inversion), 8.4% D.

**L16 (floor).** Reads coarse topics but binds content weakly:
severity-question collapses to "the itching"; template fragments carry
wrong content ("weight loss over the last 3 months" -> "a cough ...
over the last 3 months" — the temporal template survives, the content
does not); modality and polarity errors are common (13.9% D, including
"nightmares more prominent at night"). Confound: L16 is farthest from
the L32-AV checkpoint's native layer, so representation quality and
LoRA-bridgeability cannot be separated here.

## Reading

> Cue detail readability rises with depth, peaks around layer 24, and
> degrades by layer 32 — evidence begins folding into the conclusion
> between L24 and L32. Combined with the format-position results
> (theme-only at L32), this is the first measured segment of the
> "evidence -> conclusion" trajectory the layer-wise Medical-NLA program
> is after.

Caveats: single rater; L16/L24 at 2 epochs vs L32 at 3 (direction favors
the L24 conclusion); parenthesized tokens slightly deflate the automatic
soft metric (hand labels supersede); L16 conclusion is confounded with
cross-layer adapter transfer difficulty.

## Next steps

1. **L24 is the operating point.** Build the counterfactual faithfulness
   test (cue removal: removed cue disappears, retained cues stay) on the
   L24 reader.
2. Optionally complete the curve: L8 (expect <= L16) and a finer point
   (L20 or L28) to localize the fold.
3. The other half of the map: layer sweep at the FORMAT position (where
   does the conclusion/theme representation form?) — together with this
   curve it yields the full trajectory picture.
4. Correction axis remains parallel work.

## Addendum: Anti-Memorization Check

The only memorization route available to the v4/v5 readers is
nearest-train-cue regurgitation (map the vector to the closest train-cue
cluster, emit that train string). Measured per heldout row: is the
emitted text closer (content-token F1) to its GOLD heldout cue — never
present in any training target — or to the best-matching of the 117
train cues?

| layer | closer to gold | closer to a train cue | exact train-cue copy |
|---|---:|---:|---:|
| L16 | 31.1% | 54.6% | 20.5% |
| L24 | 63.2% | 35.6% | 22.1% |
| L32 | 47.9% | 34.7% | 12.1% |

Three interlocking arguments against memorization at L24/L32:

1. Composition: the majority of L24 outputs sit closest to a string that
   exists in no training target, and many (e.g. "how bad is the
   itching") exist neither in training targets nor as the gold string —
   they must be composed, not copied.
2. Layer contrast: all three layers trained on identical data and
   targets; if scores came from the training set, layers would look
   alike. They differ 34/73/56 — the only varying factor is which layer
   the input vector came from.
3. The nearest-train residue (~1/3 at L24, majority at L16,
   concentrated in location-family C errors) matches the hand-label C
   class, which is why only A+B is claimed as reading.

Standing caveat: cue-position runs are a positive control (information
present by construction); they establish mechanism capability, not final
faithfulness — that burden stays with the counterfactual tests. A
vanilla (no-LoRA) readout of the same cue-position activations is queued
as the remaining baseline: it determines how much of the v4/v5 result is
the pretrained AV's existing ability vs the LoRA's contribution, and
whether vanilla collapses on L16/L24 vectors (the LoRA-as-translator
claim).
