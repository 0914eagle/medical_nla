# Results 2026-08-16: v4 Cue-Position Positive Control

Question: was the v3 format-position failure positional (detail compressed
away at the answer position) or mechanistic (single-vector NLA readout
cannot carry case-specific detail at all)?

Setup: one layer-32 activation per (case, cue) at the cue's own token span
(`last_subtoken`), 3,200 all-cue cases x up to 4 cues = 12,800 rows.
Cue-STRING-level heldout: 41 of 164 unique cue strings never appear in any
supervised target (train 7,515 / val 1,086 / test_seen_cue 2,122 /
test_heldout_cue 438; heldout-cue rows in train/val cases dropped: 1,639).
Adapter `medical_nla_cue_position_v4_lora_e3`. Target: single-cue
`<observed>` list, no diagnosis text.

## Headline numbers

| pool | n | strict read | soft@0.5 | hand-labeled semantic read |
|---|---:|---:|---:|---:|
| test_seen_cue | 2122 | 0.9797 | 0.998 | - |
| test_heldout_cue | 438 | 0.1781 | 0.699 | **0.557 (A+B)** |

Strict substring matching is a floor on unseen cue strings: the model
cannot quote phrasings it never saw, so correct reads surface as
paraphrases. All 438 heldout rows (92 unique gold/output pairs) were
hand-labeled into four classes
(`results_snapshot/v4_heldout_pairs_hand_labeled.jsonl`):

| class | rows | rate | meaning |
|---|---:|---:|---|
| A exact | 78 | 17.8% | verbatim unseen-string reproduction |
| B correct paraphrase | 166 | 37.9% | same clinical content, own words |
| C right family, wrong detail | 157 | 35.8% | e.g. ankle->calf, weakness->tingling |
| D wrong/unrelated | 37 | 8.4% | incl. negation/polarity flips |

Representative B reads (gold -> emitted), all on never-supervised strings:

- "pain that is increased with movement" -> "pain that increases with movement"
- "had an involuntary weight loss over the last 3 months" ->
  "been unintentionally losing weight over the past 3 months"
- "their abdomen is bloated or distended (swollen due to pressure from
  inside)" -> "the abdomen is swollen or painful due to gas and/or bloating"
- "where is the affected region located iliac fossa(L)" ->
  "where is the affected region located side of the abdomen(L)"
- "is their skin much paler than usual" ->
  "pale skin tone associated with anemia or poor circulation"

## Verdict

**The v3 failure was positional, not mechanistic.** At a position that
contains the information, the single-vector NLA readout verbalizes unseen
case-specific clinical content open-vocabulary at 55.7% (A+B), with a
further 35.8% landing in the correct cue family. This is the first
positive evidence for the project's core bet — an ability neither a probe
(closed label set) nor the format-position readout (theme-only) has.

Error structure is informative, not random:

- C-errors are semantic nearest-neighbors: adjacent body parts
  (ankle->calf/leg, dorsal foot->sole), related modality
  (weakness/paralysis -> numbness/tingling), severity-question ->
  severity-assertion. Laterality (L/R) is usually preserved even when the
  body part is wrong.
- D includes polarity flips ("out of breath with minimal effort" -> "NO
  shortness of breath with minimal exertion"; "chest pain even at rest" ->
  "alleviated with rest") — negation is a weak axis of the representation
  or the readout.
- One truncation artifact: "labia majora(L)" -> "l(L)" (generation began
  the unseen token sequence but failed to complete it).

Interpretation: the cue-position vector encodes the cue's semantic
neighborhood robustly and its fine detail (exact body part, polarity)
with limited resolution — or the readout loses that resolution. The
`last_subtoken` position may under-represent mid-span detail of long
verbose cues; `span_mean` is an untested variant.

## Scoring notes (for reproducibility)

- Strict lexical recall undercounts paraphrases; soft token matching
  overcounts within templated families (location cues share most tokens).
  Hand labeling was the decision instrument; strict (0.178) and A+B
  (0.557) bracket the truth.
- Two scoring bugs fixed en route: v4 split rows store the full XML target
  in `target_text`, which had contaminated gold-token matching; run_nla
  passthrough drops `cue_text` (recovered via `gold_cue_targets`).

## What this unlocks / next steps

1. **Layer/position map (now the main track).** Same v4 recipe swept over
   layers (8/16/24/32) x positions (cue span, format last-token): where
   does detail survive, where does it fold into theme? This is the
   layer-wise Medical-NLA pilot with a validated instrument.
2. **Counterfactual faithfulness on the v4 reader** (cue removal:
   removed cue disappears AND retained cues stay) — now worth building,
   since there is a reader worth testing.
3. **Detail resolution**: span_mean extraction variant; multi-token /
   span injection if single-vector resolution proves to be the binding
   constraint.
4. The theme-vs-detail contrast (format position reads theme at ~0.31
   lexical; cue position reads content at 0.56 semantic) is itself the
   first measured point of the "evidence folds into conclusion"
   trajectory story.
