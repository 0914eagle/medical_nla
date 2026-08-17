# Results 2026-08-17: Format-Position Layer Sweep (L16/L24 + existing L32)

The other half of the trajectory map. The cue-position sweep showed WHERE
individual cue detail is readable (inverted U, peak L24). This sweep asks
whether the FORMAT position (prompt's last token — the answer-forming
state) holds readable per-cue detail at any depth: same v3 cue-first
recipe (no diagnosis text in targets), same diagnosis-heldout split,
same v2 lexical scorer, layers 16/24 newly trained, L32 from the v3 run.

## Headline: no layer rescues the format position

Mean cue_recall (v2 lexical scorer; same scorer across all cells, so the
curve shape is comparable even though the scorer undercounts paraphrase):

| layer | test_seen (727) | test_heldout (800) | seen − heldout |
|---|---:|---:|---:|
| L16 | 0.3597 | 0.1883 | +0.171 |
| L24 | **0.6839** | **0.2490** | +0.435 |
| L32 | 0.6251 | 0.1876 | +0.437 |

Heldout is flat-and-low (0.19–0.25) across all three layers. The v3
failure at L32 was NOT "wrong depth for the last token": there is no
intermediate layer where the answer position still carries case-specific
cue detail in naturally readable form. Compare the cue-position curve at
the same layers (heldout, hand-labeled A+B): 34.0 / 73.1 / 55.7. At L24
the same reader recipe reads 73% at cue tokens and ~25% at the format
token.

## Reading the seen column: the shortcut needs depth

Seen-pool recall rises L16 0.36 → L24 0.68 ≈ L32 0.63. In-distribution,
the format vector supports substantial cue-list output at L24/L32 — but
the heldout collapse shows this is the class→typical-cue-template
shortcut, not per-cue reading. Interesting sub-finding: even the shortcut
needs depth — at L16 the class identity is apparently not yet formed
enough at the last token to drive template output (0.36).

## Per-diagnosis heldout confirms template-spray, not reading

| diagnosis | L16 | L24 | interpretation |
|---|---:|---:|---|
| urti | 0.719 | 0.692 | cue vocab overlaps train classes (bronchitis — v1's collapse partner) → template spray scores |
| pulmonary_neoplasm | 0.373 | 0.359 | partial overlap (cough/weight-loss cues shared with train) |
| sle | 0.000 | 0.069 | distinctive cue set → ~zero transfer |
| inguinal_hernia | 0.000 | 0.156 | distinctive cue set → ~zero transfer |

Transfer happens exactly where heldout cue vocabulary coincides with
train cue vocabulary, and nowhere else — the signature of emitting
memorized typical-cue templates rather than reading case content.

## Caveats

- The summary's `verdict: likely seen-class classifier ... neither
  answers nor cues transfer` keys partly off answer_hit = 0, which is
  **vacuous by design** here: v3 targets contain no diagnosis text and
  the readouts emitted no answers at all (top readout answers `- (100)`,
  answer_in_train_vocab 0/800). Read only the cue columns. The good news
  inside the vacuous columns: zero diagnosis leakage at every layer —
  the no-shortcut target design held.
- Lexical scorer undercounts paraphrase (v4 showed strict 0.17 vs manual
  0.73 at L24 cue positions). Absolute format-position numbers are
  therefore floors; but the flat heldout curve and the seen−heldout gap
  are scorer-independent shape facts.

## What this closes and what it opens

Closed: the trajectory map now has both halves. Readable clinical
evidence lives at cue token positions (up to 73% at L24) and — at every
depth measured — is NOT present in naturally readable per-case form at
the answer position, where only class-level signal sits (linear probe
99%). "Evidence folds into conclusion" is a statement about POSITION as
much as depth: there is no layer at which the last token still looks
like evidence.

Opened (error-notes design consequence): the internal-conclusion read
for error notes should come from the format position via the class
channel (probe, or v2-style reader), and the evidence read from cue
positions at L24. Do not spend GPU hunting a middle layer where the
format token reads as evidence — this sweep says it does not exist.
