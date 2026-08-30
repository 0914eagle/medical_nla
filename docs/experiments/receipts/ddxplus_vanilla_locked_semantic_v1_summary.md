# DDXPlus Vanilla Locked Semantic Baseline Receipt

Aggregate-only paper receipt. No generated clinical text is included.

## Population And Mapper

- locked readout rows: **10,028**
- original / cue-deleted / value-edited: **4,543 / 4,543 / 942**
- semantic requests: **1,369**
- residual semantic claims: **10,947**
- primary mapper: `gpt-5.6-sol`
- frozen semantic protocol SHA-256:
  `12e4500fa45f90d11c0146ad12e972afd9b5bd80128f49b388b11dea360b506b`
- semantic request SHA-256:
  `0eb1799bea4000ab0ebf289d0de80f0d12dccb236268d1dfca9c6f4026d31d25`
- merged judgement SHA-256:
  `b41746b2a9fad86049f9e8299d4d4d682b3a44b038d9fd020f4ad37c79e1d21e`
- exact request population: **yes**; duplicate IDs: **0**
- parser-valid requests after targeted retry: **1,369/1,369**
- replacement judgements: **3 rows / 3 unique request IDs**

## Mapping Audit

- lexical mappings: **0**
- raw AI mappings: **0**
- accepted AI mappings: **0**
- emitted claims after case-level deduplication: **0**
- rows with at least one emitted claim: **0/10,028**
- audit interpretation: `mapper_confirmed_no_ontology_claims`

## Paper Values

| panel | metric | value | denominator |
|---|---|---:|---:|
| static | finding micro F1 | .0000 | 4,543 originals |
| static | same-diagnosis shuffled F1 | .0000 | 4,121 pairs |
| static | own-shuffled gap | +.0000 | 4,121 pairs |
| static | native-value end-to-end accuracy | .0000 | 2,136 targets |
| counterfactual | deletion original hit | .0000 | 4,540 pairs |
| counterfactual | deletion phantom | .0000 | 4,540 pairs |
| counterfactual | removal success | N/A | original-hit denominator 0 |
| counterfactual | untouched retention | N/A | original-hit denominator 0 |
| counterfactual | replacement hit | .0000 | 539 edits |
| counterfactual | old-value persistence | .0000 | 539 edits |
| counterfactual | clean switch | N/A | conditional denominator 0 |

The zero phantom rate is not evidence of deletion tracking because the method
never emitted the target finding in the corresponding original cases. The
result supports a failure-to-read boundary for the public Vanilla NLA, not an
absence of clinical information in the activation; the frozen probe and
structured reader provide the positive decodability controls.
