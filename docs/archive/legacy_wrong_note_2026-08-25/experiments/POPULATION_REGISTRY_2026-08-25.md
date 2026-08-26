# Population registry -- canonical analysis cohorts

> **Purpose.** This is the single source of truth for denominators used in the
> paper, figures, and professor presentation. A number without a cohort label is
> not portable across analyses. Historical cohorts remain reproducibility audits
> and must not be mixed with the canonical primary cohorts below.

## 1. Canonical primary cohorts

| Key | Corpus / cohort definition | n | Moved decomposition | Primary use |
|---|---|---:|---|---|
| `ddx_clean` | DDXPlus; canonical no-note correct; explicit gold name absent from presentation | **1,204** | **287 = 86 adopted + 201 other diagnosis** | Main behavior table/Figure 2, wording arms, Direct-selected CoT analysis |
| `ddx_all` | DDXPlus; canonical no-note correct, including explicit-gold rows | **1,729** | **319 = 89 adopted + 230 other diagnosis** | Trajectory, single-run detection, correction ladder, reader-trust base IDs |
| `ddx_silent` | `ddx_all` rows whose wrong-note answer does not name the suggestion | **1,628** | Label remains paired `moved`; do not infer it from silence | Silent-column channel comparison in Table 2b / Figure 4(a) |
| `ddx_explicit_gold` | `ddx_all` rows with a gold diagnosis name or alias in the presentation | **525** | **32 = 3 adopted + 29 other diagnosis** | Sensitivity analysis for explicit answer anchoring |
| `mcr_behavior` | MCR; canonical no-note correct and evaluable intervention rows | **1,452** | **427 = 127 adopted + 300 other diagnosis** | Main behavior replication / Figure 2 |

`moved` means that the wrong note causally changed the answer relative to the
same case's no-note answer. It is the union of direct suggestion adoption and
loss of the gold answer to another diagnosis. It is not synonymous with
`wrong_answer == suggestion`.

## 2. Canonical analysis-specific cohorts

| Analysis | Unit and denominator | Why it differs |
|---|---:|---|
| Reader-trust full canonical rows | **2,860 rows / 1,729 base IDs** | Multiple account conditions are evaluated for each canonical DDXPlus case. |
| Reader-trust controlled comparison | **716 rows/channel**; shuffled readout **715** | Requires a complete same-case set of no-account, real, and shuffled accounts. |
| Cue-token heldout readout | **438 cue rows per layer** | Unit is one held-out cue span, not one diagnosis case. |
| Final-prompt diagnosis-heldout readout | **727 seen / 800 heldout diagnosis rows** | Diagnosis-heldout SFT split, separate from the intervention cohort. |
| MCR wrong-note readout | **1,543 wrong-arm rows** | Readout was generated for every MCR intervention case, not only the canonical no-note-correct behavior subset. |
| MCR mixed readout file | **3,086 = 1,543 none + 1,543 wrong** | Storage artifact only. Never report a pooled faithfulness score across arms. |

The MCR wrong-arm readout population (`1,543`) and the MCR canonical behavior
population (`1,452`) answer different questions and must not be placed in the
same table without explicit labels.

## 3. Historical and sensitivity cohorts

| Cohort | n | Status |
|---|---:|---|
| DDXPlus generation-time source-correct | **1,747** | Fixed-cohort provenance/audit only. The canonical matcher retains 1,729. |
| DDXPlus historical clean | **1,220** | Fixed-cohort audit only. Canonical clean is 1,204. |
| DDXPlus historical explicit-gold | **527** | Fixed-cohort audit only. Canonical explicit-gold is 525. |
| Historical DDXPlus moved | **321 = 91 + 230** | Fixed-cohort audit only. Canonical all moved is 319 = 89 + 230. |
| Historical silent by `answer_names` | **1,641** | Fixed-cohort channel audit only. Canonical silent is 1,628. |
| Alternative silent by `not took_the_hint` | **1,656** | Sensitivity definition, not the primary silent subset. |
| MCR generation-time source-correct | **1,543** | Extraction/readout pool; canonical behavior cohort is 1,452. |
| corpus-300 non-overlap clean, fixed cohort | **2,192** | Independent IDs are confirmed; arm accuracies are an appendix audit. |
| corpus-300 non-overlap clean, canonical | **2,137 expected** | Primary refresh is pending; do not fill Table 1 from the 2,192-row audit. |
| corpus-300 non-overlap behavior moved | **563 / 3,319** | Fixed-cohort behavior audit. |
| corpus-300 non-overlap ladder moved | **571 / 3,319** | Separate archived matcher output; do not combine with the 563-row behavior label. |

Historical values may appear in experiment logs and dated audit sections, but
must be labeled `fixed-cohort`, `historical`, `audit`, or `sensitivity`. They are
not valid substitutes for missing canonical cells or confidence intervals.

## 4. Figure and table mapping

| Asset | Required cohort |
|---|---|
| Main behavior accuracy and moved destination (Figure 2) | DDXPlus `ddx_clean` **1,204**; MCR `mcr_behavior` **1,452** |
| DDXPlus trajectory (Figure 3) | `ddx_all` **1,729**, grouped as unchanged 1,410 / other diagnosis 230 / adopted 89 |
| Detection (Figure 4a / Table 2b) | all `ddx_all` **1,729**; silent `ddx_silent` **1,628** |
| Correction ladder (Figure 4b / Table 3) | `ddx_all` **1,729**; moved subset **319** |
| Wording and Direct-selected CoT robustness | `ddx_clean` **1,204** |
| Reader-trust | canonical 1,729 base IDs; controlled result n=716/channel |

Figure 2 deliberately uses the clean cohort for both its accuracy and moved
destination panels. Figure 3 and Figure 4 deliberately use all canonical-eligible
rows because their causal labels and hidden-state files were rebuilt on that
population. These are different estimands, not an accidental denominator change.

## 5. Reporting rules

1. State the cohort key or definition whenever a denominator first appears.
2. Do not call `1,747`, `1,220`, or `1,641` canonical.
3. Do not compare percentages from `ddx_clean` and `ddx_all` as if only the method
   changed; their populations differ.
4. Define a detector label offline with the paired no-note/wrong-note runs, but
   evaluate the detector using only the wrong-note run.
5. For MCR readout faithfulness, join on `(base_id, arm)` and report the wrong arm
   separately. The pooled 3,086-row score is invalid.
6. New results enter `RESULTS_CANONICAL_2026-08-24.md` first, then the paper table,
   figures, outline, and presentation in that order.

## 6. Current canonical headline values

| Result | Canonical value |
|---|---:|
| DDXPlus wrong-note accuracy, clean | **.7625** (n=1,204) |
| MCR wrong-note accuracy | **.7066** (n=1,452; 29.34 pp below no-note reference) |
| Suggestion never probe top-1 at any landmark | **262/319 = .821** |
| Detection AUROC, all: answer / rule CoT / LLM monitor / AV / probe | **.6632 / .5434 / .7305 / .7511 / .9330** |
| Detection AUROC, silent: monitor / AV / probe | **.6904 / .8319 / .9881** |
| Correction, moved: first / r3 / r4 / r5 / r6 | **.0031 / .4545 / .4044 / .6301 / .8339** |
| Reader-trust real-minus-no-account: probe / CoT / readout | **+.0692 / -.0217 / -.0998** |

If a current document disagrees with this registry, the document is stale unless
it explicitly identifies a newer canonical rebuild and records its provenance in
the canonical results ledger.
