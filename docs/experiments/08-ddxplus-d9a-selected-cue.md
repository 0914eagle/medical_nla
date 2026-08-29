# DDXPlus D9a Selected Changed-Cue Pipeline

This pipeline is a mechanism test on the one pre-existing deletion arm per
DDXPlus training case. It is not an all-cue support mask and it does not finish
the multi-claim Medical-NLA training phase.

## Frozen Scope

- Server: 125 only (`/data1/heejae`)
- Hidden state: CoT-P0 / HS32 / last token
- Train scoring: two-fold OOF finding heads, `crc32(base_id) % 2`
- Minimum fold positives: 5
- Donors: same fold, same diagnosis, candidate cue absent, at most 5
- Support rule: presence AND deletion delta AND donor margin
- Validation selection: false-support rate at most 5%, then maximum coverage
- Unsupported selected cue: exclude the case; never emit abstention
- Other input cues: untested
- Value edit: excluded from the first smoke
- Locked DDXPlus test: never read

## Phase 1: Read-Only Audit

First locate the validation-selected HS32 artifact, then run both train and
validation audits.

```bash
cd /home/eagle0914/medical_nla
git pull origin main

find /data1/heejae/medical_nla/results \
  /data1/heejae/medical_nla/data \
  -type f -name finding_value_hs32.pt -print

DATA_ROOT=/data1/heejae \
GPU=0 \
PROBE_ARTIFACT=/absolute/path/to/finding_value_hs32.pt \
MODE=audit \
bash scripts/run_ddxplus_d9a_pipeline.sh
```

Outputs:

```text
/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/
  d9a_selected_changed_cue_v1/
    train_audit/{private_scores.jsonl,report.json,summary.md}
    validation_null_audit/{private_scores.jsonl,report.json,summary.md}
```

Stop here and inspect fold coverage, eligibility, donor coverage, and positive
versus null score distributions. No cut grid is embedded in the repository.

## Phase 2: Explicit Candidate Grid

Only after Phase 1 review, provide all candidate values explicitly.

```bash
DATA_ROOT=/data1/heejae \
GPU=0 \
PROBE_ARTIFACT=/absolute/path/to/finding_value_hs32.pt \
MODE=select \
PRESENCE_THRESHOLDS="<reviewed values>" \
DELETION_THRESHOLDS="<reviewed values>" \
DONOR_THRESHOLDS="<reviewed values>" \
bash scripts/run_ddxplus_d9a_pipeline.sh
```

This writes `recommendation_unapproved.json`. It does not authorize training.

## Phase 3: Human Approval

After reviewing `cut_selection/summary.md` and the private candidate table,
record approval without changing the recommendation.

```bash
ROOT=/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/d9a_selected_changed_cue_v1

python scripts/approve_ddxplus_d9a_support_protocol.py \
  --recommendation "$ROOT/cut_selection/recommendation_unapproved.json" \
  --validation-scores "$ROOT/validation_null_audit/private_scores.jsonl" \
  --approved-by heejae \
  --approved-at '<ISO-8601 timestamp>' \
  --confirmation I_APPROVE_D9A_CUTS \
  --output "$ROOT/cut_selection/protocol_approved.json"
```

The command records both validation-score and recommendation SHA256 values.

## Phase 4: Supported Pair Dataset

```bash
DATA_ROOT=/data1/heejae \
GPU=0 \
PROBE_ARTIFACT=/absolute/path/to/finding_value_hs32.pt \
MODE=build \
APPROVED_PROTOCOL=/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/d9a_selected_changed_cue_v1/cut_selection/protocol_approved.json \
bash scripts/run_ddxplus_d9a_pipeline.sh
```

The builder emits one claim per supported case and the original/deleted
activation paths. It excludes ineligible and below-cut cases. It emits no
abstention, no value arm, and no claim for untested cues.

## Approved D10 Contract

Human approval on 2026-08-29 resolved the earlier terminology conflict. The
first smoke is one changed claim scored under original and deleted activations,
not a literal 2x2. No deleted-state target, negation, abstention, or untested cue
is invented. Literal 2x2 is deferred to D9b/value-edit work.

One retained cue is fixed for evaluation only. It must have the exact same cue
text in original and deleted rows and is selected by the minimum
`SHA256(base_id || NUL || cue_text)`. The model never sees its score during
selection or D10 training.

## Completed Results — 2026-08-29

All results below are from official DDXPlus train or validation. The locked
test split was not read.

### Train OOF audit

The train audit scored 4,655 existing original/deletion pairs with two
cross-fitted finding heads. Every retained probability was produced by the
head trained on the opposite `crc32(base_id) % 2` fold.

| quantity | result |
|---|---:|
| base cases | 4,655 |
| finding labels | 91 |
| score eligible | 4,652 |
| changed cue outside ontology | 3 |
| changed cue below fold support | 0 |
| donor available | 3,109 |
| donor unavailable | 1,546 |
| donor-available fraction of all cases | 0.6679 |

Fold support was above the frozen minimum of five for every label.

| held-out fold | train rows | held-out rows | zero-positive labels | below-five labels | minimum positives |
|---:|---:|---:|---:|---:|---:|
| 0 | 2,243 | 2,412 | 0 | 0 | 14 |
| 1 | 2,412 | 2,243 | 0 | 0 | 15 |

Train OOF selected-cue score distributions:

| score | mean | q05 | q25 | median | q75 | q95 |
|---|---:|---:|---:|---:|---:|---:|
| deletion delta | 0.4828 | 0.0000 | 0.0140 | 0.4780 | 0.9488 | 0.9997 |
| donor margin | 0.5629 | 0.0083 | 0.2233 | 0.5438 | 0.9710 | 0.9988 |

### Validation positive/null audit

Validation contained 4,525 cases. Two selected changed cues were outside the
91-label ontology, leaving 4,523 positive rows. The primary AND rule could be
evaluated for 3,034 positives because those rows had a valid cue-absent donor.
Of the corresponding 3,034 deterministic null controls, 2,964 had the extra
donor required to compute their own donor margin.

| population | eligible / available | rate |
|---|---:|---:|
| selected changed-cue positive | 3,034 / 4,523 | 0.6708 |
| cue-absent null | 2,964 / 3,034 | 0.9769 |

Validation score distributions:

| score | population | mean | q05 | q25 | median | q75 | q95 |
|---|---|---:|---:|---:|---:|---:|---:|
| deletion delta | positive | 0.6554 | 0.000005 | 0.3745 | 0.7949 | 0.9824 | 0.9999 |
| deletion delta | null | 0.0131 | -0.6775 | -0.0174 | 0.0030 | 0.1321 | 0.4375 |
| donor margin | positive | 0.7213 | 0.1896 | 0.5280 | 0.7700 | 0.9886 | 0.9993 |
| donor margin | null | 0.0094 | -0.2295 | -0.0639 | -0.0007 | 0.0391 | 0.3863 |

Presence probabilities were nearly saturated for positives but not for nulls.

| population | q01 | q05 | q25 | median | q75 | q95 | q99 |
|---|---:|---:|---:|---:|---:|---:|---:|
| positive | 0.9787 | 0.9955 | 0.9998 | 0.999997 | 1.0000 | 1.0000 | 1.0000 |
| null | 0.000007 | 0.00013 | 0.0068 | 0.1696 | 0.5065 | 0.8813 | 0.9845 |

### Frozen support cuts

After inspecting the read-only distributions, 245 explicit candidates were
evaluated:

- presence: `0.80, 0.90, 0.95, 0.975, 0.99`
- deletion delta: `0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50`
- donor margin: `0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50`

The predeclared rule selected the candidate with maximum positive coverage
among candidates with validation false-support at most 5%.

| threshold / result | value |
|---|---:|
| presence threshold | 0.9000 |
| deletion-delta threshold | 0.0000 |
| donor-margin threshold | 0.0000 |
| positive supported | 3,032 / 3,034 |
| positive coverage | 0.9993 |
| null false supported | 112 / 2,964 |
| false-support rate | 0.0378 |
| Wilson 95% CI for false-support | [0.0315, 0.0453] |

The human-approved protocol and its input hashes are recorded under
`cut_selection/protocol_approved.json`. The `.90/0/0` operating point is the
primary cut because changing to a stricter point after observing the grid would
violate the declared maximum-coverage rule. For context only, `.99/0/0` had
positive coverage `0.9763` and false-support `0.0054`; it is not the training
cut.

### Approved train pairs

Applying the frozen cuts to train produced:

| disposition | cases |
|---|---:|
| selected changed-cue supported | 3,104 |
| outside ontology / ineligible | 3 |
| donor unavailable | 1,543 |
| eligible but below cut | 5 |
| total | 4,655 |

Thus `3,104 / 4,655 = 0.6668` of the complete train population enters the D9a
mechanism smoke, while `3,104 / 3,109 = 0.9984` of donor-eligible cases pass the
support cuts. Each retained row contains one selected claim plus its original
and cue-deleted activation paths. It contains no abstention target, no value
arm, and no target for another cue.

Artifact hashes:

| artifact | SHA256 |
|---|---|
| train OOF scores | `6ac12c0a7b9347247ce2f6fe03ef092a8d1f2e8db5addb3ccda0537de7222ace` |
| validation positive/null scores | `ec9a8c6ec84ab084acd4891cf6df397e8239662eb9a7fabad611aa8444922056` |
| approved protocol | `a968a63fcfc381b27f2dae6d62c14515da43000fd6b1b6dd66c81ca13f839e86` |

## Result Interpretation

The support audit did not reveal a large unsupported selected-cue population.
Only five donor-eligible train cases failed the frozen cuts. The dominant loss
of coverage was the inability to construct a same-fold, same-diagnosis,
cue-absent donor for 1,543 otherwise eligible cases. Therefore D9a establishes
the following narrower claims:

1. For the pre-existing one-deletion-per-case sample, almost every selected cue
   with a valid donor is represented directionally at CoT-P0/HS32.
2. The result does not establish support for all 21,331 train cue occurrences.
3. The `.90/0/0` cut is mainly a presence-plus-directionality gate, not a strong
   effect-size filter. The large median deletion delta (`0.7949`) and donor
   margin (`0.7700`) describe the population, but the case-level cut accepts any
   nonnegative delta and margin.
4. The 3,104 pairs are appropriate for a changed-claim mechanism smoke. They
   are not sufficient to claim completion of a multi-claim Medical-NLA.

## Proposed Next Objective

For retained case `i`, let `y_i` be its one selected changed-cue claim,
`h_i^orig` its original activation, and `h_i^del` its cue-deleted activation.
The proposed D9a objective is:

```text
g_i = NLL(y_i | h_i^del) - NLL(y_i | h_i^orig)

L = L_SFT(y_i | h_i^orig)
    + lambda * temperature * softplus(-g_i / temperature)
```

This is explicitly **one claim by two activations (1x2)**. The deleted
activation receives no invented target. The fair smoke comparison is:

| arm | original SFT | original-vs-deleted ranking | seeds |
|---|---:|---:|---|
| original-only control | yes | no | 17, 29, 43 |
| D9a paired ranking | yes | yes | 17, 29, 43 |

Both arms must use the same initialization, 3,104-row population, optimizer
steps, pair order, and generation settings. Validation must remain on the
frozen selected-cue population. The D5 gate remains:

- all three seeds have the same improvement sign;
- disease-cluster bootstrap CI excludes zero;
- paired ranking improves changed-claim contrast by at least `0.05` over the
  original-only arm on the same subset;
- ranking improves `changed_gap - retained_gap` specificity over original-only
  and its diagnosis-cluster bootstrap interval excludes zero;
- original-arm changed-cue hit does not decrease;
- deleted-arm phantom does not increase.

The literal 2x2 objective is deferred to D9b. It requires per-cue deletion
activations so the deleted state can use another independently supported claim
instead of abstention, an untested cue, or an invented negation.

## Run D10 On Server 125

The wrapper rebuilds the 3,104 train pairs with retained-control fields, builds
the frozen validation pair population, then runs original-only and ranking arms
for seeds 17/29/43. Each wave uses GPUs 0,1 for control and 2,3 for ranking.

```bash
cd /home/eagle0914/medical_nla
git pull origin main

DATA_ROOT=/data1/heejae \
nohup bash scripts/run_ddxplus_d10_1x2_smoke_4gpu_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d10_1x2_smoke20_v1_queue.log 2>&1 &
```

Monitor:

```bash
tail -f /data1/heejae/medical_nla/logs/ddxplus_d10_1x2_smoke20_v1_queue.log
tail -f /data1/heejae/medical_nla/logs/ddxplus_d10_1x2_smoke20_v1_{original_only,ranking}_seed17_{train,specificity}.log
```

The aggregate output is:

```text
/data1/heejae/restricted/direct/e4/
  ddxplus_d10_1x2_smoke20_v1_validation_v1/
    paired_arm_comparison_summary.md
```

Teacher-forced results settle changed-gap and specificity conditions first.
Generated original-hit and deleted-phantom rates are required only if those
necessary teacher-forced conditions pass.

## D10 Result — 2026-08-29

The three-seed teacher-forced gate failed before generation was needed.

| seed | changed-gap delta | changed cluster 95% CI | retained-gap delta | specificity delta | specificity cluster 95% CI |
|---:|---:|---:|---:|---:|---:|
| 17 | +0.0005 | [-0.0006, +0.0016] | +0.0010 | -0.0005 | [-0.0020, +0.0010] |
| 29 | +0.0028 | [+0.0017, +0.0039] | -0.0000 | +0.0029 | [+0.0015, +0.0045] |
| 43 | +0.0030 | [+0.0015, +0.0048] | -0.0007 | +0.0037 | [+0.0017, +0.0059] |

All changed-gap deltas were below the frozen `0.05` minimum. Seed 17 also had
a changed-gap interval spanning zero and negative specificity. Because these
are mandatory D5 conditions, the smoke is a promotion failure regardless of
unrun generation checks. No post-result lambda, temperature, or step sweep is
authorized.
