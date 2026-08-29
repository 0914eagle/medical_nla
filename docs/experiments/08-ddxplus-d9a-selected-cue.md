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

## Deliberate Stop Before Training

The approved discussion currently says both:

1. D9a deleted/unsupported cases must not receive abstention or untested-cue
   targets.
2. D10 starts with a deletion-only 2x2 ranking objective.

A literal 2x2 needs two activation states and two valid targets. D9a defines
only the positive selected-cue target. The deleted-state target is not yet
defined. Training therefore stops after pair construction until that four-cell
contract is approved. Implementing a one-target original-versus-deleted ranker
would be a 1x2 objective, not the agreed 2x2.

