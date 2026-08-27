# DDXPlus E5 canonical population

## Purpose

DDXPlus is not a second clinical-explanation training corpus in the primary
experiment. Medical-NLA is adapted on DiReCT only, and DDXPlus is the independent
controlled testbed for asking whether a readout depends on the paired activation
and follows a dataset-native evidence change.

The old 4,900-case artifact was sampled from one CSV for the former referral-note
study. It must not be reused for E5 because it does not preserve the new
validation/test roles and its old cue-swap code can replace a finding with an
arbitrary finding from another evidence type.

## Frozen population

The new builder reads the official files separately:

| official split | population rule | role |
|---|---:|---|
| `validate.csv` | common eligible diagnoses, capped at 100 each | thresholds, control choices, validation mean activation |
| `test.csv` | same diagnosis support, capped at 100 each | locked Table 3 and Figure 3 |
| `train.csv` | not used in primary | optional DDX grounding-adaptation ablation only |

Within each split, cases are reservoir-sampled independently by diagnosis with
seed 17. A primary case must have at least three clean rendered cues and must not
literally contain the gold diagnosis or an accepted alias. Every derived variant
of a case remains in the same official split.

The diagnosis set is the intersection of labels with at least one eligible case
in every supplied official split. The builder then takes up to 100 rows per
diagnosis in each split. It does not choose the largest buckets or require every
diagnosis to reach the cap. This rule is fixed from input eligibility only and
does not inspect model outputs, activations, or evaluation scores.

The first release audit showed that validation contains 47 eligible diagnoses:
44 reach the 100-case cap, while `bronchiolitis`, `cluster_headache`, and `ebola`
provide 28, 8, and 89 eligible rows. The former 4,900-case expectation was
therefore impossible under the frozen leakage filters. Final validation and test
counts are recorded only after both official files have been scanned and common
support has been applied.

## Derived variants

For each selected base case, the builder writes the original CoT-P0 prompt and
one cue-deletion CoT-P0 prompt. It also writes a value-edit CoT-P0 prompt when
the selected evidence has another value explicitly declared by
`release_evidences.json` and that value can be rendered cleanly.

`P0` means the last hidden state at the chat-formatted prompt boundary before
any model-generated reasoning. The primary condition is explicitly **CoT-P0**,
not Direct-P0, because Medical-NLA is trained on DiReCT CoT-P0 activations. The
builder also emits `activation_rows_validation_direct_control.jsonl` for the
unchanged validation base cases. This paired Direct-P0 file measures instruction
sensitivity and is never mixed into the primary grounding denominator or the
locked test.

The value edit is constrained as follows:

```text
same base case
same evidence_id
original declared value -> another declared value
all other cues unchanged
```

Binary evidence with no declared alternative is tested by deletion. The builder
does not invent `Y` or `N`, and it does not draw an unrelated medical sentence
from a global cue vocabulary.

The hard-shuffle manifest assigns every case one unique donor from the same
diagnosis. Among deterministic cyclic derangements it first minimizes identical
evidence/value signatures and then cue-count and prompt-length differences. It
records whether the evidence/value signature differs. Source-answer agreement is
reported after generation but does not reject a pair: holding the diagnosis fixed
while changing case evidence is the intended hard negative.

## Controls

- `own pair`: readout and activation from the same base case.
- `hard shuffle`: same diagnosis, similar size, different base case activation.
- `zero activation`: language-prior floor.
- `validation mean activation`: non-case-specific activation floor.
- `cue deletion`: target cue removed, remaining cues unchanged.
- `native value edit`: one evidence value changed within its DDXPlus ontology.
- `activation swap`: metadata/text target from one case with another case state.
- `Direct-P0 sensitivity`: same validation case at the Direct-instructed prompt boundary.

The mean control is computed from validation activations. Computing it from the
test population would let locked-test information enter the control.

## Run on server 62

This is CPU preprocessing. It can run while the SFT jobs occupy GPUs 2 and 3.

```bash
cd /home/eagle0914/medical_nla
git pull origin main
source /data/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla

DATA_ROOT=/data/heejae \
nohup bash scripts/run_ddxplus_e5_data_prep.sh \
  > /data/heejae/medical_nla/logs/ddxplus_e5_data_prep_v1.log 2>&1 &
```

Monitor:

```bash
tail -f /data/heejae/medical_nla/logs/ddxplus_e5_data_prep_v1.log
```

Expected counts are intentionally not hard-coded. `summary.md` reports common
diagnosis support, per-split base cases, counterfactuals, and pair coverage after
both official files are scanned. Native value-edit coverage is also empirical.

## Verify

```bash
ROOT=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1

cat "$ROOT/summary.md"
wc -l \
  "$ROOT/cases_validation.jsonl" \
  "$ROOT/cases_test.jsonl" \
  "$ROOT/counterfactual_cases_validation.jsonl" \
  "$ROOT/counterfactual_cases_test.jsonl" \
  "$ROOT/activation_rows_validation.jsonl" \
  "$ROOT/activation_rows_test.jsonl" \
  "$ROOT/activation_rows_validation_direct_control.jsonl" \
  "$ROOT/hard_shuffle_pairs_validation.jsonl" \
  "$ROOT/hard_shuffle_pairs_test.jsonl"
```

The protocol file records source SHA-256 values, selected case hashes, pair
hashes, exclusion counts, and cross-split selected patient-ID overlap.

## Copy to server 125

Do not rerun sampling independently. Copy the frozen artifacts so both servers
use byte-identical case and pair IDs.

```bash
# Run on server 125.
mkdir -p /data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1
rsync -a --info=progress2 \
  eagle0914@165.132.76.62:/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1/ \
  /data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/
```

After the copy, compare `protocol.json` and every JSONL hash before dividing GPU
activation extraction across servers.

The activation merge utility writes `summary.json` in the merged activation
directory; it does not emit a Markdown summary.

## Next gate

Data preparation alone does not fill Table 3. After the three DiReCT SFT seeds
finish and a checkpoint is selected using DiReCT validation only:

1. extract primary CoT-P0/HS32 activations for DDXPlus validation;
2. run vanilla NLA and the frozen Medical-NLA checkpoint;
3. extract Direct-P0 for the paired validation instruction-sensitivity control;
4. choose scoring thresholds and report hard-shuffle source-answer agreement;
5. freeze the E5 scoring protocol;
6. extract and score the locked DDXPlus test population;
7. run AR round-trip only for methods that pass the paired grounding gate.
