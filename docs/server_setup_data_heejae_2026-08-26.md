# Secondary 4090 server setup (`/data/heejae`)

This procedure stages an authorized second server for the DiReCT experiments.
The source server is `165.132.76.125`, where the data root is `/data1/heejae`.
The destination data root is `/data/heejae`. Restricted DiReCT text remains
outside git under `/data/heejae/restricted/direct`.

## 1. Clone the code on the destination server

```bash
mkdir -p /data/heejae /home/eagle0914
cd /home/eagle0914
git clone https://github.com/0914eagle/medical_nla.git
cd /home/eagle0914/medical_nla
```

If the repository already exists, use `git pull origin main` instead.

## 2. Run the one-command bootstrap

The destination needs `git`, `ssh`, `rsync`, `uv`, and `python3`. The script
copies only the restricted DiReCT tree, the native Llama judge checkpoint, and
the Gemma/NLA cache entries needed by the planned experiment. It recreates the
virtual environment rather than copying it.

```bash
cd /home/eagle0914/medical_nla

SOURCE_HOST=eagle0914@165.132.76.125 \
SOURCE_DATA_ROOT=/data1/heejae \
DEST_DATA_ROOT=/data/heejae \
COPY_HF_MODELS=1 \
COPY_DDXPLUS=0 \
bash scripts/bootstrap_direct_server.sh
```

The first SSH connection asks for the source-server password. SSH connection
multiplexing reuses that authenticated connection for subsequent `rsync`
calls. Configure an SSH key if policy permits for unattended transfers.

To transfer data and checkpoints first and create the environment later, add
`INSTALL_ENV=0`. This mode does not require `uv` on the destination yet:

```bash
SOURCE_HOST=eagle0914@165.132.76.125 \
SOURCE_DATA_ROOT=/data1/heejae \
DEST_DATA_ROOT=/data/heejae \
COPY_HF_MODELS=1 COPY_DDXPLUS=1 INSTALL_ENV=0 \
bash scripts/bootstrap_direct_server.sh
```

Set `COPY_DDXPLUS=1` only when the destination will also run the DDXPlus
grounding experiments. Existing activations and old result trees are not copied
by default because they are large and are not required for DiReCT E1.

## 3. Activate the destination environment

```bash
cd /home/eagle0914/medical_nla
source /data/heejae/uv/medical_nla/bin/activate
source scripts/env.sh /data/heejae
```

Expected environment lines include:

```text
[env] data root /data/heejae
[env] hf cache  /data/heejae/hf_cache
[env] gpus      CUDA_VISIBLE_DEVICES=2,3
```

Do not set `TRANSFORMERS_CACHE`. `env.sh` deliberately uses only `HF_HOME` to
avoid duplicate checkpoint caches.

## 4. Verify GPU and copied assets

```bash
nvidia-smi
python scripts/check_gpu_setup.py

find /data/heejae/restricted/direct/samples -type f -name '*.json' | wc -l
find /data/heejae/restricted/direct/diagnostic_kg -type f -name '*.json' | wc -l
ls -lh /data/heejae/models/Meta-Llama-3-8B-Instruct/original/
```

The expected restricted counts are 511 sample files and 24 KG files. The model
directory must contain `consolidated.00.pth`, `params.json`, and
`tokenizer.model`.

## 5. Re-run the official evaluator smoke test

```bash
DATA_ROOT=/data/heejae GPU=0 LIMIT=10 FORCE=1 \
nohup bash scripts/run_direct_official_smoke.sh \
  > /data/heejae/medical_nla/logs/direct_official_smoke.log 2>&1 &

tail -f /data/heejae/medical_nla/logs/direct_official_smoke.log
```

Success requires ten evaluation JSONs, zero private errors, diagnosis/category
accuracy of 1.0, unsmoothed observation precision/recall of 1.0, and
`Obscomp`, `Expcom`, and `Expall` of 1.0. Official `Obspre` and `Obsrec` remain
below 1.0 on an oracle because the released statistics code adds one to both
denominators.

```bash
wc -l /data/heejae/restricted/direct/evaluator_smoke/oracle_10/private_errors.jsonl
find /data/heejae/restricted/direct/evaluator_smoke/oracle_10/evaluations \
  -type f -name '*.json' | wc -l
cat /data/heejae/restricted/direct/evaluator_smoke/oracle_10/reports/official_metrics_summary.md
```

Only aggregate summaries may be shared. Prediction, evaluation, manifest, and
error JSONL files contain or derive from restricted clinical text.

## 6. DiReCT E1 source-run smoke on physical GPUs 2 and 3

After the evaluator smoke passes, run ten cases through the exact E1 path. The
script generates one source CoT/diagnosis and teacher-forces that same response
to extract P0 (prompt boundary), P1 (answer boundary before the diagnosis), and
P2 (diagnosis token) at hidden-state indices 16, 24, and 32. All prompt,
transcript, manifest, and tensor files remain under the restricted tree.

```bash
cd /home/eagle0914/medical_nla
source /data/heejae/uv/medical_nla/bin/activate

DATA_ROOT=/data/heejae GPUS=2,3 LIMIT=10 BATCH_SIZE=1 \
nohup bash scripts/run_direct_e1_pipeline.sh \
  > /data/heejae/medical_nla/logs/direct_e1_smoke10_v1.log 2>&1 &

tail -f /data/heejae/medical_nla/logs/direct_e1_smoke10_v1.log
```

The expected number of extraction rows is three times the number of naturally
parsed source answers. Forced answer completion is disabled because it would
create a second source run. With ten parsed answers there are 30 rows; at each
layer their entries are divided between the `last_token` P0 manifest and the
`last_subtoken` P1/P2 manifest.

```bash
cat /data/heejae/restricted/direct/e1/direct_e1_smoke10_v1/reports/activation_rows_summary.md

find /data/heejae/restricted/direct/e1/direct_e1_smoke10_v1/activations \
  -name manifest.jsonl -print -exec wc -l {} \;
```

Do not start `LIMIT=0` until the ten-row token-position audit is complete.

### Parallel full E1 after the smoke gate

The two authorized servers can process disjoint splits. Keep the roots and
physical GPU IDs explicit; do not reuse one server's command on the other.

Server `165.132.76.62` (`/data/heejae`, physical GPUs 2 and 3) owns the
training and validation states:

```bash
DATA_ROOT=/data/heejae GPUS=2,3 LIMIT=0 BATCH_SIZE=1 \
SPLITS="train val_seen" RUN_NAME=direct_e1_trainval_v1 FORCE=1 \
nohup bash scripts/run_direct_e1_pipeline.sh \
  > /data/heejae/medical_nla/logs/direct_e1_trainval_v1.log 2>&1 &
```

Server `165.132.76.125` (`/data1/heejae`, physical GPUs 0 and 1) owns the two
test splits:

```bash
DATA_ROOT=/data1/heejae GPUS=0,1 LIMIT=0 BATCH_SIZE=1 \
SPLITS="test_seen test_pdd_heldout" RUN_NAME=direct_e1_test_v1 FORCE=1 \
nohup bash scripts/run_direct_e1_pipeline.sh \
  > /data1/heejae/medical_nla/logs/direct_e1_test_v1.log 2>&1 &
```

These runs cover 325 and 171 source cases respectively. They are shards of one
experiment, not replications. Keep their run roots distinct and preserve the
private transcripts and activation manifests under each server's restricted
tree.

These commands use `direct_patient_pdd_v1` and are now classified as the
exploratory pilot. Do not silently reuse their outputs as the frozen
downstream-confirmatory result.

## 7. Reproduce the frozen confirmatory split on server 62

Regenerate the private manifest locally so `source_path` uses `/data/heejae`
rather than server 125's `/data1/heejae`. Then reproduce the split with the
same seed and forbidden pilot labels.

```bash
python scripts/make_direct_canonical_manifest.py \
  --samples-root /data/heejae/restricted/direct/samples \
  --data-list /data/heejae/restricted/direct/official_repo/utils/data_loading_analysisi/data_list.csv \
  --output-jsonl /data/heejae/restricted/direct/manifests/direct_canonical_v3_private.jsonl \
  --summary-md /data/heejae/restricted/direct/audit/direct_canonical_v3_summary.md

python scripts/make_direct_patient_pdd_splits.py \
  --manifest /data/heejae/restricted/direct/manifests/direct_canonical_v3_private.jsonl \
  --out-dir /data/heejae/restricted/direct/splits/direct_patient_pdd_confirmatory_v1 \
  --seed 17 \
  --heldout-fraction 0.20 \
  --train-fraction 0.70 \
  --val-fraction 0.15 \
  --min-heldout-label-rows 3 \
  --min-remaining-category-rows 3 \
  --forbid-heldout-pdds HFrEF HFpEF NSTEMI "Low-risk PE" "Non-Allergic Asthma"
```

The server-local manifest file hash may differ because absolute paths differ.
The logical population hash and all split ID hashes must match server 125:

```text
population:       7d0a89a880fa868959099b7146c369cccaac5e7701d7ce5d8f01356ecfb68894
train:            0fb3e49aa8a3dd5f853399967fe2739423ceffc9ce1522533b32505c628f722c
val_seen:         5e1e6ce1b687b4f20dc9803004d3ae3b971018b166c8bdc08b283860c731068e
test_seen:        48d3c0beb2ffc13e86f5875e00415accf36154ec740f5e0a59dd770036346ed9
test_pdd_heldout: 12d2594951bbf32e20d0d140dab9f50ffdfb6e42e863aea4187ea0f903b3da7a
```

Future confirmatory E1 runs must set the split directory explicitly:

```bash
SPLIT_DIR=/data/heejae/restricted/direct/splits/direct_patient_pdd_confirmatory_v1
```
