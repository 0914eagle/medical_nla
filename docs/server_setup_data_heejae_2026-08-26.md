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
[env] gpus      CUDA_VISIBLE_DEVICES=0,1
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
