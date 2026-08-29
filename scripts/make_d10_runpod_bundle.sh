#!/usr/bin/env bash
set -euo pipefail

# Run on server 125. Collects everything the D10 budget calibration needs into
# one tarball with paths relative to /data1/heejae, so extracting it at
# /data1/heejae on the pod reproduces the exact layout the frozen runner expects.
#
# DDXPlus derivatives only. DiReCT files must never enter this bundle; the file
# list is built explicitly from the D9a artifacts and the three activation
# manifests, never from a directory glob.

DATA_ROOT="${DATA_ROOT:-/data1/heejae}"
OUT="${OUT:-${DATA_ROOT}/medical_nla/d10_runpod_bundle.tar.gz}"

PROBE_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_probe_train_v1"
CF_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_counterfactual_train_v1"
E5_ROOT="${DATA_ROOT}/medical_nla/data/ddxplus_e5_canonical_v1"
D9A="${CF_ROOT}/d9a_selected_changed_cue_v1"
TRAIN_MANIFEST="${PROBE_ROOT}/activations/ddxplus_probe_train_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"
CF_MANIFEST="${CF_ROOT}/activations/ddxplus_counterfactual_train_cot_p0_merged_v1/layer32/last_token/manifest.jsonl"
VAL_MANIFEST="${E5_ROOT}/activations/ddxplus_e5_validation_cot_p0_merged_server125_v1/layer32/last_token/manifest.jsonl"

for path in "${TRAIN_MANIFEST}" "${CF_MANIFEST}" "${VAL_MANIFEST}" \
  "${D9A}/train_audit/private_scores.jsonl" \
  "${D9A}/validation_null_audit/private_scores.jsonl" \
  "${D9A}/cut_selection/protocol_approved.json"; do
  test -s "${path}" || { echo "[error] missing ${path}" >&2; exit 2; }
done

FILELIST="$(mktemp)"
trap 'rm -f "${FILELIST}"' EXIT

python3 - "$DATA_ROOT" "$FILELIST" \
  "$TRAIN_MANIFEST" "$CF_MANIFEST" "$VAL_MANIFEST" <<'EOF'
import json
import sys
from pathlib import Path

data_root, out, *manifests = sys.argv[1:]
root = Path(data_root)
files: set[str] = set()

def add(path: str) -> None:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"[error] listed file missing: {p}")
    files.add(str(p.relative_to(root)))

d9a = root / "medical_nla/data/ddxplus_counterfactual_train_v1/d9a_selected_changed_cue_v1"
for rel in (
    "train_audit/private_scores.jsonl",
    "train_audit/report.json",
    "validation_null_audit/private_scores.jsonl",
    "validation_null_audit/report.json",
    "cut_selection/protocol_approved.json",
    "cut_selection/recommendation_unapproved.json",
):
    p = d9a / rel
    if p.is_file():
        add(str(p))

for manifest in manifests:
    add(manifest)
    guard = "/direct" in manifest.lower() or "/mimic" in manifest.lower()
    if guard:
        raise SystemExit(f"[error] refusing non-DDXPlus manifest: {manifest}")
    with open(manifest, encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            path = str(row.get("activation_path") or "")
            if "direct" in path.lower() or "mimic" in path.lower():
                raise SystemExit(f"[error] DiReCT-looking path in manifest: {path}")
            add(path)

with open(out, "w", encoding="utf-8") as handle:
    for rel in sorted(files):
        handle.write(rel + "\n")
print(f"[bundle] {len(files)} files listed")
EOF

tar -C "${DATA_ROOT}" -czf "${OUT}" --files-from="${FILELIST}"
sha256sum "${OUT}"
du -h "${OUT}"
echo "[done] transfer with: scp -P <pod-port> ${OUT} root@<pod-ip>:/root/"
