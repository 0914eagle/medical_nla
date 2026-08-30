# Locked baseline execution runbook

## 질문

Medical-NLA 성공 여부와 무관하게 필요한 DDXPlus Vanilla와 DiReCT Source CoT/Vanilla 숫자를
어떤 서버에서 어떤 순서로 만들 것인가?

## 모집단 분리

두 queue는 서로 다른 실험이며 결과 파일을 합치지 않는다.

| Queue | Dataset | Rows | Variants/pools | Paper destination |
|---|---|---:|---|---|
| DDXPlus Vanilla | E5 locked test | 10,028 | original 4,543, deletion 4,543, value edit 942 | Table 3 |
| DiReCT locked baselines | frozen confirmatory | 178 | test-seen 72, PDD-heldout 106 | Tables 1A, 1B, 2 |

DDXPlus의 기존 locked probe/structured-reader는 validation-selected HS24를 썼다. Vanilla checkpoint
`kitft/nla-gemma3-12b-L32-av`는 HS32 injection용이므로 같은 CoT-P0 prompts에서 HS32 activation을
추가 추출한다. 이 작업은 locked test에서 layer를 선택하는 것이 아니다.

## Lane A - DDXPlus 지금 실행

### 서버

기본 실행 위치는 **server 125** (`/data1/heejae`)다. 단, GPU 0-3이 모두 실제로 비어 있을 때만
실행한다. 네 장 중 일부가 다른 사용자 process에 점유돼 있으면 그 process를 종료하지 말고,
동일 파일 bundle을 올린 4-GPU RunPod 또는 두 서버 분산 extraction을 사용한다.

### A0. HS32 activation 10,028행

```bash
cd /home/eagle0914/medical_nla
git pull origin main

DATA_ROOT=/data1/heejae \
GPU_PAIR_A=0,1 GPU_PAIR_B=2,3 \
CONFIRMATION=I_ACCEPT_DDXPLUS_HS32_READOUT_EXTRACTION \
nohup bash scripts/run_ddxplus_vanilla_hs32_locked_activations_4gpu.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_vanilla_hs32_locked_activations_4gpu.log 2>&1 &
```

진행 확인:

```bash
tail -f /data1/heejae/medical_nla/logs/ddxplus_vanilla_hs32_locked_activations_4gpu.log
tail -f /data1/heejae/medical_nla/logs/ddxplus_e5_test_cot_p0_hs32_readout_v1/activation_shard0.log
tail -f /data1/heejae/medical_nla/logs/ddxplus_e5_test_cot_p0_hs32_readout_v1/activation_shard1.log
```

완료 확인:

```bash
HS32=/data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_test_cot_p0_hs32_merged_v1/layer32/last_token/manifest.jsonl
wc -l "$HS32"
cat /data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_test_cot_p0_hs32_merged_v1/provenance/output_population.json
```

정상값은 `10,028`행, missing activation `0`, CoT-P0, HS32다.

### A1. Vanilla 10,028행 generation-only seal

이 queue는 semantic mapper를 호출하지 않는다. 출력은 생성 직후 hash 봉인하고 본문을 열어보지
않는다.

```bash
HS32=/data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_test_cot_p0_hs32_merged_v1/layer32/last_token/manifest.jsonl

DATA_ROOT=/data1/heejae \
MANIFEST="$HS32" \
GPU_PAIR_A=0,1 GPU_PAIR_B=2,3 \
CONFIRMATION=I_GENERATE_SEALED_DDXPLUS_VANILLA \
OPERATOR_ATTESTATION=NO_LOCKED_TEXT_INSPECTED \
nohup bash scripts/run_ddxplus_vanilla_locked_generation_4gpu.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_vanilla_locked_generation_v1.log 2>&1 &
```

진행/완료 확인은 output text가 아니라 log, 행 수, seal만 본다.

```bash
tail -f /data1/heejae/medical_nla/logs/ddxplus_vanilla_locked_generation_v1.log
OUT=/data1/heejae/medical_nla/results/ddxplus_vanilla_locked_generation_v1
wc -l "$OUT/vanilla_shard0.jsonl" "$OUT/vanilla_shard1.jsonl" "$OUT/vanilla_locked.jsonl"
python scripts/manage_nla_generation_seal.py verify --receipt "$OUT/generation_seal.json"
cat "$OUT/population_validation.json"
```

`vanilla_locked.jsonl`을 `head`, `cat`, notebook으로 열지 않는다. 실패 후 재실행은 완결 shard는
검증 후 건너뛰며, partial shard는 자동 삭제하지 않는다.

### A2. Mapper G1-G4 이후 재생성 없는 semantic scoring

Mapper 구현이 만든 validation-only receipt와 frozen scorer가 준비된 뒤에만 실행한다.

Mapper validation은 A1 generation과 병렬로 server 125에서 실행할 수 있다.

```bash
DATA_ROOT=/data1/heejae \
MODE=prepare \
nohup bash scripts/run_ddxplus_semantic_mapper_validation_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_semantic_mapper_validation_v2.log 2>&1 &
```

`audit/dry_run_report.json`의 잔여 claim/request 규모를 확인한 뒤 `MODE=run`과
실제 서로 다른 비-Gemma `PRIMARY_MODEL`/`AUDITOR_MODEL`을 지정하면
primary/cold/auditor mapping과 receipt 생성까지 실행한다. V2 value-enriched
validation 계약의 사람 승인 후에는
`VALUE_AUDIT_CONFIRMATION=I_APPROVE_VALIDATION_VALUE_ENRICHED_G4`도 함께 지정한다.

완료 후 `summary.md`와 receipt를 확인한다.

```bash
MAP=/data1/heejae/medical_nla/results/ddxplus_semantic_mapper_validation_v2
cat "$MAP/summary.md"
python scripts/validate_semantic_mapper_freeze_receipt.py \
  --receipt "$MAP/semantic_mapper_freeze_receipt.json" \
  --expected-protocol-sha256 "$(sha256sum "$MAP/frozen/semantic_protocol.json" | awk '{print $1}')"
```

네 gate가 모두 pass한 경우에만 아래 locked scoring을 실행한다.

```bash
GEN=/data1/heejae/medical_nla/results/ddxplus_vanilla_locked_generation_v1
MAP=/data1/heejae/medical_nla/results/ddxplus_semantic_mapper_validation_v2
PROTOCOL="$MAP/frozen/semantic_protocol.json"
SCORER=scripts/score_ddxplus_semantic_readouts.py

DATA_ROOT=/data1/heejae \
GENERATION_SEAL="$GEN/generation_seal.json" \
MAPPER_RECEIPT="$MAP/semantic_mapper_freeze_receipt.json" \
SEMANTIC_PROTOCOL="$PROTOCOL" \
SEMANTIC_SCORER="$SCORER" \
EXPECTED_SEMANTIC_PROTOCOL_SHA256="$(sha256sum "$PROTOCOL" | awk '{print $1}')" \
EXPECTED_SEMANTIC_SCORER_SHA256="$(sha256sum "$SCORER" | awk '{print $1}')" \
PRIMARY_MODEL=<same-primary-model-as-receipt> \
HARD_PAIRS=/data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/hard_shuffle_pairs_test.jsonl \
OUT=/data1/heejae/medical_nla/results/ddxplus_vanilla_locked_semantic_v1 \
nohup bash scripts/score_ddxplus_vanilla_locked_from_seal.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_vanilla_locked_semantic_v1.log 2>&1 &
```

Model ID는 receipt의 실제 primary model과 같아야 한다. G1-G4 receipt가 아직 없거나
gate가 실패하면 A2가 실행되지 않는 것이 정상이다.

## Lane B - DiReCT D10 종료 후 한 번 실행

DiReCT는 DDXPlus 10,028행에 포함되지 않는다. 기존
`run_direct_locked_baseline_batch.sh`가 아래를 한 batch에서 처리한다.

1. 기존 Direct/CoT source output을 frozen 72/106으로 재index해 Table 1A를 계산한다.
2. validation에서 동결한 HS24 category/PDD probe를 test-seen 72에 적용해 Table 1B를 계산한다.
3. HS32 Vanilla 178행을 생성한다.
4. Source CoT와 Vanilla를 동일 claim extractor와 official evaluator로 평가해 Table 2를 만든다.

실행 전 필요한 값은 D10 final decision JSON, final recipe JSON, probe-control JSON, split protocol,
Vanilla actor prompt의 실제 SHA-256이다. 이 값이 아직 없는데 임시 JSON이나 임의 hash를 만들면
안 된다. D10 결과가 성공인지 실패인지는 baseline 네 행의 필요성에는 영향을 주지 않지만,
locked 72/106을 본 뒤 recipe를 바꾸지 못하도록 decision/recipe hash를 먼저 고정한다.

```bash
DATA_ROOT=/data1/heejae \
D10_DECISION=/path/to/final_d10_decision.json \
FINAL_RECIPE=/path/to/final_recipe.json \
PROBE_CONTROL_PROTOCOL=/path/to/direct_probe_control.json \
EXPECTED_D10_DECISION_SHA256=<sha256> \
EXPECTED_FINAL_RECIPE_SHA256=<sha256> \
EXPECTED_PROBE_CONTROL_SHA256=<sha256> \
EXPECTED_ACTOR_PROMPT_SHA256=<sha256> \
EXPECTED_SPLIT_PROTOCOL_SHA256=<sha256> \
GPU_PAIR=0,1 JUDGE_GPU=2 EXTRACTOR_BACKEND=codex \
nohup bash scripts/run_direct_locked_baseline_batch.sh \
  > /data1/heejae/medical_nla/logs/direct_locked_baselines_v1.log 2>&1 &
```

## 판정

- A0/A1은 Medical-NLA 성공 여부 및 D10과 독립적이다.
- A2는 mapper G1-G4에만 의존한다.
- Lane B baseline은 Medical-NLA 성공 여부와 독립적이지만 D10 decision/recipe 동결 뒤 연다.
- 성공한 Medical-NLA locked 행은 별도 조건부 queue이며 위 baseline queue에 자동 포함하지 않는다.
