# E5. DDXPlus activation grounding

## 질문

Medical-NLA의 자연어가 의료 지식 prior로 만든 그럴듯한 문장이 아니라 해당 사례
activation에 의존하는가?

## 통제

1. Matched vs hard-shuffled pair: 같은 diagnosis와 유사 길이 안에서 짝을 바꿈
2. Mean/zero activation: 언어 모델 prior 바닥
3. Activation swap: 사례 A metadata에 사례 B activation
4. Finding deletion: source prompt에서 evidence 하나 제거 후 재추출
5. Finding-value edit: DDXPlus가 정의한 native value만 변경
6. AV->text->AR identity round-trip

## Population split

DDXPlus를 평가에만 쓸지 grounding 학습에도 쓸지를 full-objective 학습 전에 고정한다. Primary transfer
설정은 DiReCT-only adaptation 뒤 DDXPlus를 cross-corpus test로 사용하는 것이다. DDXPlus
counterfactual을 학습에 쓰는 보조 설정에서는 base case, cue/value 조합과 donor pool을
train/validation/test 사이에 분리한다. Test pair나 test cue/value로 prompt, reward weight,
shuffle 난이도를 선택하지 않는다. 학습에 쓴 DDXPlus 행을 같은 Table 3 분모에 넣지 않는다.

공개 AV/AR가 hidden-state extraction index 32용이므로 primary grounding과 round-trip은
**CoT-P0/HS32**로 고정한다. 여기서 CoT-P0는 CoT instruction을 포함한 chat prompt의
경계에서, 모델이 reasoning을 생성하기 전 마지막 hidden state다. DiReCT Medical-NLA도
동일한 CoT-P0로 학습되므로 DDXPlus primary에서 instruction distribution을 맞춘다.
Direct-P0는 같은 validation base case의 paired instruction-sensitivity 통제로만 사용하며
primary 분모와 locked test에는 넣지 않는다. HS16/HS24를 같은 decoder에 넣는 값은
representation 차이와 decoder distribution shift를 분리하지 못해 appendix
sensitivity로만 다룬다.

정본 데이터는 공식 `validate.csv`와 `test.csv` 양쪽에서 eligible row가 한 건 이상 존재하는
diagnosis의 교집합을 먼저 고정하고, 각 split에서 diagnosis당 최대 100건을 독립적으로
reservoir sampling한다(seed 17). 적격 조건은 clean rendered cue가 3개 이상이고 prompt에
gold diagnosis/alias가 직접 나오지 않는 것이다. 가장 큰 bucket만 고르지 않으며 모든
diagnosis가 100건을 채운다고 가정하지 않는다. 첫 validation 감사에서는 47개 eligible
diagnosis 중 44개만 100건을 채웠고 short bucket은 28/8/89건이었다. 최종 분모는 두 official
split 스캔 후 protocol에 고정한다. 세부 생성 규칙과 실행 명령은
[`docs/data/ddxplus_e5_canonical.md`](../data/ddxplus_e5_canonical.md)에 고정한다.

Finding deletion은 모든 적격 case에 만들고, value edit은 같은 `evidence_id`에 release가
명시한 다른 value가 있으며 정상 문장으로 렌더링되는 case에만 만든다. 전역 cue 어휘에서
무관한 문장을 뽑는 과거 swap은 사용하지 않는다. Validation mean activation만 mean-control로
사용하고 test mean은 계산하지 않는다.

## 지표

Table 3은 서로 다른 의미를 하나의 `Own-case F1`로 합치지 않고 두 패널로 보고한다.

### A. Claim grounding and pair specificity

- Finding F1
- Native value accuracy
- Source-decision fidelity
- 같은 진단 hard-shuffled score
- Own-minus-shuffled paired gap

### B. Counterfactual response and reconstruction

- Edited-finding response
- 수정하지 않은 finding retention
- Matched FVE
- Shuffled FVE
- Matched-minus-shuffled FVE gap

Round-trip cosine과 MSE는 보조 지표다. Diagnosis만 같고 evidence가 다른 hard negative 구분과
paired bootstrap CI를 함께 보고한다.

## Finding/value probe gate

Natural-language readout을 더 튜닝하기 전에, 같은 CoT-P0 activation에 annotated finding과
native value가 선형적으로 읽히는지 먼저 확인한다. 이 단계는 Medical-NLA 성능 비교가 아니라
**target information availability audit**이다. Official `train.csv`로만 probe를 학습하고,
official validation은 layer, regularization, finding threshold 선택에만 사용한다. E5 official test는
이 단계에서 읽지 않는다.

Finding head는 train에서 일정 횟수 이상 관찰된 `evidence_id`를 한 번에 예측하는 multi-label
linear map이다. Value head는 하나의 linear map을 사용하되, 각 `evidence_id`에 DDXPlus release가
허용한 value들 사이에서만 conditional cross-entropy를 계산한다. 따라서 diagnosis별 probe를
여러 개 만드는 실험이 아니다. Validation score에는 train ontology coverage를 함께 쓰고,
같은 diagnosis의 다른 case label을 붙인 hard-shuffle 점수와 own-minus-shuffled gap을 반드시
보고한다.

### 1. Server 62에서 train-only population 생성 및 분할

기존 E5 validation/test 정본은 수정하지 않는다. Frozen E5 protocol의 47개 diagnosis support와
동일 eligibility만 official train에 적용한다.

```bash
cd /home/eagle0914/medical_nla
source /data/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla

E5=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1
TRAIN=/data/heejae/medical_nla/data/ddxplus_probe_train_v1

python scripts/prepare_ddxplus_probe_train.py \
  --train-csv /data/heejae/ddxplus/train.csv \
  --evidences /data/heejae/ddxplus/release_evidences.json \
  --e5-protocol "$E5/protocol.json" \
  --reference-cases "$E5/cases_validation.jsonl" \
  --reference-cases "$E5/cases_test.jsonl" \
  --out-dir "$TRAIN" \
  --examples-per-diagnosis 100 \
  --seed 17

python scripts/shard_jsonl_by_key.py \
  --input "$TRAIN/activation_rows_train.jsonl" \
  --out-dir "$TRAIN/activation_shards_cot_p0_v1" \
  --num-shards 2 \
  --key base_id
```

`summary.md`, `protocol.json`, 두 shard를 server 125로 복사한다.

```bash
rsync -a --info=progress2 \
  /data/heejae/medical_nla/data/ddxplus_probe_train_v1/ \
  eagle0914@165.132.76.125:/data1/heejae/medical_nla/data/ddxplus_probe_train_v1/
```

### 2. 두 서버에서 train CoT-P0 activation 병렬 추출

```bash
# Server 62: physical GPUs 2,3
TRAIN=/data/heejae/medical_nla/data/ddxplus_probe_train_v1
DATA_ROOT=/data/heejae GPUS=2,3 \
INPUT_FILE="$TRAIN/activation_shards_cot_p0_v1/shard_000_of_002.jsonl" \
RUN_NAME=ddxplus_probe_train_cot_p0_shard0_v1 \
OUT_DIR="$TRAIN/activations/ddxplus_probe_train_cot_p0_shard0_v1" \
  nohup bash scripts/run_ddxplus_probe_train_activations.sh \
  > /data/heejae/medical_nla/logs/ddxplus_probe_train_cot_p0_shard0_v1.log 2>&1 &

# Server 125: physical GPUs 0,1
TRAIN=/data1/heejae/medical_nla/data/ddxplus_probe_train_v1
DATA_ROOT=/data1/heejae GPUS=0,1 \
INPUT_FILE="$TRAIN/activation_shards_cot_p0_v1/shard_001_of_002.jsonl" \
RUN_NAME=ddxplus_probe_train_cot_p0_shard1_v1 \
OUT_DIR="$TRAIN/activations/ddxplus_probe_train_cot_p0_shard1_v1" \
  nohup bash scripts/run_ddxplus_probe_train_activations.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_probe_train_cot_p0_shard1_v1.log 2>&1 &
```

### 3. Server 62에서 병합하고 probe 학습

```bash
TRAIN=/data/heejae/medical_nla/data/ddxplus_probe_train_v1
E5=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1

rsync -a --info=progress2 \
  eagle0914@165.132.76.125:/data1/heejae/medical_nla/data/ddxplus_probe_train_v1/activations/ddxplus_probe_train_cot_p0_shard1_v1/ \
  "$TRAIN/activations/ddxplus_probe_train_cot_p0_shard1_v1/"

python scripts/merge_activation_shards.py \
  --shard-roots \
    "$TRAIN/activations/ddxplus_probe_train_cot_p0_shard0_v1" \
    "$TRAIN/activations/ddxplus_probe_train_cot_p0_shard1_v1" \
  --out-dir "$TRAIN/activations/ddxplus_probe_train_cot_p0_merged_v1" \
  --path-map /data1/heejae=/data/heejae \
  --expected-layers 16 24 32

DATA_ROOT=/data/heejae GPU=2 \
TRAIN_ROOT="$TRAIN/activations/ddxplus_probe_train_cot_p0_merged_v1" \
VALIDATION_ROOT="$E5/activations/ddxplus_e5_validation_cot_p0_merged_v1" \
VALIDATION_HARD_PAIRS="$E5/hard_shuffle_pairs_validation.jsonl" \
OUT_DIR=/data/heejae/restricted/direct/e2/ddxplus_finding_value_probe_val_v1 \
  nohup bash scripts/run_ddxplus_finding_value_probes.sh \
  > /data/heejae/medical_nla/logs/ddxplus_finding_value_probe_val_v1.log 2>&1 &
```

이 probe gate가 own-case에서 hard-shuffle보다 높지 않으면 P0에 target information이 있다는
근거가 없으므로, 같은 target으로 Medical-NLA SFT를 확장하지 않는다. 통과하면 validation-selected
layer와 train-supported ontology를 동결한 뒤에만 E5 test activation과 최종 평가를 실행한다.

## Validation activation extraction

Locked test를 열기 전에 official validation의 CoT-P0만 HS16/24/32에서 추출한다. Wrapper에는
test 입력 경로가 없고 `--resume`이 기본이므로 중단 후 같은 명령으로 이어서 실행할 수 있다.
Activation manifest는 finding/value probe와 counterfactual 분석에 필요한 `cue_value_ids`,
`cue_polarities`, `cf_original_*`, `cf_replacement_*` 필드를 보존해야 한다.

한 서버에서 전부 실행할 수 있지만, 두 서버를 쓸 때는 layer가 아니라 `base_id` 단위로 행을
나눈다. Layer별 분할은 같은 prompt를 두 번 forward하므로 GPU 시간을 줄이지 못한다. Original,
deletion, value-edit가 같은 shard에 남도록 다음을 server 62에서 한 번 실행한다.

```bash
E5=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1
python scripts/shard_jsonl_by_key.py \
  --input "$E5/activation_rows_validation.jsonl" \
  --out-dir "$E5/activation_shards_validation_cot_p0_v1" \
  --num-shards 2 \
  --key base_id
```

Frozen E5 directory를 server 125에 복사한 뒤 두 서버에서 한 shard씩 HS16/24/32를 함께 뽑는다.

```bash
# Server 62: physical GPUs 2,3
E5=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1
DATA_ROOT=/data/heejae GPUS=2,3 CONDITION=cot LAYERS="16 24 32" \
INPUT_FILE="$E5/activation_shards_validation_cot_p0_v1/shard_000_of_002.jsonl" \
RUN_NAME=ddxplus_e5_validation_cot_p0_shard0_v1 \
  nohup bash scripts/run_ddxplus_e5_validation_activations.sh \
  > /data/heejae/medical_nla/logs/ddxplus_e5_validation_cot_p0_shard0_v1.log 2>&1 &

# Server 125: physical GPUs 0,1
E5=/data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1
DATA_ROOT=/data1/heejae GPUS=0,1 CONDITION=cot LAYERS="16 24 32" \
INPUT_FILE="$E5/activation_shards_validation_cot_p0_v1/shard_001_of_002.jsonl" \
RUN_NAME=ddxplus_e5_validation_cot_p0_shard1_v1 \
  nohup bash scripts/run_ddxplus_e5_validation_activations.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_e5_validation_cot_p0_shard1_v1.log 2>&1 &
```

완료 확인:

```bash
find "$E5/activations/ddxplus_e5_validation_cot_p0_shard0_v1" \
  -name manifest.jsonl -print -exec wc -l {} \;
find "$E5/activations/ddxplus_e5_validation_cot_p0_shard1_v1" \
  -name manifest.jsonl -print -exec wc -l {} \;
```

Server 125의 shard output을 server 62로 복사한 뒤 `merge_activation_shards.py`로 합친다. 이때
manifest 안의 `/data1/heejae` tensor path도 `/data/heejae`로 바꾸고, 세 layer의 row ID grid가
완전히 같은지 검사한다.

```bash
# Server 62
E5=/data/heejae/medical_nla/data/ddxplus_e5_canonical_v1
rsync -a --info=progress2 \
  eagle0914@165.132.76.125:/data1/heejae/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_validation_cot_p0_shard1_v1/ \
  "$E5/activations/ddxplus_e5_validation_cot_p0_shard1_v1/"

python scripts/merge_activation_shards.py \
  --shard-roots \
    "$E5/activations/ddxplus_e5_validation_cot_p0_shard0_v1" \
    "$E5/activations/ddxplus_e5_validation_cot_p0_shard1_v1" \
  --out-dir "$E5/activations/ddxplus_e5_validation_cot_p0_merged_v1" \
  --path-map /data1/heejae=/data/heejae \
  --expected-layers 16 24 32
```

병합 결과는 `summary.md`가 아니라
`activations/ddxplus_e5_validation_cot_p0_merged_v1/summary.json`에 기록된다.

Direct-P0 control은 primary CoT-P0가 끝난 뒤 validation base population에만 실행한다. Locked
official test activation은 probe, Medical-NLA objective, threshold, checkpoint를 validation에서
동결하기 전에는 생성하지 않는다.

## 통과 조건

절대 threshold 하나로 정하지 않고 validation에서 effect-size와 bootstrap CI를 고정한다.
Test에서 pair gap과 finding-specific change가 0을 배제하고 untouched retention이 유지되어야
grounding 통과로 판정한다. AR absolute cosine만으로 faithfulness를 판정하지 않는다.

## 산출물

Table 3과 Figure 3. 실패하면 E4 결과를 좋은 explanation generation으로만 해석하고
E6 patching을 주 실험으로 진행하지 않는다.
