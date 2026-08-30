# D22: Patchscope same-layer source sweep

## 동기

HS32 source를 고정하고 target HS16/24/32만 바꾼 feature-interface calibration은 모든
셀에서 control gate를 통과하지 못했다.

| family | HS32→16 | HS32→24 | HS32→32 |
|---|---:|---:|---:|
| entity-description keyword hit | 2/5 | 2/5 | 2/5 |
| relation-specific keyword hit | 1/5 | 1/5 | 0/5 |

모든 셀의 no-patch hit는 `0/5`, exact output divergence는 `5/5`였다. 즉 target layer를
바꾸면 generation은 변하지만 expected feature extraction은 회복되지 않았다. Clinical
activation은 selection 전에 열리지 않았고 clinical generation은 0건이었다.

Patchscopes는 late source representation이 next-token prediction 쪽으로 이동하면서
feature extraction이 약해질 수 있다고 보고한다. 우리 DDXPlus probe에서도 native-value
accuracy는 HS16/24/32에서 `.7641/.7700/.6990`이었다. 따라서 target만 바꾸는 대신 source와
target을 같은 layer로 맞춘 마지막 진단을 사전 등록한다.

## 동결 설계

- source/target candidate: `HS16→HS16`, `HS24→HS24`, `HS32→HS32`
- prompt family: entity-description, relation-specific
- 총 6개 control cell
- model/mapping: 동일 `google/gemma-3-12b-it`, identity mapping, 학습 없음
- control set, keyword와 no-patch 조건: 이전 feature-interface calibration과 동일
- clinical population과 donor: 기존 Patchscope v1 protocol/generation manifest에서 고정
- layer별 train mean: DDXPlus official train 4,655 original activation으로 별도 계산
- validation-only; locked test는 읽지 않음

Control cell은 다음 세 조건을 모두 만족해야 한다.

1. keyword hit `>= 3/5`
2. no-patch 대비 keyword gain `> 0`
3. no-patch와 다른 continuation `>= 4/5`

Eligible 셀 선택 순서는 keyword hit, keyword gain, divergence, HS32와 layer 거리,
relation-specific 우선이다. Control 선택에는 DDXPlus activation content를 사용하지 않는다.

## Clinical 적용

선택된 cell이 있을 때만 그 source layer의 DDXPlus validation activation tensor를 load한다.
세 layer manifest의 경로와 SHA256은 선택 전에 protocol에 고정하지만, 그 임상 tensor 값은
control 선택에 사용하지 않는다.
각 base의 original, 기존 same-diagnosis donor의 original, 해당 layer train mean을 선택된
same target layer에 patch한다. Raw continuation, no-patch divergence, first-token KL과
unique-output count만 보고한다.

이 6개 셀이 모두 실패하면 현재 same-model identity Patchscope의 open-ended feature
verbalization 경로를 종료한다. Gate를 낮추거나 source/target layer 조합을 추가하지 않는다.
그 뒤의 생성형 경로는 domain-specific Medical-AR 또는 supervised activation-language
decoder라는 학습 기반 방법으로 넘어가야 한다.

## 실행

Server 125 GPU 2,3:

```bash
cd /home/eagle0914/medical_nla
git pull origin main
source /data1/heejae/uv/medical_nla/bin/activate

nohup env \
  DATA_ROOT=/data1/heejae \
  GPUS=2,3 \
  CASES=5 \
  bash scripts/run_ddxplus_d22_patchscope_same_layer_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_same_layer5_v1.log 2>&1 &

tail -f \
  /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_same_layer5_v1.log
```

결과:

```text
/data1/heejae/medical_nla/results/ddxplus_d22_patchscope_same_layer5_v1/
```

## 1차 결과

일반-domain control은 통과했다.

| family | HS16→16 | HS24→24 | HS32→32 |
|---|---:|---:|---:|
| entity-description | 5/5 | 0/5 | 2/5 |
| relation-specific | 3/5 | 3/5 | 0/5 |

모든 no-patch hit는 `0/5`, 모든 exact continuation divergence는 `5/5`였다. Frozen
tie-break가 고른 primary cell은 entity-description HS16→16이다.

그러나 primary clinical output은 환자별 finding을 읽지 않았다. Real 5건은 두 종류의
continuation만 만들었고, shuffled와 train mean은 각각 한 종류였다. `real == shuffled`는
`3/5`, `real == train_mean`은 `2/5`였다. 세 조건의 mean first-token KL도 각각
`16.1386/16.1562/16.1683`으로 거의 같았다. 모든 continuation은 환자 activation 대신
target prompt에 직접 적힌 `Patient A: fever and productive cough`를 설명했다.

따라서 이 결과는 HS16 Patchscope control 성공과 clinical entity prompt 실패를 함께
뜻한다. Activation에 임상 정보가 없다는 판정으로 사용하지 않는다.

## Relation-specific 후속 진단

임상 결과를 보고 layer 하나를 고르지 않도록, 1차 control에서 이미 독립적으로 gate를
통과한 relation-specific HS16→16과 HS24→24를 모두 report-only로 실행한다. 두 실행은
동일한 frozen 5사례, donor와 train mean을 사용한다. 이 후속 결과는 primary cell을
교체하거나 promotion에 사용하지 않는다.

Server 125에서 GPU 두 쌍으로 병렬 실행한다.

```bash
cd /home/eagle0914/medical_nla
git pull origin main
source /data1/heejae/uv/medical_nla/bin/activate

nohup env \
  DATA_ROOT=/data1/heejae \
  GPUS=0,1 \
  CASES=5 \
  CLINICAL_CELL=relation_specific:16 \
  bash scripts/run_ddxplus_d22_patchscope_same_layer_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_relation_hs16_5_v1.log 2>&1 &

nohup env \
  DATA_ROOT=/data1/heejae \
  GPUS=2,3 \
  CASES=5 \
  CLINICAL_CELL=relation_specific:24 \
  bash scripts/run_ddxplus_d22_patchscope_same_layer_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_relation_hs24_5_v1.log 2>&1 &
```

둘 다 real이 환자별 finding을 말하고 shuffled/train mean과 분리될 때만 50건 semantic
audit으로 확대한다. 둘 다 generic하거나 조건 간 분리가 없으면 학습 없는 identity
Patchscope의 clinical verbalization 경로를 종료한다.
