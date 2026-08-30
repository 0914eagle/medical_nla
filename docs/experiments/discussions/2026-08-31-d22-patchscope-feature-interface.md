# D22: Patchscope feature-interface calibration

## 질문

Final-marker token-identity Patchscope는 DDXPlus와 같은 backbone의 HS32에서 source
next-token top-1을 `18/20 = .9000` 복원했다. 따라서 activation extraction, target
pre-hook 위치와 same-model identity patch 자체는 작동한다. 반면 entity-description
양성 대조는 `2/5 = .4000`이었고, clinical few-shot target의 15 continuation 중 11개는
완전히 동일한 knowledge-graph 설명이었다.

다음 질문은 activation 주입 여부가 아니라 다음 두 target interface 중 어느 것이
HS32의 내용을 자연어 feature로 가장 잘 표면화하는가이다.

1. `subject: description` 형식의 entity-description interface
2. `relation of subject` 형식의 relation-specific feature interface

Clinical 결과를 보고 interface나 target layer를 고르는 것을 막기 위해 일반-domain
양성 대조만으로 셀을 먼저 선택한다.

## 동결 모집단과 경계

- source model: `google/gemma-3-12b-it`
- source activation: HS32 only
- target layers: HS16, HS24, HS32
- target model: source와 동일한 checkpoint
- mapping: identity; 학습 파라미터 없음
- clinical population: 기존 Patchscope v1 protocol의 첫 5개 DDXPlus validation base ID
- clinical arms: real, same-diagnosis shuffled, train mean, 공통 no-patch
- locked test: 읽지 않음
- clinical activation vector와 generation content는 control cell 선택이 `selection.json`으로
  기록된 뒤에만 load함; 입력 artifact SHA는 inference 전 protocol에 고정

## 일반-domain control

### Entity-description family

Target은 다음 final-marker prompt로 고정한다.

```text
Syria: Country in the Middle East, Leonardo DiCaprio: American actor,
Samsung: South Korean electronics company, foo
```

Paris, Saturn, Mozart, oxygen, Mount Everest의 source last-token HS32를 `foo`의 target
HS16/24/32에 각각 patch한다. 각 항목의 사전 고정 keyword 중 하나라도 continuation에
있으면 hit이다.

### Relation-specific family

다음 다섯 source/relation/keyword를 고정한다.

| source subject | target prompt | keyword |
|---|---|---|
| France | `The capital city of foo` | Paris |
| Japan | `The official currency of foo` | yen |
| Mozart | `The profession of foo` | composer, musician |
| Saturn | `The astronomical object type of foo` | planet |
| Mount Everest | `The mountain range containing foo` | Himalaya |

각 source는 `This text is about {subject}`의 마지막 token HS32에서 추출한다. Target
layer는 HS16/24/32를 모두 report한다.

## 셀 통과와 선택 규칙

각 `family × target layer` 셀은 다음 세 조건을 모두 만족해야 eligible이다.

1. expected keyword hit `>= 3/5`
2. no-patch 대비 keyword-hit gain `> 0`
3. no-patch와 다른 exact continuation `>= 4/5`

Eligible 셀 중 다음 순서로 단 하나를 선택한다.

1. keyword hit rate 최대
2. keyword gain 최대
3. no-patch divergence 최대
4. source HS32와 target layer 거리 최소
5. 완전 동률이면 relation-specific 우선

Eligible 셀이 없으면 clinical generation 없이 종료한다. Threshold, prompt, keyword,
layer 또는 tie-break는 control/clinical 출력을 본 뒤 바꾸지 않는다.

## Clinical application

선택 family에 따라 target을 다음 중 하나로 고정한다.

```text
Patient A: fever and productive cough, Patient B: substernal chest pain and
exertional dyspnea, Patient C: itchy swollen rash, Patient foo
```

```text
The clinical findings of patient foo
```

선택된 target layer의 final marker `foo`에 DDXPlus real, same-diagnosis shuffled,
train-mean HS32를 patch한다. Raw continuation, no-patch exact divergence, first-token KL,
max logit delta와 unique-output count를 보고한다. 이 5-case 결과는 prompt calibration
diagnostic이며 semantic score나 Medical-NLA 성공 판정이 아니다.

## 실행

Server 125의 GPU 2,3에서 실행한다.

```bash
cd /home/eagle0914/medical_nla
git pull origin main
source /data1/heejae/uv/medical_nla/bin/activate

nohup env \
  DATA_ROOT=/data1/heejae \
  GPUS=2,3 \
  CASES=5 \
  bash scripts/run_ddxplus_d22_patchscope_feature_calibration_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_feature_calibration5_v1.log 2>&1 &

tail -f \
  /data1/heejae/medical_nla/logs/ddxplus_d22_patchscope_feature_calibration5_v1.log
```

결과 위치:

```text
/data1/heejae/medical_nla/results/ddxplus_d22_patchscope_feature_calibration5_v1/
```

`protocol.json`은 모델 inference 전에 쓰고, `selection.json`은 clinical activation을
읽기 전에 쓴다.

## 결과

실행 결과 eligible cell은 없었다. Entity-description은 HS32→16/24/32에서 모두 `2/5`,
relation-specific은 각각 `1/5`, `1/5`, `0/5`였다. 모든 no-patch hit는 `0/5`, exact
continuation divergence는 `5/5`였다. 따라서 selection은 `none`, control gate는 `False`,
clinical generation은 0건으로 종료됐다.

Target layer 변경만으로는 feature fidelity가 개선되지 않았다. Source layer 자체를
포함하는 마지막 same-layer 진단은
[`2026-08-31-d22-patchscope-same-layer-source-sweep.md`](2026-08-31-d22-patchscope-same-layer-source-sweep.md)
에서 별도로 관리한다.
