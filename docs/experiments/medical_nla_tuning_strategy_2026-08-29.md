# Medical-NLA 튜닝 전략

## 문서 목적

이 문서는 현재까지 검토하거나 실행한 Medical-NLA 학습 방법을 한곳에 정리하고,
다음 구현의 우선순위를 고정한다. 단순히 의료 문장을 잘 생성하는 모델이 아니라 다음 두
조건을 동시에 만족하는 하나의 Medical-NLA를 목표로 한다.

1. 임상적으로 유용한 observation, value, relation, decision state를 자연어로 표현한다.
2. 그 표현이 언어 prior나 전형적 질환 설명이 아니라 입력 activation에 사례별로 의존한다.

최종 모델은 데이터셋마다 별도 NLA를 두는 방식이 아니다. DiReCT, DDXPlus,
MedCaseReasoning의 서로 다른 annotation을 공통 임상 claim 공간으로 정규화해 **하나의
adapter**를 학습하고, 각 데이터셋은 서로 다른 능력을 감독하거나 평가하는 역할을 맡는다.

## 결론부터

Finding deletion과 value edit는 방향 자체가 잘못된 것이 아니다. 이는 Medical-NLA의 최종
출력 범위를 finding으로 제한하려는 장치가 아니라, activation 하나를 바꿨을 때 판독이
그 변화만 따라가는지 확인하는 controlled intervention이다. 자연 임상 note에서는 무엇이
바뀌었어야 하는지 완전하게 알기 어렵기 때문에 DDXPlus의 evidence ID와 native value를
사용한다.

다만 현재처럼 original/deletion/value-edit의 전체 문장을 일반 SFT로만 학습하는 것은
충분하지 않다. 삭제된 cue는 target에서 사라지므로 그 cue를 계속 출력하는 행위를 직접
벌점 주지 못하고, 바뀌지 않은 여러 finding의 token loss가 intervention 신호를 압도한다.

따라서 다음 primary method는 다음 조합이다.

> **Changed-claim counterfactual objective + 낮은 가중치의 자연어 SFT + DDXPlus replay를
> 유지한 DiReCT clinical adaptation**

AR reconstruction과 offline preference optimization은 이 primary method가 validation에서
grounding 신호를 보인 뒤 추가할 두 번째 후보이다. 추가 epoch, LoRA lambda sweep, 더 큰
모델은 현재 병목을 해결하지 않으므로 우선순위가 낮다.

## Medical-NLA가 읽어야 하는 것

최종 출력은 특정 데이터셋의 slot을 그대로 노출하지 않는다. 내부 학습 metadata에서는
claim을 구분하되, 사용자에게는 간결한 자연어 판독을 출력한다.

| Claim family | 예시 | 주 supervision |
|---|---|---|
| Finding presence | fever, unilateral weakness | DDXPlus evidence, DiReCT observation |
| Finding value | pain location, severity, duration | DDXPlus native value |
| Relation/interpretation | finding A supports or weakens hypothesis B | DiReCT rationale |
| Decision state | favored diagnosis, considered alternative, uncertainty | source answer/CoT와 DiReCT |
| Absence/abstention | activation에서 안정적으로 읽히지 않음 | negative/control activation |

공통 출력 계약은 고정된 세 문장이나 데이터셋별 label 목록이 아니다.

```xml
<explanation>
- zero or more concise, activation-supported clinical claims
</explanation>
```

Claim 수는 activation이 지지하는 내용에 따라 달라져야 한다. 정보가 약하면 적게 말하거나
abstain할 수 있어야 한다. `exactly three clinical claims` 같은 고정 개수는 사용하지 않는다.

## 왜 probe만 계속 학습하지 않는가

Probe는 사전에 정의한 target이 activation에 존재하는지 확인하는 가장 효율적인 도구다.
실제로 DDXPlus locked test에서 HS24 finding probe는 micro F1 `.9562`, native-value
conditional probe는 accuracy `.7659`였다. 이는 finding/value 정보가 P0에 있다는 강한
근거다.

그러나 probe와 NLA의 역할은 다르다.

| Probe | Medical-NLA |
|---|---|
| 미리 정한 label마다 출력 node 필요 | 열린 자연어 claim 생성 |
| 새 attribute에는 head 재학습 필요 | 공통 언어 공간에서 새 조합을 표현 가능해야 함 |
| 분류 성능과 calibration이 명확 | 사람이 읽을 수 있으나 hallucination 통제가 필요 |
| 구조화된 monitoring에 강함 | 관계, 불확실성, 복합 설명에 유리할 가능성 |

따라서 probe는 제거할 baseline이 아니다. 다음 세 역할로 계속 사용한다.

1. NLA target 정보가 해당 layer에 존재하는지 확인하는 availability gate
2. NLA가 읽지 못한 정보와 activation 자체에 없던 정보를 분리하는 upper bound
3. 생성된 claim을 점수화하는 critic 또는 structured-reader baseline

만약 최종 Medical-NLA가 grounding gate를 계속 통과하지 못하면, probe + deterministic
verbalizer가 더 정직한 실용 baseline이라는 결론도 허용해야 한다.

## 데이터셋 역할

| 데이터셋 | 학습 역할 | 평가 역할 | 금지하는 사용 |
|---|---|---|---|
| DDXPlus train | finding/value grounding과 반사실 supervision | 없음 | validation/test를 학습에 혼합 |
| DDXPlus validation | layer, objective, threshold 선택 | paired grounding gate | 결과를 본 뒤 test protocol 수정 |
| DDXPlus test | 없음 | 최종 locked grounding | checkpoint 선택 |
| DiReCT train | physician observation/rationale의 임상 언어 adaptation | 없음 | source state와 어긋난 gold를 무조건 positive로 사용 |
| DiReCT validation | clinical semantic alignment 선택 | 공식 Obs/Exp validation | lexical 점수만으로 선택 |
| DiReCT locked split | 없음 | 최종 clinical alignment | 반복 확인 및 prompt 수정 |
| MedCaseReasoning | primary 학습에 사용하지 않음 | natural-text OOD | gold span이 있다고 가정 |

DDXPlus는 synthetic이라는 한계가 있지만 intervention 정답이 명확하다. DiReCT는 자연 임상
설명 품질을 제공하지만 physician gold가 backbone의 현재 internal state와 같다는 보장은
없다. 두 데이터셋을 섞는 이유는 하나의 데이터셋이 두 조건을 동시에 완전히 제공하지 않기
때문이다.

## Activation과 layer 고정

- Primary input: **CoT-P0**, reasoning 생성 전 prompt boundary activation
- Primary NLA index: **HS32**, 공개 AV/AR checkpoint와 맞는 extraction index
- HS16/HS24: probe sensitivity에만 사용
- P1: CoT diagnosis 문자열 누출 때문에 보조 분석
- P2: answer 이후 positive control

DDXPlus probe에서 HS24가 가장 높았다는 이유만으로 HS32용 AV에 HS24 activation을 넣지
않는다. 그렇게 하면 정보량 차이와 decoder distribution shift가 섞인다. HS24 NLA를 주
방법으로 쓰려면 layer-matched AV/AR를 새로 학습해야 한다.

## 현재까지 확인한 결과

### 1. 정보는 activation에 존재한다

| Audit | Validation | Locked test | 해석 |
|---|---:|---:|---|
| DDXPlus finding probe, micro F1 | .9607 | .9562 | finding presence는 강하게 decode 가능 |
| DDXPlus native value, accuracy | .7700 | .7659 | 일부 value도 decode 가능 |
| Same-diagnosis shuffled finding F1 | .7954 | .7938 | 진단 prior만으로도 높지만 own-case gap이 남음 |
| Same-diagnosis shuffled value accuracy | .5758 | .5791 | own-case value gap이 남음 |

따라서 NLA 실패를 `P0에 finding 정보가 전혀 없다`로 설명할 수 없다. 문제는 AV가 그
정보를 자연어로 선택적으로 꺼내는 방법이다.

### 2. Vanilla AV는 의료 판독기로 충분하지 않다

Validation 52행 x 2 prompt x 3 layers의 진단명 semantic audit에서 primary default/HS32는
source answer, gold PDD, category를 모두 `0/52` 복원했다. 기존 AV가 fluent text를 만들 수
있다는 사실과 의료 target을 읽는다는 사실은 다르다.

### 3. Original-only SFT는 schema와 일부 finding을 학습했지만 전이에 실패했다

Full-data common SFT는 DDXPlus original 4,655행과 DiReCT 248행을 사용했다.

| Method | DDX current finding | Deletion removal | DiReCT Obscomp | Expcom |
|---|---:|---:|---:|---:|
| Full SFT seed 17 | .3389 | .4052 | .0301 | 0 |
| Full SFT seed 29 | .3612 | .3232 | .0296 | 0 |
| Source CoT | N/A | N/A | .2130 | .0650 |

출력 format은 안정됐지만 DiReCT physician observation과 의미상 정렬되지 않았다. 더 많은
문장을 생성하는 것과 activation의 환자별 임상 내용을 읽는 것은 같지 않다.

### 4. 문장 전체 matched/crossed contrastive는 개선되지 않았다

DiReCT validation에서 full SFT seed 29의 symmetric alignment gap은 `+.0051`이었다.
Sentence-level contrastive smoke의 gap은 lambda `.1/1.0`에서 `+.0013/.0022`, pure
contrastive에서 `+.0030`이었다. Strong SFT+contrastive는 `+.0051`을 회복했지만 baseline을
넘지 못하고 matched win rate가 `.5333`으로 낮아졌다.

이 결과는 contrastive 방향 자체가 항상 틀렸다는 뜻이 아니다. 서로 다른 환자의 긴 target
전체를 교차하면 target 난이도, 문장 길이, 공통 질환 문구가 primary 신호에 섞인다는 뜻이다.

### 5. Counterfactual sequence SFT는 부분 신호만 만들었다

DDXPlus train 4,655 original family에 deletion과 가능한 native value edit를 추가해 seed
17/29를 학습했다. 아래는 동일 validation 435 base / 952 readout의 결과다.

| Method | Current recall | Original target hit | Deleted phantom | Deletion contrast | Removal success | Clean switch |
|---|---:|---:|---:|---:|---:|---:|
| Original-only seed 17 | .3389 | .3517 | .2138 | .1379 | .4052 | .0244 |
| Counterfactual seed 17 | .5632 | .6345 | .4253 | **.2092** | .3659 | .0488 |
| Original-only seed 29 | .3612 | .3770 | .2667 | .1103 | .3232 | .0122 |
| Counterfactual seed 29 | .3475 | .3770 | .2713 | .1057 | .4268 | 0 |

`Deletion contrast = original target hit - deleted phantom`이다. Seed 17은 recall과 deletion
contrast가 함께 증가했으므로 단순 verbosity만 생겼다고 단정할 수는 없다. 하지만 phantom
자체가 `.4253`으로 커졌고 seed 29에서 재현되지 않았다.

Value edit는 명확히 실패했다. Seed 17의 replacement hit는 `.0732`, old-value persistence는
`.4024`, clean switch는 `.0488`이었다. 새 값으로 교체하기보다 옛 값과 새 값을 함께 말하는
경향이다. 평가 가능한 value-edit base도 82개이므로 불확실성이 크다.

현재 판정은 다음과 같다.

> Counterfactual supervision에는 학습 가능한 신호가 있지만, 전체 sequence CE는 changed
> finding을 선택적으로 읽게 만드는 objective로 충분하지 않다.

## 검토한 학습 방법

### A. Prompt-only vanilla AV

- 장점: 추가 학습 없음, 공개 checkpoint 그대로 사용
- 결과: 의료 target 복원 실패
- 역할: language-prior baseline으로만 유지
- 우선순위: 종료

### B. Diagnosis 또는 physician text 직접 SFT

- 장점: 구현이 단순하고 fluent한 의료 출력 생성
- 위험: activation을 무시한 seen-class classifier 또는 전형적 설명 생성기
- 추가 위험: source-wrong 사례에서 physician gold가 현재 activation과 어긋남
- 역할: 실패 baseline
- 우선순위: 추가 epoch 중단

### C. Common-schema original-only SFT

- 장점: DDXPlus와 DiReCT를 하나의 diagnosis-free `<observed>` schema로 통합
- 결과: DDXPlus finding은 일부 개선, DiReCT semantic alignment는 낮음
- 역할: 현재 strongest generative baseline
- 우선순위: 보존하되 추가 sweep 중단

### D. Sentence-level matched/crossed contrastive

- 목표: matched `(h_i,y_i)`가 crossed `(h_i,y_j)`보다 낮은 NLL을 갖도록 학습
- 결과: 문장 NLL은 개선됐지만 사례별 gap은 baseline을 넘지 못함
- 원인 후보: target 전체 난이도와 공통 문구가 changed information보다 큼
- 우선순위: 현재 형태로 종료

### E. Counterfactual sequence SFT

- 목표: original/deletion/value-edit activation 각각에서 현재 cue 전체를 생성
- 결과: seed 17에서 deletion contrast 증가, 높은 phantom과 value persistence 발생
- 원인 후보: unchanged cue token이 changed cue를 압도하고 absence에는 positive token이 없음
- 역할: 다음 objective의 warm-start/data source
- 우선순위: 같은 방식 추가 epoch 중단

### F. Changed-claim counterfactual objective

현재 가장 가능성이 높은 방법이다. 전체 문장을 교차하지 않고 실제로 변경한 cue만
teacher-forced scoring한다.

삭제 family에서 원하는 관계는 다음과 같다.

```text
NLL(old cue | original activation)
    < NLL(old cue | deleted activation)
```

Value-edit family에서는 symmetric 2x2 관계를 사용한다.

```text
NLL(old | original) + NLL(new | edited)
    < NLL(new | original) + NLL(old | edited)
```

Loss 구성의 권장 형태는 다음과 같다.

```text
L = lambda_language * L_sequence_SFT
  + lambda_keep     * L_unchanged_claims
  + lambda_delete   * L_deleted_claim_ranking
  + lambda_edit     * L_value_swap_ranking
```

- `L_sequence_SFT`: 출력 문법과 자연스러운 claim 표현 유지, 낮은 가중치
- `L_unchanged_claims`: 개입하지 않은 finding 보존
- `L_deleted_claim_ranking`: deleted activation에서 old cue likelihood를 낮춤
- `L_value_swap_ranking`: original/edited activation과 old/new cue의 정확한 짝을 선호

단어 하나의 unlikelihood만 적용하면 `pain`, `present` 같은 공통 token까지 억제할 수 있다.
따라서 전체 cue phrase NLL 또는 old/new를 구분하는 discriminative token span을 사용한다.

장점은 현재 반사실 데이터와 activation을 그대로 재사용하며 20-step smoke로 방향을 빠르게
확인할 수 있다는 점이다. 첫 목표는 fluent generation이 아니라 paired validation에서
changed-claim margin이 실제로 커지는지 확인하는 것이다.

### G. AV-AR reconstruction regularization

NLA text가 activation 정보를 보존한다면 AR로 복원한 activation이 원 activation과 가까워야
한다. 가능한 목적은 다음과 같다.

```text
L_reconstruction = distance(h, AR(AV(h)))
```

그러나 생성 token은 discrete이므로 현재 SFT CE에 AR MSE를 단순 가산할 수 없다. 가능한
구현은 세 가지다.

1. Soft-token 또는 expected embedding을 AR에 넣는 differentiable surrogate
2. 여러 readout 후보를 생성하고 AR FVE로 순위를 매기는 offline preference 학습
3. AR score를 reward로 쓰는 policy optimization

현재 가장 현실적인 것은 2번이다. 공개 AV/AR의 index가 HS32로 맞고, candidate 생성과
scoring을 분리해 OOM과 학습 불안정을 줄일 수 있다. 다만 reconstruction만 높고 clinical
content가 틀릴 수 있으므로 DDXPlus finding/value와 DiReCT semantic score를 함께 사용해야
한다.

### H. Offline preference optimization

각 activation에서 여러 explanation 후보를 생성하고 다음 reward를 분리 계산한다.

- clinical alignment reward
- changed-claim grounding reward
- hard-shuffle specificity reward
- AR reconstruction reward
- unsupported-claim penalty

Matched activation에서 높은 joint reward를 받은 후보를 preferred, language prior 또는
shuffled activation에 더 잘 맞는 후보를 rejected로 만들어 DPO류 학습을 할 수 있다.

장점은 discrete text와 비미분 evaluator를 사용할 수 있다는 점이다. 단점은 candidate 품질과
reward calibration에 민감하고 계산량이 크다는 점이다. Changed-claim objective가 신호를
보인 뒤 두 번째 primary 후보로 둔다.

### I. Probe-guided structured reader + verbalizer

Finding/value probe가 강하므로 다음 구조는 중요한 control이다.

```text
activation -> calibrated finding/value heads -> selected claims -> frozen verbalizer
```

이는 strict한 의미의 open NLA보다 structured monitor에 가깝다. 새로운 attribute마다 head가
필요하지만, 무엇을 읽었는지와 무엇을 말했는지를 분리할 수 있고 phantom을 통제하기 쉽다.

이 방법은 두 용도로 가치가 있다.

1. Medical-NLA가 반드시 넘어야 하는 grounding upper baseline
2. generative NLA가 계속 실패할 경우 사용할 정직한 실용 대안

### J. Set decoder 또는 latent claim bottleneck

순서가 없는 clinical claim set을 먼저 예측하고 자연어로 변환하는 구조다. Sequence CE에서
발생하는 임의 순서 noise를 줄일 수 있다. Hungarian matching이나 evidence-ID set prediction을
사용할 수 있지만 DDXPlus ontology에 과도하게 맞춰질 위험이 있다. 공통 latent type과 text
span을 함께 학습할 설계가 필요해 구현 비용이 높다.

우선순위는 cue-level objective와 offline preference보다 낮지만, 장기적으로 가장 명확한
구조적 대안이다.

### K. RL/GRPO류 end-to-end 최적화

Semantic judge, AR, counterfactual score를 reward로 직접 사용할 수 있다. 그러나 reward
hacking, 높은 분산, 12B 모델의 계산 비용이 크다. Validation reward 자체가 아직 안정적으로
정의되지 않았으므로 지금 바로 시작하지 않는다. Offline preference가 먼저다.

### L. 더 큰 모델, 더 많은 epoch, LoRA sweep

현재 병목은 capacity보다 objective다. 모델은 schema와 cue 문장을 이미 생성할 수 있지만
삭제와 value change를 선택적으로 반영하지 못한다. Objective가 그대로인 상태에서 epoch,
rank, GPU를 늘리면 더 강한 template memorization이 될 가능성이 높다.

우선순위: 현재 중단.

## 권장 최종 학습 파이프라인

### Phase 0. Representation availability audit

1. Train-only probe로 finding/value/decision 정보가 존재하는지 확인
2. Validation에서 layer와 regularization 선택
3. Locked test는 설정 동결 후 한 번 평가
4. NLA target은 probe로 확인된 정보 family만 primary claim으로 사용

이 단계는 이미 DDXPlus finding/value에 대해 완료됐다.

### Phase 1. DDXPlus changed-claim grounding pretraining

1. Original/deletion/value-edit family의 CoT-P0/HS32 activation 사용
2. 동일한 common natural-language claim schema 유지
3. Sequence SFT 가중치는 낮추고 changed-claim ranking을 주 objective로 사용
4. Unchanged finding retention을 함께 최적화
5. Seed 17/29, 20 optimizer-step smoke부터 실행

승격 조건은 validation에서 다음을 모두 확인하는 것이다.

- original-vs-deleted cue margin 증가
- deletion phantom 감소 또는 최소한 증가하지 않음
- untouched finding retention 유지
- value replacement hit 증가와 old-value persistence 감소
- same-diagnosis hard shuffle보다 matched activation 우세

### Phase 2. DiReCT clinical-language adaptation with replay

Phase 1 checkpoint가 grounding gate를 통과한 경우에만 진행한다. DiReCT만 연속 fine-tune하면
DDXPlus grounding을 잊을 수 있으므로 매 epoch에 DDXPlus family replay를 유지한다.

권장 batch 구성은 다음과 같다.

```text
50% DDXPlus counterfactual families
50% DiReCT clinical targets
```

정확한 비율은 validation에서 정하지만 데이터 크기 비례의 자연 mixture는 사용하지 않는다.
DiReCT target은 다음처럼 다룬다.

- observation: note에 근거가 있고 activation-target alignment가 확인된 claim
- rationale: source decision과 모순되지 않는 관계만 primary
- diagnosis: physician gold를 무조건 넣지 않고 source state와 physician target을 분리
- source-wrong case: correction SFT가 아니라 fidelity/negative/control로 사용

Phase 2 후에는 DDXPlus와 DiReCT validation을 모두 다시 평가한다. 한쪽만 개선되면 최종
Medical-NLA로 승격하지 않는다.

### Phase 3. Reconstruction 또는 preference refinement

Phase 2가 clinical alignment와 grounding을 모두 부분적으로 통과하면 AR/offline preference를
추가한다. 목표는 fluent claim을 더 만드는 것이 아니라 unsupported claim을 줄이고 activation
정보 보존을 높이는 것이다.

### Phase 4. Freeze and evaluate

1. Prompt, checkpoint, threshold, extraction index 고정
2. DDXPlus locked test 한 번 평가
3. DiReCT locked seen/PDD-heldout 평가
4. MCR natural-text OOD 평가
5. 두 gate를 통과한 경우에만 text patching 진행

## 평가 관문

### Gate A. Information availability

- Probe own-case 성능
- same-diagnosis shuffled control
- label/value coverage
- layer sensitivity

### Gate B. Natural-language activation grounding

- current finding recall/precision
- original target hit와 deleted phantom의 paired 차이
- removal success conditional on original hit
- native replacement hit
- old-value persistence
- clean switch
- untouched finding retention
- same-diagnosis hard-shuffle gap
- zero/mean activation control

한 지표만으로 통과시키지 않는다. Recall 증가와 phantom 증가가 같이 나타나면 더 많이 말한
효과일 수 있다. 반대로 conditional removal만 오르고 original finding coverage가 떨어지면
빈 출력으로 얻은 개선일 수 있다.

### Gate C. Clinical alignment

- DiReCT official Obspre, Obsrec, Obscomp
- Expcom, Expall
- extraction coverage
- source-correct/source-wrong stratification
- seen/PDD-heldout 분리

### Gate D. Reconstruction and intervention

- matched AR FVE vs shuffled FVE
- identity round-trip drift
- target attribute change
- off-target finding preservation
- backbone diagnosis/behavior change

Gate B를 통과하지 못한 모델은 Gate D text patching에 사용하지 않는다.

## 통계와 선택 규칙

- 같은 base case의 method/variant를 paired comparison으로 유지
- DDXPlus는 base case 또는 diagnosis cluster bootstrap 사용
- DiReCT는 patient/category cluster를 보조 분석
- 세 seed 평균과 seed별 결과를 모두 보고
- lexical threshold는 validation에서 동결
- semantic scorer는 method blind로 실행
- parse 실패는 분모에서 제거하지 않음
- test는 checkpoint와 threshold 선택에 사용하지 않음

현재 counterfactual 결과는 point estimate이므로 다음 objective 전에 threshold `.3/.5/.7`
sensitivity와 paired bootstrap을 CPU에서 추가한다. 이 분석은 새 모델 선택이 아니라 현재
failure mode가 특정 lexical threshold에만 의존하는지 확인하는 용도다.

## 하지 않을 것

1. 데이터셋마다 별도 Medical-NLA adapter를 최종 방법으로 제시하지 않는다.
2. DDXPlus evidence ID를 그대로 최종 자연어 출력 slot으로 강제하지 않는다.
3. Source-wrong activation에 physician gold diagnosis를 현재 internal belief처럼 SFT하지 않는다.
4. 정확히 세 claim 같은 고정 개수를 강제하지 않는다.
5. Recall 하나가 오르면 grounding 성공이라고 하지 않는다.
6. 같은 objective에서 epoch, lambda, LoRA rank만 반복 탐색하지 않는다.
7. Locked test 결과를 본 뒤 target schema나 threshold를 수정하지 않는다.
8. Grounding을 통과하지 못한 readout으로 text patching 성능을 주장하지 않는다.

## 구현 우선순위

| Priority | 작업 | GPU | 판단 결과 |
|---:|---|---:|---|
| 1 | 현재 CF 결과 paired bootstrap/threshold sensitivity | 없음 | failure mode 확정 |
| 2 | Cue-level deletion/value 2x2 objective 20-step smoke | 4 x 4090 병렬 | margin 방향 확인 |
| 3 | 통과 arm의 full DDXPlus train run, 2 seeds | 4 x 4090 | DDX validation gate |
| 4 | DDX replay를 유지한 DiReCT adaptation | 4 x 4090 | 두 데이터셋 동시 유지 |
| 5 | AR-scored candidate generation + offline preference | GPU 다수 | reconstruction 추가 가치 |
| 6 | Locked test 및 MCR OOD | 동결 checkpoint | 논문 결과 확정 |

Cue-level smoke가 실패하면 바로 RL로 넘어가지 않는다. Probe-guided structured reader와
set-decoder baseline을 먼저 구현해 generative AV 구조 자체가 병목인지 확인한다.

## 최종 성공 정의

Medical-NLA 성공은 다음 세 문장을 모두 지지할 때만 선언한다.

1. **Availability:** 목표 임상 정보가 source activation에서 decode 가능하다.
2. **Grounding:** 자연어 claim이 matched activation과 controlled intervention을 따라간다.
3. **Clinical utility:** 그 claim이 physician reference와 정렬되고, 검증된 개입에 유용하다.

현재는 1번을 통과했고 2번에서 부분 신호를 얻었지만 통과하지 못했으며, 3번의 SFT baseline은
실패했다. 따라서 연구 방향은 findings를 무작정 빼는 것이 아니라, **변경된 임상 claim에
직접 loss를 걸어 하나의 자연어 판독기가 activation 차이를 선택적으로 표현하도록 만드는
것**이다.
