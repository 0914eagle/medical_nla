# 튜닝 전략 검토 — 보완점과 즉시 실행 목록

## 문서 목적

이 문서는 [`medical_nla_tuning_strategy_2026-08-29.md`](medical_nla_tuning_strategy_2026-08-29.md)에
대한 검토 결과다. 전략의 뼈대 — 병목은 capacity가 아니라 objective라는 진단,
changed-claim counterfactual objective, Phase/Gate 구조 — 는 유지한다. 이 문서는
원문에 남아 있는 구멍 5개를 고정하고, smoke 전에 끝내야 할 작업의 순서와 판정
기준을 확정한다.

## 방향이 옳다고 판단하는 근거

1. **세 번의 실패가 같은 곳을 가리킨다.** Original-only SFT, counterfactual
   sequence SFT, sentence-level contrastive 모두에서 바뀌지 않은 token의 CE가
   intervention 신호를 압도했고, 삭제된 cue에는 학습 신호가 아예 없었다.
   Changed-claim ranking은 이 두 결함을 직접 겨냥한다.
2. **Probe가 "정보는 있다"를 증명했다.** 선형 사상 하나로 finding micro F1
   `.9562`가 나오는 activation에서 12B decoder가 자연어로 못 꺼낸다면, 그것은
   용량이나 아키텍처가 아니라 학습 신호의 문제로 보는 것이 타당하다.
   (단, 이 논증에는 아래 보완점 1의 구멍이 있다.)
3. **교수님 반박에 닿을 지점이 없다.** 이 설계에는 인공 소견서 삽입이 없다.
   개입은 DDXPlus 케이스 기술 자체의 deletion/value edit이고(deletion은
   아무것도 지어내지 않음), 실제 임상 언어는 DiReCT physician annotation에서
   온다.
4. **RETRACTIONS의 실패 유형이 제도화되어 있다.** Same-diagnosis shuffled
   control, "recall 하나로 성공 선언 금지", "parse 실패를 분모에서 제거하지
   않음", locked test 규율.

## 보완점 5개

### 1. Gate A가 실제 NLA 입력 layer에서 통과되지 않았다 (가장 중요)

원문은 availability(성공 정의 1번)를 통과했다고 선언하지만, 근거인 probe
F1 `.9562` / value accuracy `.7659`는 **HS24**다. NLA 입력은 **HS32**로
고정되어 있고, "HS24가 가장 높았다"는 문장은 HS32 probe가 더 낮다는 뜻이다.
즉 availability는 HS24에서 증명하고 판독은 HS32에서 하는 구조다.

- **고정**: HS32 finding/value probe 수치를 Gate A의 공식 ceiling으로 표에
  추가한다. Tensor는 이미 저장되어 있으므로 저비용 작업이다.
- **판정 규칙**: HS32 probe가 HS24와 비슷하면 "objective가 병목" 논증이
  봉인된다. 크게 깎여 있으면 loss 변경으로는 해결되지 않는 layer 병목이므로,
  layer-matched HS24 AV/AR 재학습을 escape hatch로 승격한다.
- **순서**: 이 수치가 나오기 전에는 cue-level smoke를 시작하지 않는다.
  Smoke가 실패했을 때 objective 탓인지 layer 탓인지 분리할 수 없기 때문이다.

### 2. Seed 분산은 노이즈가 아니라 진단 신호다 — smoke 규칙에 반영

CF sequence SFT에서 seed 17/29는 사실상 다른 모델이 됐다(recall `.5632` vs
`.3475`, deletion contrast `.2092` vs `.1057`). 원인은 측정 지표가 전부
off-objective라는 데 있다: loss가 제약하지 않는 행동(삭제된 cue를 계속
말할지, 몇 개의 claim을 낼지)은 초기화와 데이터 순서가 채우고, 그 값이
seed 복권이 된다. 특히 두 seed의 격차는 상당 부분 **verbosity 축 하나**로
설명된다 — seed 17은 많이 말해서 recall과 phantom이 함께 높고, seed 29는
반대다. 평가 표본 노이즈(±.03~.05 수준)로는 설명되지 않는 실제 checkpoint
차이다.

- **고정 1**: 통계 절("세 seed")과 구현 절("2 seeds")의 불일치를 해소한다.
  Smoke부터 seed 3개로 돌리고, 승격 조건은 "**seed 3개 전부에서 margin 부호
  일치**"로 명시한다. 지금 분산 수준에서 seed 2개는 동전 두 번 던지기다.
- **고정 2**: 지표를 verbosity로 정규화한다. Readout당 claim 수를 함께
  보고하고, recall/phantom을 claim당 precision 형태로 병기한다. CPU 재집계로
  가능하므로 paired bootstrap에 포함한다.
- **활용**: changed-claim objective가 실제로 작동하면 해당 지표가
  on-objective가 되므로 **seed 간 spread 자체가 줄어야 한다**. Spread 축소를
  성공 증거의 하나로 기록하고, 새 objective에서도 분산이 그대로면 loss가
  여전히 그 행동을 잡지 못한다는 신호로 읽는다.

### 3. Value-edit gate는 n=82로 판정 불능

Phase 1 승격 조건에 "value replacement hit 증가 + old-value persistence
감소"가 들어 있으나, 평가 가능한 base 82개에서는 paired bootstrap CI가 겹쳐
통과도 탈락도 선언할 수 없다.

- **고정**: 둘 중 하나를 택한다. (a) 케이스당 value-edit family를 늘려 평가
  base를 확대하거나, (b) Phase 1 승격에서는 deletion 지표를 primary로 두고
  value 지표는 "악화되지 않음"으로 강등한다. 기본값은 (b)로 하고, (a)는
  full run 단계에서 병행한다.

### 4. Gate C에 합격선이 없다

Gate B는 Phase 1 승격 조건으로 방향이 고정되어 있으나 Gate C(clinical
alignment)는 지표 목록만 있다. 자연 기준선이 이미 표에 있다: **source CoT의
Obscomp `.2130` / Expcom `.0650`**.

- **고정**: "NLA 판독이 source CoT 자기설명을 넘는다"를 Gate C의 명시적
  bar로 박는다. 이것은 논문 대전제(내부 판독이 output 자기설명보다 낫다)와
  일치하며, 이 bar를 못 넘으면 논문 주장 자체가 성립하지 않는다.

### 5. Ranking loss 구현의 함정 — paired margin 필수

`L_deleted_claim_ranking`을 deleted activation에서 old cue의 NLL을 독립적으로
올리는 unlikelihood로 구현하면, 모델은 old cue의 likelihood를 original arm
에서까지 전역으로 낮춰 loss를 만족시킬 수 있다. 그러면 deletion contrast는
오르는데 original target hit이 무너지는, Gate B가 경고한 "빈 출력으로 얻은
개선"의 변종이 나온다.

- **고정**: 같은 batch에 (original, deleted) activation 쌍을 넣고 margin
  loss로 묶는다. Smoke 판정은 margin 단독이 아니라 **margin 부호 일치 +
  original hit 유지 + phantom 비증가** 세 조건을 함께 본다.

## 방법 서열에 대한 판단 — 왜 GRPO가 아니라 ranking부터인가

Changed-claim ranking(`NLL(old|orig) < NLL(old|del)`)은 SFT loss 조정이
아니라 **비교쌍을 ground-truth 반사실에서 만드는 preference optimization**
이다. DPO와의 차이는 쌍의 출처뿐이다. 따라서 전체 파이프라인은 같은
preference 계열 안에서 통제를 한 칸씩 푸는 사다리로 읽는 것이 정확하다.

| 단계 | 방법 | 비교쌍의 출처 | 통제 수준 |
|---|---|---|---|
| 지금 (F) | teacher-forced ranking | ground-truth counterfactual | 최대 — 쌍이 정답으로 보장 |
| 다음 (H) | offline preference (DPO류) | 샘플링 후보 + offline 채점 | 중간 — 학습 전에 후보와 점수를 사람이 감사 가능 |
| 마지막 (K) | GRPO/RL | on-policy 샘플 + online reward | 최소 — 정책이 reward 결함을 반복 탐색 |

GRPO를 지금 시작하지 않는 근본 이유는 계산 비용이 아니라 **현재 지표가
해킹 가능하다**는 것이다. Recall/phantom/contrast가 verbosity 축에 오염되어
있음을 방금 확인했다. CE처럼 지표를 겨냥하지도 않는 objective의 빈틈이 seed
복권으로 채워지는 상황에서, RL은 그 빈틈을 적극적으로 찾아내는 최적화다.
Lexical matcher reward는 matcher에 걸리는 기괴한 표현으로, judge reward는
judge의 관대한 구석으로 수렴한다. Offline preference(H)가 중간 단계로서
중요한 이유는 judge·AR FVE 같은 미분 불가 평가자를 처음 쓸 수 있으면서도
gradient가 흐르기 전에 후보와 reward를 사람이 감사할 수 있기 때문이다.

**GRPO로 넘어가는 조건 세 가지** (셋 다 갖춰지기 전에는 시작하지 않는다):

1. F/H에서 "ranking margin은 커졌는데 실제 샘플 출력이 안 바뀌는" 현상이
   확인될 때 — offline 방법의 알려진 한계이며, on-policy 최적화가 실제로
   필요한 유일한 상황이다.
2. Composite reward(verbosity 정규화 포함)가 held-out 후보에서 사람 판정과
   정렬됨을 사전 검증했을 때.
3. H로 gate를 통과하지 못해 최적화로 얻을 것이 남아 있을 때.

Structured reader(I)와 set decoder(J)는 loss 계열의 후속이 아니라 다른
질문에 대한 답이다. F가 실패하면 그것은 "objective를 더 바꿔라"가 아니라
"단일 decoder가 선택(무엇을 읽을지)과 발화(어떻게 말할지)를 분리하지
못한다"는 신호이며, 그때 가는 곳이 I/J다.

## 즉시 실행 목록

### 지금 — GPU 없이 (smoke의 전제 확보)

| # | 작업 | 산출물 | 판정 |
|---:|---|---|---|
| 1 | 현재 CF 결과 paired bootstrap + threshold `.3/.5/.7` sensitivity + verbosity 정규화 재집계 | CPU 스크립트 + 표 | seed 17 contrast `.2092`가 threshold 허상인지, seed 격차가 verbosity로 설명되는지 |
| 2 | HS32 finding/value probe (저장된 tensor 재사용) | Gate A ceiling 수치 | HS24 대비 유지 → objective 병목 확정 / 급락 → layer 병목, HS24 AV/AR 승격 |
| 3 | 전략 문서 보수 반영 (seed 3개 규칙, value gate 강등, Gate C bar, escape hatch) | 원문 수정 | — |

### 병렬 — bar 세우기

| # | 작업 | 산출물 | 판정 |
|---:|---|---|---|
| 4 | Probe-guided structured reader baseline (방법 I) 구현 — probe 선택 + frozen verbalizer | Gate B 지표 전체에 대한 upper baseline | generative NLA가 넘어야 할 수치 bar. 반복 실패 시 "생성형 AV 구조가 병목"이라는 결과 자체가 논문 재료 |

### 그 다음 — GPU 한 판

| # | 작업 | 산출물 | 판정 |
|---:|---|---|---|
| 5 | Changed-claim 2x2 ranking objective, 20-step smoke, **seed 3개**, paired margin 구현 | margin/hit/phantom 표 | 부호 일치 + hit 유지 + phantom 비증가 → full run / 실패 → I·J로 (같은 계열 loss 변형 추가 탐색 금지) |

### 조건부 분기

- **통과** → full DDXPlus train(seed 3개) → Gate B validation → DiReCT
  adaptation(DDX replay 유지) → offline preference(H).
- **탈락** → structured reader(I)와 set decoder(J)로 구조 병목 여부 확인.
  RL로 바로 넘어가지 않는다.

## 이 문서가 바꾸지 않는 것

원문의 데이터셋 역할 분담, claim family 정의, Phase 0-4 구조, "하지 않을 것"
8개 항목, locked test 규율은 그대로 유지한다. 이 문서는 그 위에 실행 순서와
판정 기준만 조인다.
