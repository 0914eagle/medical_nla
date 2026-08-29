# 튜닝 전략 검토 — 보완점과 즉시 실행 목록

## 문서 목적

이 문서는 [`medical_nla_tuning_strategy_2026-08-29.md`](medical_nla_tuning_strategy_2026-08-29.md)에
대한 검토 결과다. 전략의 뼈대 — 병목은 capacity가 아니라 objective라는 진단,
changed-claim counterfactual objective, Phase/Gate 구조 — 는 유지한다. 이 문서는
원문에 남아 있는 구멍 6개를 고정하고, smoke 전에 끝내야 할 작업의 순서와 판정
기준을 확정한다. (2차 검토에서 반영: probe 증명 범위 격하, HS32 판정의
validation 종결, cue support mask 단계 신설, 최소 효과 크기, DPO 서술 정정.)

## 방향이 옳다고 판단하는 근거

1. **세 번의 실패가 같은 곳을 가리킨다.** Original-only SFT, counterfactual
   sequence SFT, sentence-level contrastive 모두에서 바뀌지 않은 token의 CE가
   intervention 신호를 압도했고, 삭제된 cue에는 학습 신호가 아예 없었다.
   Changed-claim ranking은 이 두 결함을 직접 겨냥한다.
2. **Probe가 "prompt cue identity는 부분적으로 사례 특이적으로 decode
   가능하다"를 증명했다.** Probe label은 `cue_evidence_ids` — 입력에 기록된
   cue의 identity — 이므로, F1 `.9562`는 "prompt에 있던 cue가 P0에서 분류
   가능하다"까지만 증명한다. Same-diagnosis shuffled `.7938`이 보여주듯
   상당량은 질환 전형 조합으로도 맞으며, 사례 특이적 증분은 own-case gap
   (+.16)과 deletion 후 probe score 감소가 지지하는 **cue 부분집합**에만
   성립한다. "선형으로 꺼내지는 정보를 decoder가 못 꺼내면 objective 문제"
   논증도 그 부분집합에 대해서만 유효하다 — 이것이 보완점 6(support mask)의
   출발점이다.
3. **교수님 반박에 닿을 지점이 없다.** 이 설계에는 인공 소견서 삽입이 없다.
   개입은 DDXPlus 케이스 기술 자체의 deletion/value edit이고(deletion은
   아무것도 지어내지 않음), 실제 임상 언어는 DiReCT physician annotation에서
   온다.
4. **RETRACTIONS의 실패 유형이 제도화되어 있다.** Same-diagnosis shuffled
   control, "recall 하나로 성공 선언 금지", "parse 실패를 분모에서 제거하지
   않음", locked test 규율.

## 보완점 6개

### 1. Gate A의 HS32 ceiling — validation 수치로 종결 (해결됨)

원문의 probe 수치(F1 `.9562` / value `.7659`)는 HS24 locked test이고 NLA
입력은 HS32라는 불일치가 있었다. **이 판정은 validation 수치로 이미 종결됐다**:

| Probe (validation) | HS24 | HS32 | 판정 |
|---|---:|---:|---|
| finding micro F1 | .9607 | .9607 | finding availability에 layer 병목 없음 — objective 병목 확정 |
| native value accuracy | .7700 | .6990 | value는 HS32에서 ~7pp 낮은 ceiling — value gate 강등(보완점 3)과 정합 |

- **Leakage 금지**: layer 판정은 여기서 끝낸다. **Locked test의 HS32 수치를
  layer 결정에 열지 않는다** — 결과를 본 뒤 layer를 고르면 test가 선택에
  오염된다. Locked test는 설정 동결 후 최종 평가에서만 쓴다.
- **귀결**: finding 쪽 "objective가 병목" 논증은 봉인됐다. Value-edit의
  replacement hit `.0732`는 HS32 ceiling `.6990`에도 한참 못 미치므로 value
  역시 objective가 지배적이되, layer로 이미 잃고 들어가는 폭이 있음을
  해석에 반영한다.

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
  Smoke부터 seed 3개로 돌리고, 승격 조건은 부호 일치만으로는 부족하다 —
  +.0001도 통과하기 때문이다. **세 조건을 모두 요구한다**: (a) seed 3개
  전부에서 margin 부호 일치, (b) cluster-bootstrap CI가 0을 배제, (c) 사전
  동결한 최소 효과 크기 **δ_min = .05** 이상. δ_min의 근거: 1a 감사에서
  baseline(original-only)의 seed 간 contrast 격차는 +.0276 [−.0115, .0644]로
  0을 배제하지 못했고, 결함 있는 sequence CE조차 seed 17에서 +.0713의
  contrast 개선을 냈다. 새 objective가 baseline seed 노이즈 위이면서 기존
  방법의 우연한 개선보다 작지 않아야 하므로 .05로 동결한다.
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

### 6. Cue는 gold target이 아니라 candidate claim pool이다 (2차 검토 신설)

DDXPlus `cue_targets`는 다음 네 가지 중 **첫 번째**일 뿐이다: (1) 입력에
기록된 finding, (2) activation에 사례별로 보존된 finding, (3) backbone이
진단에 실제 사용한 finding, (4) 환자 상태를 완전하게 기술하는 finding.
Probe label도 `cue_evidence_ids` 그대로이므로 F1 `.9607`은 (1)→(2)의 전부가
아니라 일부만 담보한다 — same-diagnosis shuffled `.7938`이 그 증거다. 입력에
적힌 cue 전부를 positive로 SFT하면 NLA는 내부 상태 판독기가 아니라 **prompt
finding 재구성기**가 될 수 있다.

- **고정**: deletion 실험을 학습 전에 **selection 도구**로 먼저 쓴다. 기존
  probe로 cue별 `p_original`, `p_deleted`, `delta = p_original − p_deleted`,
  same-diagnosis donor margin을 산출하고, cue를 다음처럼 나눈다.

  | Cue 상태 | 학습 처리 |
  |---|---|
  | original에서 검출 + deletion 후 감소 | positive + paired ranking |
  | original에서도 미검출 | positive에서 제외 |
  | 삭제해도 score 유지 | 불확실 — SFT target에서 제외 (negative 아님: 상관 finding을 통한 중복 인코딩일 수 있음) |
  | 감소하되 진단 출력 불변 | input-retained finding |
  | 감소하고 진단 logprob/answer도 변화 | decision-relevant finding |

- **순환 차단**: support mask는 **out-of-fold probe score**로 만든다. 같은
  사례로 probe를 학습하고 그 probe로 그 사례의 target을 고르면 순환이다.
  기존 `crc32(base_id) % 2` 결정론적 2-fold 관례를 재사용한다.
- **자유도 동결**: `p_original`/`delta`/donor margin의 컷은 validation에서
  동결하고 통계 절에 기록한다. Smoke 결과를 본 뒤 움직이면 무효다.
- **이중 분모**: supported cue로 학습하면 supported cue만으로 평가한 지표는
  기계적으로 오른다. 모든 비교는 (a) 전체 cue 분모와 (b) supported cue
  분모를 병기하고, mask 정의는 비교되는 모든 method에 동일하게 적용한다.
- **한계 명시**: probe가 검출한 cue만 positive로 학습하므로, supervised set
  안에서 "NLA가 probe만큼 읽는다"는 부분적으로 동어반복이 된다. H2.2(조합적
  자연어 출력)는 출력 공간이 달라 살아남지만, "NLA가 probe보다 더 읽는다"는
  주장은 heldout/OOD에서만 할 수 있다.

## 방법 서열에 대한 판단 — 왜 GRPO가 아니라 ranking부터인가

Changed-claim ranking(`NLL(old|orig) < NLL(old|del)`)은 SFT loss 조정이
아니라 **비교쌍을 ground-truth 반사실에서 만드는 contrastive/ranking
objective**다. DPO와 같은 계열이지만 같은 것은 아니다 — DPO는 reference
policy 대비 log-ratio에 Bradley–Terry를 거는 sequence-level preference
objective이고, teacher-forced cue-phrase NLL margin에는 reference policy가
없다. 그래도 "비교쌍에 margin을 건다"는 구조가 같으므로, 전체 파이프라인은
같은 계열 안에서 통제를 한 칸씩 푸는 사다리로 읽는 것이 정확하다.

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

## 1a 감사 결과 (2026-08-29 실행 완료)

`analyze_cf_readout_uncertainty.py`를 validation 435 base에 실행했다
(`$E5_ROOT/cf_uncertainty_audit_v1/`). 판정 세 가지:

1. **Threshold 허상 아님.** CF seed 17의 deletion contrast는 `.1931/.2092/.1931`
   (threshold .3/.5/.7)로 안정적이고 CI `[.1655, .2552]`가 0을 배제한다.
   네 모델 모두 contrast CI가 0을 배제한다 — original-only SFT조차 약한
   contrast는 낸다.
2. **CF 학습의 이득은 seed 17에서만 실재하고, phantom 비용이 크다.**
   paired delta(cf17 − orig17): contrast **+.0713 [.0230, .1218]** (실재하되
   완만), phantom **+.2115 [.1609, .2621]** (phantom이 2배). seed 29에서는
   contrast delta −.0046 [−.0575, .0437]로 이득이 재현되지 않는다.
   baseline끼리의 seed 격차는 +.0276 [−.0115, .0644]로 0을 배제하지 못하므로,
   **CF 학습이 seed 분산을 증폭시켰다**(cf17−cf29 contrast +.1034
   [.0483, .1563]) — objective가 행동을 정하지 못한다는 진단과 정합.
3. **Verbosity가 격차의 일부지만 전부는 아니다.** cf17은 claim 5.01개로 가장
   많이 말하지만 unsupported-claim rate는 `.5091`로 네 모델 중 가장 낮다
   (cf29 `.7063`, orig17 `.6464`). 즉 cf17의 개선은 순수 말수 효과가 아니라
   실제 grounding 성분을 포함하되, phantom 급증이라는 대가를 치렀다.

종합: sequence CE는 "완만하고 seed-불안정한 contrast + phantom 2배"를 주는
objective다. 전략의 진단(전체 sequence CE로는 부족)이 정량 확정됐고,
cue-level ranking + support mask로 넘어갈 근거가 마련됐다.

## 즉시 실행 목록

### 완료

| # | 작업 | 결과 |
|---:|---|---|
| 1a | CF paired bootstrap + threshold sensitivity + verbosity | 위 "1a 감사 결과" — threshold 허상 아님, seed 17만 실재 이득, phantom 2배 |
| 1b | HS32 Gate A ceiling | validation으로 종결: finding .9607 = HS24, value .6990 (−.071). Locked test는 열지 않는다 |

### 지금 — GPU 없이/경량 (smoke의 전제 확보)

| # | 작업 | 산출물 | 판정 |
|---:|---|---|---|
| 2a | Cue별 `p_original`/`p_deleted`/`delta`/donor margin 산출 (기존 probe, CF activation 재사용) | cue-level support score 테이블 | cue 몇 %가 activation-supported인지 — mask 크기 자체가 결과 |
| 2b | Cross-fitted support mask 생성 (`crc32(base_id) % 2` out-of-fold), threshold는 validation에서 동결 | supported/uncertain/absent cue 분류 | 순환 없는 positive target 집합 |
| 2c | Probe-guided structured reader baseline (방법 I) — probe 선택 + frozen verbalizer | Gate B 지표의 upper bar | generative NLA가 넘어야 할 수치. 반복 실패 시 "생성형 AV 구조가 병목"이라는 결과 자체가 논문 재료 |

### 그 다음 — GPU 한 판

| # | 작업 | 산출물 | 판정 |
|---:|---|---|---|
| 3 | **Supported cue만**에 changed-claim 2x2 ranking, 20-step smoke, seed 3개, paired margin | margin/hit/phantom 표 (전체 cue / supported cue 이중 분모) | 부호 일치 + CI 0 배제 + δ_min .05 + hit 유지 + phantom 비증가 → full run / 실패 → I·J로 |

### 이후 — 분리 측정

| # | 작업 | 산출물 |
|---:|---|---|
| 4 | Deletion 시 진단 logprob/answer 변화 측정 — input-retained vs decision-relevant finding 분리 | decision-relevance 주석 (error anatomy의 입력) |
| 5 | DiReCT clinical-language adaptation (DDX replay 유지) | Gate C |

### 조건부 분기

- **통과** → full DDXPlus train(seed 3개) → Gate B validation → DiReCT
  adaptation → offline preference(H).
- **탈락** → structured reader(I)와 set decoder(J)로 구조 병목 여부 확인.
  RL로 바로 넘어가지 않는다.

## 이 문서가 바꾸지 않는 것

원문의 데이터셋 역할 분담, claim family 정의, Phase 0-4 구조, "하지 않을 것"
8개 항목, locked test 규율은 그대로 유지한다. 이 문서는 그 위에 실행 순서와
판정 기준만 조인다.
