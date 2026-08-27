# 논문 실험 상태

기준일: 2026-08-27. 제한 데이터 원문이나 개인 식별자는 기록하지 않는다.

| 단계 | 상태 | 완료 조건 | 다음 단계 의존성 |
|---|---|---|---|
| E0 DiReCT audit/evaluator | 완료 | 496행 canonical split, official oracle smoke | E1-E4 |
| E1 source/activation | 완료 | pilot 496 source outputs, P0/P1/P2 x HS16/24/32 완전성 확인 | E2 |
| E2 P0 representation audit | 부분 완료 | diagnosis/category probe와 output-head 완료; source-decision/finding/value probe 대기 | E3-E5 |
| E3 Medical-NLA train | SFT-only 완료 | DiReCT P0 seeds 17/29/43 완료; reconstruction/full objective 대기 | E4-E6 |
| E4 DiReCT explanation | validation 완료 | 공통 50-case official-compatible 평가 완료; locked 72/106 대기 | Table 2 |
| E5 DDX grounding | 데이터 빌더 완료, 실행 대기 | official validate/test 각 49x100, native counterfactual, shuffle/round-trip 통과 | RQ2, E6 gate |
| E6 text patching | 조건부 | target change + no-op preservation | RQ3 |
| E7 MCR OOD | 후순위 | frozen checkpoint external test | generalization |

## E1 실행 현황

| 서버 | data root | GPU | split | 행 |
|---|---|---|---|---:|
| `165.132.76.62` | `/data/heejae` | physical 2,3 | train + val_seen | 325, 완료 |
| `165.132.76.125` | `/data1/heejae` | physical 0,1 | test_seen + PDD-heldout | 171, 완료 |

공통 backbone은 Gemma-3-12B-IT이고 greedy decoding과 forced answer 비활성화를 쓴다.
CoT는 batch size 1, `max_new_tokens=2048`; Direct는 batch size 4, answer prefill,
`max_new_tokens=64`다. strict PDD accuracy는 실행 중
약 0.16-0.24이나, DiReCT PDD가 세분화되어 있어 category accuracy와 official semantic
evaluation을 함께 보지 않고 이 중간값만으로 모델 적합성을 판정하지 않는다.

## E1 smoke에서 확인한 것

- 10/10 answer parse 성공
- strict PDD alias hit 0/10, disease-category hit 6/10
- 모델 answer alias가 CoT 안에 이미 등장 8/10
- gold PDD alias가 CoT 안에 등장 1/10
- L16/L24/L32의 P0 10개, P1/P2 각 10개가 모두 3840차원으로 저장됨

따라서 P0를 주 비교 위치로 고정하고, P1은 answer leakage가 없는 행만 별도 분석한다.

## E1 test 완료 결과

- 171/171 answer parse 성공, forced answer 0
- strict PDD alias accuracy 33/171 = 0.1930
- disease-category accuracy 87/171 = 0.5088
- diagnosis-label token F1 = 0.1850
- 모델 answer alias가 CoT reasoning에 등장: 156/171 = 0.9123
- gold PDD alias가 CoT reasoning에 등장: 49/171 = 0.2865
- Activation rows 513 = 171 x P0/P1/P2
- Tensors 1,539 = 513 rows x 3 layers
- Prompt tokens: min 209, mean 1,619, max 4,304

P1 leakage-free test subset은 15행뿐이므로 P1의 source-decision 결과는 정성·민감도
분석으로 제한한다. P0가 주 설명 판독 위치라는 결정이 확정됐다. Strict PDD 0.193은
세부 label exact/alias 점수다. Category accuracy는 0.5088로 strict PDD보다 높았고,
PDD-heldout에서도 0.6000이었다. 따라서 strict 실패에는 넓은 질병 범주는 맞지만 세부
PDD 또는 표현이 다른 사례가 포함된다. 다만 category는 PDD보다 쉬운 계층적 지표이고,
token F1은 진단명 문자열 유사도이지 설명 품질 지표가 아니다. Official semantic score가
나오기 전에는 이 값만으로 source model의 최종 진단 성능을 확정하지 않는다.

같은 171행의 prefilled Direct baseline도 완료됐다. Strict PDD는 Direct 36/171 =
0.2105, CoT 33/171 = 0.1930이었다. 둘 다 맞은 사례 26, CoT만 맞은 사례 7,
Direct만 맞은 사례 10, 둘 다 틀린 사례 128이며 CoT-Direct는 -0.0175,
McNemar exact p=0.6291이다. Category는 Direct 0.5029, CoT 0.5088로 +0.0058이었고
paired McNemar exact p=1.0000이었다. Diagnosis-label token F1은 Direct 0.1593,
CoT 0.1850이었다. 따라서 CoT가 진단명 어휘에는 조금 더 가까웠지만 strict PDD나
category 정확도를 개선했다는 증거는 없다. CoT 설명 품질은 이 결과가 아니라 E4의
official `Obs*`/`Exp*`로 별도 평가한다.

## E1 전체 완주

62번 train+validation도 완료됐다. Source answers 325, activation rows 975,
tensor 2,925개가 생성됐다. 125번 test의 171/513/1,539와 합치면 pilot universe 전체에서
source answers 496, activation rows 1,488, tensor 4,464개다. 각 case마다 P0/P1/P2 세
위치와 HS16/HS24/HS32 세 index가 모두 존재한다. Train+validation prompt token 최대는
4,834였고 tensor 저장 dtype은 float32다.

62번에서 frozen split을 독립 재현한 결과 logical population hash와 네 split ID hash가
125번 정본과 모두 일치했다. 두 서버의 activation을 62번으로 모아 재색인했으며
266/52/72/106 cases가 각각 2,394/468/648/954 activation rows에 대응한다. 전체 4,464행에서
case x position x layer grid가 완전하고, duplicate·unassigned·missing path는 0이다.

## 즉시 할 일

1. ~~CoT와 네 NLA arm의 validation claim extraction 및 official semantic matching~~ 완료.
   SFT-only는 CoT보다 낮아 최종 방법이 아니라 실패 ablation으로 고정
2. ~~Early forced-answer candidate sequence baseline과 validation 통제~~ 완료. Raw category
   25-way `.4808/.6731/.5814`, raw PDD 61-way `.1538/.4423/.3168`, matched raw PDD
   49-way `.1538/.5192/.3250`(top-1/top-5/MRR). PDD의 한 후보 35/52 쏠림은 49-way에서도
   유지됐다. Content-free 차감은 category `.2308/.3077/.3091`, PDD
   `.0577/.1346/.1486`으로 더 악화되어 sensitivity로만 보존
3. ~~HS16/HS24 vanilla AV prompt sensitivity 집계~~ 완료
4. 기존 171행은 exploratory로 동결. 새 locked downstream split 266/52/72/106과 hash는 확정 완료
5. 새 heldout artifact 감사 완료: backbone 106/106, vanilla AV 16/106
6. ~~62번에서 logical population/split hash가 125번과 동일한지 확인~~ 완료
7. ~~기존 activation을 새 split ID로 재색인하고 join/completeness 100% 확인~~ 완료
8. ~~Validation 52행에서 probe regularization을 선택~~ 완료. Locked test 평가 전 checkpoint와 분석 코드를 고정
9. Source-decision/finding-presence/finding-value P0 probe를 validation에서 구현·실행하고
   Medical-NLA가 책임질 수 있는 target family를 고정
10. ~~Vanilla AV P0 semantic audit 312행~~ 완료. Primary default/HS32에서 source answer,
    gold PDD, category 모두 0/52; HS16 category만 두 prompt에서 1/52. 이 결과는 진단 target의
    명시적 의미 복원 실패이며 observation 설명 품질이나 grounding 결과가 아님. Exact
    readout quote를 요구한 local Llama-3-8B 판정을 정본으로 사용하며 human-validated
    score라고 부르지 않음

## 08-27 이후 실행 순서

1. **E2 target audit 완결**: source decision, finding presence, finding value가 CoT-P0에서
   decode 가능한지 probe와 shuffle control로 확인한다. HS16/24/32는 validation에서만 비교한다.
2. **E3 full objective 구현**: SFT-only 3-seed 실패를 기준선으로 두고 reconstruction과
   pair-specificity를 추가한다. DDXPlus를 학습에 쓸 경우 evaluation pair와 완전히 분리한다.
3. **E4 validation 재평가**: reconstruction/full 후보를 같은 50-case extractor와 official
   evaluator로 비교해 method를 고정한다. Seed를 골라 버리지 않고 mean/SD를 보고한다.
4. **방법 동결 후 test 1회**: seen 72와 PDD-heldout 106에서 CoT/vanilla/SFT-only/full을 같은
   evaluator로 비교한다. 이 시점 이후 prompt, layer, target schema를 수정하지 않는다.
5. **E5 DDXPlus grounding**: `prepare_ddxplus_e5.py`로 official validation/test를 분리한
   정본 population을 먼저 만들고, frozen SFT-only adapter에 matched/shuffled, zero/validation-
   mean activation, finding deletion/native-value edit, AV-to-AR round-trip을 적용한다. Test에서
   population, donor 또는 threshold를 다시 선택하지 않는다.
6. **E6 조건부 patching**: E5의 pair gap과 finding-specific change가 통과할 때만 실행한다.
7. **E7 MCR OOD**: 핵심 표가 닫힌 뒤 frozen checkpoint의 외적 일반화로만 실행한다.

현재 주 큐에서 제외하는 작업은 human audit, calibrated likelihood 추가 실행, task-aligned
prompt 및 HS16/24 추가 sweep, P1/P2 학습, E5 전 patching, MCR 조기 실행이다. Full objective는
SFT-only가 이미 임상 정렬에 실패했으므로 선택적 후속이 아니라 다음 방법 개발 단계다.

## 두 서버 병렬 실행 원칙

| lane | server | GPU | 우선 작업 |
|---|---|---|---|
| A | 62, `/data/heejae` | physical 2,3 | primary Medical-NLA/AV 학습·생성 |
| B | 125, `/data1/heejae` | physical 0,1 | output-head, 위치 통제, official evaluator, grounding control |

두 lane은 같은 population/split hash와 checkpoint를 사용한다. Validation으로 설정을 고르는
동안 locked test 72+106행은 어느 lane에서도 추가 생성하지 않는다. 한 lane이 장시간 GPU
작업을 수행할 때 다른 lane에는 그 결과에 의존하지 않는 baseline 또는 positive/negative
control을 배정한다. 동일 출력을 두 서버에서 중복 생성하지 않는다.

Gold-label-in-note audit은 raw 511행 중 28행(0.0548)으로 완료됐다. 기존 test CoT 171행에는
formatted answer marker가 여러 번 등장한 행이 0개여서 final-answer parser 수정이 pilot
수치를 바꾸지 않는다. 다음 split은 이 leakage flag를 eligibility에서 제거하지 않고 split별
민감도 분모와 ID hash를 함께 동결한다.

## E2 vanilla AV exploratory 상태

### P0 linear probe validation

Train 266행으로 학습하고 frozen `val_seen` 52행에서 validation NLL로 설정을 선택했다.
Locked test manifest는 읽지 않았다.

| Target | HS16 top-1 / top-5 / MRR | HS24 top-1 / top-5 / MRR | HS32 top-1 / top-5 / MRR | Majority |
|---|---:|---:|---:|---:|
| Canonical PDD, 49 classes | .3846 / .6923 / .5294 | **.4423 / .7692 / .5762** | .3846 / .6923 / .5335 | .0962 |
| Disease category, 25 classes | .5000 / .7885 / .6374 | **.5962 / .9038 / .7284** | .5192 / .8654 / .6609 | .0577 |

HS24가 두 target 모두에서 top-1, top-5, MRR, macro recall 및 NLL 기준으로 가장 좋았다.
이 결과는 P0 activation에 닫힌 진단 label 정보가 존재하고 supervised linear map으로 읽을
수 있음을 보인다. 특히 같은 HS32/P0에서 vanilla AV의 source-answer, gold-PDD, category
literal mention은 모두 0이므로, 현재 관찰은 `activation 정보 부재`가 아니라 `vanilla
자연어 decoder의 task mismatch`와 일치한다.

다만 probe는 train에서 정의된 49 PDD 또는 25 category 중 하나를 고를 뿐이며 관찰·관계·
근거를 열린 자연어로 설명하지 못한다. 위 수치는 validation model-selection 결과이지
locked-test 성능도 아니다. HS24 probe의 우세와 별개로 공개 AV/AR checkpoint 호환 index는
HS32이므로 Medical-NLA primary와 round-trip은 HS32를 유지하고 HS16/24는 sensitivity로 둔다.

같은 L32 AV decoder에 HS16/HS24 P0 activation을 입력하는 교차-layer sensitivity도
완료했다. Default와 task-aligned prompt 모두 HS16/24/32에서 parse 1.0이었지만 source
answer, gold PDD, category, own-donor source gap이 전부 0이었다. Prompt trigram gap은
default HS24와 task-aligned HS32에서만 +0.0007이고 나머지는 0이었다. 따라서 layer 또는
generic medical suffix 변경만으로 vanilla AV의 P0 diagnosis recovery는 개선되지 않았다.
단, HS16/24 비교에는 activation layer와 HS32 decoder의 distribution shift가 함께 들어가므로
주표의 layer 승패나 layer 정보 부재로 해석하지 않는다.

Frozen validation 52행의 HS32/P0에서도 default와 task-aligned prompt를 직접 비교했다.
두 조건 모두 parse 1.0이었지만 source answer, gold PDD, disease category의 literal
phrase/alias mention은 전부 0이었다. Same-category/different-source donor 대비 source-answer
gap도 0이고 prompt trigram gap은 default 0.0000, task-aligned +0.0007이었다. 따라서 generic
medical suffix만으로 vanilla AV가 생성 전 의료 판독기로 바뀌었다는 증거는 없다. Default를
vanilla primary prompt로 유지하고 task-aligned는 prompt sensitivity로 남긴다. 이 결과는
lexical diagnostic이므로 P0 정보 부재가 아니라 probe/output-head와의 판독 능력 차이로
검증한다.

이 0은 semantic score가 아니다. 현재 scorer는 case/punctuation/plural과 등록된 gold alias는
처리하지만 source answer/category의 약칭과 미등록 임상 동의어를 추론하지 않는다. 따라서
`GERD`, `PE` 또는 표현 수준의 동의어가 false negative일 수 있다. Frozen validation P0는
52 cases x 2 prompts x 3 layers = 312 readout이며, 이 312행 전체를 local Llama-3-8B judge로
blinded semantic audit했다. Judge에는 note를 보이지 않고 세 target의 역할과 순서를 숨기며,
판정 근거로 readout 안의 exact quote를 요구했다. 이 evidence-quoted semantic 판정을
Table 1의 AI 평가로 사용하되, 사람 검증이나 activation grounding으로 해석하지 않는다.

Validation에는 P1/P2 208행도 있어 총 520 readout을 생성했다. 별도의 old exploratory test는
171 cases x P0/P1/P2 = 513행이다. 운영상 누적 1,033행을 하나의 평가 모집단으로 합치지 않는다.
방법 선택의 primary 분모는 default/HS32/P0 52행이고, layer/prompt sensitivity 분모는 P0
312행이다.

같은 validation의 P1/P2 positive control에서는 source-answer mention이 default
0.5192/0.5962, task-aligned 0.5577/0.5000이었다. Same-category donor 대비 gap도
+0.4091~+0.5000으로 사례별 source answer와 연결됐다. 그러나 P1에서 CoT reasoning에
source-answer alias가 없던 행은 5건뿐이었고 두 prompt 모두 0/5였다. 따라서 P1 전체
결과는 주로 이미 생성된 CoT 문자열 누출로 보고, P2는 answer-exposed positive control로만
사용한다. P0/P1/P2를 서로 독립적인 내부 판독 성능처럼 비교하지 않는다.

Pilot test P0/HS32 171행 생성은 완료됐다. `parsed_explanation_tag`는 171/171, 빈 출력은
0/171이며 split은 test-seen 71, PDD-heldout 100으로 E1 test 모집단과 일치한다. 출력
길이는 637--741자(중앙값 697, 평균 696.9)로 매우 좁다. 이는 generation 안정성은
보이지만 사례별 내용 복원이나 faithfulness를 뜻하지 않는다. 동일 문구 반복률,
own-case 대 shuffled-case 격차, 이후 official claim extraction을 통과하기 전에는
vanilla AV의 설명 성능으로 인용하지 않는다.

같은 disease category 안에서 donor prompt를 한 칸 회전한 lexical pilot은 164행에서
output trigram의 own-prompt containment 0.0013, shuffled-prompt containment 0.0013,
gap -0.0001이었다. 원문 표현을 사례 특이적으로 복원한다는 증거는 없으며 generic 또는
paraphrastic prose를 우선 의심해야 한다. 다만 exact trigram은 의미 보존 paraphrase를
놓치므로 이 값은 경고용 진단이지 Table 2의 설명 점수나 E5 faithfulness 판정이 아니다.

P1/P2 L32도 각 171행 생성됐고 전부 parse됐으며 빈 출력과 exact duplicate는 없었다.
Same-category lexical pilot은 P1 own/shuffled 0.0067/0.0064(gap +0.0003), P2
0.0017/0.0018(gap -0.0001)이었다. P1의 높은 절대 overlap은 같은-category shuffled에도
거의 그대로 나타나 CoT의 공유 임상 어휘 효과로 보인다. P2에서도 사례 특이적 trigram
gap은 확인되지 않았다. 단, 짧은 진단명은 trigram을 만들지 못하므로 source-answer 및
gold-PDD phrase mention과 semantic extraction을 별도로 검사한다.

Phrase-level 검사는 위치 차이를 분리했다. P0에서는 source answer, gold PDD, disease
category mention이 모두 0/171이었다. P1은 source answer 0.4912, gold PDD 0.1404,
category 0.5848이었고 same-category donor answer 대비 source-answer mention gap은
+0.4146(164행)이었다. 그러나 P1 leakage-free subset은 15행뿐이며 source-answer
mention은 1/15=0.0667이었다. 따라서 P1의 높은 값은 대부분 CoT 안에 이미 등장한 답
문자열의 영향으로 해석한다. P2는 source answer 0.3918, gold PDD 0.0819, category
0.4854였고 donor 대비 gap은 +0.3598이었다. P2 positive control은 vanilla AV가 답이
노출된 activation에서 사례별 source-answer 정보를 부분적으로 읽을 수 있음을 보이지만,
P0의 생성 전 diagnosis recovery는 0이었다. P0가 evidence를 의미 수준에서 복원하는지는
별도 claim extraction으로 평가한다.

이 171행은 위치 선택과 vanilla prompt 진단에 이미 사용됐으므로 최종 untouched test가
아니다. 이후 방법 선택의 근거로 수치를 계속 추가하지 않고 exploratory 결과로 보존한다.
최종 주표는 새 confirmatory protocol에서 다시 계산한다.

## Downstream-confirmatory split freeze

Seed 17, pilot-heldout PDD component 금지 조건으로 266 train / 52 val-seen /
72 test-seen / 106 test-PDD-heldout을 동결했다. Logical population SHA-256은
`7d0a89a880fa868959099b7146c369cccaac5e7701d7ce5d8f01356ecfb68894`다. Held-out은
12 PDD와 10 categories로 구성되고 gold-label-in-note는 5/106이다.

이 split은 앞으로의 downstream Medical-NLA 선택과 평가에는 고정됐지만, 과거 source
실행 universe와 같은 496행을 재분할한 것이다. 감사 결과 heldout 106/106 모두 과거
backbone output이 존재했고, old test-seen에서 온 16행은 CoT와 vanilla AV output도
존재했다. 따라서 `dataset-level untouched`나 pristine confirmatory test라고 쓰지 않고
`locked downstream evaluation`으로 제한한다.
