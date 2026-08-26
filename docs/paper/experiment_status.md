# 논문 실험 상태

기준일: 2026-08-26. 제한 데이터 원문이나 개인 식별자는 기록하지 않는다.

| 단계 | 상태 | 완료 조건 | 다음 단계 의존성 |
|---|---|---|---|
| E0 DiReCT audit/evaluator | 완료 | 496행 canonical split, official oracle smoke | E1-E4 |
| E1 source/activation | 완료 | pilot 496 source outputs, P0/P1/P2 x HS16/24/32 완전성 확인 | E2 |
| E2 capability baselines | 실행 중 | exploratory P0/HS32 vanilla AV 171행 완료; output head/probe 대기 | E3-E5 |
| E3 Medical-NLA train | 설계 차단 | SFT-only 실행 가능; reconstruction/full objective 미구현 | E4-E6 |
| E4 DiReCT explanation | 대기 | official metrics + human audit | Table 2 |
| E5 DDX grounding | 대기 | shuffle/counterfactual/round-trip 통과 | RQ2, E6 gate |
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

## 즉시 할 일

1. Train/validation의 strict PDD, category, token-F1을 같은 방식으로 집계
2. P1의 `diagnosis_alias_in_reasoning`에 따른 clean/leaky 민감도 분석
3. CoT를 DiReCT official prediction schema로 변환한 뒤 official semantic matching 실행
4. 기존 171행은 exploratory로 동결. 새 locked downstream split 266/52/72/106과 hash는 확정 완료
5. 새 heldout artifact 감사 완료: backbone 106/106, vanilla AV 16/106
6. 62번에서 logical population/split hash가 125번과 동일한지 확인
7. 기존 activation을 새 split ID로 재색인하고 join/completeness 100% 확인
8. Validation에서 primary index와 probe regularization을 선택하고 locked test에는 고정 적용
9. E3 전에 full objective를 RL/preference 방식으로 구현할지, SFT-only 논문으로 제한할지 결정

Gold-label-in-note audit은 raw 511행 중 28행(0.0548)으로 완료됐다. 기존 test CoT 171행에는
formatted answer marker가 여러 번 등장한 행이 0개여서 final-answer parser 수정이 pilot
수치를 바꾸지 않는다. 다음 split은 이 leakage flag를 eligibility에서 제거하지 않고 split별
민감도 분모와 ID hash를 함께 동결한다.

## E2 vanilla AV exploratory 상태

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
