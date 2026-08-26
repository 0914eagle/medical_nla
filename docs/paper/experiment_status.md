# 논문 실험 상태

기준일: 2026-08-26. 제한 데이터 원문이나 개인 식별자는 기록하지 않는다.

| 단계 | 상태 | 완료 조건 | 다음 단계 의존성 |
|---|---|---|---|
| E0 DiReCT audit/evaluator | 완료 | 496행 canonical split, official oracle smoke | E1-E4 |
| E1 source/activation | 실행 중 | 496 source outputs, P0/P1/P2 x L16/24/32 | E2 |
| E2 capability baselines | 대기 | output head, probe, vanilla NLA | E3-E5 |
| E3 Medical-NLA train | 대기 | SFT-only, recon-only, full, 3 seeds | E4-E6 |
| E4 DiReCT explanation | 대기 | official metrics + human audit | Table 2 |
| E5 DDX grounding | 대기 | shuffle/counterfactual/round-trip 통과 | RQ2, E6 gate |
| E6 text patching | 조건부 | target change + no-op preservation | RQ3 |
| E7 MCR OOD | 후순위 | frozen checkpoint external test | generalization |

## E1 실행 현황

| 서버 | data root | GPU | split | 행 |
|---|---|---|---|---:|
| `165.132.76.62` | `/data/heejae` | physical 2,3 | train + val_seen | 325, 실행 중 |
| `165.132.76.125` | `/data1/heejae` | physical 0,1 | test_seen + PDD-heldout | 171, 완료 |

공통 설정은 Gemma-3-12B-IT, greedy decoding, batch size 1,
`max_new_tokens=2048`, forced answer 비활성화다. strict PDD accuracy는 실행 중
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
- 모델 answer alias가 CoT reasoning에 등장: 156/171 = 0.9123
- gold PDD alias가 CoT reasoning에 등장: 49/171 = 0.2865
- Activation rows 513 = 171 x P0/P1/P2
- Tensors 1,539 = 513 rows x 3 layers
- Prompt tokens: min 209, mean 1,619, max 4,304

P1 leakage-free test subset은 15행뿐이므로 P1의 source-decision 결과는 정성·민감도
분석으로 제한한다. P0가 주 설명 판독 위치라는 결정이 확정됐다. Strict PDD 0.193은
세부 label exact/alias 점수이며, by-split category와 official semantic score가 나오기
전에는 source model의 최종 진단 성능으로 해석하지 않는다.

## 즉시 할 일

1. 62번 train+val 실행의 `source_cot_answers.jsonl`, activation, manifest 완주 확인
2. 두 서버 결과를 합쳐 strict PDD, category, token-F1을 split별로 집계
3. P1의 `diagnosis_alias_in_reasoning`에 따른 clean/leaky 민감도 분석
4. CoT를 DiReCT official prediction schema로 변환한 뒤 official semantic matching 실행
5. Validation에서 primary layer와 probe regularization을 선택하고 test에는 고정 적용
