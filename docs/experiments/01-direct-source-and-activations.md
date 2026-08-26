# E1. DiReCT source CoT and activation extraction

## 질문

같은 임상 사례에서 source CoT, 최종 진단, 생성 전·후 activation을 누락 없이 만들 수
있는가? 어느 위치가 공정한 CoT-vs-NLA 비교점인가?

## 설정

- Backbone: `google/gemma-3-12b-it`
- Decoding: greedy, `do_sample=false`
- `max_new_tokens=2048`, batch size 1, forced answer 비활성화
- Layers: 16, 24, 32
- P0: prompt 마지막 토큰, 생성 전
- P1: assistant의 마지막 `The answer is` marker 마지막 subtoken, diagnosis 전
- P2: parsed diagnosis 마지막 subtoken
- Hidden size: 3840

## 실행 분할

| 서버 | root | physical GPU | rows |
|---|---|---|---:|
| 62 | `/data/heejae` | 2,3 | train+val 325 |
| 125 | `/data1/heejae` | 0,1 | test 171 |

완료 시 각 서버에서 source answers는 각각 325/171행, activation rows는 975/513행이어야
한다. 각 layer에서 P0 manifest는 325/171행, P1+P2 manifest는 650/342행이어야 한다.

## Smoke 결과와 설계 수정

10행 smoke에서 parse는 10/10이었지만 strict PDD alias hit는 0/10, category hit는 6/10이었다.
모델 최종 answer alias는 CoT에 이미 8/10 등장했고 gold PDD alias는 1/10 등장했다.
Activation은 모든 layer/position에서 누락 없이 3840차원으로 저장됐다.

따라서 P1은 최종 answer 문자열이 reasoning 안에 이미 존재할 가능성이 높다. P0를 주
Medical-NLA 입력과 CoT 비교 위치로 고정한다. P1은 `diagnosis_alias_in_reasoning=false`
subset 또는 leakage sensitivity로만 보고하고 P2는 positive control이다.

## 완료 후 집계

- strict PDD alias accuracy
- disease-category accuracy
- answer token-F1과 official semantic match
- answer/gold alias in CoT rate
- split, layer, position별 manifest completeness

Strict PDD 중간 정확도만으로 source model을 탈락시키지 않는다. PDD가 매우 세분화되어
있으므로 category와 official semantic evaluator를 함께 본다.

## 125번 test 실행 결과

2026-08-26에 test_seen + test_pdd_heldout 171행이 완료됐다.

| 항목 | 결과 |
|---|---:|
| Answer parse | 171/171 (1.0000) |
| Forced answer | 0/171 |
| Strict PDD alias hit | 33/171 (0.1930) |
| Disease-category hit | 87/171 (0.5088) |
| Diagnosis-label token F1 | 0.1850 |
| Model answer alias already in CoT | 156/171 (0.9123) |
| Gold PDD alias already in CoT | 49/171 (0.2865) |
| Extraction rows | 513 |
| Stored tensors | 1,539 |

`1,539 = 171 cases x 3 positions x 3 layers`로 완전하다. P1 clean subset은 15행뿐이므로
P1을 독립적인 pre-answer readout 결과로 일반화하지 않는다. P0를 primary로 사용한다.

같은 171행에서 direct prefill baseline은 strict PDD 36/171 = 0.2105였다. CoT는
33/171 = 0.1930으로 -1.75%p였고, paired discordant cases는 CoT rescue 7 대 CoT break
10, McNemar exact p=0.6291이었다. 이는 strict PDD상 CoT가 Direct보다 낫거나 나쁘다는
증거가 없다는 뜻이다. Disease category는 Direct 0.5029, CoT 0.5088
(CoT-Direct +0.0058, paired McNemar exact p=1.0000)로 사실상 같았다. Label token F1은
Direct 0.1593, CoT 0.1850으로 CoT가 조금 높았지만, 이는 진단명 문자열 유사도일 뿐
explanation quality가 아니다. 설명 품질 비교는 E4의 official `Obs*`/`Exp*`로 별도
수행한다.

Split별로도 같은 패턴이다. PDD-heldout 100행에서 Direct/CoT strict PDD는
0.1800/0.1700, category는 0.5800/0.6000, token F1은 0.0650/0.1367이었다. Test-seen
71행에서는 strict PDD 0.2535/0.2254, category 0.3944/0.3803, token F1
0.2921/0.2530이었다. Heldout과 seen은 label 및 category 구성이 다르므로 두 pool의
절대 정확도를 난이도 차이로 단정하지 않는다.
