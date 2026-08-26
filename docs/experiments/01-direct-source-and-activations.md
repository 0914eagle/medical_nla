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
