# Medical-NLA 현재 연구 요약

## 최종 목표

설명가능성과 성능을 모두 개선하는 의료용 NLA를 만드는 것이 최종 목표다. 새로운
구조든 SFT-NLA든 형식은 열어두되, 자연어 출력이 의학적으로 좋아 보인다는 사실만으로
성공으로 판정하지 않는다.

## 출발점

기존 연구는 CoT가 모델이 실제 사용한 이유를 항상 드러내지 않으며 사후 합리화할 수
있음을 보였다. 의료에서도 설명의 임상적 타당성과 내부 계산 충실성을 구분해야 한다.
한편 linear probe는 사전에 정한 진단 label을 잘 읽지만, 새로운 증상·속성·관계를
하나의 열린 자연어 판독으로 보여주지는 않는다.

따라서 Medical-NLA의 역할은 probe를 단순히 정확도로 이기는 것이 아니다. 내부 상태를
사람이 읽을 수 있는 임상 언어로 표현하되, 그 문장이 실제 activation에서 나온 것인지
검증할 수 있게 만드는 것이다.

## 가설

1. CoT의 임상적 그럴듯함과 내부 상태 충실성은 같지 않다.
2. Probe의 닫힌 label detection과 NLA의 열린 자연어 readout은 서로 다른 능력이다.
3. 단순 의료 SFT는 분류기·문구 암기로 붕괴할 수 있어 reconstruction과 grounding이 필요하다.
4. 임상 정렬과 activation grounding을 모두 통과한 판독만 성능 개선에 사용해야 한다.

## 평가 구조

| 관문 | 데이터 | 질문 |
|---|---|---|
| 1. Clinical alignment | DiReCT | 의사 observation-rationale-diagnosis를 얼마나 복원하는가 |
| 2. Activation grounding | DDXPlus | 해당 사례 activation에 실제로 의존하는가 |
| 3. Causal utility | DDXPlus | text edit가 목표 상태와 행동을 선택적으로 바꾸는가 |

DiReCT는 511개 raw note 중 충돌·식별 실패·중복 15행을 제외한 496행을 쓴다. 최초
263/62/71/100 split의 test 171행은 설계 pilot로 사용했다. 최종 downstream protocol은
train 266, validation 52, seen test 72, PDD-heldout test 106으로 다시 동결했으며 환자와
held-out PDD component는 split 사이에 겹치지 않는다.
Gold PDD/root의 정규화된 완전 구문이 note에 직접 등장한 행은 raw 28/511(5.48%)였다.
이를 주 모집단에서 사후 제거하지 않고, split별 gold-label-absent 민감도 분석을 병기한다.

## 현재 실행

Gemma-3-12B-IT에서 source CoT와 layer 16/24/32 activation을 추출 중이다. 생성 전 prompt
마지막 토큰 P0를 주 비교 위치로 쓰고, CoT 뒤 P1은 answer 문자열 누출을 분석하는 보조
위치, answer 뒤 P2는 positive control로 쓴다.

- Server 62 `/data/heejae`: train+val 325행, GPU 2/3, 실행 중
- Server 125 `/data1/heejae`: exploratory test 171행, GPU 0/1, 완료
- Greedy, max new tokens 2048, batch size 1

10행 smoke에서 answer parse는 100%, strict PDD alias는 0%, disease category는 60%였다.
세부 PDD가 매우 좁아 strict 값만으로 모델을 판단하지 않고 official semantic evaluator와
category를 함께 본다. 중요한 설계 결과는 모델 answer alias가 CoT에 8/10 등장했다는
점이다. 그래서 P1이 아니라 P0를 주 Medical-NLA 입력으로 확정했다.

Test 171행에서는 parse 100%, strict PDD alias accuracy 19.30%, disease-category
accuracy 50.88%, diagnosis-label token F1 0.1850이었고 activation tensor 1,539개가
모두 생성됐다. 모델 answer alias는 CoT에 156/171(91.23%) 등장해 P1의
leakage-free subset이 15행뿐이었다. 따라서 P1은 민감도 분석으로만 두고 P0를 주
비교 위치로 확정한다. 같은 사례의 Direct/CoT strict PDD는 21.05%/19.30%
(McNemar p=.6291), category는 50.29%/50.88%(p=1.0)로 우열의 증거가 없다. CoT의
label token F1은 0.1850으로 Direct 0.1593보다 높았지만 이는 진단명 문자열 유사도이며
설명 품질은 아니다. 최종 설명 비교는 official DiReCT `Obs*`/`Exp*`로 수행한다.

단, 이 171행은 P0/P1/P2 위치 선택과 vanilla AV 진단에 이미 사용했으므로 untouched final
test가 아니다. 위 수치는 pilot로만 보존한다. 새 106행 PDD-heldout은 pilot-heldout 5개
PDD를 금지하고 선택했으며, logical population SHA-256과 split ID SHA-256을 동결했다.
다만 이 106행 중 일부는 과거 train/validation source 실행에서 output이 이미 생성됐을 수
있다. 그러므로 정확한 overlap을 집계하기 전에는 `dataset-level untouched`라고 하지 않고,
이 시점 이후 Medical-NLA 선택과 평가에 대해 고정한 downstream-confirmatory split이라고 한다.

## 다음 순서

1. E1 source/activation 완주와 official source score
2. 새 confirmatory heldout 106행이 과거 artifact에 얼마나 materialize됐는지 aggregate 감사
3. 62번에서 같은 logical population/split ID hash 재현 후 서버별 source path 정규화
4. Output head, linear probe, default/task-aligned vanilla NLA baseline. 공개 AV/AR 호환 때문에 HS32를 primary로 사용
5. SFT-only를 3 seeds로 학습. Reconstruction/full은 objective 구현 후에만 추가
6. DiReCT Table 2로 clinical alignment 평가
7. DDXPlus Table 3으로 shuffle/counterfactual/round-trip 검증
8. Table 3 통과 시에만 text patching과 성능 개선 평가

이 구조에서 설명 점수만 오르면 `좋은 의료 설명 생성기`, grounding까지 통과하면
`내부 상태 판독기`, patching까지 성공하면 `설명과 성능을 함께 개선하는 방법`이라고
단계적으로 주장한다.
