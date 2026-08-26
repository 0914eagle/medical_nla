# Medical-NLA 논문 정본

## 논문 한 문장

> Medical CoT can be clinically plausible without faithfully exposing the model state;
> we develop Medical-NLA and require both expert-reference alignment and activation-level
> grounding before using its natural-language readouts for intervention.

한국어로는 다음과 같다.

> 의료 CoT는 그럴듯하더라도 모델 내부 상태를 충실하게 드러낸다고 보장할 수 없다.
> 우리는 내부 activation을 자연어로 판독하는 Medical-NLA를 만들고, 의사 주석과의
> 정렬 및 activation 수준 검증을 모두 통과한 판독만 개입에 사용한다.

## 논문의 목표

최종 목표는 설명가능성과 성능을 함께 개선하는 것이다. 다만 두 목표를 한 번에
주장하지 않는다.

1. Medical-NLA가 CoT와 vanilla NLA보다 임상적으로 필요한 정보를 잘 보존하는지 본다.
2. 그 문장이 해당 activation에 근거하는지 별도의 통제로 검증한다.
3. 앞의 두 관문을 통과한 뒤 text patching 또는 선택적 재추론으로 성능 개선을 시험한다.

## 가설

### H1. CoT의 임상적 그럴듯함과 내부 상태 충실성은 다르다

CoT는 설명 품질 비교군이다. 기존 연구가 제기한 비충실성 문제를 이 논문에서 처음부터
다시 증명하려 하지 않는다. 대신 같은 source run에서 CoT와 pre-generation activation
판독을 만들고, 의사 주석 정보의 복원과 사례 특이적 grounding을 나란히 비교한다.

### H2. 닫힌 내부 판독과 열린 자연어 판독의 능력 경계가 다르다

선형 probe는 미리 정한 진단 label을 매우 정확하게 읽을 수 있다. 그러나 하나의 probe가
학습 때 정의하지 않은 관찰, 속성, 관계, 불확실성을 자연어로 서술하지는 않는다.
NLA가 필요한 이유는 probe를 이기기 위해서가 아니라 이 열린 판독 공간을 다루기 위해서다.

### H3. 단순 의료 SFT는 충분하지 않으며 검증 가능한 Medical-NLA가 필요하다

정답 진단과 전형적 설명을 직접 생성하도록 SFT하면 seen-class 분류기나 의료 문구
생성기로 붕괴할 수 있다. 따라서 SFT-only, reconstruction-only, 임상 supervision과
activation grounding을 함께 쓰는 full method를 분리해 비교한다.

### H4. 검증된 판독만 인과적 개입에 사용해야 한다

설명 점수가 높다는 이유만으로 patching하지 않는다. 짝 깨기, evidence counterfactual,
identity round-trip을 통과한 뒤에만 dataset-native text edit가 목표 activation 및
backbone 행동을 선택적으로 바꾸는지 평가한다.

## 연구 질문

| RQ | 질문 | 핵심 산출물 |
|---|---|---|
| RQ1 | CoT, output head, probe, vanilla NLA의 능력 경계는 무엇인가 | Table 1 |
| RQ2 | Medical-NLA가 임상 정렬과 activation grounding을 함께 개선하는가 | Tables 2-3, Figures 2-3 |
| RQ3 | 검증된 판독을 편집·복원해 안전하게 행동을 개선할 수 있는가 | Table 4, Figure 4 |

## 데이터셋 역할

| 데이터셋 | 역할 | 하지 않는 주장 |
|---|---|---|
| DiReCT | 의사 observation-rationale-diagnosis tree에 대한 설명 품질 | activation ground truth |
| DDXPlus | label/evidence가 명시된 통제된 grounding·반사실·patching | 자연 임상 설명의 최종 품질 |
| MedCaseReasoning | 학습하지 않은 긴 자연 임상 텍스트의 OOD 검증 | gold evidence span이 있다는 주장 |

## Activation 위치

E1에서는 Gemma-3-12B-IT의 layer 16/24/32에서 세 위치를 추출한다.

| 위치 | 정의 | 논문 역할 |
|---|---|---|
| P0 | 임상 prompt 마지막 토큰, 생성 전 | CoT와 NLA의 주 비교 및 Medical-NLA 입력 |
| P1 | 생성된 CoT 뒤, 최종 answer 전 | reasoning 이후 상태의 보조 trajectory |
| P2 | 최종 answer 뒤 | 정답 문자열 누출을 확인하는 positive control |

P1은 CoT 안에 최종 진단이 이미 등장할 수 있다. smoke 10건에서 모델 답 alias가 CoT에
8건 등장했으므로 P1을 주 결과로 쓰지 않는다. P0가 정본 위치다.

## 주장 관문

- Table 2만 성공: `clinically aligned explanation`까지 가능
- Table 3까지 성공: `activation-grounded readout` 가능
- Table 4까지 성공: `causally useful for intervention` 가능
- Table 2가 높고 Table 3이 실패: 좋은 의료 설명 생성기일 뿐 faithful reader가 아님
- Table 4가 실패: 설명·grounding 기여는 남지만 성능 개선 주장은 철회

## 현재 상태

- E0 DiReCT 감사·환자 분리 split·공식 evaluator smoke: 완료
- E1 source CoT와 P0/P1/P2 activation: 두 서버에서 실행 중
- E2 이후 baseline, Medical-NLA 학습, 설명 평가, grounding, patching: 대기

상세 상태는 [`experiment_status.md`](experiment_status.md), 표·그림은
[`tables_and_figures.md`](tables_and_figures.md)를 따른다. 재현 설정은
[`prompts_and_hyperparameters.md`](prompts_and_hyperparameters.md), 문헌 위치는
[`related_work_map.md`](related_work_map.md)에 정리한다.
