# 06 — CoT의 이중성

**질문**: 추론을 시키면 앵커링이 막히는가.

**상태**: ✅ 전수 생성 및 canonical matcher 재집계 완료. 이전 보고의
"CoT가 무력화"는 n=381의 과대평가였고, 전수 정본에서는 "완화"로 정정됐다.

---

## 설정

같은 케이스 파일이 `prompt`(direct)와 `prompt_cot`를 함께 나른다. 두 조건은
**바이트 단위로 동일한 접두사**를 공유하고 지시문만 다르다 — 지시문이 제시
뒤에 오므로 인과 어텐션 하에서 cue 위치 활성값이 두 조건에서 같다.

- `DIRECT_INSTRUCTION`: 질문 + `"Give the diagnosis only. Do not explain your
  reasoning."` + 형식 제약
- `COT_INSTRUCTION`: arXiv:2605.28301의 문구를 따르되 USMLE 시험 프레이밍은 뺌
- 두 조건 공통: `You MUST end your response with exactly "The answer is <diagnosis>."`

**"설명하지 말라"가 왜 필요한가**: 자유롭게 두면 gemma-3-12b-it가 "Okay,
let's break down this case"로 시작해 수백 토큰을 추론한다. 그러면 direct
조건이 사실상 CoT 조건이 되고 논문이 기대는 대비가 사라진다. 이것은 계측
조건이지 배포 프롬프트가 아니며, **네 arm 전부에 동일하게** 들어가므로
arm 간 차이를 만들 수 없다.

## 결과

| | direct | CoT |
|---|---:|---:|
| no-note accuracy | **.9897** | **.7464** |
| wrong-note accuracy | **.8117** | **.7018** |
| 오답 소견서 낙폭 | **−17.80%p** | **−4.46%p** |
| 제안 채택률, moved 중 | 28.3% | **43.0%** |

## 읽는 법

**추론은 피해를 1/4로 줄이지만 없애지 못한다.** 그리고 **채택률은 오히려
늘어난다** — 답이 덜 바뀌는데, 바뀔 때는 제안 쪽으로 더 자주 간다.

추론은 **방패인 동시에 합리화의 지면**이다. 이 이중성이 §4.2가 CoT를 해법으로
제시하지 않는 이유이고, [07](07-chain-attribution-rule-based.md)·[08](08-cot-llm-monitor.md)이
"그 지면에 원인이 적히는가"를 따로 묻는 이유다.

**Table/Figure에서의 역할**: 이 결과는 CoT가 정답을 보장하는지 묻는 표가
아니라, 같은 오답 소견서 개입의 행동 효과가 응답 모드에 따라 어떻게 달라지는지
보는 robustness 분석이다. 낙폭은 전체 취약성, 채택률은 움직인 사례 중 제안으로
간 비율이므로 같은 분모가 아니다.

**말하면 안 되는 것**: `채택률 43%`만으로 CoT가 제안을 더 원인으로 사용했다고
단정하지 않는다. CoT arm의 moved 모집단 자체가 달라진 조건부 비율이다.

## 한정어

형식 제약(`ANSWER_FORMAT`)은 실제 배포 프롬프트가 아니다. arm 상수라 결론을
흔들지 않지만 **절대 정확도를 눌렀을 수 있다**. 그리고 2605.29889가 출력 형식
제약 자체가 실패를 유발할 수 있음을 보였으므로, §3.2에 통제로 **명시**한다.

## 재현 조건

- direct와 CoT 모두 deterministic greedy decoding(`do_sample=false`)을 썼다.
- Direct는 최대 64 new tokens, CoT는 최대 2,048 new tokens이며, CoT가 형식
  문장 전에 잘리면 동일 모델로 answer-only completion을 붙인다.
- 따라서 표의 차이는 sampling seed 변동이 아니라 prompt instruction 차이다.
