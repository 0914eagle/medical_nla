# 06 — CoT의 이중성

**질문**: 추론을 시키면 앵커링이 막히는가.

**상태**: ⚠️ Direct-selected cohort의 탐색적 재집계 완료. 이전 보고의
"CoT가 무력화"는 n=381의 과대평가였지만, 현재 1,204건만으로는 CoT가
anchoring을 완화한다고 정식 비교할 수 없다. Direct/CoT 2×2 matched 비교가 남아
있다.

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

## 결과 — 동일한 Direct-defined clean 1,204건

| | direct | CoT |
|---|---:|---:|
| no-note accuracy | **1.0000** | **.7068** |
| wrong-note accuracy | **.7625** | **.6628** |
| 오답 소견서 낙폭 | **−23.75%p** | **−4.40%p** |
| moved | 287 | 220 |
| 제안 채택률, moved 중 | 30.0% | **49.1%** |

## 읽는 법 — 탐색적 관찰

이 Direct-selected 집합에서는 CoT의 none-to-wrong gap이 4.40%p로 Direct의
23.75%p보다 작게 관측됐다. 그러나 이것만으로 **추론이 피해를 약 1/5로 줄인다**고
말할 수 없다. Direct no-note 정답으로 사례를 골랐기 때문에 Direct는 1.0에서
시작하지만 CoT는 이미 .7068이고, CoT에는 wrong note가 추가로 망가뜨릴 여지가
작은 floor effect가 생길 수 있다.

채택률도 Direct 30.0%, CoT 49.1%로 관측됐지만 moved 분모가 서로 다르다. 따라서
현재 결과는 `CoT 조건에서도 anchoring과 suggestion adoption이 남는다`는
탐색적 관찰이지, CoT가 더 강건하거나 더 순응적이라는 정식 비교가 아니다.

CoT가 **방패인 동시에 합리화의 지면일 가능성**은 남지만, 방패의 크기는 아래
confirmatory experiment 전에는 확정하지 않는다. [07](07-chain-attribution-rule-based.md)·
[08](08-cot-llm-monitor.md)은 별도로 "그 지면에 원인이 적히는가"를 묻는다.

**Table/Figure에서의 역할**: 이 결과는 CoT가 정답을 보장하는지 묻는 표가
아니라, 같은 오답 소견서 개입의 행동 효과가 응답 모드에 따라 어떻게 달라지는지
보는 robustness 분석이다. 낙폭은 전체 취약성, 채택률은 움직인 사례 중 제안으로
간 비율이므로 같은 분모가 아니다.

**말하면 안 되는 것**: `채택률 49.1%`만으로 CoT가 제안을 더 원인으로 사용했다고
단정하지 않는다. CoT arm의 moved 모집단 자체가 달라진 조건부 비율이다.

## 한정어

형식 제약(`ANSWER_FORMAT`)은 실제 배포 프롬프트가 아니다. arm 상수라 결론을
흔들지 않지만 **절대 정확도를 눌렀을 수 있다**. 그리고 2605.29889가 출력 형식
제약 자체가 실패를 유발할 수 있음을 보였으므로, §3.2에 통제로 **명시**한다.

## 표본이 선택됐다는 것 — 정확도 비교에 쓸 수 없는 이유

Primary 비교의 1,204건은 canonical matcher에서 **직답 no-note가 맞고 정답명이
presentation에 직접 나오지 않는 케이스만** 고른 집합이다. 소견서가 답을
움직였는지 판정하려면 원래 맞히던 케이스여야 하므로 그 선택은 옳다. 다만 그
대가로 direct의 무소견서 정확도는 **1.0 by construction**이다. CoT correctness는
eligibility에 쓰지 않았으므로 CoT no-note `.7068`은 1.0일 필요가 없다.

그래서 선택 집합에서 direct와 CoT의 no-note 정확도 차이를
**"CoT가 정확도를 해친다"로 읽으면 안 된다.** 종속변수로 표본을 고른 비교라
CoT는 내려갈 곳밖에 없다.

편향 없는 표본의 답은 이미 나와 있다 —
`docs/data/ddxplus_as_a_benchmark_2026-08-22.md`: 맨 DDXPlus 320건 짝지음에서
직답 **.3375** vs CoT **.3187**, 살림 24 / 깨뜨림 30, **exact p = 0.50**.
이 320건에서는 **유의한 정확도 차이를 검출하지 못했다**. 이는 동등성 검정이
아니므로 CoT와 direct가 정확히 동등하다고 확정하지 않는다.

별도의 1,747건 fixed-cohort 감사가 뒷받침하는 명제는 이것이며, 이 형태로만 쓴다:

> **추론은 모델이 이미 맞혔던 답을 흔든다.** 1,747 base case의 no-note와
> wrong-note를 각각 비교한 **3,494 paired prompt instances** 중
> 25.1%(877/3,494)에서 Direct와 CoT의 정오가 엇갈린다. 그중 CoT가 깬 것은
> 747건, 구한 것은 130건이다. 총 정확도 차이만 보면 이 뒤바뀜을 놓친다.

## 재현 조건

- direct와 CoT 모두 deterministic greedy decoding(`do_sample=false`)을 썼다.
- Direct는 최대 64 new tokens, CoT는 최대 2,048 new tokens이며, CoT가 형식
  문장 전에 잘리면 동일 모델로 answer-only completion을 붙인다.
- 따라서 표의 차이는 sampling seed 변동이 아니라 prompt instruction 차이다.

이 보정은 arm 간 비대칭이다 — direct에는 answer-only completion 경로가
없다. 위의 정오 불일치 25.1% 중 일부가 그 경로에서 왔을 가능성을 배제하지
않는다.

## 남은 것

- ~~direct/CoT 두 arm 모두 동일 canonical clean ID에서 낙폭·moved·채택을
  다시 기록한다.~~ **완료 (08-25: −23.75 → −4.40%p, n=1,204)**
- ~~CoT 문구 효과와 decoding 변동 분리~~ **완료** — 위 「재현 조건」이
  greedy·deterministic임을 명시한다.
- ▢ **Direct × CoT matched 2×2 confirmatory experiment**를 실행한다.

### Confirmatory 2×2 설계

같은 base ID에 다음 네 출력을 모두 요구한다.

| generation | no note | wrong note |
|---|---|---|
| Direct | Direct-none | Direct-wrong |
| CoT | CoT-none | CoT-wrong |

환자 presentation, source checkpoint, chat template, wrong-note 문장, answer format,
greedy decoding은 동일하게 두고 instruction과 CoT token budget만 사전 정의대로
다르게 한다. CoT가 budget 안에 closing answer를 내지 못한 `answer_forced` 사례는
표시하고, 제외/포함 민감도 분석을 함께 보고한다.

두 모집단은 서로 다른 질문에 답한다.

1. **Unbiased common cohort**: gold가 prompt에 직접 없고 네 셀이 모두 정상
   파싱되며 같은 base ID를 갖는 사례. 정답 여부로 선정하지 않는다. 여기서 네 셀의
   accuracy와 difference-in-differences를 계산한다.
2. **Shared-solvable cohort**: Direct-none과 CoT-none이 모두 정답인 교집합.
   두 방법 모두 no-note accuracy가 1.0이므로 wrong note 뒤 harmful flip rate를
   직접 비교한다.

주 interaction은 다음과 같다.

```text
Direct effect = Acc(Direct-wrong) - Acc(Direct-none)
CoT effect    = Acc(CoT-wrong)    - Acc(CoT-none)
Interaction   = CoT effect - Direct effect
```

양의 interaction은 CoT에서 note-induced accuracy loss가 더 작다는 뜻이다. 같은
사례를 네 번 쓰므로 case-level paired bootstrap 95% CI와 paired permutation test를
사용한다. No-note/wrong-note accuracy, harmful flip, suggestion adoption,
third-diagnosis movement, newly corrected, answer-forced rate를 모두 같은 공통
cohort에서 보고한다.

먼저 기존 출력에서 네 셀이 겹치는 전체 gold-absent 공통 cohort를 재분석한다.
기존 unbiased 320건은 코드와 통계 검증용 예비 분석으로만 쓰고, coverage가 부족한
셀만 새 GPU 생성으로 보완한다.
