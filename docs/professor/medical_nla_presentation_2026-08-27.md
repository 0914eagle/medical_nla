# Medical-NLA 교수님 발표 구성 (2026-08-27)

이 문서는 현재 연구 방향, DiReCT 데이터 구성, 실제 baseline prompt와 실행 설정,
P0/P1/P2 activation 위치, 현재까지 나온 결과, 최종 논문 표를 처음 듣는 사람에게 설명하기
위한 **슬라이드 구성과 발표 원고**다. Restricted DiReCT 원문과 환자 식별자는 포함하지 않는다.

Legacy 발표 원고의 형식만 차용해 전체를 `Introduction -> Methodology -> Data and
Experimental Setup -> Experimental Results(RQ1 -> RQ2 -> RQ3) -> Conclusion` 순서로
구성한다. 과거 wrong-note 연구의 문제, 가설, 수치, 표는 현재 발표에 가져오지 않는다.

각 슬라이드의 표, code block, 짧은 bullet은 **화면에 실제로 놓을 내용**이다. 뒤의 줄글은
**발표자 노트**다. 발표할 때는 표의 모든 숫자를 읽지 않고 먼저 모집단과 비교축을 설명한 뒤,
굵은 셀과 그 셀이 답하는 RQ만 연결한다.

| 대단원 | 발표에서 답하는 질문 |
|---|---|
| Introduction | 왜 CoT만으로 부족하며 왜 open-text internal readout이 필요한가 |
| Methodology | 무엇을 어디서 읽고, 어떤 조작·통제·중단 기준으로 검증하는가 |
| Data and Experimental Setup | 각 검증에 어떤 데이터와 모집단·split을 사용하는가 |
| Experimental Results | RQ1, RQ2, RQ3에 현재 데이터가 각각 무엇이라고 답하는가 |
| Conclusion | 확립된 기여, 아직 성립하지 않은 주장, 다음 실행은 무엇인가 |

## 발표에서 먼저 구분할 것

- **현재 결과가 없는 것이 아니다.** E1 exploratory backbone 결과, E2 validation diagnosis
  probe와 forced-answer likelihood, vanilla AV semantic audit, E4 SFT-only validation 결과가 있다.
- 논문 주표의 빈칸은 `test_seen=72`, `test_pdd_heldout=106` locked evaluation과 아직 구현하지
  않은 reconstruction/full Medical-NLA 결과를 validation SFT-only 값으로 미리 채우지 않았기 때문이다.
- 발표에서는 `현재까지 확인한 결과`와 `최종 locked-test 표`를 서로 다른 슬라이드로 보여준다.
- `PDD`는 **Primary Discharge Diagnosis**다. 퇴원 기록 자체가 아니라 disease category보다
  세분화된 최종 진단 label이다. 예를 들어 category가 `Heart Failure`일 때 PDD는 `HFrEF`,
  `HFpEF`가 될 수 있다.
- `Seen PDD`는 같은 환자를 봤다는 뜻이 아니다. 환자는 split 간 분리되어 있고, PDD label이
  train에 있었던 새 환자다. `Held-out PDD`는 PDD label 자체가 train에 없었던 새 환자다.

---

# Part I. Introduction

Introduction에서는 데이터셋 세부사항을 먼저 말하지 않는다. `CoT의 충실성 문제 -> closed
internal tool과 open explanation의 간극 -> 검증 가능한 Medical-NLA -> 세 RQ`까지만 세운다.
데이터셋 역할, prompt, activation 위치, evaluator는 Methodology에서 소개한다.

## Slide 1. 두괄식 연구 요약

### 화면에 넣을 내용

- CoT는 유용한 임상 설명을 생성하지만 모델의 실제 내부 판단 과정을 충실하게 보고한다고
  보장할 수 없다.
- Linear probe는 사전에 정의한 진단 label을 내부에서 잘 탐지하지만, 환자 고유 관찰·속성·
  관계를 하나의 열린 자연어 설명으로 직접 제공하지 않는다.
- Vanilla AV는 현재 DiReCT validation의 생성 전 P0에서 진단 target과 physician observation을
  안정적으로 복원하지 못했다.
- 본 연구는 activation을 임상 자연어로 판독하는 Medical-NLA를 만들고, **임상 설명 품질**과
  **해당 activation에 대한 사례 특이적 의존성**을 별도로 검증한다.
- 검증된 판독이 확보된 뒤에만 selective correction과 text patching을 통해 진단 성능 개선을
  평가한다.

### 발표 줄글

연구의 최종 목표는 설명가능성과 진단 성능을 모두 개선하는 것이다. 그러나 자연어 설명이
의학적으로 그럴듯하다는 이유만으로 내부 상태를 충실하게 읽었다고 할 수는 없다. 따라서 먼저
읽을 수 있고 검증할 수 있는 Medical-NLA를 만들고, 그 다음에만 해당 판독을 교정에 사용한다.
성능 향상은 사전 결론이 아니라 마지막 실험에서 판정할 downstream hypothesis다.

---

## Slide 2. 기존 연구에서 남은 문제

### 화면에 넣을 내용

1. CoT faithfulness 연구는 모델이 말한 설명과 실제 판단 원인이 다를 수 있음을 보였다.
2. Probe와 mechanistic interpretability 연구는 내부에 특정 label 또는 속성 신호가 있는지
   탐지할 수 있음을 보였다.
3. 그러나 의료 사례의 관찰·속성·관계를 **열린 자연어로 판독**하면서, 그 문장이 실제
   activation에서 나왔는지를 함께 검증하는 문제는 남아 있다.

### 발표 줄글

우리의 신규성은 단순히 같은 사례에서 CoT와 activation을 비교하는 데 있지 않다. 핵심은
환자 고유의 observation, attribute, relation을 open-vocabulary clinical text로 읽고,
임상적 정답성과 activation 의존성을 서로 다른 시험으로 평가한 뒤, 검증된 판독만 개입에
사용하는 데 있다.

---

## Slide 3. 대전제, 가설, 연구 질문

### 화면에 넣을 내용

> **대전제:** 믿을 수 있는 의료 내부 설명은 임상적으로 타당한 문장일 뿐 아니라,
> 실제 source-model activation에 사례 특이적으로 근거해야 한다.

| 가설 | 핵심 주장 | 대응 연구 질문 |
|---|---|---|
| H1 | Medical-NLA는 CoT와 vanilla NLA보다 의사가 표시한 관찰과 관찰-진단 연결을 더 잘 복원한다 | RQ1: Medical-NLA가 CoT·vanilla NLA보다 임상 설명을 잘 복원하는가? |
| H2 | 임상적으로 그럴듯한 문장만으로는 내부 판독이라 할 수 없으며, Medical-NLA 설명은 해당 사례 activation에 의존해야 한다 | RQ2: 설명이 해당 사례 activation에 의존하는가? |
| H3 | Activation-grounded한 설명은 dataset-native claim 편집을 통해 내부 상태와 진단 출력을 선택적으로 바꿀 수 있다 | RQ3: 설명을 편집해 상태와 진단을 선택적으로 바꿀 수 있는가? |

### 발표 줄글

세 질문은 병렬 체크리스트가 아니라 단계적 자격 조건이다. RQ1만 통과하면 Medical-NLA는
의사 기준에 잘 맞는 **의료 설명 생성기**다. RQ2까지 통과해야 그 설명이 현재 사례의 내부
상태에서 읽혔다고 말할 수 있는 **내부 판독기**가 된다. RQ3까지 통과해야 판독을 이용해
내부 상태와 진단을 선택적으로 제어하는 **성능 개선 방법**이라고 부를 수 있다. CoT의 한계,
probe의 closed-label 능력, P0의 정보 존재 여부는 이 세 RQ를 정당화하고 해석하기 위한 선행
근거이지 별도의 RQ가 아니다.

| 단계 | 표 | 통과했을 때 가능한 주장 |
|---|---|---|
| 선행 representation audit | Table 1 | P0에서 진단·source decision·finding/value가 decode 가능한지 설명 가능 |
| RQ1: clinical explanation quality | Table 2 | Medical-NLA는 임상 설명 생성기로서 CoT·vanilla NLA보다 우수 |
| RQ2: activation grounding | Table 3 | Medical-NLA는 해당 사례 activation을 읽는 내부 판독기 |
| RQ3: text-mediated intervention | Table 4 | Medical-NLA는 상태와 진단을 선택적으로 바꾸는 성능 개선 방법 |

---

# Part II. Methodology

Methodology는 데이터셋을 먼저 나열하지 않고, **무엇을 검증하며 어떤 반론을 어떤 통제로
막을 것인지**를 먼저 고정한다.

```text
무엇을 성공으로 부를 것인가
 -> source model의 어느 상태를 읽는가
 -> P0에 목표 정보가 실제로 decode 가능한가
 -> Medical-NLA가 임상적으로 맞는 설명을 만드는가
 -> 그 설명이 자기 activation에 사례 특이적으로 의존하는가
 -> 자연어 병목을 거친 개입이 identity와 비목표 상태를 보존하는가
```

핵심은 physician annotation과 activation ground truth를 혼동하지 않는 것이다. Methodology에서는
각 가설의 조작, 비교군, failure condition을 먼저 설명한다. 구체적인 데이터 구조와 split은 이
검증 설계를 이해한 뒤 Experimental Results 바로 앞의 Data and Experimental Setup에서 제시한다.

## Slide 4. 평가 프로토콜

| 단계 | 데이터 | 질문 | 통과했을 때 가능한 표현 |
|---|---|---|---|
| Clinical alignment | DiReCT | 의사 observation-rationale-diagnosis를 복원하는가 | clinically aligned explanation |
| Activation grounding | DDXPlus | 판독이 해당 사례 activation 변화에 따라가는가 | activation-grounded readout |
| Causal utility | DDXPlus | 판독 기반 개입이 목표 상태와 행동을 선택적으로 바꾸는가 | useful correction/patching |

이 세 단계는 한 기존 논문의 이름을 가져온 표준 관문이 아니라, 본 연구가 기존 평가 개념을
조합해 만든 단계적 검증 프로토콜이다. Clinical alignment만 통과하면 좋은 의료 설명
생성기라고 말할 수 있지만 내부 판독기라고 부르지 않는다. Grounding까지 통과해야 내부
판독기라고 말하며, utility까지 통과해야 설명과 성능을 함께 개선한다고 주장한다.

### Methodology에서 고정하는 검증 질문

| 검증 질문 | 주 조작 | 반드시 필요한 비교군 | 실패하면 남는 주장 |
|---|---|---|---|
| P0에 목표 정보가 존재하는가 | Target별 probe decoding | Majority, label/answer/value shuffle | 해당 target을 NLA 평가 범위에서 제외 |
| 임상 설명이 맞는가 | 동일 case의 CoT/NLA를 physician tree와 비교 | 공통 method-blind extractor | 좋은 설명 생성 여부만 판정 |
| 자기 activation을 읽는가 | Matched pair 대 hard shuffle, finding edit | Mean/zero, activation swap | Activation-grounded 주장을 철회 |
| 자연어 병목이 상태를 보존하는가 | AV->text->AR identity와 native-value edit | Raw/oracle patch, no intervention | Patching 기여를 철회 |
| 전체 행동이 개선되는가 | Validation-gated selective intervention | Patch-all, probe gate, oracle gate | 조건부 정보 가치만 유지 |

이제 이 질문들을 실제로 구현하기 위해 source prompt와 activation 위치부터 정의한다.

---

## Slide 5A. Backbone에 실제로 넣은 prompt

### 공통 임상 prefix

```text
You are an expert physician. A patient presents as follows:

[Chief complaint, HPI, PMH, family history, physical exam,
and pertinent results assembled as one restricted clinical note]
```

### Direct instruction

```text
What is the single most likely diagnosis?

Give the diagnosis only. Do not explain your reasoning.

You MUST end your response with exactly "The answer is <diagnosis>."
```

Direct에서는 assistant turn을 `The answer is`로 미리 시작한다. 단순히 “설명하지 말라”고
적어도 Gemma가 긴 reasoning을 생성해 Direct arm이 CoT arm으로 변했기 때문이다. Prefill은
답만 완성하게 하며, causal attention상 prefill 이전의 P0 activation을 바꾸지 않는다.

### CoT instruction

```text
Work through this case as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Which diagnoses fit the presentation and which do not
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is <diagnosis>."
```

두 조건은 임상 note까지 byte-identical한 prefix를 공유하고 instruction suffix만 다르다.

**왜 Direct와 CoT를 둘 다 생성하는가.** Direct는 진단 행동 기준선이고 CoT는 RQ1의 설명
기준선이다. 두 arm이 같은 사례에서 어떤 답을 내는지 기록해야 CoT 설명 품질과 단순 진단 정확도를
혼동하지 않고, instruction이 P0 상태를 바꾸는지도 sensitivity로 확인할 수 있다.

---

## Slide 5B. Source generation 실행 설정

| 항목 | Direct | CoT |
|---|---:|---:|
| Backbone | Gemma-3-12B-IT | Gemma-3-12B-IT |
| Decoding | greedy, `do_sample=false` | greedy, `do_sample=false` |
| Temperature / top-p | 미사용 | 미사용 |
| Assistant prefill | `The answer is` | 없음 |
| Max new tokens | 64 | 2,048 |
| Batch size | 4 | 1 |
| Answer parser | 마지막 `The answer is <diagnosis>.` | 동일 |
| Forced second answer | 없음 | 이번 E1에서는 비활성화 |

Greedy decoding을 사용한 이유는 sampling variance를 설명 방법의 차이로 오인하지 않기 위해서다.
Direct와 CoT의 token budget이 다른 것은 두 arm의 목적이 다르기 때문이다. Direct는 진단명만
완성하고, CoT는 자연스러운 reasoning과 마지막 진단까지 생성한다.

**왜 다음에 P0/P1/P2가 필요한가.** “Activation을 읽었다”는 주장은 activation을 어느 token에서
뽑았는지에 따라 완전히 달라진다. 특히 CoT나 진단명이 이미 출력된 뒤의 activation을 읽으면,
내부 판단을 발견한 것이 아니라 방금 생성된 문자열을 재독해한 것일 수 있다.

---

## Slide 6. P0, P1, P2는 무엇인가

### 이 슬라이드가 필요한 이유

NLA 성능은 같은 layer라도 **언제 뽑은 activation인가**에 따라 다른 질문에 답한다. 생성 전
상태를 읽는 것과, CoT나 진단명이 이미 context에 적힌 뒤 그 문자열을 읽는 것은 같은
faithfulness 증거가 아니다. Slide 6은 좋은 결과가 output leakage에서 생겼다는 반론을 막고,
논문의 primary readout 위치를 사전에 고정하기 위해 필요하다.

```text
[clinical note + CoT instruction]  <- P0: prompt의 마지막 token
                 ↓ free generation
[model reasoning ... The answer is] <- P1: 마지막 answer marker의 마지막 subtoken
[diagnosis]                          <- P2: 실제 생성 진단명의 마지막 subtoken
```

| 위치 | 정확한 정의 | 해석 | 논문에서의 역할 |
|---|---|---|---|
| P0 | source prompt 마지막 token, 생성 전 | note를 읽고 답하기 직전의 통합 상태 | 주 Medical-NLA 입력 |
| P1 | assistant의 마지막 `The answer is` marker 직후, diagnosis 전 | reasoning을 마친 뒤 답을 쓰기 직전 상태 | leakage sensitivity |
| P2 | parsed diagnosis의 마지막 subtoken | 답 문자열이 이미 생성된 상태 | positive control |

P0는 모델이 note와 instruction을 모두 읽었지만 아직 한 글자도 답하지 않은 시점이다. 따라서
P0에서 복원된 정보는 출력 문자열의 직접적인 흔적만으로 설명할 수 없다. P1과 P2는 source CoT
response를 teacher-force하여 같은 trajectory에서 추출한다. P1에는 최종 diagnosis token이 아직
없지만 앞선 reasoning에 진단명이 이미 적혔을 수 있고, P2에는 진단명 자체가 있다.

실제로 171행 pilot에서 모델이 최종 선택한 diagnosis alias가 CoT reasoning에 이미 등장한
사례가 156/171(.9123)이었다. `diagnosis_alias_in_reasoning=false`인 P1 clean subset은 15행뿐이다.
따라서 P1/P2의 높은 진단 회수율을 곧바로 내부 추론 판독 능력으로 해석하지 않는다.

### 본 논문의 위치 규칙

- **Primary:** CoT instruction을 본 직후의 `P0/HS32`. RQ1의 CoT와 같은 source condition에서
  아직 CoT가 출력되기 전 내부 상태를 읽는다.
- **Instruction sensitivity:** Direct instruction의 P0. P0 자체가 CoT 전용 위치는 아니지만,
  instruction suffix가 다르면 activation도 달라지므로 primary와 섞지 않는다.
- **Leakage sensitivity:** P1과 diagnosis alias가 reasoning에 없던 P1 clean subset.
- **Positive control:** P2. 이미 출력된 진단 정보를 NLA가 읽을 수 있는지 확인한다.

**왜 다음에 HS32를 설명하는가.** P0/P1/P2가 시간축의 선택이라면 HS16/24/32는 network 깊이축의
선택이다. Token 위치를 고정한 뒤 어느 layer의 벡터를 AV에 넣는지 정해야 동일한 NLA checkpoint와
공정하게 비교할 수 있다.

---

## Slide 7. HS32는 무엇이며 왜 primary인가

```text
P0 = 어느 token 위치에서 activation을 뽑는가
HS32 = 어느 hidden-state layer/index에서 activation을 뽑는가

h(P0, HS32) in R^3840
```

| 축 | 후보 | 역할 |
|---|---|---|
| Token position | P0, P1, P2 | 생성 전, answer boundary, 생성 진단 뒤 상태 비교 |
| Hidden state | HS16, HS24, HS32 | layer sensitivity |
| Primary NLA input | P0/HS32 | 공개 AV/AR checkpoint와 학습 위치를 맞춤 |

`HS32`는 32차원이라는 뜻이 아니라 hidden-state index 32의 **3,840차원 벡터**다. Validation
probe에서는 HS24가 가장 좋았지만 공개 `nla-gemma3-12b-L32-av/ar`가 HS32용이므로 NLA와
round-trip의 primary는 HS32로 고정한다. HS16/24 NLA는 다른 layer activation을 HS32 decoder에
넣는 distribution-shift sensitivity이지 공정한 primary 비교가 아니다.

**왜 다음에 baseline을 비교하는가.** 이제 동일한 P0에서 likelihood, probe, vanilla NLA가 각각
무엇을 출력할 수 있는지 비교해야 Medical-NLA가 해결하려는 간극이 closed-label detection인지
open-text explanation인지 명확해진다.

---

## Slide 8. RQ에 앞서 비교하는 내부 측정 채널

| Method | 입력 | 출력 공간 | 할 수 있는 것 | 구조적 한계 |
|---|---|---|---|---|
| Forced-answer likelihood | source prompt와 고정 후보 문자열 | supplied diagnosis ontology | 후보 간 행동 선호 순위 | 열린 observation을 생성하지 못함 |
| Linear/multi-label probe | P0 activation | 고정 diagnosis, source answer, finding/value ontology | 읽으려는 정보의 decodability 감사 | 새 속성·관계·문장을 출력하지 못함 |
| Vanilla NLA | P0 activation | 자유 자연어 | open-text 판독 | 길고 잡음이 많고 통합 상태 복원 실패 가능 |
| Medical-NLA | P0 activation | 구조화 임상 자연어 | observation·관계·source answer 판독 목표 | 별도 grounding 검증이 필요 |

네 방법의 숫자는 모두 같은 종류의 accuracy가 아니다. Likelihood와 probe는 닫힌 후보 공간,
NLA는 열린 생성 공간을 사용한다. 따라서 Table 1은 probe 기반 representation audit만 담당하고,
open-text NLA는 Table 2와 Table 3에서 평가한다. Probe는 diagnosis마다 따로 만들지 않는다.
Diagnosis/category/source decision은 multiclass, finding presence는 multi-label, finding value는
native value를 대상으로 한 conditional decoder다.

**왜 다음에 학습을 설명하는가.** Baseline 분석은 P0에 정보가 있다는 것과 vanilla NLA의 한계를
보여줄 뿐이다. 그 간극을 메우려면 어떤 activation-text pair로 AV를 적응시키고, 어떤 데이터는
평가에만 남기는지 공개해야 한다.

---

## Slide 9A. Medical-NLA를 어떤 supervision으로 학습하는가

### 현재 실행 가능한 SFT-only v1

| 항목 | 현재 설정 |
|---|---|
| 데이터 | DiReCT confirmatory `train`과 `val_seen`만 사용 |
| 원 split | Train 266 / validation 52 |
| Primary 학습 분모 | Gold PDD 문자열이 note에 직접 노출된 18/2행 제외 후 248/50 |
| Activation 입력 | 같은 사례 source trajectory의 P0/HS32, 3,840차원 |
| `<observed>` target | Note에서 exact substring으로 확인된 physician observations, 최대 12개 |
| `<answer>` target | Physician gold가 아니라 backbone이 같은 source run에서 실제 생성한 진단 |
| 사용하지 않는 데이터 | Locked `test_seen=72`, `test_pdd_heldout=106`, DDXPlus heldout, MCR |
| 현재 loss | Target token next-token cross-entropy, LoRA SFT |
| Checkpoint 선택 | `val_seen`의 `<observed>` content-token loss |
| 반복 | 동일 ID와 recipe로 seeds 17, 29, 43 |

실제 target schema는 다음과 같다.

```text
<explanation>
<readout>
<observed>
- note에 문자 그대로 근거한 physician observation 1
- note에 문자 그대로 근거한 physician observation 2
</observed>
<answer>backbone source answer</answer>
</readout>
</explanation>
```

`<answer>`에 physician gold를 넣지 않는 이유는 source model이 틀린 사례의 activation을 정답
상태처럼 강제로 설명하지 않기 위해서다. 하지만 현재 v1은 source-wrong 행의 `<observed>`에도
physician gold observations를 사용한다. Observation이 note에 존재한다는 사실은 source model이
그 정보를 실제로 사용했다는 뜻이 아니므로, 이 부분에는 여전히 clinical target과 source-state
target의 충돌 가능성이 있다.

따라서 SFT-only v1은 최종 faithful reader가 아니라 다음을 확인하는 **ablation**이다.

1. 의료 형식과 physician observation supervision이 vanilla NLA의 설명 품질을 높이는가
2. Seen PDD 문구나 상투적인 임상 표현을 암기하는 분류기로 붕괴하는가
3. RQ1은 좋아지지만 RQ2의 shuffled/counterfactual grounding에는 실패하는가

Final recipe에서는 source-correct 행에만 physician clinical target을 적용하거나, source-wrong
행의 observation loss를 mask하고 source-answer/grounding loss만 적용하는 field-level ablation을
비교해야 한다. 현재 코드에는 이 field mask가 구현돼 있지 않으므로 완료된 방법처럼 말하지 않는다.

---

## Slide 9B. Vanilla, SFT-only, Reconstruction, Full은 무엇이 다른가

### Clinical supervision이 뜻하는 것

여기서 `Clinical supervision`은 단순히 의료 용어를 유창하게 말하게 하는 것보다 구체적이다.
Activation을 입력받았을 때 **어떤 환자 관찰과 진단을 어떤 임상 구조로 출력해야 하는지**를
DiReCT physician annotation으로 가르치는 것이다.

| Supervision target | 모델에게 가르치는 것 | 이것만으로 보장되지 않는 것 |
|---|---|---|
| `<observed>` physician observations | 환자별 임상 finding을 짧고 구조적으로 표현 | Source model이 그 finding을 실제 판단에 사용했는지 |
| `<answer>` source-model diagnosis | 현재 backbone이 내릴 준비가 된 진단을 명시 | 진단에 이른 내부 근거 전체가 충실하게 설명됐는지 |
| XML output schema | 일관된 형식, 짧은 출력, 자동 평가 가능성 | 설명 내용의 사례 특이성과 activation 의존성 |

따라서 clinical supervision이 통과하면 “의학적으로 적절한 내용을 구조화해 말할 수 있다”고
할 수 있지만, “바로 이 activation을 읽어서 말했다”고 할 수는 없다. 질환별 전형적인 문장이나
학습에서 자주 본 PDD 표현을 암기해도 SFT cross-entropy는 낮아질 수 있기 때문이다.

이 때문에 아래 표의 세 열은 서로 다른 질문이다.

```text
Clinical supervision: 무엇을 어떤 의료 언어로 말해야 하는가?
Reconstruction:       그 설명이 원 activation 정보를 보존하는가?
Pair specificity:     다른 비슷한 환자가 아니라 바로 이 activation에 해당하는가?
```

| Method | Clinical supervision | AV-AR reconstruction | Pair/counterfactual grounding | 실험상 역할 |
|---|---:|---:|---:|---|
| Vanilla NLA | No | 일반 도메인 pretrained | No | 의료 적응 전 공개 baseline |
| Medical-AV, SFT only | Yes | No | No | 의료 문장 supervision만의 효과와 classifier collapse 검사 |
| Medical-NLA, reconstruction | Yes | Yes | No | 자연어가 원 activation 정보를 보존하도록 강제하는 ablation |
| Medical-NLA, full | Yes | Yes | Yes | 사례 특이성과 evidence 변화 추종까지 강제하는 제안법 |

표의 `No`는 해당 능력이 절대로 없다는 뜻이 아니라, 현재 의료 학습 objective가 그 능력을
명시적으로 강제하지 않는다는 뜻이다. `Medical-NLA, reconstruction`과 `Medical-NLA, full`의
`Yes`도 현재 완료된 결과가 아니라 아래 objective를 실제로 구현했을 때의 제안 설계다.

### Reconstruction과 Full의 핵심 차이

Reconstruction 모델은 생성한 자연어 `z`만으로 원 activation `h`를 복원하도록 AV와 AR을
학습한다.

```text
h -> AV -> z -> AR -> h_hat
L_recon = ||h - h_hat||^2
```

이 objective는 설명이 activation 복원에 필요한 정보를 담도록 강제한다. 그러나 같은 질환의
여러 환자에게 비슷한 전형적 설명을 생성하거나, AR이 질환별 평균 activation을 복원해도 matched
reconstruction이 높을 수 있다. 따라서 reconstruction만 통과했다고 해서 그 설명이 바로 해당
환자의 activation에 고유하다고 말하지 않는다.

Full 모델은 같은 clinical supervision과 reconstruction 위에 다음 grounding objective를
추가한다.

1. 같은 activation-text pair가 같은 진단의 shuffled pair보다 높은 점수를 받아야 함
2. Finding을 삭제하면 해당 claim이 감소해야 함
3. DDXPlus native value를 바꾸면 해당 value claim이 같은 방향으로 바뀌어야 함
4. 편집하지 않은 finding은 유지되어야 함
5. Zero, mean activation과 activation swap이 실제 matched activation보다 낮아야 함

따라서 두 모델의 차이는 다음 한 문장으로 요약한다.

> Reconstruction은 자연어가 activation 정보를 보존하도록 학습하고, Full은 그 정보가 질환별
> 상투 문구가 아니라 해당 사례의 activation에 고유하며 evidence 변화에 반응하도록 추가로
> 학습한다.

Full Medical-NLA의 제안 학습은 두 데이터 역할을 결합한다.

| 학습 단계 | 데이터 | 제공하는 supervision | 목적 |
|---|---|---|---|
| Medical warm-start | DiReCT train/val | Physician observations와 source decision text | 임상적으로 읽을 수 있는 출력 형식과 내용 |
| Grounding development | DDXPlus train/validation | Evidence ID/value, matched activation, hard negative와 cue counterfactual | Case specificity와 evidence 변화 추종 |
| AV-AR reconstruction | DiReCT/DDXPlus의 train activation-text pair | `h -> text -> h_hat` | 자연어가 원 activation 정보를 보존하도록 강제 |
| Locked evaluation only | DiReCT 72/106, DDXPlus heldout, MCR | 학습에 사용하지 않음 | Seen/PDD-OOD, grounding, natural-text OOD 평가 |

원 NLA 구조는 `h -> AV -> text -> AR -> h_hat`이다. Reconstruction 모델은 clinical text를 잘
생성하는 것뿐 아니라 생성 text로부터 원 activation을 복원해야 한다. Full 모델은 여기에 matched
pair가 shuffled pair보다 높은 점수를 받고 evidence counterfactual을 따라가도록 하는 조건을
추가한다. Reconstruction은 FVE로, pair specificity는 matched-shuffled와 counterfactual 변화로
평가한다.

현재 구현된 코드는 **DiReCT SFT-only의 CE loss까지**다. DDXPlus paired training,
AR regression, reconstruction reward, pair-specific objective는 아직 구현·smoke 전이다. 그러므로
논문의 full row는 이 objective가 실제 구현되고 다음 조건을 통과한 뒤에만 유지한다.

1. Matched text가 shuffled text보다 activation을 더 잘 복원함
2. Mean/zero activation이 실제 matched activation보다 높은 reward를 받지 않음
3. Cue를 제거하거나 값을 바꾸면 해당 설명 claim만 따라 변함
4. SFT-only보다 RQ1 설명 점수와 RQ2 grounding을 모두 개선함

**왜 다음에 평가기를 설명하는가.** 학습 target의 physician observation과 source answer는 서로
다른 의미를 갖는다. 따라서 결과도 하나의 loss나 accuracy로 합치지 않고 DiReCT clinical
alignment와 DDXPlus activation grounding으로 다시 분리해 채점한다.

---

## Slide 10. 설명과 activation을 서로 다른 평가기로 검증한다

```text
DiReCT free-text output
   -> 공통 method-blind quote-constrained claim extractor
   -> official prediction JSON
   -> 제공된 native Llama-3-8B Yes/No semantic matcher
   -> Accdiag, Obs*, Exp*

DDXPlus paired activation
   -> own pair / same-diagnosis hard shuffle
   -> finding deletion / native value edit / activation swap
   -> pair gap, target change, untouched retention
```

### 검증 단위와 사전 판정 규칙

| 축 | 한 행의 단위 | 주 효과 | 함께 봐야 하는 보존 지표 | 통과 뒤 가능한 주장 |
|---|---|---|---|---|
| Clinical alignment | 동일 DiReCT case-method output | Physician observation/rationale/diagnosis match | Extraction coverage, unsupported prediction | Clinically aligned explanation |
| Pair specificity | 동일 DDXPlus case의 matched/shuffled pair | `score(own)-score(shuffled)` | Diagnosis와 finding-count가 같은 hard donor | Case-specific readout |
| Finding counterfactual | 같은 base case의 original/edited prompt pair | 편집한 finding claim의 방향성 있는 변화 | Untouched-finding retention | Counterfactual grounding |
| AV-AR reconstruction | 같은 activation의 matched/shuffled text | `FVE_matched-FVE_shuffled` | Mean/zero activation floor | Text preserves activation information |
| Text intervention | 같은 case의 original/no-op/edited activation | Target value/logit change | Identity preservation, off-target KL | Selective state intervention |
| Final policy | 같은 locked case의 keep/intervene outcome | Wrong-to-right minus right-to-wrong | Intervention rate, correct-case preservation | Net behavioral improvement |

Pair gap과 net correction은 전체 평균을 따로 빼지 않고 case-level paired difference로 계산한다.
환자 반복 note가 있는 DiReCT CI는 `patient_group`을 cluster 단위로 bootstrap한다. Test에서
threshold나 donor 난이도를 다시 고르지 않는다.

Codex는 자유 산문을 official schema로 바꾸는 **앞단 claim extractor**에만 사용할 수 있다.
Observation/rationale 의미 일치 판정을 Codex로 바꾸면 official DiReCT metric이 아니다. Primary
semantic matcher는 제공된 `Meta-Llama-3-8B-Instruct` native checkpoint, temperature 0,
top-p 1, exact `Yes`, official prompt와 greedy one-to-one matching을 유지한다.

DiReCT는 physician-reference clinical alignment를, DDXPlus는 paired activation dependence를
평가한다. 어느 한 데이터셋의 점수를 다른 축의 ground truth로 부르지 않는다.

**왜 다음에 개입 규칙을 미리 정하는가.** 설명이 좋아 보인 뒤 사후적으로 유리한 patch 사례만
고르면 RQ3가 성립한 것처럼 보일 수 있다. RQ2 grounding 결과를 보기 전에 identity 보존,
target selectivity, net correction의 순서와 중단 기준을 고정한다.

---

## Slide 11. RQ3 개입은 grounding 통과 후에만 평가한다

| 단계 | 조작 | 성공 조건 |
|---|---|---|
| Decode-encode identity | text를 고치지 않고 AV->text->AR | 원 답과 비목표 상태 보존 |
| Text patch | dataset-native evidence value 하나만 편집 | 목표 속성/logit만 선택적으로 변화 |
| Selective correction | validation에서 고정한 detector가 flag한 사례만 재검토 | net correction 양수, correct-case preservation 유지 |
| Oracle activation patch | 실제 paired activation을 주입 | 달성 가능한 인과 효과의 상한 |

DDXPlus primary activation은 DiReCT 학습과 동일한 **CoT-P0/HS32**로 고정했다. E5 builder는
primary `activation_rows_{validation,test}.jsonl`에 CoT-P0 prompt를 쓰고, Direct-P0는 validation
base case만 별도 파일로 내보내 instruction sensitivity로 분리한다. P1/P2는 grounding 주결과가
아니라 leakage/positive control이므로 필요 시 subset에서만 추가한다.

**Methodology에서 Data/Setup으로 넘어가는 논리.** 이제 source prompt, activation 위치,
hidden-state index, 학습 objective, 평가기와 중단 기준이 고정됐다. 다음 Data/Setup에서는
각 검증에 쓰는 모집단과 split을 연결한다. 그 뒤 Results에서 Table 1의 representation audit,
Table 2의 RQ1, Table 3의 RQ2, Table 4의 RQ3를 순서대로 판정한다.

---

# Part III. Data and Experimental Setup

Method에서 정의한 각 검증 질문을 실제 데이터와 분모에 연결한다. 이 절에서는 데이터가
무엇을 제공하는지, 어떤 행이 제외됐는지, train/validation/test가 어떻게 분리됐는지만
설명한다. 성능 수치는 다음 Results 절에서만 제시한다.

## Slide 12. 데이터셋별 역할

| 데이터셋 | 원래 제공하는 정보 | 본 연구의 역할 | 사용하지 않을 주장 |
|---|---|---|---|
| DiReCT | 임상 note, physician observation, rationale, diagnosis tree | Clinical warm-start와 RQ1 설명 품질, seen/PDD-heldout 평가 | activation ground truth |
| DDXPlus | pathology, evidence ID/value, differential | Grounding objective 개발, matched/shuffled, finding 반사실, patching | 자연 임상 산문의 최종 품질 |
| MedCaseReasoning | case-report 산문과 diagnosis/reasoning | 향후 frozen natural-text OOD | 정확한 gold evidence span |

Intro에서는 “임상 설명과 activation grounding을 서로 다른 적합한 데이터로 검증한다”는
한 문장만 말한다. 이 역할표와 데이터 구조는 Data and Experimental Setup에서 설명한다.

**왜 DiReCT를 먼저 자세히 보는가.** DiReCT는 RQ1 평가 자료일 뿐 아니라 현재 실행 중인
Medical-NLA의 clinical warm-start 자료다. 따라서 note의 입력 섹션, physician deduction,
PDD label이 각각 backbone 입력과 학습 target에서 무엇이 되는지 먼저 정의해야 한다.

---

## Slide 13A. DiReCT 한 사례에는 무엇이 들어 있는가

DiReCT restricted release는 **511개의 임상 note JSON**과 **24개의 diagnostic KG JSON**을
포함한다. 임상 note 한 행이 본 연구의 한 사례이며, 다음 여섯 섹션을 합쳐 backbone prompt의
환자 기록으로 사용한다.

| 원 필드 | 임상 섹션 | 실제로 담는 정보 | 진단에서의 역할 |
|---|---|---|---|
| `input1` | Chief Complaint, 주호소 | 환자가 병원에 온 가장 직접적인 이유 | 문제의 출발점과 가장 두드러진 증상 |
| `input2` | History of Present Illness, 현병력 | 증상의 시작, 기간, 변화, 유발·완화 요인, 동반 증상 | 현재 질환의 시간적 경과와 증상 조합 |
| `input3` | Past Medical History, 과거력 | 기존 질환, 과거 입원·수술 등 | 기저 위험과 감별진단의 사전확률 |
| `input4` | Family History, 가족력 | 가족의 질환과 유전적 위험 | 유전성·가족성 질환 가능성 |
| `input5` | Physical Exam, 신체검진 | 활력징후와 의료진이 관찰·측정한 징후 | 환자 진술과 구분되는 객관적 임상 소견 |
| `input6` | Pertinent Results, 주요 검사 결과 | 혈액검사, 영상, 심전도 등 관련 검사 | 진단을 지지하거나 배제하는 검사 증거 |

여섯 필드는 서로 다른 여섯 환자가 아니라 **같은 환자의 한 임상 기록을 구성하는 여섯 부분**이다.
모델은 특정 cue 하나만 보는 것이 아니라 주호소, 경과, 위험요인, 진찰, 검사를 함께 읽고 진단해야
한다. 일부 필드는 원자료에서 비어 있을 수 있으며, 빈 필드에는 내용을 보충하지 않고 그대로 둔다.

---

## Slide 13B. Disease category, PDD, physician deduction

### Disease category와 PDD는 같은 것이 아니다

```text
넓은 질환군: Disease category = Heart Failure
구체적 주 퇴원 진단: Primary Discharge Diagnosis (PDD) = HFrEF 또는 HFpEF
```

`PDD`는 환자 퇴원 기록 전체를 뜻하지 않고, 그 기록에 부여된 **구체적인 주 퇴원 진단 label**을
뜻한다. DiReCT에는 25개 disease category와 이를 세분화한 61개 canonical PDD가 있다. 따라서
category accuracy는 넓은 질환군을 맞혔는지, strict PDD accuracy는 세부 진단까지 맞혔는지를
서로 다르게 측정한다.

### 의사 주석이 제공하는 설명 구조

```text
환자 기록의 observation
        -> 그 관찰이 진단을 지지하는 rationale
        -> intermediate 또는 final diagnosis
```

예를 들어 실제 restricted 원문이 아닌 일반적인 예시는 다음과 같다.

```text
observation: 휴식 중에도 발생하는 흉통
rationale:   휴식 시 흉통은 안정형보다 급성 관상동맥 증후군 가능성을 더 지지함
diagnosis:   불안정 협심증
```

본 연구에서는 annotation tree를 `observation -> rationale -> diagnosis` deduction으로
정규화한다. 이 구조 덕분에 단순히 최종 진단을 맞혔는지만 보지 않고 다음을 분리해 평가한다.

1. 설명이 진단에 필요한 환자 관찰을 회수했는가
2. 기록이나 의사 주석에 없는 관찰을 불필요하게 추가했는가
3. 관찰과 진단 사이의 임상적 이유를 올바르게 연결했는가
4. 최종 diagnosis를 올바른 specificity로 제시했는가

---

## Slide 13C. 데이터 감사 수치는 무엇을 보장하는가

여기서 **감사(audit)**는 CoT나 Medical-NLA의 성능 평가가 아니다. 모델 실험 전에 원자료를
정상적으로 읽을 수 있는지, label 정의가 일관적인지, 같은 환자가 train과 test에 섞일 위험은
없는지, 의사 observation이 실제 note에 근거하는지를 확인한 데이터 품질 검사다.

| 감사 항목 | 값 | 이 값이 의미하는 것 | 이 값이 의미하지 않는 것 |
|---|---:|---|---|
| Raw notes / valid JSON | 511 / 511 | 제공된 511개 note가 모두 JSON으로 정상 파싱됨 | 511개 내용과 label이 모두 오류 없이 완벽함 |
| Disease categories | 25 | 넓은 상위 질환군이 25개임 | 서로 균등한 25-class 데이터임 |
| Canonical PDD labels | 61 | 공식 목록과 annotation root를 정규화한 세부 주 퇴원 진단 수 | 폴더 이름만 세어 얻은 최초 62개가 정본임 |
| Manifest patient-group keys | 469 | 환자 ID가 확인된 468개 그룹과 unparsed 4행의 공통 placeholder 1개 | 469명의 환자가 모두 확인됨 |
| Physician deductions | 5,109 | 전체 note에 존재하는 observation-rationale-diagnosis 연결 수 | 5,109명의 환자 또는 독립 표본이 존재함 |
| Exact-substring grounded observations | 4,965/5,109 (.9718) | 의사가 표시한 observation의 97.18%가 note 원문에서 문자 그대로 확인됨 | 모델이 observation을 97.18% 정확도로 복원함 |

### 각 감사가 필요한 이유

- **JSON 유효성:** parse 실패 사례가 조건별로 조용히 빠져 분모가 달라지는 문제를 막는다.
- **Category/PDD 정규화:** 복수형, 개행, 표기 차이와 폴더-label 충돌을 별도 진단으로 잘못 세는
  문제를 막는다. 정본은 공식 `data_list.csv`와 annotation root를 기준으로 한 61 PDD다.
- **Patient grouping:** 같은 환자의 반복 note가 train과 test에 동시에 들어가 환자 고유 표현을
  기억하는 일을 막는다. 이후 split은 note-random이 아니라 patient-disjoint로 만든다.
- **Deduction 수:** note 하나에 여러 임상 관찰과 추론 연결이 있음을 보여준다. 설명 평가의 단위는
  환자 수와 같지 않으며, case-level 집계와 deduction-level 집계를 구분해야 한다.
- **Exact-substring grounding:** physician observation이 원문에서 실제로 추적 가능한지 확인한다.
  남은 144개는 곧바로 잘못된 주석으로 처리하지 않고 약어, 문장 변형, 정규화 차이를 별도 감사한다.

### Raw 511행과 실제 실험 496행의 차이

| 제외 사유 | 행 수 | 이유 |
|---|---:|---|
| Canonical PDD 의미 충돌 | 10 | 폴더 PDD와 annotation root가 임상적으로 다른 label을 가리킴 |
| Patient ID parse 실패 | 4 | 환자 단위 분리를 보장할 수 없음 |
| Exact duplicate copy | 1 | 동일 사례를 두 번 세는 것을 방지 |
| **최종 eligible population** | **496** | 이후 patient-disjoint split의 고정 모집단 |

따라서 `511`은 배포본 감사의 분모이고, `496`은 primary split과 후속 실험의 모집단이다.
두 숫자를 같은 표의 분모로 섞지 않는다. 원 audit의 469 grouping keys는 환자 ID가 확인된
468개 그룹과 unparsed 4행을 묶은 placeholder 하나이며, 제외 후 split의 458 patient groups와
같은 수가 아니다.

마지막으로 physician annotation은 **임상적으로 바람직한 설명의 reference**이지 source model
activation의 정답이 아니다. 모델이 오답을 선택한 사례에서는 의사 gold diagnosis와 모델 내부의
현재 결론이 다를 수 있다. 그러므로 다음 두 질문을 하나의 점수로 합치지 않는다.

```text
Clinical alignment: 설명이 의사 주석과 임상적으로 일치하는가?
Source-decision fidelity: 설명이 실제 source model의 현재 판단을 충실하게 읽는가?
```

**왜 다음에 split을 고정하는가.** 같은 환자의 반복 note나 같은 PDD가 학습과 시험에 함께
들어가면 Medical-NLA가 activation을 읽은 것이 아니라 환자 표현이나 label 문구를 외운 결과가
될 수 있다. 데이터 구조를 확인한 뒤 patient-disjoint와 PDD-heldout split을 먼저 동결한다.

---

## Slide 14. DiReCT 모집단과 split

Canonical PDD 의미 충돌 10행, patient ID parse 실패 4행, duplicate copy 1행을 제외해
496행을 사용한다. 같은 환자가 여러 PDD에 걸치는 경우 연결된 PDD들을 component로 묶어
환자와 label leakage를 동시에 막았다.

| Split | Notes | Patient groups | PDDs | 역할 |
|---|---:|---:|---:|---|
| Train | 266 | 244 | 49 | probe 및 Medical-NLA 학습 |
| Validation seen | 52 | 47 | 24 | layer, prompt, epoch 선택 |
| Test seen PDD | 72 | 64 | 25 | 학습에서 본 PDD의 새 환자 |
| Test PDD held-out | 106 | 103 | 12 | 학습에 없던 PDD의 새 환자 |

이 72/106행은 Medical-NLA 선택 이후 평가를 고정한 `locked downstream evaluation`이다.
과거 backbone output이 일부 또는 전부 materialize되어 완전히 untouched dataset test라고
부르지는 않는다. 하지만 이 시점 이후 test readout을 보고 prompt, layer, epoch, threshold를
바꾸지 않는다.

**왜 다음에 Results로 넘어가는가.** 데이터 역할, raw/eligible 분모, patient-disjoint split과 locked-test 분모가 모두 고정됐다. 이제 Results에서는 각 수치가 exploratory 171, validation 52/50, locked 72/106 중 어디에서 나온 것인지 표 제목에서 분리한다.

---


# Part IV. Experimental Results

결과는 먼저 capability baseline을 확인한 뒤 RQ1, RQ2, RQ3 순서로 읽는다. 현재 값이 있는
exploratory/validation 결과와 아직 비어 있는 locked-test 주표를 같은 종류의 증거처럼 섞지
않는다.

## Preliminary baseline. 생성 전 내부에는 무엇이 있으며 기존 채널은 무엇을 읽는가

**왜 세 RQ 전에 이 분석이 필요한가.** P0 activation에 임상·진단 정보가 없거나 vanilla NLA가
이미 통합 설명을 충분히 복원한다면 새로운 Medical-NLA를 만들 이유가 약하다. 먼저 backbone
행동, closed-label decodability, open readout의 성공과 실패 범위를 같은 사례에서 확인한다.
이 분석은 Medical-NLA의 필요성과 비교 기준을 세우지만, 그 자체가 RQ1의 답은 아니다.

## Slide 15. E1 backbone behavior: 현재 나온 exploratory 결과

이 표는 현재 locked 72/106 split 이전에 실행한 exploratory pilot의 전체 pool별 결과다.

### A. Overall exploratory pool, n=171

| Generation | Parse | Strict PDD | Disease category | Diagnosis token F1 |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | 1.0000 | .2105 | .5029 | .1593 |
| Source CoT | 1.0000 | .1930 | .5088 | .1850 |

### B. Pilot seen-PDD pool, n=71

| Generation | Parse | Strict PDD | Disease category | Diagnosis token F1 |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | 1.0000 | .2535 | .3944 | .2921 |
| Source CoT | 1.0000 | .2254 | .3803 | .2530 |

### C. Pilot PDD-heldout pool, n=100

| Generation | Parse | Strict PDD | Disease category | Diagnosis token F1 |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | 1.0000 | .1800 | .5800 | .0650 |
| Source CoT | 1.0000 | .1700 | .6000 | .1367 |

### D. Paired Direct-versus-CoT breakdown, n=171

| Target | Both correct | Direct only | CoT only | Neither | CoT-Direct | McNemar exact p |
|---|---:|---:|---:|---:|---:|---:|
| Strict PDD | 26 | 10 | 7 | 128 | -.0175 | .6291 |
| Disease category | 77 | 9 | 10 | 75 | +.0058 | 1.0000 |

### E. E1 source/activation artifact completeness

| Artifact universe | Source answers | Position rows | Stored tensors | Positions | HS indices | Max prompt tokens |
|---|---:|---:|---:|---|---|---:|
| Train + validation | 325 | 975 | 2,925 | P0/P1/P2 | 16/24/32 | 4,834 |
| Exploratory test | 171 | 513 | 1,539 | P0/P1/P2 | 16/24/32 | 4,304 |
| **Merged eligible population** | **496** | **1,488** | **4,464** | **P0/P1/P2** | **16/24/32** | **4,834** |

재색인된 confirmatory split에서도 266/52/72/106 cases가 2,394/468/648/954 tensors와 정확히
대응하며 전체 case x position x layer grid의 누락과 중복은 0이다.

### 왜 CoT strict PDD가 Direct보다 낮았는가

현재 결과만으로 CoT가 진단을 악화시켰다고 말할 수 없다.

1. 차이는 3건, 1.75%p이며 paired test에서 유의하지 않다.
2. Disease category에서는 오히려 CoT가 `.5088`로 Direct `.5029`보다 1건 높다.
3. Strict PDD는 매우 세분화된 label exact/alias 판정이다. CoT가 같은 disease family 안에서
   다른 subtype이나 더 구체적·서술적인 진단명을 출력하면 category는 맞고 strict PDD는
   틀릴 수 있다.
4. Direct는 answer cue에서 즉시 진단명을 완성하지만 CoT는 2,048-token reasoning을 거친다.
   이 과정이 어떤 사례는 구하고 7건, 다른 사례는 바꾸어 10건을 잃었다. 평균만 보면 이
   사례 교체가 가려진다.
5. CoT의 diagnosis token F1은 오히려 더 높아, 단순히 gold label에서 더 멀어진 것으로도
   해석할 수 없다.

따라서 발표 문장은 다음으로 제한한다.

> 이 exploratory cohort에서는 Direct와 CoT의 진단 성능 차이가 확인되지 않았다. CoT는
> 정답 사례의 구성을 바꾸었지만 strict PDD와 category 모두 통계적 우열이 없었다. CoT의
> 설명 품질과 faithfulness는 이 정확도 표가 아니라 Table 2와 grounding 실험에서 평가한다.

---

## Slide 16. 생성 전 P0에 닫힌 진단 정보가 있는가

아래는 locked test가 아니라 `val_seen=52`에서 layer와 baseline을 선택하기 위한 결과다.
모든 행의 분모가 52이므로 `n`은 열에서 제거한다.

### A. Early forced-answer candidate ranking 전체 결과

| Variant | Target | Candidates | Top-1 | Top-5 | MRR | Mean gold rank |
|---|---|---:|---:|---:|---:|---:|
| Raw | Disease category | 25 | .4808 | .6731 | .5814 | 5.02 |
| Content-free calibrated | Disease category | 25 | .2308 | .3077 | .3091 | 9.58 |
| Raw, train-matched ontology | Canonical PDD | 49 | .1538 | .5192 | .3250 | 7.92 |
| Content-free calibrated | Canonical PDD | 49 | .0577 | .1346 | .1486 | 15.83 |

### B. Linear probe layer sensitivity 전체 결과

| Target | HS | Classes | Majority | Top-1 | Top-5 | MRR | Macro recall | Val NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Disease category | 16 | 25 | .0577 | .5000 | .7885 | .6374 | .4833 | 1.9679 |
| Disease category | **24** | 25 | .0577 | **.5962** | **.9038** | **.7284** | **.5000** | **1.3961** |
| Disease category | 32 | 25 | .0577 | .5192 | .8654 | .6609 | .4426 | 1.6869 |
| Canonical PDD | 16 | 49 | .0962 | .3846 | .6923 | .5294 | .3597 | 2.5533 |
| Canonical PDD | **24** | 49 | .0962 | **.4423** | **.7692** | **.5762** | **.3868** | **2.0489** |
| Canonical PDD | 32 | 49 | .0962 | .3846 | .6923 | .5335 | .2771 | 2.3784 |

세 hidden-state index를 모두 비교했고 HS24가 두 closed-label target에서 가장 높았다. Table 1B에는
validation 선택 절차를 반복하지 않고 고정된 probe의 locked-test 값만 넣는다. Medical-NLA는
공개 AV/AR checkpoint 호환 때문에 HS32를 primary로 유지하므로, HS24 probe의 우세를
Medical-NLA layer 선택으로 전용하지 않는다.

Forced-answer likelihood는 P0 prompt 뒤에 `The answer is`를 붙이고 후보 문자열을
teacher-force하여 평균 token log probability로 순위를 매긴 행동 기준선이다. 저장된 P0
벡터를 직접 unembed한 값이 아니다. PDD raw ranking은 한 희귀 후보를 35/52행에서 top-1으로
선택해 label surface prior에 취약했다. Content-free prior subtraction은 category top-1
`.2308`, PDD `.0577`로 더 악화되어 appendix sensitivity로만 둔다.

Probe의 결과는 P0에 진단 정보가 없지 않다는 증거다. 그러나 probe는 train에서 정의한 49개
PDD 또는 25개 category 중 하나를 고르는 분류기이므로 열린 observation 설명 기준선이 아니다.

---

## Slide 17. P0 representation audit: 확인된 것과 남은 것

Table 1은 probe 기반 decodability만 보고한다. 서로 다른 출력 공간인 NLA 생성 점수를 같은
표에 섞지 않는다.

| Target | Decoder | Validation | Locked test | 상태 |
|---|---|---:|---:|---|
| Gold disease category | 25-way linear probe | HS24 Top-1 .5962 | TBD | layer 선택 완료 |
| Gold canonical PDD | 49-way linear probe | HS24 Top-1 .4423 | TBD | layer 선택 완료 |
| Source decision | multiclass probe | label-space audit 중 | TBD | 자유 생성 답의 ontology 정규화율을 먼저 확인 |
| Finding presence | multi-label probe | DDXPlus validation 예정 | DDXPlus test TBD | evidence ID 고정, 구현·실행 필요 |
| Finding value | conditional probe | DDXPlus validation 예정 | DDXPlus test TBD | native value ID 고정, 구현·실행 필요 |

Diagnosis마다 별도 probe를 만들지 않는다. Gold diagnosis와 source decision도 합치지 않는다.
앞의 둘은 physician label이 P0에서 decode되는지, source-decision probe는 모델이 실제로 낼
답이 P0에서 decode되는지를 묻는다. 다만 실제 source answer는 자유 문자열이므로 normalized
answer를 그대로 class로 쓰지 않는다. Train에서 validation label을 얼마나 덮는지와 49 PDD/25
category ontology로 유일하게 정규화되는 비율을 aggregate audit으로 먼저 확인한다. Coverage가
낮거나 ambiguity가 크면 source decision은 closed probe에서 제외하고 open-text fidelity로만
평가한다. Finding/value probe는 fixed evidence/value ID가 있는 DDXPlus CoT-P0에서 학습·평가한다.

---

## Slide 18. Vanilla AV의 CoT-P0 validation 결과

Validation 52 cases x 2 prompts x 3 layers의 312 readout을 모두 quote-constrained semantic
judge로 판정했다.

| Prompt | HS | Source answer | Gold PDD | Category |
|---|---:|---:|---:|---:|
| Default | 16 | 0/52 | 0/52 | 1/52 |
| Default | 24 | 0/52 | 0/52 | 0/52 |
| **Default** | **32** | **0/52** | **0/52** | **0/52** |
| Task-aligned | 16 | 0/52 | 0/52 | 1/52 |
| Task-aligned | 24 | 0/52 | 0/52 | 0/52 |
| Task-aligned | 32 | 0/52 | 0/52 | 0/52 |

Judge에는 환자 note를 주지 않고 source answer, gold PDD, category를 순서를 섞어 제시했다.
`match=true`에는 실제 readout 안의 quote를 요구했다. 이 결과는 진단 target의 명시적 복원
실패를 보이지만 physician observation 품질이나 activation faithfulness를 판정하지 않는다.

### Exploratory position sensitivity, n=171

| Position | Source-answer mention | Gold-PDD mention | Category mention | Hard-shuffle rows | Own source | Shuffled source | Pair gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0, pre-generation | .0000 | .0000 | .0000 | 164 | .0000 | .0000 | .0000 |
| P1, after CoT | .4912 | .1404 | .5848 | 164 | .4939 | .0793 | .4146 |
| P2, after diagnosis | .3918 | .0819 | .4854 | 164 | .4024 | .0427 | .3598 |

P1에서 source answer alias가 reasoning에 없었던 leakage-free subset은 15행뿐이며 source-answer
mention은 1/15=.0667이었다. 따라서 P1/P2의 높은 전체 mention과 pair gap은 생성된 CoT 또는
diagnosis 문자열의 재독해가 섞인 positive/leakage control로 해석하고 P0 결과와 합치지 않는다.

---

## Slide 19. SFT-only validation 결과: 의료 형식 학습만으로 충분하지 않다

동일한 50-case validation에서 공통 quote extractor와 official DiReCT evaluator를 적용했다.
이 값은 method selection 결과이며 locked-test Table 2가 아니다.

| Method | Obs. rows | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| CoT | 50/50 | 0 | **.3009** | **.3903** | **.2349** | **.0573** | **.0144** |
| Vanilla AV | 0/50 | 0 | 0 | 0 | 0 | 0 | 0 |
| Medical-AV SFT, seed 17 | 50/50 | 0 | .0771 | .0435 | .0343 | 0 | 0 |
| Medical-AV SFT, seed 29 | 50/50 | 0 | .0133 | .0047 | .0047 | 0 | 0 |
| Medical-AV SFT, seed 43 | 50/50 | 0 | .0200 | .0029 | .0032 | 0 | 0 |

SFT-only는 출력을 의료 observation 형식으로 바꾸었지만 CoT보다 임상 alignment가 낮고 seed
간 편차가 크다. 현재 target에 rationale가 없으므로 Expcom/Expall 0은 구조상 예상되지만,
observation 지표도 충분하지 않다. 따라서 이 결과는 Full Medical-NLA의 성공이 아니라
**clinical CE만으로는 부족하며 reconstruction과 pair specificity가 필요하다는 실패 분석**이다.

---

## Slide 20. 최종 Table 1: locked-test에서 채울 표

### Panel A1. Backbone behavior, test seen PDD, n=72

| Generation | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | TBD | TBD | TBD | TBD |
| Source CoT | TBD | TBD | TBD | TBD |

### Panel A2. Backbone behavior, test PDD held-out, n=106

| Generation | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | TBD | TBD | TBD | TBD |
| Source CoT | TBD | TBD | TBD | TBD |

### Panel B1. DiReCT CoT-P0 diagnosis decodability on locked test

| Target | Decoder | Output space | Test seen | Test OOD | Control |
|---|---|---|---:|---:|---|
| Gold category | Linear probe | 25-way | TBD | N/A | label shuffle |
| Gold PDD | Linear probe | 49-way train labels | TBD | N/A | label shuffle |
| Source decision | Linear probe | frozen source labels | TBD | TBD | answer shuffle |

### Panel B2. DDXPlus CoT-P0 finding decodability

| Target | Decoder | Output space | Validation | Locked test | Control |
|---|---|---|---:|---:|---|
| Finding presence | Multi-label probe | frozen evidence IDs | TBD | TBD | label/hard shuffle |
| Finding value | Conditional probe | native values per evidence ID | TBD | TBD | value shuffle |

`Layer`는 validation에서 고정하므로 주표 열에서 제외한다. Figure 2에는 HS16/24/32 validation
sensitivity를 모두 보여주고, caption에는 target별 선택 mapping을 적는다. 현재 category와
canonical PDD는 HS24이며 source decision/finding/value는 아직 TBD다. `N/A`는 실패가 아니라 closed probe에 unseen output node가 없어
과제가 정의되지 않았다는 뜻이다.

**RQ1로 넘어가는 이유.** Table 1은 P0에서 어떤 의료 정보가 decode 가능한지 확인하는
선행 감사다. 이제 open-text Medical-NLA가 그 정보를 의사 annotation에 맞는 임상 설명으로
표현하는지를 Table 2에서 평가한다.

---

## RQ1. Medical-NLA가 CoT·vanilla NLA보다 임상 설명을 잘 복원하는가

RQ1은 설명의 **임상적 내용**만 평가한다. 같은 DiReCT 사례에서 CoT, vanilla NLA,
Medical-NLA가 의사가 표시한 observation, rationale, diagnosis 구조를 얼마나 복원하는지
Table 2로 비교한다. 여기서 Medical-NLA가 가장 높더라도 아직 activation을 읽었다는 뜻은
아니다. RQ1만 통과하면 좋은 의료 설명 생성기라고 부를 수 있다.

이 표의 reconstruction과 full은 clinical supervision의 양을 달리한 이름이 아니다.
Reconstruction은 AV-AR 정보 보존 objective까지 추가한 모델이고, full은 여기에 pair specificity와
evidence counterfactual grounding까지 추가한 모델이다. RQ1에서는 grounding objective가 임상
설명 품질을 손상시키지 않거나 개선하는지를 함께 확인한다. 두 방법의 결정적 차이는 다음 RQ2에서
평가한다.

## Slide 21. RQ1: DiReCT clinical explanation quality

### Panel A. Test seen PDD, n=72

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

### Panel B. Test PDD held-out, n=106

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- `Accdiag`: 생성 진단과 physician diagnosis의 일치
- `Obspre`: 생성한 observation 중 physician observation과 맞는 정도
- `Obsrec`: physician observation을 얼마나 회수했는가
- `Obscomp`: observation 집합의 completeness
- `Expcom`: 대응된 observation에서 rationale와 diagnosis edge가 맞는 정도
- `Expall`: 누락·추가·관계·진단 오류를 포함한 end-to-end explanation alignment

모든 method의 free text를 동일한 claim extractor로 공식 schema에 변환한다. Extractor에는
method 이름, 원 note, gold annotation을 주지 않는다. Parse/extraction 실패를 삭제하지 않고
0점 처리하며 coverage를 함께 보고한다. 이 표가 개선되면 clinical alignment를 주장할 수
있지만 activation faithfulness는 아직 주장하지 않는다. 다음 슬라이드에서 claim extraction과
official semantic matching이 서로 다른 단계임을 구분한다.

현재 SFT-only v1 target은 `<observed>`와 `<answer>`만 포함하고 rationale 생성은 학습하지
않는다. 따라서 이 버전에서는 observation 계열과 Accdiag가 주 평가이고, `Expcom/Expall`은
탐색적으로만 보고한다. Rationale를 포함한 full objective가 실제 구현되기 전부터
`Expcom/Expall` 개선을 주가설로 쓰지 않는다.

---

## Slide 22A. Official evaluator 전체 흐름

### 전체 평가 흐름

```text
CoT / Vanilla NLA / Medical-NLA free text
    ↓ 우리 연구의 공통 quote-constrained claim extractor
Official prediction JSON
{observation: [rationale, note_section, diagnosis], ..., "chain": [...]}
    ↓ DiReCT 제공 Llama-3-8B semantic matcher
observation match와 rationale match의 Yes/No 기록
    ↓ official statistics-compatible aggregation
Accdiag, Obspre, Obsrec, Obscomp, Expcom, Expall
```

이 파이프라인에서 **claim extraction**과 **official semantic matching**을 같은 LLM-as-a-judge
단계로 부르면 안 된다.

| 구성요소 | 역할 | 현재 구현 | Official DiReCT metric의 일부인가 |
|---|---|---|---|
| Claim extractor | 자유 산문을 official structured prediction으로 변환 | Codex 기본, local Llama 대체 가능 | 아니며 본 연구의 adaptation |
| Observation matcher | gold/predicted observation 의미 대응 | 제공된 native Meta-Llama-3-8B-Instruct | 예 |
| Rationale matcher | 대응 observation의 rationale 의미 대응 | 동일 official Llama-3-8B | 예 |
| Metric aggregator | official 산식으로 집계 | `statistics.py` 호환 local wrapper | 예 |
| E2 semantic diagnostic audit | readout에 target diagnosis가 명시됐는지 검사 | 같은 local Llama checkpoint, 별도 custom prompt | 아니며 Table 1 보조 진단 |

---

## Slide 22B. 공통 quote-constrained claim extraction

CoT와 NLA는 자유 산문이지만 DiReCT evaluator는 구조화 prediction JSON을 요구한다. 따라서
모든 방법에 같은 extractor를 적용한다. 현재 wrapper의 기본 backend는 Codex이며, 필요하면
같은 prompt를 local Llama-3-8B로 실행할 수 있다.

현재 validation 실행 단위는 gold-label phrase가 note에 직접 노출된 2행을 제외한 50개
사례다. 비교 method는 CoT, vanilla NLA, Medical-NLA seed 17/29/43의 다섯 개이며 총
250개의 method-blind extraction request를 만든다. Request 순서는 seed 17로 섞는다.

Extractor가 받는 정보:

- 해당 method가 생성한 output text
- 사전 고정한 canonical PDD candidate label 목록

Extractor가 받지 않는 정보:

- 원 환자 note
- physician gold observations와 rationale
- case ID와 split 이름
- output을 만든 method 이름

Extractor는 최대 12개의 patient-specific observation을 뽑는다. Observation, rationale,
diagnosis에는 method output에 실제로 연속해서 존재하는 exact quote가 필요하다. 외부 의학
지식으로 관찰이나 이유를 보충하면 안 되며, diagnosis는 candidate label과 같은 specificity로
명시된 경우에만 선택한다. 후처리는 quote가 실제 output에 없으면 claim을 거절하고, JSON parse
실패와 빈 extraction을 숨기지 않고 extraction coverage와 failure로 기록한다.

Codex는 이 **구조 변환 단계**에 사용할 수 있다. 그러나 Codex가 observation 또는 rationale의
gold-prediction 의미 일치를 판정하도록 바꾸면 더 이상 official DiReCT semantic matcher가
아니다. 논문에서는 이를 다음처럼 표기한다.

> DiReCT official semantic scores computed after a common, method-blind,
> quote-constrained extraction step.

즉 산식과 semantic matcher는 official이지만, 자유 산문을 공식 schema로 옮기는 앞단은 본
연구가 추가한 adaptation임을 숨기지 않는다. Extractor backend, model version, prompt hash,
parse coverage를 결과와 함께 고정한다.

현재 shell wrapper는 extractor backend를 Codex로 기본 설정하지만 model 이름을 환경 변수로
받아 비워둘 수도 있다. 이는 full evaluation 전에 닫아야 할 재현성 항목이다. Confirmatory
실행에서는 Codex model/version을 명시적으로 고정하고, 동일 request를 local Llama extractor로
처리한 결과를 sensitivity로 남긴다. Model을 기록하지 않은 결과는 최종 표에 넣지 않는다.

현재 schema converter는 명시적으로 추출된 diagnosis label을 canonical PDD ontology의 대표
chain에 매핑한다. 따라서 chain category는 모델이 자유롭게 생성한 별도 category 예측이 아니라
진단 label에서 결정론적으로 정규화된 값이다. 이를 독립적인 reasoning 성능처럼 해석하지 않고,
ontology mapping 규칙과 candidate 수를 공개한다. Observation과 rationale는 여전히 method
output의 exact quote만 허용한다.

---

## Slide 22C. 제공된 Llama-3-8B official semantic matching

| 설정 | 값 |
|---|---|
| Judge checkpoint | `Meta-Llama-3-8B-Instruct` native original weights |
| Runtime | DiReCT 제공 native Llama code + FairScale, `torchrun --nproc_per_node 1` |
| Max sequence length | 8,192 |
| Max batch size | 4 |
| Temperature | 0 |
| Top-p | 1.0 |
| Observation prompt | official `discriminate_similarity_observation()` |
| Rationale prompt | official `discriminate_similarity_reason()` |
| Positive decision | response가 정확히 `"Yes"`인 경우만 인정 |
| Matching | gold observation 순서의 greedy one-to-one matching |

Wrapper는 원본 evaluator의 GPU 번호 하드코딩만 제거했다. Greedy matching, exact `Yes`, 공식
prompt와 native checkpoint는 유지하고 raw judge response와 exception을 restricted audit에
남긴다. `strip/casefold Yes`나 maximum bipartite matching은 primary가 아니라 sensitivity다.

### Official statistics-compatible aggregation

```text
O     = physician gold observation set
O_hat = extracted predicted observation set
M     = official matcher가 만든 one-to-one observation pairs
m     = rationale가 Yes이고 연결 diagnosis도 맞는 pairs
```

| Metric | Official 계산과 해석 |
|---|---|
| Accdiag | predicted chain의 마지막 diagnosis와 gold final diagnosis 비교 |
| Obspre | `\|M\| / (\|O_hat\| + 1)` |
| Obsrec | `\|M\| / (\|O\| + 1)` |
| Obscomp | `\|M\| / \|O union O_hat\|` |
| Expcom | `m / \|M\|` |
| Expall | `m / \|O union O_hat\|` |

`Obspre`와 `Obsrec`은 공식 코드가 denominator에 `+1`을 넣으므로 oracle도 1.0이 되지 않는다.
평가 JSON이 누락되거나 invalid하면 공식 통계 동작과 동일하게 모든 metric을 0으로 처리한다.
Unsmoothed observation precision/recall은 별도 sensitivity이며 official score라고 부르지 않는다.

---

## Slide 22D. Official evaluator 재현 smoke와 민감도

Gold annotation에서 만든 oracle prediction 10건을 전체 evaluator에 넣었다.

| Metric | Oracle-10 mean | 해석 |
|---|---:|---|
| Acccat | 1.0000 | category chain 일치 |
| Accdiag | 1.0000 | final diagnosis 일치 |
| Obspre | .8104 | 공식 `+1` denominator |
| Obsrec | .8104 | 공식 `+1` denominator |
| Obscomp | 1.0000 | observation set 완전 일치 |
| Expcom | 1.0000 | matched observation의 relation 일치 |
| Expall | 1.0000 | end-to-end chain 일치 |

10/10 evaluation JSON이 생성됐고 missing/invalid는 0이었다. 이 smoke는 evaluator wrapper가
공식 동작을 재현한다는 검사이지, CoT나 NLA의 성능 결과가 아니다.

### 알려진 민감도

- Observation은 gold 순서대로 첫 `Yes` prediction을 선택하므로 dictionary 순서에 민감할 수 있다.
- 응답이 `Yes.` 또는 ` yes `이면 primary official mode에서는 불일치다.
- Rationale가 의미상 맞아도 연결 diagnosis가 exact rule을 통과하지 못하면 Expcom/Expall에
  포함되지 않는다.
- Claim extractor가 explicit quote만 허용하므로 implicit하지만 타당한 설명은 누락될 수 있다.
- 따라서 prediction-order permutation, normalized-Yes, alternative matching, extractor backend,
  일부 human/clinician audit은 보조 민감도로 보고하고 primary official score를 사후 교체하지 않는다.

**RQ1의 현재 답.** 공식 evaluator 재현과 공통 extraction protocol, vanilla/pilot baseline은
준비됐지만, 동일한 locked test에서 CoT·vanilla NLA·Medical-NLA 3 seeds를 비교한 Table 2는
아직 완성되지 않았다. 따라서 현재 Medical-NLA가 임상 설명을 더 잘 복원한다고 결론 내리지
않는다. Table 2에서 Medical-NLA가 observation과 explanation 지표를 개선하면 RQ1을 통과하며,
이때의 자격은 **임상 설명 생성기**까지다.

**RQ2로 넘어가는 이유.** 전문가 설명과 잘 맞는 출력도 언어 모델이 의학 지식으로 그럴듯하게
만든 문장일 수 있다. 다음에는 설명과 activation의 짝을 깨고 evidence를 바꾸어, 그 설명이
해당 사례 내부 상태에 실제로 의존하는지를 검사한다.

---

## RQ2. 설명이 해당 사례 activation에 의존하는가

RQ2는 설명의 **activation faithfulness**를 평가한다. RQ1의 physician-reference 점수를 반복하지
않고, matched-vs-shuffled, finding deletion/value edit, activation swap, AV-AR round-trip으로 같은 설명이
자기 사례 activation에만 결합되는지를 시험한다. RQ2까지 통과해야 Medical-NLA를 내부 판독기라고
부를 수 있다.

## Slide 23A. RQ2 데이터와 DDXPlus native value

RQ2의 primary controlled testbed는 DDXPlus다. 최신 primary 설정에서는 Medical-NLA를
DiReCT에만 적응하고 DDXPlus를 cross-corpus grounding test로 사용한다. DDXPlus
`validate.csv`는 scoring threshold, hard-shuffle 규칙, counterfactual 생성 규칙과 mean
activation control을 고정하는 데 사용하고, `test.csv`는 locked Table 3에만 사용한다.
`train.csv`는 primary에서 사용하지 않으며, 향후 DDXPlus grounding-adaptation ablation에서만
별도 사용한다. MCR은 정확한 finding/value annotation이 없으므로 Table 3 전체가 아니라
natural-text OOD 보조 평가만 담당한다.

DDXPlus의 `value`는 본 연구가 임의로 만든 label이 아니다. 환자 CSV의 `EVIDENCES`에는 다음
두 형태가 있고, `release_evidences.json`이 evidence 질문과 가능한 value ID의 의미를 정의한다.

```text
E_DYSPNEA             값 없는 bare evidence ID: 해당 이진 finding이 기록됨
E_TRAVEL_@_N          evidence ID와 native value ID가 함께 기록됨

release_evidences.json 예시
E_TRAVEL.value_meaning = {N: "no", Y: "yes"}
```

Value edit은 현재 case에 실제 value ID가 있고, **같은 evidence ID의 사전에 다른 value가 명시돼
있으며**, 두 값을 모두 정상적인 cue 문장으로 렌더링할 수 있을 때만 만든다.

```text
같은 base case, 같은 evidence ID
E_TRAVEL: N -> Y
나머지 findings는 모두 그대로 유지
```

값 없이 bare ID로만 등장하는 binary evidence는 absence를 원자료가 기록하지 않는다. 따라서
`E_DYSPNEA -> no dyspnea`처럼 음성 값을 발명하지 않고 해당 cue를 삭제하는 deletion만 적용한다.
결과적으로 value accuracy와 value-edit response의 분모는 모든 test case가 아니라 **native
value가 실제로 선언되고 대안 값이 존재하는 적격 subset**이다. Deletion과 value edit은 별도
분모로 보고하고, 하나의 평균으로 합친 값은 보조 분석으로만 둔다.

Population은 official validation과 test 모두에 적격 사례가 존재하는 diagnosis 교집합으로
고정한다. 각 split에서 diagnosis별 최대 100건을 seed 17로 독립 sampling하고, clean rendered
findings가 3개 이상이며 prompt에 gold diagnosis/alias가 직접 노출되지 않은 사례만 사용한다.
최종 분모는 두 split 전체 audit 후 protocol에 기록하며 4,900건이라고 미리 가정하지 않는다.

---

## Slide 23B. Claim grounding and pair specificity

### Panel A. Claim grounding and pair specificity

| Method | Finding F1 | Value accuracy | Source-decision fidelity | Hard shuffle | Pair gap |
|---|---:|---:|---:|---:|---:|
| CoT | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD |

### Finding F1

Readout에서 추출한 finding ID 집합을 같은 case의 DDXPlus gold evidence ID 집합과 비교한다.
Precision은 생성한 finding 중 맞는 비율, recall은 gold finding을 회수한 비율이며 F1은 두 값의
조화평균이다. 누락과 불필요한 finding을 동시에 감점한다.

### Native value accuracy

서로 대응된 value-bearing finding에서 readout이 DDXPlus 사전의 정확한 native value를
복원했는지 본다. Finding 종류를 맞히고 `mild`를 `severe`로 읽은 경우 finding hit는 맞지만
value는 오답이다. Bare binary evidence와 대안 값이 없는 evidence는 이 분모에 넣지 않는다.

### Source-decision fidelity

Readout의 진단이 physician gold가 아니라 같은 source backbone run이 실제 생성한 진단과
일치하는지 본다. Backbone이 gold를 틀렸더라도 readout이 backbone 답을 정확히 읽었다면 이
지표에서는 성공이다. RQ2는 정답 교정 전에 현재 내부 판단의 충실한 판독을 먼저 묻기 때문이다.

### Hard shuffle과 Pair gap

Hard shuffle은 case `i`의 readout을 같은 diagnosis, 비슷한 finding 수와 prompt 길이를 가진
다른 case `j`의 finding/value target에 대조한다. 같은 진단 안에서 섞어 질환명과 설명 길이만으로
점수를 얻는 shortcut을 막는다.

```text
own_i      = score(readout_i, findings_i)
shuffled_i = score(readout_i, findings_j)
pair_gap_i = own_i - shuffled_i
Pair gap   = mean_i(pair_gap_i)
```

Hard-shuffle score는 낮고 Pair gap은 높아야 한다. Pair gap의 case-level paired bootstrap CI가
0을 배제해야 사례 특이성의 근거로 인정한다. Finding F1이 높더라도 Pair gap이 0이면 같은 질환의
전형적인 설명을 생성했을 가능성이 남는다.

---

## Slide 23C. Counterfactual response and reconstruction

### Panel B. Counterfactual response and reconstruction

| Method | Edited-finding response | Untouched retention | Matched FVE | Shuffled FVE | FVE gap |
|---|---:|---:|---:|---:|---:|
| CoT | TBD | TBD | N/A | N/A | N/A |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | N/A | N/A | N/A |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD |

이 Panel에서 reconstruction과 full의 차이가 직접 검증된다. Reconstruction 모델은 matched FVE가
높아도 shuffled FVE가 함께 높거나 finding edit에 반응하지 않을 수 있다. Full 모델은 양의 FVE
gap, 더 높은 edited-finding response, 높은 untouched retention을 함께 보여야 한다. 즉
reconstruction은 **자연어가 activation 정보를 담는가**, full은 **그 정보가 바로 이 사례에
고유하고 evidence 변화에 선택적으로 반응하는가**를 묻는다.

### Edited-finding response

같은 base case에서 finding 하나만 삭제하거나 native value 하나만 바꾼 뒤 CoT-P0 activation을
다시 추출한다. Deletion이면 원 readout에서 존재하던 target claim이 사라져야 하고, value edit이면
원 value에서 replacement value로 정해진 방향으로 변해야 한다. 단순히 설명 전체가 달라졌는지가
아니라 **편집한 finding에 대응하는 claim이 올바르게 반응했는지**를 측정한다.

### Untouched retention

편집하지 않은 나머지 findings가 readout에 얼마나 보존되는지 본다. 이 지표가 없으면 모든
claim을 지우는 reader가 deletion response에서 높은 점수를 받을 수 있다. Edited-finding
response와 untouched retention이 함께 높아야 선택적 counterfactual grounding이다.

### Matched FVE, Shuffled FVE, FVE gap

AV가 생성한 문장 `z_i`만 AR에 넣어 원 activation `h_i`를 복원한다.

```text
h_i -> AV -> z_i -> AR -> h_hat_i

FVE_matched
= 1 - MSE(h_i, AR(z_i)) / MSE(h_i, h_mean_validation)

FVE_shuffled
= 1 - MSE(h_i, AR(z_j)) / MSE(h_i, h_mean_validation)

FVE gap = FVE_matched - FVE_shuffled
```

`FVE=1`은 완전 복원, `FVE=0`은 validation mean activation을 예측한 수준이며 음수는 mean보다도
못한 복원이다. Matched FVE는 자기 설명으로 자기 activation을 복원하고, Shuffled FVE는 같은
진단의 다른 case 설명으로 복원한다. FVE gap이 양수이고 paired CI가 0을 배제해야 자연어가
사례 고유 activation 정보를 보존한다고 본다. AR이 없는 CoT와 Medical-AV SFT-only에는 FVE를
계산하지 않아 `N/A`다.

### 표 밖의 필수 controls

| Control | 무엇을 바꾸는가 | 실패하면 의미하는 것 |
|---|---|---|
| Zero activation | 실제 activation 대신 영벡터 입력 | 출력이 AV의 language prior만으로 생성됨 |
| Validation mean activation | 모든 case에 같은 평균벡터 입력 | 사례 고유 정보 없이 평균 상태만 읽음 |
| Activation swap | Case A metadata에 case B activation 결합 | Readout이 activation보다 metadata/prompt를 따름 |
| Direct-P0 sensitivity | 같은 validation case에서 instruction만 Direct로 변경 | CoT-P0 결과가 instruction에 과도하게 의존 |

RQ2 통과는 한 metric으로 결정하지 않는다. Own finding/value score, 양의 Pair gap, 방향성 있는
edited-finding response, 높은 untouched retention, 그리고 AR 모델의 matched-over-shuffled FVE가
함께 필요하다. Panel A는 **자기 환자의 내용을 읽었는가**, Panel B는 **상태를 바꾸면 설명이
선택적으로 따라 변하고 그 설명이 activation 정보를 보존하는가**를 답한다.

**RQ2의 현재 답.** DDXPlus CoT-P0에서 Medical-NLA matched/shuffled, finding edit,
untouched retention, round-trip Table 3은 아직
완료되지 않았다. 따라서 현재는 Medical-NLA 설명이 자기 activation에 의존한다고 결론 내리지
않는다. RQ1의 Table 2와 별개로 Table 3의 grounding 통제를 통과해야 RQ2가 닫힌다.

**RQ3로 넘어가는 조건.** Table 2만 높은 방법은 임상 문장 생성기일 수 있고, Table 3만 높은
방법은 의미가 빈약한 activation 식별기일 수 있다. 두 관문을 모두 통과한 방법만 text patching
또는 selective correction의 입력으로 사용한다.

---

## RQ3. 설명을 편집해 상태와 진단을 선택적으로 바꿀 수 있는가

RQ3는 grounding을 통과한 설명의 dataset-native claim을 편집하고 AR을 통해 activation으로
되돌렸을 때, 목표 속성과 진단 likelihood가 선택적으로 변하는지를 묻는다. 먼저 표상 수준의
선택성을 확인하고 그 다음 최종 진단 행동과 순이득을 본다. 개입으로 기존 오답이 줄더라도 원래
정답을 더 많이 깨뜨리면 성능 개선 방법이라고 부르지 않는다.

## Slide 24. RQ3: text patching과 selective correction

Table 3 grounding을 통과한 방법만 평가한다.
Reconstruction 모델이 FVE만 통과하고 pair specificity 또는 counterfactual grounding에 실패하면
비교군으로는 남길 수 있지만 faithful text intervention의 주 방법으로 사용하지 않는다. RQ2에서
clinical alignment와 사례 특이적 grounding을 모두 통과한 full 모델만 RQ3의 주 text-patching
방법이 된다.

### Panel A. Identity preservation and target selectivity

| Intervention | Identity preservation | Edited-value decoding | Target logit delta | Off-target KL |
|---|---:|---:|---:|---:|
| Raw activation patch | TBD | TBD | TBD | TBD |
| Vanilla NLA round-trip | TBD | TBD | TBD | TBD |
| Medical-NLA round-trip | TBD | TBD | TBD | TBD |
| Oracle counterfactual activation | TBD | TBD | TBD | TBD |

먼저 아무것도 편집하지 않은 decode-encode identity가 원 답과 비목표 logits를 보존해야 한다.
그 다음 DDXPlus가 정의한 evidence value만 편집한다. Text patching이 불안정하면 먼저 detector가
위험하다고 판단한 사례에서 readout을 재검토 prompt로 제공하는 selective correction을
평가한다.

### Panel B. Final behavioral utility

| Policy | Overall accuracy | Wrong-to-right | Right-to-wrong | Net correction | Intervention rate |
|---|---:|---:|---:|---:|---:|
| No intervention | TBD | TBD | TBD | 0 | 0 |
| Patch all | TBD | TBD | TBD | TBD | 1.0 |
| Probe-gated | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA-gated | TBD | TBD | TBD | TBD | TBD |
| Oracle-gated | TBD | TBD | TBD | TBD | TBD |

**RQ3의 현재 답.** 아직 Table 3을 통과한 Medical-NLA와 frozen intervention policy가 없으므로
`진단 성능을 개선했다`는 결론은 성립하지 않는다. 현재 성립하는 것은 평가 기준과 실행 순서다.
Identity preservation -> target selectivity -> behavioral net correction을 순서대로 통과해야 하며,
어느 단계든 실패하면 성능 개선 주장은 해당 단계에서 중단한다.

---

# Part V. Conclusion

Conclusion에서는 빈 표를 숨기지 않는다. 현재 확립된 답, 아직 미결인 답, 실패 시에도 남는
기여를 같은 화면에서 구분한다.

## Slide 25. 세 RQ에 대한 현재 답

**화면에 넣을 내용**

| RQ | 현재 답 | 가장 강한 현재 근거 | 아직 필요한 증거 |
|---|---|---|---|
| RQ1: CoT·vanilla NLA 대비 임상 설명 복원 | 현재 SFT-only 실패 | 50-case validation에서 seed 17이 최고였지만 Obscomp .0343으로 CoT .2349보다 낮음 | Reconstruction/full 모델과 locked 72/106 Table 2 |
| RQ2: 해당 사례 activation 의존성 | 미결 | 평가 population·hard shuffle·counterfactual·FVE 정의 고정 | DDXPlus CoT-P0 matched/shuffled·counterfactual Table 3 |
| RQ3: 설명 편집을 통한 선택적 상태·진단 제어 | 미결 | 개입 protocol과 policy metric 고정 | Identity, target-selective patching, positive net correction Table 4 |

**발표자 노트.** Table 1의 선행 baseline에서는 P0 내부 진단 신호와 vanilla reader의 범위를
확인했다. 그러나 이것은 RQ1의 성공 결과가 아니다. RQ1은 Medical-NLA가 CoT와 vanilla NLA보다
임상 설명을 잘 복원해야 닫히고, RQ2는 그 개선된 설명이 자기 activation에 의존해야 닫힌다.
RQ3는 그 후 설명 편집이 목표 상태와 진단을 선택적으로 바꾸면서 비목표 정보와 기존 정답을
보존할 때 닫힌다.

---

## Slide 26. 현재 논문의 기여

**화면에 넣을 내용**

1. CoT, likelihood, linear probe, vanilla NLA, Medical-NLA를 같은 source trajectory의 위치에
   정렬해 비교하는 의료 internal-readout protocol을 구성했다.
2. Closed diagnosis detection과 open clinical explanation을 같은 능력처럼 평균하지 않고
   별도의 평가 열로 분리했다.
3. DiReCT physician reference와 DDXPlus paired counterfactual을 분리해 clinical alignment와
   activation grounding의 혼동을 막았다.
4. 자유 산문 claim extraction과 official DiReCT Llama-3 semantic matching의 경계를 명시하고
   official evaluator를 oracle smoke로 재현했다.
5. 설명가능성과 진단 성능을 함께 주장하려면 grounding 이후 identity 보존과 selective net
   correction까지 필요하다는 검증 순서를 제시했다.

**발표자 노트.** 기여를 `Medical-NLA를 이미 완성했다`고 쓰지 않는다. 현재 가장 강한 기여는
어떤 자연어 판독을 믿을 수 있는 내부 설명이라고 부르기 위한 실험 구조와 baseline 경계를
정확히 세운 것이다. 최종 모델 기여는 Table 2와 Table 3이 채워진 뒤 결정한다.

---

## Slide 27. 한계와 주장 경계

| 현재 말할 수 있는 것 | 반드시 함께 말할 제한 |
|---|---|
| P0에서 진단 label이 선형 decode됨 | DiReCT validation 52행, supervised closed ontology |
| HS24 probe가 likelihood보다 높음 | NLA primary는 HS32 checkpoint 호환; 지표 공간도 다름 |
| Vanilla AV가 P0 진단 target을 명시적으로 복원하지 못함 | observation 전체의 부재나 activation 정보 부재를 뜻하지 않음 |
| DiReCT evaluator를 재현함 | 앞단 quote-constrained extractor는 본 연구 adaptation |
| SFT-only Medical-AV가 의료 형식으로 출력함 | validation 임상 alignment는 CoT보다 낮고 seed 불안정성이 큼 |

Backbone은 현재 Gemma-3-12B-IT 하나다. DiReCT는 511 notes의 제한된 corpus이고 supervised
train은 266행, exact-label 노출 제외 후 SFT train은 248행이다. DDXPlus는 큰 통제 실험에는
적합하지만 synthetic fixed ontology이므로 자연 임상 산문의 외적 타당성을 대신하지 않는다.
MCR 또는 추가 natural-text OOD가 필요하다.

---

## Slide 28. 현재 완료 상태와 남은 작업

### 완료

- DiReCT 511행 schema, PDD, 중복, patient grouping, official evaluator 감사
- Eligible 496행 및 frozen 266/52/72/106 split
- Gemma source CoT 496행과 exploratory Direct 비교
- P0/P1/P2 x HS16/24/32 activation tensor 4,464개
- Validation diagnosis/category linear probe와 early forced-answer likelihood
- Vanilla NLA P0 312 outputs의 blinded semantic diagnostic audit
- Medical-AV SFT-only seeds 17/29/43 학습
- CoT, vanilla NLA, SFT-only의 공통 50-case DiReCT validation 평가

### 아직 최종 표를 채우기 위해 필요한 것

1. DiReCT source-decision free-text label-space audit
2. DDXPlus validation CoT-P0 HS16/24/32 activation 추출
3. Finding-presence/value probe 구현과 validation layer/hyperparameter 선택
4. Medical-AR와 reconstruction/pair-specific Medical-NLA 구현
5. Validation grounding gate를 통과한 recipe만 동결
6. Frozen recipe로 DiReCT 72/106 Table 1·2와 DDXPlus locked Table 3을 한 번 평가
7. Grounding 통과 시 round-trip, patching, selective correction Table 4
8. MCR frozen OOD 또는 추가 외부 데이터에서 일반화 확인

### 교수님께 확인받을 결정

> DiReCT로 physician-reference clinical alignment를 평가하고, DDXPlus로 activation grounding과
> 개입 가능성을 검증하는 역할 분담이 적절한지, 그리고 SFT-only validation 실패를 근거로
> probe audit 뒤 AR reconstruction/pair-specific objective로 넘어가는 실행 순서가 타당한지
> 확인을 부탁드립니다.

---

## Slide 29. 최종 결론

**화면에 넣을 내용**

> 생성 전 의료 LLM activation에는 닫힌 진단 정보가 존재하지만, 기존 vanilla natural-language
> reader는 긴 사례의 통합 상태를 안정적인 임상 설명으로 복원하지 못했다. 본 연구는 이 간극을
> Medical-NLA로 메우되, 의사 설명과의 일치만으로 faithfulness를 선언하지 않고 paired
> activation grounding을 별도로 요구한다.

> Medical-NLA가 clinical alignment와 activation grounding을 모두 통과하고, 그 판독을 사용한
> 선택적 개입이 기존 정답을 보존하면서 positive net correction을 만들 때에만 설명가능성과
> 진단 성능을 함께 개선했다고 결론 내린다.

**발표자 노트.** 첫 문장은 세 RQ에 앞선 capability baseline과 연구 필요성을 요약한다. 둘째
문장은 RQ1·RQ2·RQ3의 단계적 성공 조건이며 아직 결과가 아니라 사전 고정한 판정 기준이다.
Table 2만 통과하면 임상 설명 생성기, Table 3까지 통과하면 내부 판독기, Table 4까지 통과하면
선택적 성능 개선 방법이라고 결론 내린다.

---

## Appendix A. 실험별 재현 설정

### A1. Source backbone generation

| 항목 | 설정 |
|---|---|
| Model | `google/gemma-3-12b-it` |
| Precision | bfloat16 |
| Placement | 2 x RTX 4090, 각 GPU `max_memory=22GiB` |
| Chat formatting | Hugging Face model chat template, `add_generation_prompt=true` |
| Sampling | greedy, `do_sample=false`, temperature/top-p 미사용 |
| Direct | assistant prefill `The answer is`, max new tokens 64, batch 4 |
| CoT | free generation, max new tokens 2,048, batch 1 |
| Parse | 마지막 `The answer is <diagnosis>.` pattern |
| Random seed | 17 |

Direct prefill과 CoT instruction은 출력 조건을 다르게 하지만, 각 조건 안에서 correctness label과
activation은 동일한 stored prompt/source transcript에 조인한다. 이번 E1 CoT에서는 truncated
response의 forced second-answer를 비활성화했다.

### A2. Activation extraction

| 항목 | 설정 |
|---|---|
| Hidden-state layers | HS16, HS24, HS32 |
| Hidden dimension | 3,840 |
| Primary position | P0 `last_token` |
| P1/P2 selection | target string의 `last_subtoken` |
| Forward precision | backbone bfloat16 |
| Stored tensor dtype | float32 |
| Extraction batch | 1 |
| Completeness | 496 cases x 3 positions x 3 layers = 4,464 tensors |

P1/P2는 새 응답을 생성하지 않고 source CoT transcript를 teacher-force한다. 동일 prompt의 여러
position과 layer는 가능한 한 같은 forward pass의 hidden states에서 저장한다.

### A3. Linear probe validation

| 항목 | 설정 |
|---|---|
| Input | P0 activations, HS16/24/32 |
| Train / validation | 266 / 52, patient-disjoint |
| Targets | canonical PDD 49-way, disease category 25-way |
| Model | single `torch.nn.Linear` layer |
| Feature preprocessing | train feature mean/std로 dimension-wise standardization |
| Optimizer | AdamW |
| Learning-rate grid | `3e-4`, `1e-3` |
| Weight-decay grid | `0`, `1e-4`, `1e-3`, `1e-2` |
| Class weighting | unweighted와 inverse-frequency balanced 모두 탐색 |
| Maximum epochs / patience | 300 / 30 |
| Selection | validation NLL 최소, tie-break Top-1과 weight decay |
| Seed | 17 |

Locked test manifest는 probe training interface에 존재하지 않는다. HS24가 validation에서 가장
좋았지만, NLA/AR의 primary layer는 공개 checkpoint 호환 때문에 HS32로 유지한다.

### A4. Early forced-answer likelihood

| 항목 | 설정 |
|---|---|
| Source prompt | CoT P0 prompt |
| Completion prefix | `The answer is` |
| Candidate scoring | candidate diagnosis 전체 문자열 teacher forcing |
| Primary score | label token mean log probability |
| Candidate sets | category 25, canonical PDD 61, probe-matched PDD 49 |
| Cases | validation 52 only |
| Candidate batch | 실제 완료 run 4 |
| Calibration sensitivity | fixed content-free prompt prior subtraction |

이 값은 raw next-token logit도, 저장된 P0 vector의 직접 unembedding도 아니다. 후보 ontology를
제공한 closed ranking이며 PDD 문자열 prior에 강하게 오염된 결과를 함께 보고한다.

### A5. Vanilla AV validation

| 항목 | 설정 |
|---|---|
| Decoder | `kitft/nla-gemma3-12b-L32-av` |
| Precision / placement | bfloat16, 2 x RTX 4090 |
| Primary activation | HS32/P0 |
| Sensitivity | HS16/P0, HS24/P0, P1, P2 |
| Prompts | checkpoint sidecar default, medical task-aligned suffix |
| Decoding | greedy, max new tokens 256, batch 4 |
| Validation outputs | P0 312 + P1/P2 208 = 520 |

HS16/24는 HS32에서 학습된 decoder에 다른 layer activation을 넣는 distribution-shift sensitivity다.
따라서 HS24 probe가 좋다는 이유만으로 HS24 vanilla AV를 primary 결과로 바꾸지 않는다.

### A6. Medical-NLA SFT-only v1

| 항목 | 설정 |
|---|---|
| Base decoder | released HS32 AV checkpoint |
| Activation | DiReCT P0/HS32 |
| Train / validation | 248 / 50 after exact gold-label-in-note exclusion |
| Target | grounded `<observed>` items + source-model `<answer>` |
| Max observations | 12 |
| Seeds | 17, 29, 43 |
| Epochs | 3 |
| Batch / gradient accumulation | 1 / 8 |
| Learning rate / weight decay | `2e-4` / `0` |
| LoRA | rank 16, alpha 32, dropout .05 |
| Target modules | q/k/v/o projections and gate/up/down projections |
| Memory | gradient checkpointing enabled |
| Checkpoint selection | validation clinical content-token loss |

XML scaffold token loss가 빠르게 낮아지는 현상을 피하기 위해 전체 validation loss가 아니라 실제
clinical content span의 token loss로 epoch를 선택한다. SFT-only는 reconstruction 또는
pair-specificity objective를 포함하지 않는다.

### A7. E4 validation evaluation

| 항목 | 설정 |
|---|---|
| Cases | 50 validation cases |
| Methods | CoT, vanilla, Medical-NLA seeds 17/29/43 |
| Extraction requests | 250, method-blind shuffled order |
| Extractor | quote-constrained Codex primary; model/version pin 필수 |
| Extractor sensitivity | same prompt with local Llama-3-8B |
| Official matcher | native Meta-Llama-3-8B-Instruct, temperature 0, top-p 1 |
| Failure handling | parse/extraction failure 유지, missing eval은 official rule대로 0 |

모든 private note, generated output, extraction request, judge response, prediction JSON은 restricted
data root에만 저장한다. Git에는 aggregate summary, code, prompt template만 올린다.

---

## 현재 수치와 최종 표를 혼동하지 않기 위한 원칙

- `n=171` Direct/CoT 결과는 exploratory pilot이다.
- `n=52` probe, likelihood, vanilla NLA 결과는 validation method-selection 결과다.
- `n=72/106` 결과만 frozen downstream main table에 들어간다.
- 과거 DDXPlus `.7247/.7506`은 local cue-readout positive control이며 DiReCT P0 점수와 직접
  평균하거나 같은 과제로 부르지 않는다.
- `N/A`는 실패 0점이 아니라 해당 방법에 출력 공간이 정의되지 않았다는 뜻이다.
- Clinical alignment, source-decision fidelity, activation grounding을 하나의 faithfulness
  점수로 합치지 않는다.

## 정본 결과 기록 위치

- DiReCT data/split: `docs/data/direct_dataset_and_split.md`
- E1 source 및 activation: `docs/experiments/01-direct-source-and-activations.md`
- E2 capability baselines: `docs/experiments/02-capability-baselines.md`
- Medical-NLA training: `docs/experiments/03-medical-nla-training.md`
- DiReCT explanation evaluation: `docs/experiments/04-direct-explanation-evaluation.md`
- 전체 실행 상태: `docs/paper/experiment_status.md`
- 최종 논문 표와 그림: `docs/paper/tables_and_figures.md`
