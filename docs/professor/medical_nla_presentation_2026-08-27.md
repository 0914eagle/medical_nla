# Medical-NLA 교수님 발표 구성 (2026-08-27)

이 문서는 현재 연구 방향, DiReCT 데이터 구성, 실제 baseline prompt와 실행 설정,
P0/P1/P2 activation 위치, 현재까지 나온 결과, 최종 논문 표를 처음 듣는 사람에게 설명하기
위한 발표 원고다. Restricted DiReCT 원문과 환자 식별자는 포함하지 않는다.

## 발표에서 먼저 구분할 것

- **현재 결과가 없는 것이 아니다.** E1 exploratory backbone 결과, E2 validation probe,
  forced-answer likelihood, vanilla AV semantic audit, 과거 DDXPlus local-cue 결과가 있다.
- 논문 주표의 빈칸은 `test_seen=72`, `test_pdd_heldout=106` locked evaluation과 아직 학습하지
  않은 Medical-NLA 결과를 validation 값으로 미리 채우지 않았기 때문이다.
- 발표에서는 `현재까지 확인한 결과`와 `최종 locked-test 표`를 서로 다른 슬라이드로 보여준다.
- `PDD`는 **Primary Discharge Diagnosis**다. 퇴원 기록 자체가 아니라 disease category보다
  세분화된 최종 진단 label이다. 예를 들어 category가 `Heart Failure`일 때 PDD는 `HFrEF`,
  `HFpEF`가 될 수 있다.
- `Seen PDD`는 같은 환자를 봤다는 뜻이 아니다. 환자는 split 간 분리되어 있고, PDD label이
  train에 있었던 새 환자다. `Held-out PDD`는 PDD label 자체가 train에 없었던 새 환자다.

---

## Slide 1. 두괄식 연구 요약

### 화면에 넣을 내용

- CoT는 유용한 임상 설명을 생성하지만 모델의 실제 내부 판단 과정을 충실하게 보고한다고
  보장할 수 없다.
- Linear probe는 사전에 정의한 진단 label을 내부에서 잘 탐지하지만, 환자 고유 관찰·속성·
  관계를 하나의 열린 자연어 설명으로 직접 제공하지 않는다.
- Vanilla AV는 개별 cue 근처의 의료 정보를 읽을 수 있었지만, 긴 임상 기록을 통합한 생성 전
  상태를 구조적으로 복원하지 못했다.
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

## Slide 3. 연구 논리와 질문

### 화면에 넣을 내용

- CoT의 임상적 그럴듯함은 내부 상태 충실성을 보장하지 않는다.
- Probe의 closed-label detection과 NLA의 open-text readout은 서로 다른 능력이다.
- 단순 의료 SFT는 진단 class 또는 상투 문구를 생성하는 모델로 붕괴할 수 있다.
- Medical-NLA는 clinical alignment와 activation grounding을 모두 통과해야 한다.
- 검증된 readout이 생긴 뒤에만 correction 또는 patching의 순이득을 평가한다.

### 발표 줄글

이를 세 질문으로 정리한다. 첫째, 생성 전 activation에는 어떤 임상 정보가 존재하고 기존
도구들이 무엇을 읽는가. 둘째, Medical-NLA가 CoT와 vanilla NLA보다 의사의 관찰-근거-진단
구조를 잘 설명하면서 해당 사례 activation에 실제로 의존하는가. 셋째, 그렇게 검증한 판독을
사용해 기존 정답을 보존하면서 오답을 순수하게 줄일 수 있는가.

---

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

---

## Slide 5. 데이터셋별 역할

| 데이터셋 | 원래 제공하는 정보 | 본 연구의 역할 | 사용하지 않을 주장 |
|---|---|---|---|
| DiReCT | 임상 note, physician observation, rationale, diagnosis tree | 주 임상 설명 품질, seen/PDD-heldout 평가 | activation ground truth |
| DDXPlus | pathology, evidence ID/value, differential | matched/shuffled, cue 반사실, patching | 자연 임상 산문의 최종 품질 |
| MedCaseReasoning | case-report 산문과 diagnosis/reasoning | 향후 frozen natural-text OOD | 정확한 gold evidence span |

Intro에서는 “임상 설명과 activation grounding을 서로 다른 적합한 데이터로 검증한다”는
한 문장만 말한다. 이 역할표와 데이터 구조는 Methodology에서 설명한다.

---

## Slide 6A. DiReCT의 배포 단위

DiReCT restricted release에서 본 연구가 확인한 구성은 다음과 같다.

| 구성 | 수 | 본 연구에서의 용도 |
|---|---:|---|
| Clinical note JSON | 511 | backbone 입력과 설명 reference |
| Diagnostic KG JSON | 24 | 진단 ontology와 구조 감사 |
| Disease category | 25 | 넓은 질환군 평가 |
| Canonical PDD | 61 | 구체적인 최종 진단 평가 |

**임상 note 한 행이 한 사례**다. 한 행 안에는 같은 환자의 주호소, 현병력, 과거력, 가족력,
신체검진, 검사 결과가 함께 들어 있다. 여섯 입력 필드는 서로 다른 환자나 여섯 독립 표본이 아니다.

발표 핵심 문장:

> DiReCT는 진단명만 제공하는 QA 데이터가 아니라, 한 환자의 임상 기록과 의사가 표시한
> observation-rationale-diagnosis 구조를 함께 제공한다.

---

## Slide 6B. 임상 입력 1: 환자가 무엇을 호소했는가

| 필드 | 임상 섹션 | 담는 정보 | 진단에서의 역할 |
|---|---|---|---|
| `input1` | Chief Complaint, 주호소 | 병원에 온 가장 직접적인 이유 | 문제의 출발점과 대표 증상 |
| `input2` | History of Present Illness, 현병력 | 시작, 기간, 변화, 유발·완화 요인, 동반 증상 | 현재 질환의 시간적 경과와 증상 조합 |
| `input3` | Past Medical History, 과거력 | 기존 질환, 과거 입원·수술 등 | 기저 위험과 감별진단의 사전확률 |

쉽게 구분하면 다음과 같다.

```text
주호소: 지금 가장 불편해서 온 이유
현병력: 그 문제가 언제부터 어떻게 진행됐는지
과거력: 현재 판단에 영향을 줄 기존 질환과 이전 병력
```

---

## Slide 6C. 임상 입력 2: 의료진이 무엇을 확인했는가

| 필드 | 임상 섹션 | 담는 정보 | 진단에서의 역할 |
|---|---|---|---|
| `input4` | Family History, 가족력 | 가족의 질환과 유전적 위험 | 유전성·가족성 질환 가능성 |
| `input5` | Physical Exam, 신체검진 | 활력징후와 의료진이 관찰·측정한 징후 | 환자 진술과 구분되는 객관적 소견 |
| `input6` | Pertinent Results, 주요 검사 | 혈액검사, 영상, 심전도 등 | 진단을 지지하거나 배제하는 검사 증거 |

```text
input1--3: 환자가 말한 현재 문제와 배경
input4--6: 가족 위험, 의료진 관찰, 객관적 검사 결과
                     ↓
              하나의 clinical note
                     ↓
              동일 backbone prompt
```

일부 필드는 원자료에서 비어 있을 수 있다. 빈 필드에 임의의 정상 소견을 보충하지 않고 그대로
두며, 모든 방법이 같은 조립 규칙으로 만든 note를 입력받는다.

---

## Slide 6D. Disease category와 PDD의 차이

`PDD`는 **Primary Discharge Diagnosis**다. 환자의 퇴원 기록 전체가 아니라, 그 기록에 부여된
구체적인 주 퇴원 진단 label을 뜻한다.

```text
Disease category: Heart Failure
        ├─ PDD: HFrEF
        └─ PDD: HFpEF
```

| 평가 | 묻는 질문 | 난이도 |
|---|---|---|
| Disease-category accuracy | 넓은 질환군을 맞혔는가 | 상대적으로 거친 분류 |
| Strict-PDD accuracy | 구체적인 세부 진단까지 맞혔는가 | 더 엄격한 분류 |

DiReCT에는 25 categories와 61 canonical PDD가 있다. 폴더 이름만 세어 얻은 초기 62개가 아니라,
공식 `data_list.csv`와 annotation root를 정규화한 **61개가 정본**이다.

---

## Slide 6E. 의사 주석은 무엇을 제공하는가

DiReCT의 의사 주석은 다음 연결을 제공한다.

```text
observation
환자 기록에서 진단에 사용되는 사실
        ↓
rationale
그 사실이 해당 진단을 지지하는 임상적 이유
        ↓
diagnosis
중간 진단 또는 최종 진단
```

실제 restricted 원문이 아닌 일반적인 예시는 다음과 같다.

```text
observation: 휴식 중에도 발생하는 흉통
rationale:   휴식 시 흉통은 급성 관상동맥 증후군 가능성을 더 지지함
diagnosis:   불안정 협심증
```

---

## Slide 6F. 이 구조로 설명의 무엇을 평가하는가

Annotation tree를 `observation -> rationale -> diagnosis` deduction으로 정규화한다.

| 평가 질문 | 확인하는 오류 |
|---|---|
| 필요한 observation을 회수했는가 | 중요한 임상 정보 누락 |
| 불필요한 observation을 추가했는가 | 기록에 없는 내용 또는 과잉 설명 |
| observation과 diagnosis를 올바르게 연결했는가 | 그럴듯하지만 잘못된 임상 관계 |
| diagnosis를 올바른 specificity로 제시했는가 | 넓은 category만 맞히고 세부 PDD를 놓침 |

따라서 DiReCT는 단순 진단 정확도 외에 **무엇을 관찰했고, 왜 그 진단으로 연결했는지**를
CoT와 Medical-NLA 사이에서 비교하게 해준다.

---

## Slide 6G. 데이터 감사 결과

감사(audit)는 CoT나 Medical-NLA의 성능 평가가 아니라, 모델 실험 전에 수행한 데이터 품질
검사다.

| 감사 항목 | 값 | 직접 확인한 것 |
|---|---:|---|
| Raw notes / valid JSON | 511 / 511 | 모든 배포 note가 정상 파싱됨 |
| Disease categories | 25 | 상위 질환군 vocabulary |
| Canonical PDD labels | 61 | 정규화된 세부 진단 vocabulary |
| Parsed patient groups | 469 | 반복 note를 환자 단위로 묶을 수 있음 |
| Physician deductions | 5,109 | observation-rationale-diagnosis 연결 수 |
| Grounded observations | 4,965/5,109 (.9718) | observation이 note에 exact substring으로 존재 |

`5,109 deductions`는 환자 수가 아니다. 한 note에 여러 관찰과 추론 연결이 있기 때문에 생긴
annotation 단위의 총수다.

---

## Slide 6H. 감사 결과를 어떻게 해석해야 하는가

| 값 | 올바른 해석 | 잘못된 해석 |
|---|---|---|
| 511/511 valid JSON | parse 누락 없이 전체 파일을 읽음 | 모든 내용과 label이 완벽함 |
| 25 categories / 61 PDDs | label vocabulary의 크기 | 클래스별 표본 수가 균등함 |
| 469 patient groups | patient-disjoint split이 필요함 | 511 notes가 511명의 독립 환자임 |
| 4,965/5,109 grounded | physician observation의 97.18%가 원문에서 추적됨 | 모델의 observation 정확도가 97.18%임 |

남은 144개 observation은 즉시 잘못된 주석으로 처리하지 않는다. 약어, 문장 변형, 정규화 차이
때문에 exact match가 실패했을 수 있으므로 별도 감사 대상으로 남긴다.

---

## Slide 6I. 왜 데이터 감사를 먼저 했는가

- **JSON 감사:** 조건별 parse 실패로 분모가 달라지는 것을 방지한다.
- **Label 정규화:** 복수형·개행·폴더 표기 차이를 별도 진단으로 잘못 세지 않는다.
- **Patient grouping:** 같은 환자의 반복 note가 train과 test에 동시에 들어가는 것을 막는다.
- **Deduction grounding:** 설명 평가의 정답 observation이 실제 note에서 추적되는지 확인한다.

즉 이 감사는 높은 모델 점수를 만들기 위한 필터가 아니라, **모든 방법을 같은 모집단과 같은
정답 정의로 비교하기 위한 선행 조건**이다.

---

## Slide 6J. Raw 511행에서 실험 496행으로

| 제외 사유 | 행 수 | 제외 이유 |
|---|---:|---|
| Canonical PDD 의미 충돌 | 10 | 폴더 PDD와 annotation root가 다른 임상 label을 가리킴 |
| Patient ID parse 실패 | 4 | 환자 단위 분리를 보장할 수 없음 |
| Exact duplicate copy | 1 | 동일 사례의 중복 집계를 방지 |
| **최종 eligible population** | **496** | patient-disjoint split의 고정 모집단 |

```text
511 = restricted release 전체를 감사할 때의 분모
496 = primary split과 후속 실험에서 사용하는 분모
```

원 audit의 469 patient groups와 제외 후 split의 458 patient groups도 같은 이유로 구분한다.
이후 표에서 511과 496을 같은 모집단처럼 섞지 않는다.

---

## Slide 6K. 의사 설명과 모델 내부 상태는 다른 정답이다

Physician annotation은 **임상적으로 바람직한 설명의 reference**이지 source-model activation의
정답이 아니다. 모델이 오답을 선택한 사례에서는 의사 gold와 모델 내부의 현재 결론이 다를 수 있다.

| 평가 축 | 질문 | 주된 평가 자원 |
|---|---|---|
| Clinical alignment | 설명이 의사 주석과 임상적으로 일치하는가 | DiReCT physician annotation |
| Source-decision fidelity | 설명이 source model의 현재 판단을 충실하게 읽는가 | source answer와 paired controls |
| Activation grounding | 설명이 해당 activation에 사례 특이적으로 의존하는가 | matched/shuffled와 counterfactual |

세 질문을 하나의 faithfulness 점수로 합치지 않는다. DiReCT의 높은 설명 점수만으로 activation을
충실하게 읽었다고 주장하지 않고, 별도의 activation grounding 실험을 통과해야 한다.

---

## Slide 7. DiReCT 모집단과 split

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

---

## Slide 8. Backbone에 실제로 넣은 prompt

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

### 실행 설정

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

---

## Slide 9. P0, P1, P2는 무엇인가

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

P1과 P2는 source CoT response를 teacher-force하여 동일한 source trajectory에서 추출한다.
171행 pilot에서 모델이 최종 선택한 diagnosis alias가 이미 CoT reasoning 안에 등장한 경우가
156/171(.9123)이었다. 따라서 P1에서 높은 진단 판독률이 나와도 내부 결론을 새로 읽은 것인지,
이미 적힌 진단 문자열을 읽은 것인지 분리하기 어렵다. `diagnosis_alias_in_reasoning=false`인
P1 clean subset은 15행뿐이었다. 이 때문에 P0를 primary로 고정하고 P2를 positive control로
사용한다.

---

## Slide 10. E1 backbone behavior: 현재 나온 exploratory 결과

| Generation | Pool | n | Parse | Strict PDD | Disease category | Diagnosis token F1 |
|---|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled | exploratory test | 171 | 1.0000 | .2105 | .5029 | .1593 |
| Source CoT | exploratory test | 171 | 1.0000 | .1930 | .5088 | .1850 |

Paired breakdown은 둘 다 정답 26, Direct만 정답 10, CoT만 정답 7, 둘 다 오답 128이다.
Strict PDD McNemar exact `p=.6291`, category `p=1.0`이다.

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

## Slide 11. 생성 전 P0에 닫힌 진단 정보가 있는가

아래는 locked test가 아니라 `val_seen=52`에서 layer와 baseline을 선택하기 위한 결과다.

| Method | Target | Classes | n | Top-1 | Top-5 | MRR |
|---|---|---:|---:|---:|---:|---:|
| Early forced-answer likelihood | Disease category | 25 | 52 | .4808 | .6731 | .5814 |
| Linear probe, HS24 | Disease category | 25 | 52 | **.5962** | **.9038** | **.7284** |
| Early forced-answer likelihood | Canonical PDD, train ontology | 49 | 52 | .1538 | .5192 | .3250 |
| Linear probe, HS24 | Canonical PDD | 49 | 52 | **.4423** | **.7692** | **.5762** |

Forced-answer likelihood는 P0 prompt 뒤에 `The answer is`를 붙이고 후보 문자열을
teacher-force하여 평균 token log probability로 순위를 매긴 행동 기준선이다. 저장된 P0
벡터를 직접 unembed한 값이 아니다. PDD raw ranking은 한 희귀 후보를 35/52행에서 top-1으로
선택해 label surface prior에 취약했다. Content-free prior subtraction은 category top-1
`.2308`, PDD `.0577`로 더 악화되어 appendix sensitivity로만 둔다.

Probe의 결과는 P0에 진단 정보가 없지 않다는 증거다. 그러나 probe는 train에서 정의한 49개
PDD 또는 25개 category 중 하나를 고르는 분류기이므로 열린 observation 설명 기준선이 아니다.

---

## Slide 12. Table 1B를 단순화한 현재 validation snapshot

`Trained task head`와 `Eval ontology` 열은 표에서 제거한다. 방법 차이는 캡션에 적는다.

| Method | n | PDD signal | Category signal | Source-decision fidelity | Open evidence |
|---|---:|---:|---:|---:|---:|
| Early forced-answer likelihood | 52 | .1538 | .4808 | N/A | N/A |
| Linear probe, HS24 | 52 | .4423 | .5962 | N/A | N/A |
| Vanilla NLA, default/HS32/P0 | 52 | 0/52 | 0/52 | 0/52 | 공식 평가 전 |
| Vanilla NLA, task-aligned/HS32/P0 | 52 | 0/52 | 0/52 | 0/52 | 공식 평가 전 |
| Medical-NLA | locked test |  |  |  |  |

**표 캡션:** PDD/category signal은 likelihood와 probe에서는 Top-1이고 NLA에서는 exact quote를
요구한 semantic diagnostic match다. 동일한 출력 지표가 아니므로 한 순위로 평균하지 않는다.
Probe는 supervised closed-label classifier, likelihood는 supplied ontology ranking,
NLA는 open-text generation이다. Vanilla NLA의 0/52는 physician observation/rationale
품질이 0이라는 뜻이 아니라, 생성 전 P0에서 진단 target을 명시적으로 복원하지 못했다는
좁은 validation 진단이다.

이 0/52를 확인한 E2 semantic audit은 공식 DiReCT 설명 metric과 별도다. Validation
52 cases x 2 prompts x 3 layers = 312 readout을 모두 판정했다. Judge에는 patient note를
주지 않고 source answer, gold PDD, category 세 target을 A/B/C로 섞어 제시했다. Match가
참이면 readout에 실제로 존재하는 exact quote를 반환하게 했고, 후처리에서 quote가 없으면
불일치로 처리했다. Native Llama-3-8B, max sequence 8,192, max generation 192,
temperature 0, top-p 1.0을 사용했다. 같은 checkpoint를 사용했지만 prompt와 판정 목적이
공식 evaluator와 다르므로 `official DiReCT metric`이라고 부르지 않는다.

---

## Slide 13. Vanilla AV가 못 읽는가: local과 integrated readout 구분

| 실험 | Activation 위치 | 목표 | Vanilla AV 결과 |
|---|---|---|---:|
| 기존 DDXPlus | 개별 cue가 적힌 token | 해당 cue 내용 | .7247 |
| 기존 DDXPlus hard shuffle | 다른 사례 cue | 통제 | .0880 |
| 현재 DiReCT | P0 | 생성 전 통합 진단 | 0/52 |
| 현재 DiReCT | P1 | CoT 뒤 source answer | .5192 |
| 현재 DiReCT | P2 | 답 뒤 source answer | .5962 |

DDXPlus `.7247`은 바닐라 AV가 의료 정보를 전혀 읽지 못한다는 가설을 반박하는 positive
result다. 다만 이는 `chest pain at rest` 같은 cue가 입력된 바로 그 token에서 내용을 다시
읽는 local readout이다. DiReCT P0는 긴 기록 전체를 읽고 아직 reasoning이나 답을 쓰기 전
하나의 상태에서 통합된 진단 근거와 결론을 설명하는 더 어려운 과제다.

따라서 정확한 주장은 다음이다.

> Vanilla AV는 token-local 의료 정보를 읽을 수 있지만 출력이 길고 잡음이 많으며, 생성 전
> 통합 상태에서 구조화된 사례별 설명을 안정적으로 복원하지 못했다.

---

## Slide 14. 기존 Medical-NLA pilot에서 LoRA가 개선한 것

| 지표 | Vanilla AV | Pilot Medical-NLA |
|---|---:|---:|
| Held-out local cue readout | .7247 | .7506 |
| Format compliance | .0494 | **1.0000** |
| Mean output length | 1,557 characters | **52 characters** |
| Whole-output precision | .0752 | **.6707** |

LoRA가 의료 정보를 처음 읽게 만든 것은 아니다. 주요 효과는 이미 존재하던 local readout을
짧고 구조적이며 정밀한 판독으로 정제한 것이다. 새 연구에서 필요한 추가 기여는 이 능력을
DiReCT P0의 integrated clinical state로 확장하고, shuffled/counterfactual 검증을 통과하게
하는 것이다.

---

## Slide 15. 새 Medical-NLA 학습 설계

| Method | Clinical text | Reconstruction | Pair specificity | 현재 상태 |
|---|---:|---:|---:|---|
| Vanilla NLA | No | pretrained | No | baseline 완료 |
| Medical-NLA SFT only | Yes | No | No | 주 실행 대상 |
| Full Medical-NLA | Yes | Yes | Yes | objective 구현 후에만 포함 |

SFT-only v1은 P0 activation을 입력으로 사용한다. `<observed>`에는 note에서 exact grounding된
physician observation만 넣고, `<answer>`에는 physician gold가 아니라 같은 trajectory에서
backbone이 실제 생성한 source answer를 넣는다. Source-wrong 행에 gold correction을 현재
상태인 것처럼 강제로 매핑하는 오류를 피하기 위해서다. Train 266과 validation 52만 사용하며,
test 72/106은 model selection에 사용하지 않는다.

---

## Slide 16. 최종 Table 1: backbone과 readout capability

### Panel A. Backbone behavior on locked identical case IDs

| Method | Pool | n | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled | Seen PDD | 72 |  |  |  |  |
| Direct, answer-prefilled | Held-out PDD | 106 |  |  |  |  |
| Source CoT | Seen PDD | 72 |  |  |  |  |
| Source CoT | Held-out PDD | 106 |  |  |  |  |

### Panel B. CoT-P0 internal readout

| Method | Coverage | Seen-PDD gold | Held-out-PDD gold | Category gold | Source-decision fidelity | Open evidence |
|---|---:|---:|---:|---:|---:|---:|
| Early forced-answer likelihood |  |  |  |  |  | N/A |
| Linear probe |  |  | N/A |  |  | N/A |
| Vanilla NLA |  |  |  |  |  |  |
| Medical-NLA SFT only |  |  |  |  |  |  |
| Full Medical-NLA |  |  |  |  |  |  |

Panel A는 모델이 실제로 무엇을 답했는지, Panel B는 생성 전 activation에서 각 방법이 무엇을
읽을 수 있는지를 묻는다. 서로 다른 질문과 출력 공간이므로 하나의 평균 점수로 합치지 않는다.

---

## Slide 17. 최종 Table 2: DiReCT clinical explanation quality

| Method | Pool | n | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | Seen / Held-out | 72 / 106 |  |  |  |  |  |  |  |
| Vanilla NLA | Seen / Held-out | 72 / 106 |  |  |  |  |  |  |  |
| Medical-NLA SFT only | Seen / Held-out | 72 / 106 |  |  |  |  |  |  |  |
| Full Medical-NLA | Seen / Held-out | 72 / 106 |  |  |  |  |  |  |  |

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

## Slide 18. DiReCT official evaluator를 어떻게 재현하는가

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

### 1단계: 공통 quote-constrained claim extraction

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

### 2단계: 제공된 Llama-3-8B official semantic matching

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

### 3단계: official statistics-compatible aggregation

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

### Official evaluator 재현 smoke

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

### Official evaluator의 알려진 민감도

- Observation은 gold 순서대로 첫 `Yes` prediction을 선택하므로 dictionary 순서에 민감할 수 있다.
- 응답이 `Yes.` 또는 ` yes `이면 primary official mode에서는 불일치다.
- Rationale가 의미상 맞아도 연결 diagnosis가 exact rule을 통과하지 못하면 Expcom/Expall에
  포함되지 않는다.
- Claim extractor가 explicit quote만 허용하므로 implicit하지만 타당한 설명은 누락될 수 있다.
- 따라서 prediction-order permutation, normalized-Yes, alternative matching, extractor backend,
  일부 human/clinician audit은 보조 민감도로 보고하고 primary official score를 사후 교체하지 않는다.

---

## Slide 19. 최종 Table 3: activation grounding

| Method | Own pair | Hard shuffle | Pair gap | Cue deletion | Untouched retention | Round-trip FVE |
|---|---:|---:|---:|---:|---:|---:|
| CoT |  |  |  |  |  | N/A |
| Vanilla NLA |  |  |  |  |  |  |
| Medical-NLA SFT only |  |  |  |  |  |  |
| Full Medical-NLA |  |  |  |  |  |  |

Hard shuffle은 같은 진단과 비슷한 cue 수를 가진 다른 사례의 activation-text 짝으로 바꾼다.
Cue deletion은 evidence 하나를 prompt에서 삭제한 뒤 해당 판독 내용만 감소하는지 본다.
Untouched retention은 삭제하지 않은 evidence가 보존되는지 측정한다. Round-trip FVE는 판독을
AR로 다시 activation으로 만들었을 때 원 activation의 분산을 얼마나 설명하는지 측정한다.

---

## Slide 20. 최종 Table 4: text patching과 성능 개선

Table 3 grounding을 통과한 방법만 평가한다.

| Intervention | No-op preservation | Edited attribute | Target logit delta | Off-target KL | Diagnostic change |
|---|---:|---:|---:|---:|---:|
| Original activation |  | N/A | 0 | 0 | 0 |
| Decode-encode identity |  | N/A |  |  |  |
| Plain-text prompt edit |  |  |  |  |  |
| Medical-NLA text patch |  |  |  |  |  |
| Oracle activation patch |  |  |  |  |  |

먼저 아무것도 편집하지 않은 decode-encode identity가 원 답과 비목표 logits를 보존해야 한다.
그 다음 DDXPlus가 정의한 evidence value만 편집한다. Text patching이 불안정하면 먼저 detector가
위험하다고 판단한 사례에서 readout을 재검토 prompt로 제공하는 selective correction을
평가한다.

성능 개선은 overall accuracy 하나만 보지 않고 다음을 함께 보고한다.

| Policy metric | 의미 |
|---|---|
| Overall accuracy | 전체 순이득 |
| Wrong-case recovery | 기존 오답 복구율 |
| Correct-case preservation | 기존 정답 보존율 |
| Newly broken | 개입으로 새로 틀린 사례 |
| Net correction | wrong-to-right minus right-to-wrong |
| Intervention rate | 실제 개입 규모 |

---

## Slide 21. 현재 완료 상태와 남은 작업

### 완료

- DiReCT 511행 schema, PDD, 중복, patient grouping, official evaluator 감사
- Eligible 496행 및 frozen 266/52/72/106 split
- Gemma source CoT 496행과 exploratory Direct 비교
- P0/P1/P2 x HS16/24/32 activation tensor 4,464개
- Validation linear probe와 early forced-answer likelihood
- Vanilla NLA P0 312 outputs의 blinded semantic diagnostic audit
- 기존 DDXPlus local cue readout positive control

### 아직 최종 표를 채우기 위해 필요한 것

1. Medical-NLA SFT-only 3 seeds
2. CoT, vanilla NLA, SFT-only의 공통 DiReCT claim extraction
3. Locked test 72/106의 Table 1과 Table 2 평가
4. DDXPlus matched/shuffled, cue deletion, untouched retention
5. Grounding 통과 시 round-trip과 patching
6. Selective correction의 net correction 및 preservation 평가
7. MCR frozen OOD 또는 추가 외부 데이터에서 일반화 확인

### 교수님께 확인받을 결정

> DiReCT로 physician-reference clinical alignment를 평가하고, DDXPlus로 activation grounding과
> 개입 가능성을 검증하는 역할 분담이 적절한지, 그리고 먼저 SFT-only를 확정한 뒤 grounding
> 실패 시 reconstruction/pair-specific objective를 추가하는 단계적 실행이 타당한지 확인을
> 부탁드립니다.

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
