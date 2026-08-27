# Medical-NLA 교수님 발표 구성 (2026-08-27)

이 문서는 현재 연구 방향, DiReCT 데이터 구성, 실제 baseline prompt와 실행 설정,
P0/P1/P2 activation 위치, 현재까지 나온 결과, 최종 논문 표를 처음 듣는 사람에게 설명하기
위한 **슬라이드 구성과 발표 원고**다. Restricted DiReCT 원문과 환자 식별자는 포함하지 않는다.

Legacy 발표 원고의 형식만 차용해 전체를 `Introduction -> Methodology -> Experimental
Results(RQ1 -> RQ2 -> RQ3) -> Conclusion` 순서로 구성한다. 과거 wrong-note 연구의 문제,
가설, 수치, 표는 현재 발표에 가져오지 않는다.

각 슬라이드의 표, code block, 짧은 bullet은 **화면에 실제로 놓을 내용**이다. 뒤의 줄글은
**발표자 노트**다. 발표할 때는 표의 모든 숫자를 읽지 않고 먼저 모집단과 비교축을 설명한 뒤,
굵은 셀과 그 셀이 답하는 RQ만 연결한다.

| 대단원 | 발표에서 답하는 질문 |
|---|---|
| Introduction | 왜 CoT만으로 부족하며 왜 open-text internal readout이 필요한가 |
| Methodology | 무엇을 어디서 읽고, 어떤 데이터와 통제로 검증하는가 |
| Experimental Results | RQ1, RQ2, RQ3에 현재 데이터가 각각 무엇이라고 답하는가 |
| Conclusion | 확립된 기여, 아직 성립하지 않은 주장, 다음 실행은 무엇인가 |

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

## Slide 3. 대전제, 가설, 연구 질문

### 화면에 넣을 내용

> **대전제:** 믿을 수 있는 의료 내부 설명은 임상적으로 타당한 문장일 뿐 아니라,
> 실제 source-model activation에 사례 특이적으로 근거해야 한다.

| 가설 | 핵심 주장 | 대응 연구 질문 |
|---|---|---|
| H1 | CoT의 임상적 그럴듯함은 내부 상태 충실성을 보장하지 않는다 | RQ1: 생성 전 내부에는 무엇이 있고 기존 채널은 무엇을 읽는가 |
| H2 | Closed-label probe와 open-text NLA는 다른 능력이며 vanilla NLA는 통합 설명에 실패할 수 있다 | RQ2: Medical-NLA가 clinically aligned하고 activation-grounded한 설명을 만들 수 있는가 |
| H3 | 두 검증을 통과한 판독만 선택적 개입에 사용해야 한다 | RQ3: 검증된 판독이 정답 보존과 순수 교정을 동시에 달성하는가 |

### 발표 줄글

세 질문은 병렬 체크리스트가 아니라 의존 관계를 가진다. RQ1에서 P0에 읽을 정보가 없거나
baseline이 이미 충분하다면 Medical-NLA의 필요성이 약해진다. RQ2에서 임상 설명 품질과
activation grounding을 모두 통과하지 못하면 그 판독으로 성능을 고치는 RQ3를 강하게 주장할
수 없다. 따라서 결과도 RQ1, RQ2, RQ3 순서로 제시한다.

---

# Part II. Methodology

Methodology에서는 세 RQ를 채점할 공통 모집단과 측정 채널을 고정한다. 특히 physician
annotation과 activation ground truth를 혼동하지 않고, DiReCT와 DDXPlus에 서로 다른 역할을
준다.

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

## Slide 6A. DiReCT 한 사례에는 무엇이 들어 있는가

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

## Slide 6B. Disease category, PDD, physician deduction

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

## Slide 6C. 데이터 감사 수치는 무엇을 보장하는가

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

## Slide 8A. Backbone에 실제로 넣은 prompt

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

---

## Slide 8B. Source generation 실행 설정

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

## Slide 10. HS32는 무엇이며 왜 primary인가

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

---

## Slide 11. RQ1에서 비교하는 내부 측정 채널

| Method | 입력 | 출력 공간 | 할 수 있는 것 | 구조적 한계 |
|---|---|---|---|---|
| Forced-answer likelihood | source prompt와 고정 후보 문자열 | supplied diagnosis ontology | 후보 간 행동 선호 순위 | 열린 observation을 생성하지 못함 |
| Linear probe | P0 activation | 학습 때 정한 25 category 또는 49 PDD | closed-label 신호 탐지 | 새 속성·관계·문장을 출력하지 못함 |
| Vanilla NLA | P0 activation | 자유 자연어 | open-text 판독 | 길고 잡음이 많고 통합 상태 복원 실패 가능 |
| Medical-NLA | P0 activation | 구조화 임상 자연어 | observation·관계·source answer 판독 목표 | 별도 grounding 검증이 필요 |

네 방법의 숫자는 모두 같은 종류의 accuracy가 아니다. Likelihood와 probe는 닫힌 후보 공간,
NLA는 열린 생성 공간을 사용한다. 따라서 Table 1에서 한 평균 점수로 순위를 만들지 않고,
closed diagnosis signal과 open evidence를 분리한다.

---

## Slide 12. Medical-NLA 학습 변형

| Method | Clinical supervision | Reconstruction | Pair specificity | 실험상 역할 |
|---|---:|---:|---:|---|
| Vanilla NLA | No | pretrained | No | 공개 baseline |
| Medical-NLA SFT only | Yes | No | No | 의료 SFT만의 효과와 classifier collapse 검사 |
| Full Medical-NLA | Yes | Yes | Yes | reconstruction/contrastive grounding의 추가 가치 |

SFT-only는 DiReCT `train=266`, `val=52`만 사용하고 locked test 72/106을 보지 않는다.
Gold label이 note에 exact phrase로 노출된 행을 제외한 실제 SFT 입력은 train 248, validation
50이다. `<observed>`는 note에서 exact-grounded한 physician observation, `<answer>`는 physician
gold가 아니라 같은 CoT-P0 trajectory에서 backbone이 실제 생성한 source answer다.

Source-wrong 사례에 gold를 현재 내부 상태인 것처럼 강제로 넣지 않는 이유는 clinical target과
source-decision target을 구분하기 위해서다. SFT-only가 임상적으로 그럴듯한 상투 문구나 seen
class 생성기로 붕괴하면 full objective로 넘어가며, 이름만 full Medical-NLA라고 미리 붙이지 않는다.

---

## Slide 13. 설명과 activation을 서로 다른 평가기로 검증한다

```text
DiReCT free-text output
   -> 공통 method-blind quote-constrained claim extractor
   -> official prediction JSON
   -> 제공된 native Llama-3-8B Yes/No semantic matcher
   -> Accdiag, Obs*, Exp*

DDXPlus paired activation
   -> own pair / same-diagnosis hard shuffle
   -> cue deletion / native value edit / activation swap
   -> pair gap, target change, untouched retention
```

Codex는 자유 산문을 official schema로 바꾸는 **앞단 claim extractor**에만 사용할 수 있다.
Observation/rationale 의미 일치 판정을 Codex로 바꾸면 official DiReCT metric이 아니다. Primary
semantic matcher는 제공된 `Meta-Llama-3-8B-Instruct` native checkpoint, temperature 0,
top-p 1, exact `Yes`, official prompt와 greedy one-to-one matching을 유지한다.

DiReCT는 physician-reference clinical alignment를, DDXPlus는 paired activation dependence를
평가한다. 어느 한 데이터셋의 점수를 다른 축의 ground truth로 부르지 않는다.

---

## Slide 14. RQ3 개입은 grounding 통과 후에만 평가한다

| 단계 | 조작 | 성공 조건 |
|---|---|---|
| Decode-encode identity | text를 고치지 않고 AV->text->AR | 원 답과 비목표 상태 보존 |
| Text patch | dataset-native evidence value 하나만 편집 | 목표 속성/logit만 선택적으로 변화 |
| Selective correction | validation에서 고정한 detector가 flag한 사례만 재검토 | net correction 양수, correct-case preservation 유지 |
| Oracle activation patch | 실제 paired activation을 주입 | 달성 가능한 인과 효과의 상한 |

DDXPlus primary activation은 DiReCT 학습과 동일한 **CoT-P0/HS32**로 맞춰야 한다. 현재 E5
builder의 Direct-P0 표기는 activation 추출 전에 수정하며, Direct-P0는 instruction sensitivity로
분리한다. P1/P2는 grounding 주결과가 아니라 leakage/positive control이므로 필요 시 subset에서
추가한다.

---

# Part III. Experimental Results

결과는 RQ 순서로 읽는다. 현재 값이 있는 exploratory/validation 결과와 아직 비어 있는 locked-test
주표를 같은 종류의 증거처럼 섞지 않는다.

## RQ1. 생성 전 내부에는 무엇이 있으며 기존 채널은 무엇을 읽는가

**왜 RQ1부터 시작하는가.** P0 activation에 임상·진단 정보가 없거나 vanilla NLA가 이미
통합 설명을 충분히 복원한다면 새로운 Medical-NLA를 만들 이유가 약하다. 먼저 backbone 행동,
closed-label decodability, open readout의 성공과 실패 범위를 같은 사례에서 확인한다.

## Slide 15. E1 backbone behavior: 현재 나온 exploratory 결과

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

## Slide 16. 생성 전 P0에 닫힌 진단 정보가 있는가

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

## Slide 17. Table 1B를 단순화한 현재 validation snapshot

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

## Slide 18. Vanilla AV가 못 읽는가: local과 integrated readout 구분

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

## Slide 19. 기존 Medical-NLA pilot에서 LoRA가 개선한 것

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

## Slide 20. RQ1 최종 Table 1: backbone과 readout capability

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

**RQ1의 현재 답.** Validation에서는 P0에 닫힌 category/PDD 정보가 선형적으로 decode되며,
HS24 probe가 forced-answer likelihood보다 강했다. Vanilla AV는 token-local cue를 읽는 positive
control은 통과했지만 DiReCT CoT-P0의 통합 진단을 명시적으로 복원하지 못했다. 따라서
`P0에 정보가 없다`가 아니라 **closed-label signal은 있으나 기존 open reader가 이를 안정적인
통합 설명으로 꺼내지 못한다**가 현재 결론이다.

**RQ2로 넘어가는 이유.** RQ1은 Medical-NLA가 필요한 간극을 확인했지만 새로운 reader가 그
간극을 실제로 메웠다는 증거는 아니다. 다음에는 Medical-NLA가 CoT와 vanilla NLA보다 의사
annotation을 잘 설명하는지, 그리고 그 설명이 paired activation에 실제로 의존하는지를 함께
검증한다.

---

## RQ2. Medical-NLA는 clinically aligned하고 activation-grounded한 설명을 만드는가

RQ2는 두 시험을 모두 요구한다. DiReCT 점수만 높으면 좋은 의료 설명 생성기일 수 있지만 내부
판독기라고 할 수 없고, shuffled gap만 높지만 임상 내용이 틀리면 충실한 오류 설명으로 사용할 수
없다. 따라서 Table 2의 clinical alignment와 Table 3의 activation grounding을 순서대로 본다.

## Slide 21. RQ2-A: DiReCT clinical explanation quality

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

---

## Slide 23. RQ2-B: activation grounding

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

**RQ2의 현재 답.** Vanilla AV의 local-cue positive control과 DiReCT baseline 감사는 완료됐지만,
Medical-NLA 3 seeds의 공통 official Table 2와 DDXPlus CoT-P0 matched/shuffled Table 3은 아직
완료되지 않았다. 따라서 현재는 `Medical-NLA가 CoT보다 더 좋은 faithful explanation을
만들었다`고 결론 내리지 않는다. RQ2는 이 두 locked 평가가 모두 채워져야 닫힌다.

**RQ3로 넘어가는 조건.** Table 2만 높은 방법은 임상 문장 생성기일 수 있고, Table 3만 높은
방법은 의미가 빈약한 activation 식별기일 수 있다. 두 관문을 모두 통과한 방법만 text patching
또는 selective correction의 입력으로 사용한다.

---

## RQ3. 검증된 readout이 설명가능성과 진단 성능을 함께 개선하는가

RQ3는 자연어를 생성했다는 사실이 아니라 **그 자연어를 이용한 개입의 순이득**을 묻는다.
개입으로 기존 오답이 줄더라도 원래 정답을 더 많이 깨뜨리면 성능 개선이 아니다.

## Slide 24. RQ3: text patching과 selective correction

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

**RQ3의 현재 답.** 아직 Table 3을 통과한 Medical-NLA와 frozen intervention policy가 없으므로
`진단 성능을 개선했다`는 결론은 성립하지 않는다. 현재 성립하는 것은 평가 기준과 실행 순서다.
Identity preservation -> target selectivity -> behavioral net correction을 순서대로 통과해야 하며,
어느 단계든 실패하면 성능 개선 주장은 해당 단계에서 중단한다.

---

# Part IV. Conclusion

Conclusion에서는 빈 표를 숨기지 않는다. 현재 확립된 답, 아직 미결인 답, 실패 시에도 남는
기여를 같은 화면에서 구분한다.

## Slide 25. 세 RQ에 대한 현재 답

**화면에 넣을 내용**

| RQ | 현재 답 | 가장 강한 현재 근거 | 아직 필요한 증거 |
|---|---|---|---|
| RQ1: P0 정보와 기존 채널 | 부분적으로 확인 | HS24 probe category `.5962`, PDD `.4423`; local AV `.7247` vs shuffle `.0880` | locked 72/106 Table 1 |
| RQ2: clinically aligned + grounded Medical-NLA | 미결 | evaluator smoke와 vanilla/pilot baseline 완료 | Medical-NLA 3 seeds의 Table 2와 DDXPlus Table 3 |
| RQ3: 설명과 성능의 동시 개선 | 미결 | 개입 protocol과 policy metric 고정 | identity, patching, selective correction의 locked net gain |

**발표자 노트.** RQ1에서 내부 진단 신호의 존재와 vanilla reader의 범위는 확인했다. 그러나
RQ2는 새 Medical-NLA가 CoT보다 더 좋은 임상 설명이면서 activation에도 근거한다는 두 결과가
모두 필요하다. RQ3는 그 후에만 평가한다. 따라서 오늘 발표의 결론은 완성된 성능 향상이 아니라,
가설을 무너뜨리지 않고 검증할 수 있는 모집단·baseline·평가기를 만들고 RQ1의 핵심 간극을
확인했다는 것이다.

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
| Vanilla AV가 local medical cue를 읽음 | token-local 결과이며 integrated P0 explanation과 다름 |
| DiReCT evaluator를 재현함 | 앞단 quote-constrained extractor는 본 연구 adaptation |
| Medical-NLA가 필요한 간극이 존재함 | 새 reader가 그 간극을 메웠다는 locked 결과는 아직 없음 |

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

## Slide 29. 최종 결론

**화면에 넣을 내용**

> 생성 전 의료 LLM activation에는 닫힌 진단 정보가 존재하지만, 기존 vanilla natural-language
> reader는 긴 사례의 통합 상태를 안정적인 임상 설명으로 복원하지 못했다. 본 연구는 이 간극을
> Medical-NLA로 메우되, 의사 설명과의 일치만으로 faithfulness를 선언하지 않고 paired
> activation grounding을 별도로 요구한다.

> Medical-NLA가 clinical alignment와 activation grounding을 모두 통과하고, 그 판독을 사용한
> 선택적 개입이 기존 정답을 보존하면서 positive net correction을 만들 때에만 설명가능성과
> 진단 성능을 함께 개선했다고 결론 내린다.

**발표자 노트.** 첫 문장은 현재 RQ1과 연구 설계를 요약한다. 둘째 문장은 RQ2·RQ3의 성공
조건이며 아직 결과가 아니라 사전 고정한 판정 기준이다. 최종 제출에서는 Table 2와 Table 3이
성공하면 강한 결론으로 전환하고, 실패하면 RQ1과 검증 protocol, failure analysis를 중심으로
주장 범위를 줄인다.

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
