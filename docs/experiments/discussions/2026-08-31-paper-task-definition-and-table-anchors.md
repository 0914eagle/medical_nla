# 논문 task 정의와 외부 표 anchor

## 출발점: 교수님 피드백

현재 DiReCT/DDXPlus 결과는 Medical-NLA 개발 과정과 내부 진단에는 유용하지만, 그 자체가
논문의 외부 task를 정의하지는 않는다. 다음 세 질문을 먼저 닫아야 한다.

1. 이 논문이 푸는 **task**는 무엇인가? 현재 결과를 어느 공개 task의 표에 놓을 수 있는가?
2. Medical-NLA는 CoT, text-only explanation, probe, SAE 등과 비교해 무엇이 더 좋은가?
3. 2025--2026 의료 explanation/interpretability 논문과 DiReCT를 사용하는 연구는 어떤
   benchmark와 표를 쓰며, 우리는 무엇을 재현 또는 확장할 수 있는가?

이 문서는 답을 동결하는 결정 원장이 아니다. 주장, task, 표의 관계를 먼저 분리하고, 이후
사람 승인을 받아 benchmark와 final recipe를 고정하기 위한 discussion이다.

## 먼저 구분할 것: 주장과 task

| 구분 | 이 논문에서의 내용 |
|---|---|
| 문제 | 의료 LLM의 hidden activation에는 환자별 임상 상태가 있을 수 있지만, 사용자는 그것을 직접 읽을 수 없다. Visible CoT는 그 상태를 완전히 또는 충실하게 드러낸다는 보장이 없다. |
| 방법 | Medical-NLA: fixed target medical LLM의 activation을 받아 자연어 clinical-state text로 변환하는 activation-conditioned verbalizer. |
| 논문 주장 | Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 자연어로 **faithfully verbalize**한다. CoT 대비 우위와 auditing utility는 이 중심 주장에서 파생되는 별도 검증 가설이다. |
| task | 위 주장을 판별할 수 있도록 고정한 입력, 출력, reference, metric, control의 묶음. |
| 표 | 각 task에서 어떤 주장 성분을 검증했는지 보고하는 결과 형식. |

따라서 "Medical-NLA가 내부 신호를 자연어로 바꾼다"는 방법 설명이고,
"다른 LLM output보다 더 많은 설명을 제공한다"는 검증할 가설이다. 둘 자체가 task는 아니다.

## 일반 도메인에서 AV/NLA를 만드는 이유

표를 고르기 전에 원 방법 계열이 해결하려는 문제를 고정한다. 일반 도메인의 AV/NLA는
더 좋은 정답이나 더 좋은 CoT를 직접 생성하는 solver가 아니다. 핵심 목적은 **모델이 출력으로
말하지 않은 hidden state를 사람이 감사할 수 있는 자연어 interface로 바꾸는 것**이다.

### AV와 NLA의 역할

```text
target activation h
    -> activation verbalizer AV
    -> natural-language state report z
    -> activation reconstructor AR
    -> reconstructed activation h_hat
```

- **AV**는 activation을 사람이 읽을 수 있는 자연어로 바꾸는 실제 판독기다.
- **AR**은 `z`만 보고 원 activation을 복원하는 학습용 측정기다.
- **NLA**는 AV와 AR을 묶어 `h -> z -> h_hat` round trip을 학습하는 체계다. 정답 설명이
  없어도 `z`가 activation 정보를 보존하도록 reconstruction reward를 제공하는 것이 AR의
  목적이다.
- 따라서 최종 사용자가 보는 것은 AV의 state report이며, AR은 그 state report가 activation을
  무시한 일반 문장으로 퇴화하지 않도록 하는 학습 신호다. 단, reconstruction이 높다는 사실만으로
  `z`의 인간 해석이나 사실성이 자동 보장되지는 않는다.

원 NLA 연구는 이 구조를 `activation -> text explanation -> reconstructed activation`으로
정의한다: <https://transformer-circuits.pub/2026/nla/index.html>.

### 일반 도메인의 주된 사용처

| 목적 | 묻는 질문 | 대표 예시 |
|---|---|---|
| 출력되지 않은 상태 읽기 | 모델이 말하지 않았지만 내부적으로 무엇을 알고, 가정하고, 계획하는가? | 답변 전에 계획한 rhyme, 사용자에 대한 추정, future response propensity |
| 안전·정렬 감사 | 출력에서는 숨긴 목표, 평가 인식, secret knowledge, misalignment가 있는가? | evaluation awareness, 탐지 회피 동기, fine-tuning으로 주입된 hidden objective |
| 오류 원인 조사 | 이상 출력 직전 내부 상태에 어떤 잘못된 가정이나 계획이 있었는가? | 잘못된 언어로 답하는 현상, multi-hop reasoning 오류의 중간 상태 |
| 유연한 activation 질의 | 사전 지정 label 하나가 아니라 자연어로 여러 질문을 할 수 있는가? | "무엇을 알고 있는가?", "어떤 목표가 있는가?", "다음 행동 성향은 무엇인가?" |
| 모델 비교·diffing | 원 모델과 fine-tuned 모델의 activation 차이가 어떤 의미인가? | emergent misalignment를 유발한 propensity 설명 |
| 후속 steering/control | 읽은 상태를 이용해 편향이나 오류 행동을 줄일 수 있는가? | debiasing, sentiment/persona control, reasoning self-correction |

이 계열의 대표적인 위치는 다음과 같다.

- [Patchscopes](https://research.google/pubs/patchscopes-a-unifying-framework-for-inspecting-hidden-representations-of-language-models/):
  hidden representation을 적절한 target context에 patch해 자연어 continuation으로 검사한다.
  harmful belief extraction과 multi-hop reasoning self-correction을 응용으로 보인다.
- [LatentQA](https://arxiv.org/abs/2412.08686): activation과 자연어 질문을 함께 받아 답하는
  open-ended activation QA를 정의한다. 판독뿐 아니라 differentiable loss를 통한 steering도
  보인다.
- [Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/): activation을
  text와 나란한 추가 input modality로 취급하고, secret knowledge와 fine-tuned misalignment를
  찾는 auditing task에서 평가한다.
- [Natural Language Autoencoders](https://www.anthropic.com/research/natural-language-autoencoders):
  별도 자연어 label 없이 reconstruction을 이용해 activation verbalization을 학습하고,
  evaluation awareness, hidden motivation, 비정상 출력 원인 조사에 적용한다.

공통점은 **좋은 benchmark rationale을 쓰는 것 자체가 최종 목적이 아니라, 관찰되지 않는
모델 상태를 읽어 감사하는 것**이다. 일부 연구가 읽은 상태를 사용해 모델을 교정하지만, 이는
판독기가 유효하다는 것이 확인된 뒤의 downstream application이다.

### Medical-NLA로 옮긴 큰 그림

의료 도메인에서의 직접 대응은 **diagnostic-state auditing**이다.

```text
clinical case X
    -> frozen target medical LLM
    -> patient-specific activation h(X)
    -> Medical-NLA
    -> natural-language diagnostic-state report Z
    -> clinician or fixed auditor
```

`Z`의 이상적인 내용은 gold rationale의 모사가 아니라 target model 내부 상태에 대한 보고다.

1. activation에 표현된 patient findings와 values
2. 현재 activation이 기울어 있는 diagnostic hypothesis 또는 disposition
3. 표현되지 않았거나 약한 근거, 상충하는 근거, 불확실성
4. 최종 출력이나 visible CoT에는 나타나지 않은 환자별 상태

모델이 틀렸다면 Medical-NLA도 전문가 정답 설명을 만들어서는 안 된다. 대신 모델 activation에
실제로 들어 있는 잘못된 finding, 누락, diagnostic direction을 충실하게 드러내야 한다. 따라서
Medical-NLA의 중심 출력은 **expert rationale**보다 **model-state report**라고 부르는 것이
정확하다.

### 논문 task와 표에 주는 수정

이 큰 그림을 따르면 논문의 우선순위는 다음과 같다.

1. **Clinical state-report adequacy:** 보고서가 환자별 임상 내용을 읽을 수 있는 자연어로
   제시하는가?
2. **Activation faithfulness:** 보고서가 의료 상식이나 diagnosis template이 아니라 own
   activation에 의존하는가?
3. **Diagnostic auditing utility:** 이 보고서를 받은 독립 auditor가 CoT만 받은 경우보다
   target model의 진단 오류와 환자별 원인을 더 잘 발견·교정하는가?

기존 의료 rationale 표의 BLEU/ROUGE/METEOR 또는 expert-reference similarity는 1번의
**clinical adequacy sanity check**로는 사용할 수 있다. 그러나 그것만으로 2번 activation
faithfulness를 증명할 수 없으며, 원 NLA 계열의 중심 목적을 직접 평가하지도 않는다.

- CoT가 전문가 설명을 그럴듯하게 모사하면 activation과 무관해도 높은 점수를 받을 수 있다.
- Medical-NLA가 잘못된 model state를 정확히 보고하면 expert rationale과 달라 낮은 점수를
  받을 수 있다.
- 따라서 "CoT보다 더 좋은 reasoning을 생성한다"를 중심 주장으로 두면 structured clinical
  reasoning 방법과 구분이 약해진다.

이 문서의 이후 Table 1 설계는 현재 **activation-augmented rationale utility 후보**로 남겨
두되, final main task로 자동 확정하지 않는다. 중심 주장을 원 NLA의 목적에 맞출 경우 Table 1은
clinical adequacy 보조표로 내리고, Table 2 activation faithfulness와 diagnostic-error auditing을
주표로 올리는 재구성이 더 자연스럽다. 이 우선순위 변경은 별도 사람 결정으로 동결한다.

### 중심 claim의 세 검증 요소 (2026-09-01)

중심 claim을 해석하기 위한 과학적 검증 요소는 다음과 같다. 아래 세 항목은 문서 하단의
교수 회신 기반 최종 Table 1--3 구성을 자동으로 대체하지 않는다. 동일한 mapping `h(X) -> Z`가
무엇을 만족해야 하는지를 분해한 것이며, final table과의 대응은 아래 consistency note에서
명시한다.

| task | 입력과 출력 | 핵심 질문 | primary metric |
|---|---|---|---|
| T1. Medical diagnostic-state decoding | `h(X) -> Z` | activation에 표현된 환자 finding, value, diagnostic disposition을 자연어로 읽는가? | finding precision/recall/F1, value accuracy, diagnostic-disposition accuracy, unsupported-claim rate |
| T2. Causal activation faithfulness | `do(h) -> Z_do` | activation만 바꿨을 때 report가 해당 변화만 선택적으로 따라가는가? | own-vs-shuffled gap, deletion removal/phantom, retained preservation, value replacement/old persistence/clean switch, specificity |
| T3. Diagnostic error auditing | `X + proposed diagnosis + evidence E -> auditor decision` | Medical-NLA report가 CoT보다 target model의 오진과 환자별 원인을 잘 찾게 하는가? | error-detection AUROC/F1, evidence-error localization F1, correction accuracy, false-alarm rate |

T1에서 CoT는 activation decoder가 아니라 **visible-output baseline**이다. Linear probe,
Patchscope, Vanilla NLA, Medical-NLA는 activation 접근 방법이다. 모든 자연어 출력은 동일한
method-blind mapper 또는 동일 fixed auditor로 평가하고, probe는 closed-ontology upper-bound로
분리해 해석한다.

T3의 권장 비교 조건은 다음과 같다.

| fixed auditor에 주는 추가 evidence `E` | 목적 |
|---|---|
| 없음 | case와 proposed diagnosis만으로 가능한 black-box audit |
| target model CoT | visible reasoning baseline |
| Vanilla NLA | 일반-domain activation verbalizer baseline |
| Medical-NLA | proposed medical state report |
| same-stratum shuffled Medical-NLA | 더 긴 의료 text나 diagnosis template 효과 통제 |

과학적 목적상 T3는 independent solver의 일반적인 정답률 향상보다 **diagnostic-error detection
and localization**이 더 직접적이다. 외부 task anchor로는 의료 오류 탐지·수정 형식을 제공하는
[MEDEC](https://arxiv.org/abs/2412.19260)와, DDXPlus/CupCase/MedCase에서 evidence subset
intervention을 사용하는 [Auditing Evidence Use in Medical LLM Diagnosis](https://arxiv.org/abs/2607.20848)을
검토한다. 어느 것도 activation report row를 그대로 제공하지 않으므로, 데이터·target backbone·
auditor를 고정하고 모든 조건을 재실행해야 한다. 교수 회신 기반 최종 Task 2가 일반 diagnosis
utility로 유지된다면 diagnostic-error auditing은 main table이 아니라 후속/appendix 후보로 둔다.

### 이 목적에 맞는 학습 원칙

Gold diagnosis rationale만 SFT하면 activation reader가 아니라 case/diagnosis template을
모사하는 clinical reasoner가 될 수 있다. Medical-NLA 학습 단위는 단순한 의료 문서 수가 아니라
**activation 변화와 target text 변화가 연결된 예제**여야 한다.

1. **Query-conditioned activation QA:** `h + question -> answer` 형식으로 finding, value,
   diagnostic disposition, conflicting/absent evidence를 묻는다. Unconditional 장문 하나보다
   LatentQA/Activation-Oracle 계열의 on-demand 판독에 가깝다.
2. **Controlled clinical pairs:** DDXPlus original, deletion, value edit, same-diagnosis donor,
   negated/absent finding을 사용해 어떤 text 조각이 바뀌어야 하는지 고정한다.
3. **Correct/error balance:** target model이 맞힌 사례와 틀린 사례를 모두 포함한다. Gold answer는
   downstream correctness label로 사용할 수 있지만 AV가 전문가 정답 rationale을 복사하는
   target으로 사용하지 않는다.
4. **Medical self-supervision/AR adaptation:** 의료 activation에서 context/evidence prediction을
   학습하고, reconstruction을 쓸 경우 medical-distribution AR을 먼저 확보한다. 공개 general-domain
   AR의 cosine/FVE만으로 promotion하지 않는다.
5. **Counterfactual objective:** changed evidence의 반응과 unchanged evidence 보존을 loss와
   validation gate에 모두 둔다. 일반 의료 문장 생성 CE만으로는 사례 특이성을 보장하지 않는다.

권장 학습 계보는 `general activation-reader initialization -> medical activation-QA SFT ->
counterfactual consistency/specificity training -> optional medical-AR reconstruction`이다. 이
구조를 사용하면 방법 이름은 strict한 unsupervised NLA인지 query-conditioned Medical Activation
Oracle인지 명확히 구분해 써야 한다. AR round trip을 실제 최종 objective로 사용하지 않는다면
NLA라는 이름만 유지하지 않는다.

## 제안하는 중심 주장

논문 중심 문장은 다음으로 좁힌다. 이것은 방법 정의가 아니라 T1--T3로 검증해야 하는 중심
hypothesis다.

> **Medical-NLA faithfully verbalizes patient-specific clinical state encoded in the hidden
> activations of a medical LLM.**
>
> **Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 자연어로
> faithful하게 언어화한다.**

이 문장은 세 검증 조항을 갖는다.

1. **Decodable clinical state:** 환자별 finding/value/diagnostic disposition을 자연어로
   회수할 수 있는가?
2. **Faithful to activation:** 그 설명이 그럴듯한 의료 상식이 아니라 해당 환자의 activation과
   counterfactual 변화에 실제로 의존하는가?
3. **Useful for auditing:** 그 설명이 visible CoT보다 target model의 오진과 환자별 원인을
   찾는 데 도움이 되는가?

"Medical-NLA가 모델을 faithful하게 만든다"고 주장하지 않는다. Target backbone을 바꾸거나
더 정답으로 만드는 것이 아니라, 이미 존재하는 내부 상태를 읽는 interface를 만드는 것이다.

### 중심 claim 검증 요약 (2026-09-01)

> **Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 자연어로
> faithfully verbalize한다.** 이를 검증하기 위해 (1) diagnostic-state decoding, (2) causal
> activation faithfulness, (3) diagnostic-error auditing utility를 분리한다.

첫 번째 task는 report의 의료 내용을, 두 번째 task는 그 내용의 activation 의존성을, 세 번째
task는 그 state report를 실제 감사에 쓰는 이유를 검사한다. CoT보다 expert rationale과 더
비슷한 문장을 쓰는지는 clinical adequacy 보조평가이며 중심 claim 자체가 아니다.

Table 1의 비교 대상은 표의 **행**이다. Published LLM 행은 benchmark 맥락을 제공할 수 있지만,
primary comparison은 같은 Gemma backbone에서의 text-only/CoT, Vanilla NLA, Medical-NLA다.
같은 answer condition, rationale prompt, decoding budget을 고정해 activation-derived information
외의 차이를 통제한다.

## 중심 task: Medical Activation Verbalization

### 공통 입출력 계약

임상 사례를 `X`, 고정된 target medical LLM의 P0 activation을 `h_P0(X)`, Medical-NLA
출력을 `Z`라고 둔다.

```text
X --target medical LLM, P0--> h_P0(X) --Medical-NLA--> Z
```

- `P0`는 **최종 rationale prompt가 아니라 activation 추출 위치/protocol**이다.
- P0는 사례를 읽은 target model의 마지막 prompt-token state이며, source answer나 source CoT
  token을 activation에 넣지 않는 조기 상태로 고정한다.
- `Z`는 diagnosis-free patient-state text 또는 finding/value claim set이다.
- 각 benchmark에서 source model, P0 prompt, layer, token position은 validation에서 한 번
  정하고 test에서는 변경하지 않는다. 현재 DDXPlus/DiReCT 축적물의 P0와 HS24/HS32는
  개발 자산이지, 새 benchmark test에서 layer를 고르는 근거가 아니다.

이 하나의 mapping이 논문의 중심 task다. 아래 Table 1과 Table 2는 같은 mapping의 서로
다른 성공 조건을 평가한다.

## 기존 Table 1 후보: activation-augmented clinical rationale generation

### 질문

`Z`가 실제로 최종 의료 explanation을 더 좋게 만드는가? 이것이 외부 의료 explanation
benchmark에 놓을 수 있는 task다.

### 기존 표 anchor

가장 직접적인 peer-reviewed anchor는 Chen et al., **"Benchmarking Large Language Models on
Answering and Explaining Challenging Medical Questions"** (NAACL 2025)다.

- 공식 논문: <https://aclanthology.org/2025.naacl-long.182/>
- benchmark: JAMA Clinical Challenge와 Medbullets
- 원 task: clinical case/question과 정답을 바탕으로 rationale을 생성하고 expert explanation과
  비교
- Table 3 계열 metric: ROUGE-L, BERTScore, BLEURT, CTC relevance/preservation/consistency,
  G-Eval relevance/coherence/consistency, BARTScore
- 논문 자체도 automatic metric과 human judgment의 불일치를 보고하므로, 재현 시에는
  공개 평가 protocol과 별도로 제한된 expert/clinician audit을 명시해야 한다.

이 표의 원래 입출력은 `X, Y* -> R`이다. Medical-NLA row는 다음으로 확장한다.

```text
X, Y*, h_P0(X) -> Z -> R
```

여기서 `Y*`는 benchmark의 **explanation-only setting**에서 주어진 target answer다. 따라서
이 task의 answer accuracy는 Medical-NLA의 점수가 아니며, "정답을 안 뒤 얼마나 좋은
rationale을 쓰는가"를 분리해 보는 setting이다.

### 권장 실행 구조

```text
case X
  └─ frozen target model + P0 -> h_P0(X)
       └─ Medical-NLA -> patient-state text Z

case X + benchmark answer Y* + Z
  └─ frozen rationale actor -> final clinical rationale R
```

동일한 rationale actor, prompt template, temperature, maximum tokens, answer `Y*`를 모든 행에
고정한다. 바뀌는 것은 actor에 주는 추가 evidence뿐이다.

| Table 1 row | rationale actor의 추가 evidence |
|---|---|
| Text-only | 없음: `X + Y*` |
| Source CoT | target model의 visible CoT |
| Vanilla NLA | frozen vanilla activation verbalization `Z_vanilla` |
| Medical-AV SFT | supervised activation verbalization `Z_sft` |
| Medical-NLA | proposed `Z_medical` |
| Shuffled Medical-NLA control | 같은 stratum의 다른 사례 activation에서 만든 `Z_shuffle` |

같은-backbone text-only row와 shuffled-activation row가 없으면, 더 좋은 base model 또는 더 긴
prompt가 이긴 것인지 activation evidence가 이긴 것인지 분리할 수 없다.

이 설계는 Chen et al.의 표를 숫자까지 그대로 복사하는 것이 아니다. 원래 explanation-only
계약 `X + Y* -> R`를 `X + Y* + activation-derived state Z -> R`로 확장하고, published model
결과는 재현 가능한 경우에만 reference rows로 둔다. Medical-NLA의 핵심 비교는 동일 Gemma
backbone의 ablation rows다.

### Table 1이 말하는 것과 말하지 않는 것

- 높은 explanation metric: `Z`가 rationale 생성에 임상적으로 유용한 evidence일 수 있다.
- 낮은 metric: Medical-NLA가 좋은 natural-language interface라는 주장을 지지하지 못한다.
- 높은 Table 1 점수만으로 activation faithfulness는 증명되지 않는다. CoT나 상식 문장이
  충분히 좋은 rationale을 만들 수 있기 때문이다.
- 현재 DiReCT `Obs*`/`Exp*` 표는 유용한 development evidence이지만, 이 외부 task Table 1을
  대체하지 않는다. DiReCT는 physician observation/rationale tree라는 다른 schema를 쓴다.

## Table 2: patient-specific activation faithfulness

### 질문

`Z`가 해당 환자의 `h_P0(X)`를 읽는가, 아니면 진단군의 전형적 문장을 생성하는가?

### 기존 표 anchor와 한계

완전히 동일한 **의료** 표는 현재 확인하지 못했다. 가장 가까운 두 2026 연구는 evaluation
원리를 제공하지만, 의료 benchmark를 제공하지는 않는다.

| 연구 | 원 task | 가져올 수 있는 것 | 그대로 가져올 수 없는 것 |
|---|---|---|---|
| [PRISM](https://arxiv.org/abs/2606.09563), 2026 preprint | activation에서 active instruction set 복원 | set recovery, coverage, hallucination, text-only/activation decoder 비교 표 형식 | instruction label과 의료 finding/value label은 다름; 의료 결과를 재실행해야 함 |
| [CHIVE](https://arxiv.org/abs/2608.16747), 2026 preprint | tool output이 counterfactual behavior 예측을 개선하는지 | transcript-only, NLA, activation oracle, SAE 비교와 counterfactual utility 원리 | 실제 환자 observation/finding ground truth가 없음 |

따라서 Table 2는 PRISM의 **set retrieval**과 CHIVE의 **counterfactual control**을 DDXPlus의
structured evidence에 적용한 새 의료 task로 명시해야 한다. 기존 논문의 숫자에 Medical-NLA
행 하나를 붙이는 표가 아니다.

### 공통 평가 단위

모든 방법의 출력을 공통 `(evidence_id, value_id)` claim set으로 정규화한다.

| 방법군 | 공통 claim set으로의 변환 |
|---|---|
| Linear probe | threshold를 넘긴 label/value를 직접 claim으로 사용 |
| SAE | train-only feature-to-finding mapping을 거친 claim set |
| Patchscope | fixed parser/mapper가 continuation에서 읽은 claim set |
| Probe-guided structured reader | probe claim set을 train-only lexicon으로 결정론적으로 렌더링 |
| Vanilla NLA / Medical-NLA | frozen method-blind mapper가 open text를 ontology claim으로 정규화 |

Structured reader는 독립 decoder가 아니라 **closed probe + deterministic renderer**다. 이 행은
"probe score를 자연어 형식으로 보여줄 수 있는가"의 upper-bound/monitor control이며,
open-ended Medical-NLA와 같은 학습 방법으로 부르지 않는다.

### 필요한 지표

| panel | 지표 | 판별하는 실패 |
|---|---|---|
| Static recovery | finding coverage/F1, unsupported-claim rate, conditional value accuracy | gold finding을 안 읽거나 환각하는가 |
| Same-diagnosis control | own activation vs 같은 진단 다른 환자 activation의 recovery gap | 진단 전형 문장만 말하는가 |
| Cue deletion | deleted claim removal, deletion phantom, retained-finding preservation | activation에서 삭제 cue만 선택적으로 잊는가 |
| Value edit | replacement hit, old-value persistence, clean value switch | value를 실제로 업데이트하는가 |

Table 2의 성공 기준은 "probe보다 모든 metric에서 이김"이 아니다. Probe는 정해진 ontology의
닫힌 분류기로 설계되어 static label F1에서 우세할 수 있다. Medical-NLA의 필요 주장은
**열린 자연어를 생성하면서도** 최소한 same-diagnosis control과 counterfactual specificity를
통과하고, 공통 claim metric에서 probe/SAE류와 비교 가능한 수준의 fidelity를 보인다는 것이다.

### 현재 결과의 위치

현재 frozen HS24 probe와 structured reader의 DDXPlus locked 결과는 Table 2의 control/
upper-bound 자산이다. 예를 들어 structured reader는 finding F1 `.9587`, deletion removal
`.6407`, retained preservation `.9987`, clean value switch `.0804`를 기록했다. 이것은
Medical-NLA 성공 결과가 아니다. Vanilla NLA semantic row는 ontology claim `0`으로 확인되어
같은 task에서 음성 baseline으로만 사용 가능하다.

현재 D10/D20 SFT/ranking/anchor 실패는 Table 2의 Medical-NLA 행으로 test를 열지 않았으며,
main-table comparison result로 바꾸어 적지 않는다. 이것들은 method-development appendix의
promotion-failure evidence다.

## 기존 Table 3 후보: explanation의 downstream solver utility

Table 1은 explanation quality, Table 2는 activation faithfulness다. "설명을 다른 clinical
solver에게 주었을 때 실제 의사결정이 좋아지는가"까지 주장하려면 세 번째 task가 필요하다.

```text
case X + additional evidence E -> independent solver -> diagnosis / answer A
```

`E`를 없음, visible CoT, source CoT, probe labels, shuffled Medical-NLA, own Medical-NLA로
바꾸고 answer accuracy와 expert reasoning-step coverage를 비교한다. 이 task의 가장 중요한
규칙은 `E`와 target model activation을 **gold answer를 보기 전에** 생성하는 것이다.

MedThink-Bench (500 high-difficulty questions, expert step references)는 explanation step
coverage의 candidate anchor다: <https://www.nature.com/articles/s41746-025-02208-7>. 다만
현재는 Medical-NLA method도, benchmark conversion도, solver protocol도 동결되지 않았다.
따라서 Table 3은 main claim에 필요하다면 새로 설계할 task이지, 현재 표를 빈칸으로 유지할
이유는 아니다. Table 1 + Table 2만으로도 "clinically informative and activation-faithful
verbalization"이라는 더 좁은 논문 주장은 가능하다.

## DiReCT의 올바른 역할과 citation audit

DiReCT는 2024 NeurIPS Datasets and Benchmarks Track benchmark이며, physician observation,
rationale, diagnosis tree를 제공한다.

- 원 논문: <https://proceedings.neurips.cc/paper_files/paper/2024/file/892850bf793e03b5c410dfd9425b94c8-Paper-Datasets_and_Benchmarks_Track.pdf>
- 현재 우리 사용: PDD-disjoint development/locked clinical-alignment audit.
- 강점: `Obs*`/`Exp*`가 자유문 clinical explanation을 observation/rationale structure와
  비교할 수 있게 한다.
- 한계: activation ground truth나 counterfactual activation pair를 주지 않으며,
  ChallengeClinicalQA와 같은 공개 explanation table의 직접 대체물이 아니다.

"DiReCT를 citation한 2025--2026 논문이 어떤 표를 썼는가"는 별도 literature audit으로
완료해야 한다. 그 audit은 논문마다 (a) DiReCT 전체 benchmark인지 custom split인지,
(b) `Accdiag`만 쓰는지 `Obs*`/`Exp*`도 쓰는지, (c) target answer exposure가 있는지,
(d) original evaluator인지 새 judge인지를 표로 기록한다. 현재 이 citation census가 완료되기
전에는 "DiReCT 관행이 이 표 구조를 요구한다"고 쓰지 않는다.

## 2025--2026 literature/benchmark audit: 현재 결론

| 필요 | 가장 직접적인 근거 | 논문 설계에 주는 결론 |
|---|---|---|
| 의료 rationale quality의 기존 표 | ChallengeClinicalQA, NAACL 2025 | Table 1은 동일 benchmark/protocol을 재현하고 Medical-NLA-assisted row를 추가하는 경로가 가장 보수적이다. |
| 의료 reasoning-step completeness | MedThink-Bench, 2026 | Table 1의 보조 지표 또는 Table 3 utility task 후보. |
| activation-to-language set fidelity | PRISM, 2026 preprint | Table 2의 coverage/hallucination/set-retrieval schema를 제공하지만 medical row는 새로 측정해야 한다. |
| activation tool의 utility/counterfactual test | CHIVE, 2026 preprint | Table 3 또는 Table 2 counterfactual panel의 control 철학을 제공한다. |
| 의료 hidden-state intervention | SAE/probing 의료 연구들 | 의료 activation을 분류/steer하는 baseline은 있으나, probe+SAE+Patchscope+NLA를 같은 환자별 자연어 verbalization task에서 비교한 2025--2026 표는 현재 확인하지 못했다. |

따라서 교수님 요구를 엄밀히 충족하는 범위는 다음과 같다.

1. **Table 1:** 기존 peer-reviewed 의료 explanation task를 재현해 Medical-NLA row를 추가할 수 있다.
2. **Table 2:** 기존 activation literature의 평가 원리를 가져오되, 의료 activation verbalization
   benchmark로 새로 정의하고 모든 baseline을 같은 data/backbone에서 재실행해야 한다.
3. **Table 3:** downstream utility까지 주장하려면 별도 solver task를 구축해야 하며, 현재 결과만으로는
   만들 수 없다.

## 논문이 피해야 할 주장

1. 현재 DDXPlus probe F1을 Medical-NLA의 explanation quality로 부르지 않는다.
2. DiReCT `Obs*`/`Exp*`를 activation faithfulness로 부르지 않는다.
3. Gold answer를 rationale actor에 준 Table 1에서 Medical-NLA가 diagnosis accuracy를 개선했다고
   주장하지 않는다.
4. Published model numbers와 Medical-NLA row를 같은 표에 놓더라도 model, prompt, evaluator가
   다르면 직접 SOTA 비교라고 쓰지 않는다. 동일 Gemma text-only/CoT control이 primary다.
5. 현재 promotion을 통과하지 못한 SFT/ranking checkpoint에 locked-test Medical-NLA 행을 만들지 않는다.

## 다음 결정과 실행 순서

### Decision A: 논문 범위

사람이 다음 중 하나를 선택해야 한다.

| 선택 | 주장 | 필요한 main results |
|---|---|---|
| A. Two-table core paper | activation verbalization의 clinical quality + faithfulness | Table 1 ChallengeClinicalQA 계열 + Table 2 DDXPlus/DiReCT faithfulness |
| B. Three-task paper | 위 두 주장 + independent solver utility | A + Table 3 MedThink/medical QA decision-support |
| C. Development/negative-results paper | 현재 objective failure와 probe-reader boundary | 현재 DiReCT/DDXPlus 표 중심, Medical-NLA 성공 주장은 하지 않음 |

현재 사용자 목표인 "Medical-NLA를 성공시켜 논문 방법으로 제시"는 A가 최소 범위이며,
B는 더 강한 주장 대신 새로운 benchmark work가 필요하다.

### Decision B: Table 1 actor contract

Table 1 실행 전 다음을 frozen protocol로 기록한다.

1. Target medical LLM, P0 prompt, layer, token position.
2. Medical-NLA output schema `Z`와 max length.
3. Rationale actor model, system/user prompt, decoding params.
4. `Y*`를 주는 explanation-only setting인지, `Y*` 없이 answer+rationale를 내는 utility setting인지.
5. Same-backbone text-only, CoT, shuffled activation controls.
6. Published benchmark evaluator 재현 범위와 human/clinician audit 범위.

### Decision C: Table 2 baseline feasibility

Probe, structured reader, vanilla NLA는 이미 일부 자산이 있다. SAE, Patchscope, LatentQA/
Activation-Oracle 계열은 같은 target backbone, same P0, same ontology mapper 아래에서 새로
실행할 수 있을 때만 main-table row가 된다. 구현 또는 checkpoint가 없으면 빈 행으로 두지 않고
관련 work/appendix comparison으로 내린다.

## 판정

현재 논문의 task를 단순히 "Medical-NLA"로 두면 안 된다. 권장되는 중심 task는
**Medical Activation Verbalization**이며, 최소 두 가지 관측 가능한 성공 조건은
**activation-augmented clinical rationale quality**와 **patient-specific activation faithfulness**다.

Table 1은 ChallengeClinicalQA라는 기존 의료 explanation table을 확장하는 경로로 설계할 수 있다.
Table 2는 기존 의료 표를 복사하는 것이 아니라 PRISM/CHIVE의 activation evaluation 원리를
DDXPlus/DiReCT에 맞게 구현한 새로운 medical faithfulness benchmark로 명시해야 한다.

이 문서가 승인되기 전에는 기존 DiReCT locked Table 1A/1B/2를 새 논문의 main table이라고
부르지 않는다. 그것들은 current baseline/development evidence이며, 새 task protocol을 결정한
뒤 재사용 범위를 명시적으로 정한다.

## 교수 회신 반영: 3-task 확정 구조 (2026-08-31, Claude 작성)

교수 회신 요지: ① Table 1 방향 승인, 단 25/26년 baseline이 **이미 채워진 더 최신
표**를 가져올 것(우리가 최신 모델들을 직접 돌려 채우지 말 것). ② Task/표는 3개 —
예컨대 임상 설명(rationale 생성) task와 진단 task를 분리. ③ Table 2(faithfulness)는
**우리가 벤치마크를 직접 새로 구축해도 됨**. 즉 기존 논문 표 2개 + 신규 제안 표 1개.

### 확정 구조

| Task | 출처 | Anchor | 우리 행 |
|---|---|---|---|
| 1. 임상 rationale 생성 | 기존 표 재현 | **MedThink-Bench** (npj Digital Medicine 2025) | Gemma CoT / +Z(vanilla) / +Z(Medical-NLA) / +Z(shuffled) |
| 2. 진단 | 기존 표 재현 | **DiagnosisArena** (ACL Findings 2026), 보조 MedXpertQA (ICML 2025) | 동일 ablation, 단 정답 비노출 계약 |
| 3. Activation faithfulness | **신규 구축 (교수 허가)** | 가칭 **MAV-Bench** — PRISM set-retrieval + CHIVE counterfactual 원리를 DDXPlus에 구현 | probe / reader / vanilla / (SAE) / (LatentQA류) / Medical-NLA |

### Task 1 anchor 교체 근거 (검증 완료)

ChallengeClinicalQA(NAACL 2025)의 Table 3 모델 행은 GPT-3.5/GPT-4/PaLM 2/Llama 2/
Llama 3/MedAlpaca/Meerkat — 전부 2023–24년 모델이라 교수 조건에 미달한다(원표
스크린샷으로 확인). MedThink-Bench는 12개 모델이 기평가돼 있고 그중 o3,
Gemini-2.5-Flash, DeepSeek-R1, Qwen3-32B, MedGemma-27B, HuatuoGPT-o1-70B,
Llama-3.3-70B 등 **2025년 모델 행이 이미 채워져 있다.** 지표는 answer accuracy +
LLM-w/o-Rationale + LLM-w-Rationale(전문가 단계 coverage, 전문가 상관 .87)로
reference 기반 설명 평가라는 표 성격도 동일하다. JAMA 콘텐츠 라이선스 문제도
함께 회피된다.

- 공식: <https://www.nature.com/articles/s41746-025-02208-7>,
  <https://github.com/plusnli/MedThink-Bench>
- ChallengeClinicalQA는 related work와 protocol 선례 인용으로 유지한다.

### 2026년 모델 행 현황 (검증 완료)

Rationale 생성 계열 표에 2026 frontier 모델(GPT-5 계열, Gemini 3, Opus 4.x)이
채워진 발표 논문은 현재 없다. 26년 모델 행을 가진 의료 벤치마크는
PhysAssistBench(2606.18613, 대화형 EHR 보조)와 stress-testing(2606.07929, 안전성)
뿐이며 둘 다 task가 다르다. 따라서 **"26년 baseline" 충족 경로는 MedThink-Bench
공개 evaluator로 26년 모델 1행(예: GPT-5 계열 또는 Gemini 3)을 동일 프로토콜로
추가하는 것**이다(500문항 API 소액 작업). 이는 표 전체 재구축이 아니라 공개 표에
행을 더하는 최소 보강이며, 교수의 "직접 채우지 말라" 취지(표 재구축 금지)와
"26년 baseline 포함"을 동시에 만족한다. 진단 task도 동일 논리로 DiagnosisArena
published 행 + MedXpertQA live leaderboard의 최신 갱신분을 확인한다.

### Task 2 실행 계약 요점

고정 solver(Gemma backbone) 하나가 동일 문항을 추가 evidence 조건만 바꿔 푼다:
text-only / +CoT / +Z(Medical-NLA) / +Z_shuffled. 네 행은 추가 evidence 외 전부
동일하므로, +Z만 오르고 +Z_shuffled가 오르지 않을 때 개선을 환자별 activation
정보에 귀속할 수 있다. Published 행(o1/R1/QwQ 등)은 난이도 맥락으로만 두고 직접
SOTA 비교라 쓰지 않는다("피해야 할 주장 4"). Task 1과 달리 **gold answer 비노출
상태에서 Z를 생성**한다 — 두 task의 계약 차이가 교수가 요구한 task 분리의 실질이다.

### Task 3: MAV-Bench 구축 범위

기존 locked 자산이 벤치마크 구축물의 골격이다: 4-패널(static recovery /
same-diagnosis control / cue deletion / value edit) × frozen split × semantic
mapper(G1–G4 통과, hash 동결) × counterfactual activation family(원본 4,543 /
삭제 4,540 / value-edit 942, locked 채점 완료). 남은 구축 작업은 baseline 행 확충
(SAE 1개 학습, LatentQA/AO류 1개 실행 — Decision C 규칙대로 실행 불가하면 행을
비우지 않고 appendix로 내림)과 패키징이다.

- **공개 범위**: release 본체는 재배포 가능한 DDXPlus 파생만(activation,
  counterfactual pair, mapper, 채점 코드). DiReCT 패널은 DUA 보유자용 확장으로
  분리 — 기존 반출 금지 규칙과 일치.
- **선택 패널(시간 허용 시)**: CoT 조작 강건성(유도 힌트/sycophancy 조건 activation
  추가 추출; MedOmni-45 원리 참조) — 독립 4번째 task로 세우지 않고 MAV-Bench의
  패널로 흡수해 표 3개 제약을 지킨다. CHIVE의 "activation 도구가 transcript를 못
  이겼다" null을 의료에서 검증/반전하는 지점이 정확히 이 패널이다.

### 우선순위와 fallback

1. Task 1–3이 주장의 필요충분이다. 선택 패널은 일정이 밀리면 첫 번째로 잘라
   후속 논문으로 넘긴다.
2. Task 1/2 파이프라인(actor contract, 채점기 재현, text-only/CoT/vanilla/shuffled
   행)은 **Medical-NLA 성공 없이도 지금 구축·검증 가능**하다. 성공한 Medical-NLA를
   만드는 방법 개발(supervised prefix mapper 등)은 별개 트랙이며, 끝내 실패하면
   Decision A의 C(개발·음성 결과 논문)로 내려간다 — 기존 locked 자산은 그 경우의
   주표로 그대로 복원된다.
3. Walk the Talk(ICLR 2025)은 이번 구조에서 anchor로 채택하지 않는다 — F 지표가
   답변 모델의 자기 설명에 정의돼 있어 행 의미가 바뀌기 때문. Task 3 counterfactual
   패널의 방법론 인용으로 유지한다.
4. MedThink-Bench 연도 표기는 2025로 통일한다(s41746-025-02208).

### 즉시 실행 가능한 preflight (방법 개발과 병렬)

1. MedThink-Bench evaluator/데이터 다운로드, published 수치 재현 확인(judge 모델
   포함), split·프롬프트 hash 동결.
2. DiagnosisArena 데이터/평가 프로토콜 다운로드와 동일 확인.
3. Task 1/2 공용 actor contract 초안(Decision B 6항목) 작성.
4. MAV-Bench 패키징 명세(공개 파일 목록, 라이선스, DiReCT 분리) 초안.

이 절은 교수 회신의 실행 번역이며, Decision A는 사실상 A(two-table core)에서
**3-task 구조로 갱신**된 것으로 본다. 최종 동결은 사람 승인으로 한다.

## 중심 claim과 교수 회신 3-task 구조의 consistency note (2026-09-01)

현재 논문의 중심 claim은 다음으로 고정 후보를 둔다.

> **Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 자연어로
> faithful하게 언어화한다.**

여기서 `faithful`은 방법 이름이나 학습 objective만으로 보장되는 수식어가 아니다. 중심
faithfulness claim은 다음 1--2를 만족했을 때 주장할 수 있고, 3은 그 faithful report를 실제로
쓸 이유를 보이는 별도 utility claim이다.

1. report가 환자별 임상 상태를 실제로 회수한다(static state decoding).
2. activation counterfactual에 따라 해당 내용만 선택적으로 변한다(causal faithfulness).
3. visible CoT 또는 text-only 정보에 없는 감사 가치를 제공한다(differential utility;
   faithfulness의 필요조건은 아님).

교수 회신 기반 최종 표와의 대응은 다음과 같다.

| 교수 회신 table | 중심 claim에서의 역할 | 단독으로 증명하지 못하는 것 |
|---|---|---|
| Task 1 MedThink rationale | state report가 임상적으로 유용한 explanation evidence인지 보는 external adequacy/utility | activation을 실제로 읽었다는 인과적 faithfulness |
| Task 2 DiagnosisArena diagnosis | `Z`가 fixed solver의 진단에 유용한지 보는 downstream utility | `Z` 내용 자체가 true model state인지 여부 |
| Task 3 MAV-Bench | static recovery와 intervention consistency로 state decoding + activation faithfulness를 직접 검증 | 실제 사용자/auditor에게 유용한지 여부 |

따라서 세 표는 서로 대체 관계가 아니다. Task 3가 중심 claim을 직접 검증하고, Task 1과 Task 2가
그 faithful state report가 readable하고 useful한 이유를 외부 의료 task에서 보강한다. Task 1에서
CoT보다 점수가 높더라도 MAV-Bench를 통과하지 못하면 Medical-NLA의 faithfulness 성공으로
판정하지 않는다. 반대로 MAV-Bench를 통과하더라도 Task 1/2가 낮으면 faithful하지만 현재 형식의
clinical utility가 제한된 도구로 해석한다.

Diagnostic-error auditing은 NLA의 일반-domain auditing 목적을 의료에 가장 직접적으로 옮긴
추가 후보지만, 현재 교수 회신의 세 표에는 포함되지 않았다. DiagnosisArena utility와 교체하거나
네 번째 task로 추가하려면 별도 사람 승인이 필요하다.

## MAV-Bench 패널 구조: 왜 Part A/B는 task 두 개가 아닌가 (2026-09-01, Claude)

사람 질문 — "counterfactual faithfulness(Part A)와 manipulation audit(Part B)는 task
두 개가 되어야 하는 것 아닌가" — 에 대한 기록.

**Task 경계는 개입 종류가 아니라 입출력 계약이다.** 이 문서의 task 정의("고정한 입력,
출력, reference, metric, control의 묶음")를 적용하면 두 패널은 같은 task다:

| | Part A (counterfactual faithfulness) | Part B (manipulation audit) |
|---|---|---|
| 입력 | activation `h` | activation `h` |
| 출력 | state report `Z` → frozen mapper claim set | 동일 |
| 채점기 | frozen semantic mapper | 동일 |
| 질문 | 알려진 evidence 개입을 report가 선택적으로 따라가는가 | 임상 evidence는 고정한 채 문맥 압력이 만든 내부 diagnostic-state 변화를 report가 추적하는가 |
| **개입** | 사례 내용을 바꿈 (cue 삭제 / value 교체) | 임상 evidence는 고정, 문맥 압력 추가 (유도 힌트 / sycophancy); activation은 달라질 수 있음 |

두 패널은 같은 intervention-consistency task의 보완 축이다.

- **Part A:** cue/value가 바뀌면 해당 report component는 바뀌고, untouched finding은
  보존되어야 한다.
- **Part B:** patient facts는 그대로 보존되어야 한다. 그러나 hint/sycophancy가 target model의
  activation과 diagnostic disposition을 실제로 바꿨다면 faithful report의 disposition도 그
  변화를 보여야 한다. 따라서 `report 전체가 안 바뀌어야 한다`는 gate는 사용하지 않는다.

서로가 invariant와 variant component를 나누어 검사하므로 한 벤치마크의 두 panel로 두는 것은
타당하다. 선례도 동일하다 — perturbation 계열 벤치마크는 개입 종류별로 task를 쪼개지 않고
(ImageNet-C 15 corruption = 한 벤치마크, PRISM의 정상/injection/hidden-objective = 한 표),
우리 현행 Table 3도 이미 Panel A(static)/Panel B(counterfactual) 구조다.

**교수 제약과의 정합**: 표 3개 = 기존 2 + 신규 1이 유지된다. Part B를 독립 task로
승격하면 신규 벤치마크가 2개가 되어 회신 구조를 벗어난다. 교수가 B를 별도 표로 보고
싶어 하면 그때 패널을 표로 승격하면 되는 표현 층위의 결정이며, 지금의 task 구조
결정과는 독립이다.

**T3(diagnostic-error auditing)와의 관계**: Part B는 별도 auditor가 없으므로 T3의 최소
구현이 아니다. Part B가 직접 측정하는 것은 T2 causal activation faithfulness의 manipulation
panel이다. 숨은 요인을 실험적으로 주입해 ground truth를 아는 것은 이후 T3를 설계할 때 유용한
선행 자산이지만, diagnostic-error auditing이라 부르려면 fixed auditor가 report를 받아 오류
여부, 원인, evidence localization, correction을 판정해야 한다. MEDEC류 완전한 error-auditing
task는 위 절대로 별도 사람 승인이 필요한 4번째 후보로 남는다.

**채점기 범위**: 현재 frozen DDXPlus semantic mapper는 finding/value ontology를 채점한다.
따라서 그대로 측정 가능한 Part B 지표는 patient-fact preservation과 ontology 안 claim의
변화다. Diagnostic disposition, hint influence, sycophancy, manipulation awareness를 채점하려면
별도의 frozen diagnostic-state ontology/mapper 또는 fixed auditor가 필요하다. 이를 만들지
않고 기존 mapper만 사용하면 Part B를 full manipulation audit이라고 주장하지 않는다.

**실행 순서**: 구조상 한 task지만 비용이 다르다. Part A는 locked 자산 재사용으로 거의
완성이므로 MAV-Bench v1 본체로 먼저 패키징한다. Part B는 (1) 조작 조건과 target-model
diagnostic shift reference 생성, (2) activation 추출, (3) invariant patient facts와 variant
diagnostic disposition을 분리해 채점할 protocol을 먼저 동결한 뒤 추가한다. 이 protocol을
완성하지 못하면 Part B를 첫 삭제 후보로 두는 기존 우선순위 규칙을 따른다.
