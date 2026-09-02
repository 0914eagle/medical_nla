# Medical-NLA 9월 3일 발표용 논문형 서사

이 문서는 슬라이드 문구 모음이 아니라, 9월 3일 미팅에서 논문의 문제의식부터
Related Work, 방법, 세 실험 task와 예상 표까지 한 흐름으로 설명하기 위한 발표 원고다.
현재 동결된 결과와 아직 성공하지 않은 방법을 구분하며, Medical-NLA의 성공을
가정한 최종 논문 구조와 실제 남은 구현 범위를 동시에 기록한다.

## 발표에서 먼저 말할 한 문장

> **Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 사람이
> 읽을 수 있는 자연어로 faithful하게 언어화하는 activation-conditioned reader다.**

이 논문의 목적은 Medical-NLA가 backbone의 진단 정확도나 reasoning 능력을 직접
향상시킨다고 주장하는 것이 아니다. 목표는 모델이 이미 내부적으로 표현하고 있는
finding, value, diagnostic disposition을 자연어 state report로 드러내고, 그 보고가
그럴듯한 의료 문장을 생성한 것이 아니라 실제 환자 activation에서 나온 것인지
검증하는 것이다.

## 1. Introduction

### 1.1 문제의식

최근 의료 LLM은 의료 질의응답과 진단 benchmark에서 높은 정확도를 보인다. 그러나
최종 답이 맞는 것과, 모델이 환자에게 어떤 임상 근거를 실제로 사용했는지는 같은
문제가 아니다. 모델은 맞는 답과 함께 잘못된 근거를 제시할 수 있고, 틀린 답을
그럴듯한 임상 서술로 정당화할 수도 있다. 의료 환경에서 설명은 단순한 부가 출력이
아니라 사용자가 모델의 판단 근거와 오류 원인을 검토하는 인터페이스이므로, 설명의
faithfulness를 별도로 확인해야 한다.

가장 흔한 설명 인터페이스는 Chain-of-Thought(CoT)다. 하지만 CoT는 모델이 이미 수행한
내부 계산을 그대로 전사한 기록이 아니라, 출력 시점에 생성되는 또 하나의 텍스트다.
[Right Diagnoses, Decorative Reasoning](https://arxiv.org/abs/2608.24790)은 의료 입력을
임상적으로 조작해도 visible chain과 최종 답이 함께 반응하지 않는 현상을 perturbation
audit으로 측정한다. [Better Accuracies, Worse Reasoning](https://arxiv.org/abs/2605.28301)은
CoT distillation 이후 정답 정확도는 올라가지만 reasoning segment의 의료 오류율은 오히려
증가할 수 있음을 step-level audit으로 보인다. 두 결과는 높은 답 정확도나 그럴듯한 CoT만으로
내부 reasoning의 충실성을 보장할 수 없다는 문제를 제기한다.

### 1.2 기존 내부 관찰 도구의 한계

내부표현을 직접 보는 방법도 이미 존재한다. Linear probe는 activation에서 미리 정한
진단 또는 finding label을 정확히 예측할 수 있다. Logit lens와 tuned lens는 중간 layer가
어떤 vocabulary prediction으로 향하는지를 보여준다. SAE는 dense activation을 sparse
feature로 분해하고, transcoder는 MLP 계산을 feature 단위로 근사한다. 이들은 모두 유용하지만
환자별 상태를 열린 자연어 보고서로 제공하는 데에는 다음과 같은 제약이 있다.

1. Probe는 사전에 정의한 label만 출력한다. ontology 밖의 내용이나 여러 임상 근거의 관계를
   자유롭게 서술하지 못한다.
2. Lens는 주로 vocabulary 또는 next-token distribution을 읽는다. 다중 finding과 수치형
   value가 결합된 환자 상태 보고서와 출력 단위가 다르다.
3. SAE와 transcoder feature는 feature ID 자체로는 임상 의미가 없다. top-activating context와
   별도 검증을 통해 feature-to-concept mapping을 먼저 만들어야 한다.
4. Patchscope와 SelfIE는 hidden representation을 언어 모델의 target context에 주입해
   자연어 continuation으로 해석하지만, target prompt와 patch 위치에 민감하고 임상 finding
   집합을 안정적으로 복원한다는 보장은 없다.

따라서 필요한 것은 닫힌 분류기만도, visible CoT만도 아니다. 의료 LLM의 activation을 직접
입력으로 받아 환자별 임상 상태를 자연어로 기술하고, 그 자연어가 activation 변화에
선택적으로 반응하는지 검증할 수 있는 reader가 필요하다.

### 1.3 제안

우리는 Natural Language Autoencoder의 activation-to-text interface를 의료 도메인으로
확장한 Medical-NLA를 제안한다. 원 [Natural Language Autoencoder](https://transformer-circuits.pub/2026/nla/)
구조에서 activation verbalizer(AV)는 hidden activation을 자연어로 바꾸고, activation
reconstructor(AR)는 생성된 텍스트에서 원 activation을 복원하여 정보 보존 신호를 제공한다.
Medical-NLA의 최종 출력은 정답 rationale이 아니라 다음 내용을 포함할 수 있는
**diagnostic-state report**다.

- activation에 표현된 환자 finding과 value
- 현재 모델 상태가 기울어 있는 diagnostic hypothesis 또는 disposition
- 약하게 표현되거나 누락된 근거
- 상충하는 근거와 불확실성
- visible CoT 또는 최종 답에는 드러나지 않은 환자별 내부 상태

모델이 틀린 경우에도 Medical-NLA가 전문가 정답 rationale을 대신 생성하면 안 된다.
그 경우 Medical-NLA의 역할은 모델 activation에 실제로 존재하는 잘못된 가정, 누락 또는
진단 방향을 충실하게 드러내는 것이다.

### 1.4 Research Questions

본 논문은 하나의 중심 주장과 세 개의 정량 task로 구성한다.

> **중심 주장:** Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를
> 자연어로 faithful하게 언어화한다.

**RQ1. Perturbation responsiveness:** 환자의 특정 임상 근거를 바꾸었을 때 설명은 변경된
근거를 등록하는가? 그리고 설명과 최종 answer가 함께 그 변경을 무시하는 decoupling은
얼마나 자주 발생하는가?

**RQ2. Clinical factuality:** 생성된 설명에서 실제로 발화한 각 임상 문장은 의료적으로
정확한가, 아니면 그럴듯하지만 잘못된 clinical segment를 포함하는가?

**RQ3. Direct activation dependence:** 동일한 진단의 다른 환자 activation으로 교체했을 때
설명의 finding 대응성이 감소하는가? 즉 설명이 환자 text나 질환 template이 아니라 바로 그
환자의 activation에 의존하는가?

RQ1과 RQ2는 둘 다 임상 텍스트를 평가하지만 같은 질문이 아니다. RQ1은 **무엇이 바뀌었을 때
설명이 움직이는지**를 보고, RQ2는 **설명이 실제로 말한 내용이 맞는지**를 본다. RQ3는 입력
문장 변경을 넘어 activation 자체를 통제하므로 중심 faithfulness claim을 가장 직접적으로
검증한다.

### 1.5 Contributions

1. 의료 LLM activation을 환자별 자연어 diagnostic-state report로 변환하는 Medical-NLA를
   정의한다.
2. 기존 의료 CoT audit의 perturbation과 sentence-level factuality 평가를 method-neutral
   explanation audit으로 확장하여 CoT, Vanilla NLA, Medical-NLA를 동일 조건에서 비교한다.
3. DDXPlus의 명시적 finding/value와 same-diagnosis counterfactual을 사용해 activation reader의
   환자 대응성을 직접 측정하는 activation-swapping benchmark를 제안한다.
4. Linear probe, SAE, lens, Patchscope, Vanilla NLA가 각각 무엇을 읽고 무엇을 출력하는지
   분리하고, 자연어 유창성·임상 정확성·activation dependence를 하나의 점수로 혼합하지 않는
   평가 원칙을 제시한다.

## 2. Related Work

### 2.1 Medical CoT faithfulness

의료 CoT 연구는 높은 정답 정확도와 explanation faithfulness가 분리될 수 있음을 보여준다.
Perturbation audit은 입력의 특정 임상 사실을 바꾸고 chain과 answer가 그 변화에 반응하는지
측정한다. Step-level audit은 reasoning을 segment로 분할한 뒤 각 segment가 의료적으로
correct, error, uncertain인지 판정한다. Medical-NLA는 CoT보다 더 좋은 reasoning solver를
만드는 방법이 아니라, 이 연구들이 드러낸 visible explanation의 한계를 hidden-state
readout으로 보완하려는 방법이다.

### 2.2 Probes and sparse feature methods

Linear probe는 frozen activation $h$에 대해

\[
p(y\mid h)=\sigma(Wh+b)
\]

를 학습해 finding이나 diagnosis처럼 미리 정한 label을 예측한다. Probe가 높은 성능을 보이면
그 정보가 activation에서 linearly decodable하다는 근거가 된다. 그러나 probe는 자연어를
생성하지 않으며, 높은 decodability가 backbone이 실제로 그 정보를 인과적으로 사용했다는
뜻도 아니다.

SAE는 activation을 sparse code $z$로 바꾼다. SAE feature를 Task 3에서 finding으로
채점하려면 train-only top-activating contexts 또는 label association으로 각 feature의 임상
의미를 정하고, held-out validation에서 mapping precision/AUROC를 검증해야 한다. SAE 뒤에
supervised multilabel classifier를 붙이면 그 행은 `SAE`가 아니라 `linear probe on SAE codes`로
표기해야 한다. Transcoder는 한 MLP의 입력으로부터 출력을 예측해 계산 경로를 분석하는 도구이므로,
환자 상태 자체를 출력하는 Task 3 주표보다는 Related Work 또는 별도 mechanism analysis에
더 적합하다.

### 2.3 Lens 계열은 무엇을 보는가

`lens`는 하나의 방법 이름이 아니라 중간 hidden state를 사람이 이해할 수 있는 출력 공간으로
투영하는 계열을 가리킨다.

- **Logit lens:** 중간 residual $h_l$을 모델의 최종 normalization/unembedding에 통과시켜
  layer $l$이 어떤 다음 token을 선호하는지 본다. 학습이 없지만 early/middle layer의
  representation이 최종 출력 공간과 정렬되지 않아 해석이 불안정할 수 있다.
- **Tuned lens:** layer별 affine translator를 학습한 뒤 unembedding하여 중간 layer의 latent
  prediction을 더 안정적으로 복원한다. [Tuned Lens](https://arxiv.org/abs/2303.08112)는
  logit lens의 brittle한 직접 투영을 보정한다.
- **Future-lens류:** 현재 hidden state에 미래 token 또는 이후 계산에 관한 정보가 있는지를
  읽는다. Patchscopes 논문에서는 이런 lens도 source representation과 target context를 정하는
  patching 문제의 한 형태로 설명한다.

Lens가 Task 3 주표에 자동으로 들어가지는 않는다. Lens의 기본 출력은 vocabulary distribution이고,
우리 표의 reference는 91개 finding과 value 집합이다. 특정 finding phrase의 token logit을
모아 concept score로 변환할 수는 있지만, multi-token phrase, 동의어, negation, categorical value를
처리하는 추가 mapping이 필요하다. 이 mapping을 validation에서 동결해 구현한다면 appendix의
`tuned-lens concept score` baseline으로 둘 수 있다. 그렇지 않으면 lens는 기능 비교와
layer-selection diagnostic으로만 사용한다.

### 2.4 Activation-to-language methods

[Patchscopes](https://arxiv.org/abs/2401.06102)는 source prompt의 특정 token/layer activation을
target prompt의 token/layer 위치에 삽입한 뒤 target model의 continuation으로 의미를 읽는다.
[SelfIE](https://arxiv.org/abs/2403.10949)도 LLM이 자신의 hidden embedding을 open-world
자연어로 해석하게 한다. [LatentQA](https://proceedings.iclr.cc/paper_files/paper/2026/hash/7f19b99e63762d20e9df91144056f1ee-Abstract-Conference.html)와
[Activation Oracles](https://alignment.anthropic.com/2025/activation-oracles/)는 activation과
자연어 질문을 함께 입력해 activation에 관한 질문에 답하는 supervised reader를 학습한다.
NLA는 정답 annotation 대신 activation-text-activation reconstruction을 중심 학습 신호로
사용한다.

이 방법들은 모두 activation에서 자연어를 얻지만 출력 계약이 다르다. Patchscope는 target
prompt에 의존하는 inference-time intervention이고, LatentQA/AO는 query-conditioned QA이며,
NLA는 activation을 자연어 bottleneck으로 압축한다. Medical-NLA는 이 가운데 NLA의
state-report 형식을 유지하되 의료 finding/value grounding을 추가하려는 시도다.

### 2.5 Patchscope를 우리 비교에 넣을 수 있는가

원칙적으로는 가능하다. Patchscope continuation을 동일한 frozen semantic mapper로 DDXPlus
evidence ID 집합으로 바꾼 뒤, Medical-NLA와 동일한 own/shuffled F1을 계산하면 된다. 그러나
이 경우 다음이 충족돼야 한다.

1. general-domain positive control에서 patch가 의도한 attribute를 복원해야 한다.
2. clinical output contract가 유효해야 한다.
3. own activation 출력이 same-diagnosis donor보다 own finding에 더 잘 대응해야 한다.
4. target prompt, source/target layer와 token 위치를 validation에서 고정해야 한다.

현재 프로젝트의 same-layer Patchscope smoke에서는 general-domain short entity control은
성공했지만, DDXPlus clinical own/donor correspondence는 선택한 HS16/HS24 cell에서 0/5였다.
따라서 현재 Patchscope 결과를 Medical-NLA와 동등한 완성 baseline 행으로 주표에 넣으면 안 된다.
가장 정직한 위치는 Related Work와 feasibility appendix의 음성 결과다.

다만 Patchscope의 **방법론적 원리**는 Task 3에 유용하다. source representation만 바꾸고
나머지 decoding 조건을 고정하여 출력의 원인을 activation에 귀속한다는 intervention 철학을
가져올 수 있다. Task 3 자체는 Patchscope처럼 activation을 target LM의 임의 residual 위치에
삽입하지 않는다. NLA가 원래 입력으로 받는 activation을 own/donor로 교체하는
`activation swapping`이다.

## 3. Methodology

### 3.1 Problem formulation

환자 사례를 $X_i$, frozen target medical LLM을 $M$, 고정된 extraction site에서 얻은
activation을 $h_i$라고 한다.

\[
h_i=M_l(X_i)
\]

Medical-NLA verbalizer $G_\theta$는 activation을 자연어 state report로 바꾼다.

\[
Z_i=G_\theta(h_i)
\]

목표는 $Z_i$가 gold physician rationale을 모사하는 것이 아니라, $h_i$에 표현된 환자별
임상 상태를 읽을 수 있는 언어로 보존하는 것이다. 따라서 좋은 $Z_i$는 다음 세 조건을
함께 만족해야 한다.

1. 발화한 임상 내용이 정확해야 한다.
2. 환자 또는 activation이 바뀌면 관련 내용이 선택적으로 바뀌어야 한다.
3. 같은 질환의 다른 환자 activation보다 own activation에 더 잘 대응해야 한다.

### 3.2 Activation site and output contract

주 실험은 Gemma-3-12B-IT의 P0 시점, hidden-state layer 32, last-token residual을 사용한다.
P0는 target model이 임상 사례를 모두 읽었지만 아직 visible answer나 CoT를 생성하기 전의
상태다. 이 위치를 사용하면 output leakage 없이 입력 사례를 통합한 내부표현을 읽을 수 있다.

Medical-NLA 출력은 free natural-language clinical state report다. 평가를 위해 모든 자연어
출력은 method-blind semantic mapper를 통해 공통 DDXPlus ontology로 변환한다.

\[
Z_i \longrightarrow \widehat F_i,\widehat V_i,\widehat D_i
\]

여기서 $\widehat F_i$는 predicted finding set, $\widehat V_i$는 categorical value,
$\widehat D_i$는 diagnostic disposition이다. Probe는 이미 evidence ID를 출력하므로 mapper를
통과하지 않는다.

### 3.3 Medical-NLA training status

최종 방법의 학습 recipe는 아직 성공으로 동결되지 않았다. 지금까지의 SFT, sequence-level
counterfactual ranking, specificity anchor, soft bottleneck은 모두 validation promotion gate를
통과하지 못했다. 따라서 발표에서는 성공하지 않은 checkpoint를 `Medical-NLA` 결과로
표기하지 않는다.

최종 방법은 최소한 다음 학습 신호를 함께 가져야 한다.

1. 의료 문장 형식과 finding/value 표현을 위한 clinical language supervision
2. own activation과 report 사이의 정보 보존
3. cue deletion/value edit에 대한 changed-component response
4. untouched finding을 보존하는 specificity constraint
5. 같은 진단의 다른 환자 activation과 구별하는 patient-level contrast

AR reconstruction을 실제 objective로 사용하지 않으면 strict한 의미의 NLA인지,
query-conditioned Medical Activation Oracle인지 이름을 다시 정해야 한다. 이 명칭은 최종
학습 구조가 확정된 뒤 고정한다.

## 4. Evaluation Tasks

### 4.1 Task 1: Perturbation-based explanation decoupling

#### 질문

환자 입력의 finding $A$를 $A'$로 바꿨을 때 설명이 그 변경을 등록하는가? 그리고
설명과 최종 answer가 모두 이 perturbation을 무시하는가?

#### 실행

원본 사례 $X_i$와 하나의 임상 요소를 변경한 $X'_i$를 같은 target model에 넣는다.
각 조건에서 P0 activation을 추출하고 CoT, Vanilla NLA, Medical-NLA 설명을 생성한다.
모든 방법은 동일한 case pair와 deterministic perturbation을 사용한다.

원 perturbation audit의 M-block은 fact ablation, demographic swap, irrelevant distractor,
negation flip, severity reversal, temporal shift를 포함한다. 각 operator가 answer를 반드시
바꿔야 한다고 가정하지 않고, 원 논문의 destructive/preserving 구분과 clinician audit을
그대로 기록한다. NLA 비교에서는 perturbation 정의를 다시 만들지 않고 동일한 원본/변형
case pair에서 explanation channel만 CoT, Vanilla NLA, Medical-NLA로 교체한다.

#### 핵심 지표

`Explanation Registration`은 설명이 변경된 사실을 명시적으로 반영한 비율이다.

\[
\mathrm{Registration}
=P(\text{explanation registers the edited evidence})
\]

`Explanation-Decoupling Rate (EDR)`은 설명이 변경을 등록하지 않고 최종 answer도 바뀌지 않은
비율이다.

\[
\mathrm{EDR}
=P(\neg U_Z \land \neg U_Y)
\]

$U_Z$는 explanation update, $U_Y$는 answer flip이다. 낮을수록 좋다. CoT 행에서는 원
논문의 Chain-Decoupling Rate와 같은 의미가 되고, NLA 행에서는 `chain`을 일반적인
`explanation`으로 확장한 새 이름을 사용한다.

Answer Flip Rate, FCS, CHS, DFG 같은 answer-side metric은 fixed target backbone과 같은
perturbation을 사용하면 explanation method별로 달라지지 않는다. 따라서 표의 method 열마다
반복하지 않고, perturbation population이 실제 backbone 행동에 미친 영향을 설명하는 별도
context block으로 한 번만 보고한다.

#### Task 1이 증명하지 못하는 것

변경된 $A\to A'$를 잘 등록했다는 사실만으로 설명 전체의 $B,C,D,E$가 모두 정확하다는
뜻은 아니다. 또한 입력 변경으로 activation도 함께 바뀌므로, 설명이 activation 자체를
읽었다는 직접 증거는 아니다. 이 두 빈칸을 각각 Task 2와 Task 3가 채운다.

### 4.2 Task 2: Sentence-chunk clinical factuality

#### 질문

CoT 또는 NLA가 실제로 발화한 임상 문장 중 의료적으로 잘못된 문장의 비율은 얼마인가?

#### 실행

[Better Accuracies, Worse Reasoning](https://arxiv.org/abs/2605.28301)의 Appendix E
sentence-chunk control을 사용한다. 번호가 있는 CoT step을 가정하지 않고, CoT와 NLA 모두를
동일한 deterministic sentence splitter로 분할한다. Method name과 문체 정보를 가린
style-blind judge가 각 chunk를 `correct`, `error`, `uncertain`으로 판정한다.

#### 지표

\[
\mathrm{SentenceChunkError}
=\frac{N_{error}}{N_{correct}+N_{error}}
\]

\[
\mathrm{UncertainRate}
=\frac{N_{uncertain}}{N_{correct}+N_{error}+N_{uncertain}}
\]

`Chunks per case`도 함께 보고한다. 설명을 거의 생성하지 않아 오류율을 낮추는 trivial solution을
확인하기 위해서다. 원 논문의 Qwen 결과는 `Reported results` block으로 두고, 핵심 직접 비교는
동일한 Gemma에서 생성한 CoT, Vanilla NLA, Medical-NLA를 같은 splitter와 judge로 다시 채점한
결과로 한다.

#### Task 2가 증명하지 못하는 것

Sentence-chunk error는 **말한 내용의 사실성**을 측정한다. 중요한 finding을 빠뜨리지 않았는지에
대한 completeness는 직접 측정하지 않는다. 또한 모든 문장이 정확해도 같은 진단에 공통적인
상투 문장만 생성했을 수 있다. 환자별 coverage와 activation dependence는 Task 3에서 본다.

### 4.3 Task 3: Direct activation dependence

#### 질문

생성 설명은 해당 환자 activation에서 나온 것인가, 아니면 같은 질환에 공통적인 임상 template을
생성한 것인가?

#### 데이터와 donor

DDXPlus에서 환자 $i$와 $j$를 다음 조건으로 짝짓는다.

\[
D_i=D_j,\qquad F_i\neq F_j
\]

즉 진단은 같지만 finding 집합이 다른 환자다. 동일 진단을 강제하는 이유는 decoder가 질환명이나
질환 전형만 말해도 성공하는 것을 막기 위해서다. 실제 실행에서는 finding-set 차이의 수와 종류,
donor 가용률, 진단별 pair 수를 함께 보고한다.

#### activation swapping

두 환자를 frozen target model에 각각 forward하여 같은 P0/HS32 위치의 activation
$h_i,h_j$를 얻는다. 그 다음 동일한 reader $G$에 own activation과 donor activation을
각각 넣는다.

\[
Z_i^{own}=G(h_i),\qquad Z_i^{shuffle}=G(h_j)
\]

decoding prompt, temperature, max tokens와 semantic mapper는 고정하고 activation만 교체한다.
이것은 target LM의 임의 layer에 vector를 삽입하는 Patchscope가 아니라, activation reader의
정상 입력을 교환하는 실험이다.

#### 단방향 점수

환자 $i$의 reference finding $F_i$에 대해 다음을 계산한다.

\[
F1_{own}=F1(\widehat F(h_i),F_i)
\]

\[
F1_{shuffled}=F1(\widehat F(h_j),F_i)
\]

\[
\Delta_{activation}=F1_{own}-F1_{shuffled}
\]

Own F1이 높고 gap도 높아야 정확하면서 환자 activation에 특이적인 reader다. Own과 shuffled가
모두 높으면 같은 질환의 전형적 설명을 생성한 것일 수 있다. Own이 낮고 gap만 높으면 activation에는
민감하지만 임상 판독기로 유용하지 않다.

#### 대칭 2x2 점수

환자 난이도 차이를 줄이기 위해 주 분석은 양방향으로 계산한다.

\[
S_{matched}=\frac{S(G(h_i),F_i)+S(G(h_j),F_j)}{2}
\]

\[
S_{crossed}=\frac{S(G(h_i),F_j)+S(G(h_j),F_i)}{2}
\]

\[
\Delta_{pair}=S_{matched}-S_{crossed}
\]

진단 category cluster bootstrap 95% CI가 0보다 완전히 큰지를 함께 보고한다. 이는 개별 환자
수를 크게 보이게 하는 대신, 특정 진단 category 몇 개가 전체 효과를 만든 것은 아닌지 확인한다.

#### Linear probe 측정

Linear probe는 자연어를 생성하지 않고 activation에서 91개 finding probability를 직접 낸다.
Validation에서 동결한 threshold를 적용해 evidence set을 만든 뒤 동일한 own/shuffled/pair F1을
계산한다.

\[
\widehat F_i^{probe}=\{E_k:\sigma(W_kh_i+b_k)\ge \tau_k\}
\]

Probe는 activation에 임상 정보가 존재하고 closed ontology에서 판독 가능한지를 보여주는
상한선이다. Medical-NLA가 probe보다 반드시 높은 F1을 가져야 한다는 뜻은 아니다. Medical-NLA의
추가 가치는 probe에 가까운 환자 판독을 사람이 읽을 수 있는 열린 자연어로 제공하는 데 있다.

#### SAE 측정

SAE는 $h_i\to z_i$ sparse code만 제공하므로 feature-to-finding mapping 없이는 F1을 계산할
수 없다. Train-only top-activating cases로 feature 의미 후보를 만들고, validation에서 mapping과
feature threshold를 동결한 뒤 활성 feature를 evidence ID 집합으로 변환한다.

\[
\widehat F_i^{SAE}=\{M(k):z_{ik}\ge\tau_k,\;k\text{ is validated}\}
\]

이 mapping을 확보하지 못하면 SAE 행은 빈 수치로 두지 않고 capability table 또는 appendix로
내린다. SAE code에 supervised linear classifier를 학습하면 행 이름은 `Linear probe on SAE
codes`로 명시한다.

#### 자연어 NLA 측정

Vanilla NLA와 Medical-NLA는 생성 text를 frozen semantic mapper로 evidence ID와 value에
매핑한 후 같은 F1을 계산한다. Mapper는 method name, patient reference와 gold label을 받지
않아야 하며, validation gate를 통과한 prompt/model/cache hash를 test 전에 동결한다.

#### cue-level 보조 panel

Main Task 3는 own/shuffled correspondence로 단순화한다. Cue deletion과 value edit은 보조 panel
또는 appendix로 둔다.

- Deletion phantom: 삭제한 finding을 설명에서 계속 말하는 비율
- Removal success: 원본에서 말했던 finding이 삭제 activation에서 사라진 비율
- Retained preservation: 바꾸지 않은 finding이 유지되는 비율
- Replacement hit: 새 value를 언급하는 비율
- Old persistence: 이전 value를 계속 언급하는 비율
- Clean switch: old value가 사라지고 new value만 나타나는 비율

이 panel은 Task 1과 유사해 보이지만 개입 위치가 다르다. Task 1은 외부 clinical input
perturbation에 explanation이 반응하는지를 기존 CoT audit 계약으로 본다. Task 3 보조 panel은
그 결과로 생성된 activation family를 reader에 직접 넣어 activation-conditioned output의
선택성을 본다.

## 5. Baselines and Capability Table

아래 표의 `O`는 해당 방법을 평가할 수 있다는 뜻이지, 이미 gate를 통과했다는 뜻이 아니다.

| Method | Patient-level natural language | Task 1 perturbation | Task 2 factuality | Task 3 activation dependence | Medical adaptation |
|---|---:|---:|---:|---:|---:|
| CoT | O | O | O | X | X |
| Logit/Tuned lens | X | X | X | △ | X |
| Linear probe | X | △ | X | O | O |
| SAE | X | X | X | O* | X |
| Transcoder | X | X | X | △ | X |
| Patchscope | O | △ | O | O* | X |
| Vanilla NLA | O | O | O | O | X |
| Medical-NLA | O | O | O | O | O |

`SAE O*`는 validated feature-to-clinical-concept mapping이 있을 때만 성립한다.
`Patchscope O*`는 output contract와 clinical positive control을 통과할 때만 성립한다. 현재
프로젝트 결과에서는 이 조건을 통과하지 못했으므로 main quantitative row가 아니다.

이 capability table이 보여주려는 핵심은 Medical-NLA가 모든 개별 metric에서 무조건 최고라는
것이 아니다. 기존 방법은 일부 기능을 강하게 수행하지만, **환자별 자연어**, **임상 사실성**,
**직접 activation dependence**, **의료 적응**을 하나의 interface에서 동시에 제공하지 못한다.

## 6. Expected Result Tables

### Table 1. Perturbation-based explanation responsiveness

| Explanation source | Explanation registration ↑ | EDR ↓ |
|---|---:|---:|
| Gemma CoT | 측정 | 측정 |
| Vanilla NLA | 측정 | 측정 |
| Medical-NLA | 성공 모델 확정 후 측정 | 성공 모델 확정 후 측정 |

같은 Gemma와 perturbation pair를 사용하므로 answer-side AFR/FCS/CHS/DFG는 별도 context block에
한 번만 기록한다. 원 2026 논문의 다른 모델 수치는 `Reported results` block으로 분리하고,
우리 세 행과 절대값 SOTA 비교라고 쓰지 않는다.

### Table 2. Sentence-chunk clinical factuality

| Explanation source | Sentence-chunk error ↓ | Uncertain rate ↓ | Chunks/case ↑ |
|---|---:|---:|---:|
| Qwen3-8B Base CoT | 60.1 (reported) | N/R in anchor table | N/R in anchor table |
| Qwen3-8B Distilled CoT | 77.5 (reported) | N/R in anchor table | N/R in anchor table |
| Gemma CoT | 측정 | 측정 | 측정 |
| Vanilla NLA | 측정 | 측정 | 측정 |
| Medical-NLA | 성공 모델 확정 후 측정 | 성공 모델 확정 후 측정 | 성공 모델 확정 후 측정 |

Published Qwen 수치는 원 논문의 evaluator 환경을 그대로 재현했을 때만 같은 numeric block에서
직접 비교한다. 그렇지 않으면 위와 같이 reported block과 our evaluation block을 시각적으로
분리한다.

### Table 3. Patient-specific activation dependence

| Method | Own finding F1 ↑ | Shuffled finding F1 ↓ | Own-shuffled gap ↑ | Symmetric pair gap ↑ | Cluster 95% CI |
|---|---:|---:|---:|---:|---:|
| Linear probe, HS32 | 측정 | 측정 | 측정 | 측정 | 측정 |
| SAE, HS32 | mapping 확보 시 측정 | 측정 | 측정 | 측정 | 측정 |
| Vanilla NLA, HS32 | 측정 | 측정 | 측정 | 측정 | 측정 |
| Medical-NLA, HS32 | 성공 모델 확정 후 측정 | 측정 | 측정 | 측정 | 측정 |

Task 3에서는 모든 행이 같은 activation site와 donor population을 사용해야 한다. 기존 HS24
probe 수치와 HS32 NLA 수치를 한 표에 그대로 혼합하지 않는다.

### Appendix. Cue-level activation counterfactuals

| Method | Original hit ↑ | Deletion phantom ↓ | Removal ↑ | Retained preservation ↑ | Replacement hit ↑ | Old persistence ↓ | Clean switch ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear probe | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| Vanilla NLA | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |
| Medical-NLA | 성공 모델 확정 후 측정 | 측정 | 측정 | 측정 | 측정 | 측정 | 측정 |

## 7. Existing Evidence and Honest Status

현재까지 다음 사실은 이미 확인됐다.

1. DDXPlus frozen probe는 환자 finding과 value를 높은 정확도로 읽는다. 따라서 P0 activation에
   임상 정보가 존재한다는 전제는 성립한다.
2. Probe prediction을 deterministic text로 렌더링하면 높은 finding F1을 얻지만, 이것은
   free-generating NLA가 아니라 closed ontology monitor다.
3. 공개 Vanilla NLA는 DDXPlus locked 10,028개 readout에서 frozen mapper가 인정한 ontology
   claim이 0개였다. 20-case 원문 감사에서는 mapper miss보다 generic clinical text 생성이
   원인으로 판정됐다.
4. SFT, counterfactual sequence SFT, ranking, soft bottleneck, specificity-anchor를 포함한
   개발 실험은 아직 three-seed promotion gate를 통과하지 못했다.
5. Public AR는 의료 분포에서 reconstruction reward 측정기로 인정되지 않았고, same-layer
   Patchscope는 general-domain control은 통과했지만 clinical correspondence에 실패했다.

따라서 현재 논문은 평가 문제와 benchmark 설계는 정리됐지만, 최종 Medical-NLA 행을 채울
성공 checkpoint는 아직 없다. 발표에서 기존 음성 결과를 숨기지 않되, 그것을 중심 주장처럼
앞세우지는 않는다. 이 결과들은 왜 단순 의료 SFT나 global ranking만으로 activation reader가
만들어지지 않는지 설명하는 method-development appendix가 된다.

## 8. Experimental Order

논문 결과 절과 실제 실행 순서는 다음처럼 맞춘다.

1. **Instrumentation:** 동일 P0/HS32 extraction, semantic mapper, donor split과 bootstrap 단위를
   동결한다.
2. **Task 1 controls:** Gemma CoT와 Vanilla NLA를 perturbation protocol로 먼저 평가한다.
3. **Task 2 controls:** 같은 두 방법을 sentence-chunk factuality judge로 평가한다.
4. **Task 3 controls:** frozen HS32 probe와 Vanilla NLA의 own/shuffled 표를 완성한다.
5. **SAE feasibility:** exact activation hook과 호환되는 SAE를 확보하고 feature mapping이
   validation을 통과할 때만 Task 3 행을 연다.
6. **Medical-NLA development:** 성공 recipe를 validation-only로 결정하고 three-seed gate를
   통과한 checkpoint만 locked evaluation에 올린다.
7. **Final comparison:** 세 task의 Medical-NLA 행을 같은 frozen checkpoint로 채운다.

Medical-NLA가 Task 2에서 낮은 오류율을 보여도 Task 3 own-shuffled gap이 없으면 faithful이라고
주장하지 않는다. 반대로 Task 3 gap이 높아도 Own F1이 낮으면 activation-sensitive하지만
임상적으로 부정확한 reader다. 세 task를 함께 두는 이유가 여기에 있다.

## 9. Paper Structure

### 1. Introduction

의료 LLM 정확도 향상, visible explanation faithfulness 문제, closed internal tools의 제약,
Medical-NLA의 필요성과 세 RQ를 제시한다.

### 2. Related Work

2.1 Medical CoT faithfulness와 perturbation audit

2.2 Probing, lens, SAE, transcoder를 이용한 internal-state analysis

2.3 Patchscope, SelfIE, LatentQA, Activation Oracle, NLA의 activation-to-language interface

### 3. Methodology

3.1 Medical-NLA: activation site, AV/AR 또는 최종 reader architecture, output contract

3.2 Faithfulness Evaluation Framework

- 3.2.1 Perturbation-based explanation decoupling
- 3.2.2 Sentence-chunk clinical factuality
- 3.2.3 Direct activation dependence

### 4. Experimental Setup

Backbone, P0/HS32, DDXPlus와 의료 QA population, perturbation operators, donor construction,
semantic mapper, judge, baselines와 statistical protocol을 설명한다.

### 5. Results

5.1 Task 1: 설명이 임상 변경을 선택적으로 등록하는가?

5.2 Task 2: 설명이 발화한 임상 문장은 정확한가?

5.3 Task 3: 설명이 해당 환자의 activation에 직접 의존하는가?

5.4 Ablation and failure analysis: SFT, ranking, anchor, AR, Patchscope 음성 결과

### 6. Discussion

Natural-language reader와 closed probe의 역할 차이, activation decodability와 causal use의
차이, 하나의 backbone과 P0/HS32 사용의 한계, semantic judge 의존성, 의료 배포로 일반화할 수
없는 범위를 논의한다.

### 7. Conclusion

Medical-NLA를 더 좋은 CoT 생성기로 정의하지 않고, 환자별 model state를 자연어로 감사하는
interface로 정의한다. 성공 여부는 유창성이나 reconstruction 하나가 아니라 perturbation
responsiveness, clinical factuality, direct activation dependence의 결합으로 판정한다.

## 10. 교수님께 확인할 결정

1. 중심 주장을 `reasoning improvement`가 아니라 `patient-specific activation
   verbalization`으로 고정해도 되는가?
2. 세 task를 각각 perturbation responsiveness, sentence-level factuality, direct activation
   dependence로 두는 구성이 충분한가?
3. 2026 두 CoT audit 논문의 표를 그대로 복사하는 것이 아니라, 공개 protocol을 동일 Gemma의
   CoT/NLA에 적용한 extension table로 만드는 것을 허용할 수 있는가?
4. Task 3 주표는 Linear probe, SAE(가능한 경우), Vanilla NLA, Medical-NLA로 제한하고 lens,
   transcoder, Patchscope는 Related Work와 appendix로 두어도 되는가?
5. 최종 Medical-NLA architecture가 아직 validation gate를 통과하지 못한 상태에서, 평가
   benchmark와 baseline 파이프라인을 먼저 완성하는 순서를 승인할 수 있는가?

## 발표 마무리 문장

이 연구가 답하려는 질문은 “Medical-NLA가 CoT보다 더 영리하게 reasoning하는가”가 아니다.
우리가 답하려는 질문은 “의료 LLM이 환자를 읽은 직후 내부에 형성한 상태를 자연어로 꺼냈을 때,
그 보고가 임상적으로 정확하고 입력 변화에 선택적으로 반응하며 바로 그 환자의 activation에
실제로 의존하는가”이다. 이 세 조건을 모두 만족할 때만 Medical-NLA를 faithful clinical
activation reader라고 부른다.
