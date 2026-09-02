# Medical-NLA 9월 3일 발표용 논문형 원고

이 문서는 슬라이드 초안이 아니라 논문 순서에 맞춘 발표 원고다. 공개 선행 연구의 보고값,
이 프로젝트에서 이미 측정한 값, 아직 측정하지 않은 값을 구분한다. `미측정`은 0이 아니며,
성공한 Medical-NLA checkpoint가 없어서 의도적으로 비워 둔 셀이다.

## 발표에서 먼저 말할 한 문장

> **Medical-NLA는 의료 LLM의 hidden activation에 표현된 환자별 임상 상태를 사람이 읽을 수
> 있는 자연어로 faithful하게 언어화하는 activation-conditioned reader다.**

이 연구의 목적은 NLA로 backbone의 진단 정확도나 reasoning 능력을 직접 올리는 것이 아니다.
목표는 모델이 환자 정보를 읽은 뒤 내부에 형성한 finding, value, diagnostic disposition을
자연어 state report로 드러내고, 그 report가 그럴듯한 의료 문장이 아니라 실제 activation에
근거했는지 검증하는 것이다.

## 1. Introduction

### 1.1 정답 정확도에서 reasoning faithfulness로

최근 의료 LLM은 의료 질의응답과 진단 benchmark에서 높은 정답률을 보인다. 그러나 정답이
맞는 것과 그 답을 만드는 과정에서 환자 근거를 올바르게 사용한 것은 같은 문제가 아니다.
모델은 맞는 답에 잘못된 근거를 붙일 수 있고, 틀린 답을 유창한 임상 서술로 정당화할 수도 있다.
따라서 의료 LLM 평가는 answer accuracy뿐 아니라 **reasoning과 explanation의 정확성 및
faithfulness**를 별도로 측정하는 방향으로 확장되고 있다.

[MedOmni-45°, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40864)은
의료 reasoning 모델의 안전성 축에 CoT faithfulness와 sycophancy를 포함한다.
[Trustworthy Medical Question Answering, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1398/)도
의료 QA의 신뢰성을 factuality, robustness, safety, explainability, calibration 등으로 나누며
정답률 하나로 평가할 수 없음을 정리한다. 이 흐름에서 CoT는 단순한 성능 향상 기법을 넘어,
임상의가 모델의 판단 근거를 검토하는 **visible explanation interface**로 사용된다.

다만 CoT를 “모델 내부 생각의 기록”이라고 단정해서는 안 된다. CoT는 hidden state를 직접
관측한 값이 아니라 출력 시점에 모델이 생성한 **자기보고형 설명(self-reported explanation)** 이다.
따라서 좋은 문장과 faithful한 설명을 구분해야 한다.

### 1.2 의료 CoT에 제기된 구체적인 문제

최근 연구는 서로 다른 방식으로 CoT 자기보고의 한계를 확인한다.

1. [Walk the Talk?, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b5ec50eb177908f21f78ed0d76ed525c-Abstract-Conference.html)는
   MedQA 임상 개념을 counterfactual하게 바꾸고 모델의 설명이 실제 decision-driving evidence를
   올바르게 밝히는지 검사했다. 그럴듯한 설명이 어떤 근거가 결정에 영향을 주었는지 잘못 말할 수
   있음을 보였다.
2. [Faithful or Just Plausible?, NeurIPS 2025 Workshop/2026 preprint](https://arxiv.org/abs/2603.13988)는
   causal ablation, positional bias, hint injection으로 의료 CoT를 검사했다. CoT step이 예측을
   인과적으로 만들지 않거나, 외부 hint를 사용하고도 이를 설명에서 밝히지 않는 현상을 보고했다.
3. [Evaluating Reasoning Faithfulness in Medical VLMs, 2025](https://arxiv.org/abs/2510.11196)는
   임상 text와 image cue를 통제해 answer accuracy와 explanation quality가 분리될 수 있음을
   확인했다. injected cue를 언급하는 것만으로 실제 grounding이 보장되지도 않았다.
4. [Better Accuracies, Worse Reasoning, 2026](https://arxiv.org/abs/2605.28301)은
   CoT distillation으로 MedQA 정답 성능과 calibration은 좋아졌지만, 같은 style-blind audit에서
   non-abstained reasoning-step 오류율이 Qwen3-8B 기준 30.6%에서 50.3%로 증가했음을 보였다.
   즉 더 높은 accuracy와 더 정확한 reasoning trace가 반대 방향으로 움직일 수 있다.
5. [Right Diagnoses, Decorative Reasoning, 2026](https://arxiv.org/abs/2608.24790)은
   14개 모델, 4개 의료 QA benchmark, 30개 임상 perturbation operator를 사용했다. 임상적으로
   의미 있는 destructive edit에서 chain이 변경을 등록하지 않고 answer도 유지한 CDR이 전체
   평균 72.9%였다. CoT corruption과 CoT prompt 제거도 accuracy를 거의 떨어뜨리지 않았다.
6. [Auditing Evidence Use in Medical LLM Diagnosis, 2026](https://arxiv.org/abs/2607.20848)은
   DDXPlus, CupCase, MedCase의 evidence subset을 통제하고 diagnostic margin을 분석했다. 높은
   진단 정확도만으로 환자 근거를 적절히 사용했는지 알 수 없음을 행동 수준에서 보였다.

이 논문들이 공통으로 말하는 것은 “CoT가 나쁘다”가 아니다. **CoT가 유창하고 답이 맞다는
사실만으로 그 CoT가 실제 model state 또는 decision process를 faithful하게 설명했다고 볼 수
없다**는 것이다.

### 1.3 왜 내부 상태를 관찰해야 하는가

기존 CoT audit은 input, visible chain, answer를 조작하여 faithfulness를 행동적으로 추론한다.
이는 중요하지만 hidden activation을 직접 읽지는 않는다. 내부 상태를 관찰하는 도구에도 각자의
강점과 한계가 있다.

- **Linear probe**는 activation에 특정 diagnosis/finding label이 linearly decodable한지 정확히
  측정할 수 있다. 하지만 사전에 정한 label 공간만 출력하므로 ontology 밖의 임상 상태나 여러
  finding의 관계를 열린 문장으로 설명하지 못한다.
- **Logit/Tuned lens**는 중간 layer를 vocabulary prediction 공간으로 투영하지만, 여러 finding과
  categorical value를 묶은 환자별 state report와 출력 단위가 다르다.
- **SAE**는 dense activation을 sparse feature로 분해하지만 feature ID에는 임상 의미가 자동으로
  붙지 않는다. top-activating context와 held-out 검증으로 feature-to-clinical-concept mapping을
  별도로 구축해야 한다.
- **Transcoder**는 주로 한 MLP의 입력-출력 계산을 feature 단위로 근사한다. 계산 경로 분석에는
  유용하지만 환자 상태를 자연어로 보고하는 장치는 아니다.
- **Patchscope/SelfIE**는 activation을 target context에 삽입해 자연어 continuation으로 읽지만,
  prompt, patch 위치와 decoding geometry에 민감하다.

따라서 필요한 것은 닫힌 label 분류기만도, visible CoT만도 아니다. 환자 activation을 직접
입력받아 읽을 수 있는 자연어로 만들고, activation을 바꾸었을 때 출력이 선택적으로 바뀌는지
검증할 수 있는 reader가 필요하다.

### 1.4 제안과 Research Questions

Natural Language Autoencoder의 activation-to-text interface를 의료 도메인으로 확장한
Medical-NLA를 제안한다. 출력은 전문가 정답 rationale이 아니라 activation에 표현된 환자
상태를 보고하는 **diagnostic-state report**다. 모델이 틀렸다면 정답 rationale을 대신 만들어
주는 것이 아니라, activation 안의 잘못된 가정, 누락, 상충 근거와 진단 방향을 그대로 드러내야
한다.

중심 주장은 다음 세 RQ로 나눈다.

**RQ1. Perturbation responsiveness:** 환자의 특정 임상 근거가 바뀌었을 때 설명은 그 근거의
변화를 선택적으로 등록하는가? 설명과 최종 answer가 모두 변경을 무시하는 decoupling은 얼마나
자주 발생하는가?

**RQ2. Clinical factuality:** 생성된 설명이 실제로 발화한 각 임상 문장은 의료적으로 정확한가,
아니면 유창하지만 잘못된 clinical segment를 포함하는가?

**RQ3. Direct activation dependence:** 동일한 진단을 가진 다른 환자의 activation으로 교체했을
때 설명의 finding 대응성이 감소하는가? 즉 설명은 질환 template이나 input wording이 아니라
바로 그 환자의 activation에 의존하는가?

RQ1은 **변경에 대한 반응성**, RQ2는 **발화 내용의 사실성**, RQ3는 **출력의 직접 원천**을
묻는다. 세 질문 중 하나만 통과해서는 faithful clinical activation reader라고 부르지 않는다.

### 1.5 Contributions

1. 의료 LLM의 hidden activation을 환자별 자연어 diagnostic-state report로 변환하는
   Medical-NLA를 정의한다.
2. 2025-2026 의료 CoT 연구의 perturbation 및 segment factuality protocol을 CoT와 NLA에
   공통으로 적용할 수 있는 method-neutral explanation audit으로 확장한다.
3. DDXPlus의 명시적 finding/value와 same-diagnosis donor를 이용해 explanation의 activation
   dependence를 직접 측정하는 paired activation-swapping benchmark를 제안한다.
4. 자연어 유창성, 임상 사실성, 환자별 coverage, activation dependence를 분리해 보고하고,
   하나의 reconstruction 또는 LLM-judge 점수만으로 faithfulness를 주장하지 않는다.

## 2. Related Work

### 2.1 Faithfulness of Chain-of-Thought

일반-domain 연구는 CoT를 모델이 답을 만든 과정을 설명하는 자기보고로 사용하면서도 그
faithfulness를 별도로 검증해야 한다고 지적했다. [Reasoning Models Don't Always Say What They
Think](https://www.anthropic.com/research/reasoning-models-dont-say-think)는 모델이 prompt hint를
사용하고도 CoT에서 이를 드러내지 않는 경우가 많음을 보였다. [Measuring CoT Faithfulness by
Unlearning Reasoning Steps, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.504/)는 특정
reasoning step에 해당하는 정보를 모델 parameter에서 제거한 뒤 prediction이 변하는지로
parametric faithfulness를 검사한다. [FaithCoT-Bench, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/6c7154e394e24c69409256ccf8bf0804-Abstract-Conference.html)는
1,000개 이상의 trajectory와 300개 이상의 unfaithful 사례를 이용해 instance-level detection을
평가한다.

의료-domain 연구는 이 문제를 임상 근거와 위험에 맞게 구체화한다. `Walk the Talk?`와
`Faithful or Just Plausible?`는 counterfactual 및 causal probe를 사용하고, `Right Diagnoses,
Decorative Reasoning`은 severity, negation, demographic, evidence ablation을 포함한 의료
operator와 CDR을 제안한다. `Better Accuracies, Worse Reasoning`은 answer-level 향상이 local
clinical factuality 향상을 뜻하지 않는다는 별도의 실패 축을 보인다.

Medical-NLA는 이 연구들을 “CoT보다 더 잘 reasoning하는 모델”로 대체하려는 것이 아니다.
이들이 행동 수준에서 발견한 visible self-report의 간극을 **hidden-state readout**으로 직접
측정하려는 것이다.

### 2.2 Natural Language Autoencoders and activation readout

[Natural Language Autoencoders, 2026](https://transformer-circuits.pub/2026/nla/)는 activation
verbalizer(AV)가 hidden activation을 자연어로 바꾸고 activation reconstructor(AR)가 그
텍스트에서 activation을 복원하는 자연어 bottleneck을 제안했다. 공개 checkpoint에는
Gemma-3-12B layer 32뿐 아니라 **Qwen2.5-7B layer 20 AV/AR**도 포함된다. 따라서 Qwen NLA는
실재하는 중요한 외부 baseline이다. 다만 일반-domain activation으로 학습된 공개 NLA이지,
의료 finding supervision으로 다시 학습된 Medical-NLA는 아니다.

[Medical Language Autoencoders](https://github.com/BlakeMasters/medical_language_autoencoders)는
공개 Qwen2.5-7B L20 AV/AR를 MedQA 200문항, 3 prompt variant, 총 600개 prediction에 적용했다.
보고값은 answer accuracy 57.5%, reconstruction cosine 0.828, heuristic alignment 5.5%,
MedGemma-judge alignment 77.7%다. 이는 의료에서 Qwen NLA를 실행한 선행 사례지만 새 의료 NLA를
학습한 것이 아니고, scorer 선택에 따라 alignment가 크게 달라진다는 경고로 읽어야 한다.

[NLA-KTH](https://github.com/mohamedibrahim26/nla-kth)는 Qwen2.5-0.5B layer 16에서 NLA를
재학습한 비심사 재현 구현이다. 8,000개 activation, Qwen2.5-3B teacher summary, AR/AV SFT와
GRPO를 사용한다. 공개 구현 선례로는 유용하지만 의료 benchmark의 peer-reviewed baseline은
아니다. 따라서 본문 결과표의 published SOTA 숫자로 사용하지 않고 구현 참고 또는 appendix
baseline으로 구분한다.

Activation-to-language의 인접 계열에는 Patchscope, SelfIE, LatentQA와 Activation Oracle이
있다. 이들은 각각 prompt-conditioned patching, self-interpretation, query-conditioned
activation QA를 수행한다. 반면 NLA는 activation을 독립적인 자연어 bottleneck으로 만들고
reconstruction을 정보 보존 신호로 쓴다. 본 논문은 이 NLA 인터페이스에 의료 임상성 및
patient-specific grounding constraint를 추가한다.

## 3. Methodology

### 3.1 Medical-NLA

#### 3.1.1 Problem formulation

환자 사례를 $X_i$, frozen target medical LLM을 $M$, 고정한 extraction layer를 $l$이라고 하자.

\[
h_i=M_l(X_i), \qquad Z_i=G_\theta(h_i)
\]

$h_i$는 환자 사례를 읽은 target model의 hidden activation이고, $G_\theta$는 activation을
자연어 report $Z_i$로 바꾸는 Medical-NLA verbalizer다. 목표는 physician gold rationale을
그대로 모사하는 것이 아니라 $h_i$에 표현된 다음 상태를 읽을 수 있게 보존하는 것이다.

- 환자 finding과 categorical/ordinal value
- 현재 activation의 diagnostic hypothesis 또는 disposition
- 약하게 표현되거나 누락된 근거
- 상충하는 근거와 불확실성

#### 3.1.2 Activation site and output contract

주 실험은 Gemma-3-12B-IT의 P0, hidden-state layer 32, last-token residual을 사용한다. P0는
모델이 임상 사례를 모두 읽었지만 visible answer나 CoT를 생성하기 전이다. 따라서 answer text
leakage 없이 사례를 통합한 pre-generation state를 읽는다.

Medical-NLA 출력은 free natural-language clinical state report다. 평가할 때만 method-blind
semantic mapper를 사용해 공통 DDXPlus ontology로 변환한다.

\[
Z_i \longrightarrow \widehat F_i,\widehat V_i,\widehat D_i
\]

$\widehat F_i$는 finding set, $\widehat V_i$는 finding value, $\widehat D_i$는 diagnostic
disposition이다. Mapper는 method 이름, gold label과 patient reference를 받지 않는다.

#### 3.1.3 Training contract and current status

최종 Medical-NLA는 다음 학습 신호를 함께 만족해야 한다.

1. 임상 문장 형식과 finding/value 표현을 위한 clinical language supervision
2. own activation과 report 사이의 정보 보존 또는 reconstruction
3. cue deletion/value edit에서 changed component만 반응시키는 constraint
4. untouched finding을 보존하는 specificity constraint
5. 같은 진단의 다른 환자 activation과 구별하는 patient-level contrast

현재 SFT, counterfactual sequence SFT, ranking, soft bottleneck, specificity anchor는 모두
validation promotion gate를 통과하지 못했다. 공개 AR도 이 의료 분포에서 valid reconstruction
instrument로 인정되지 않았다. 따라서 아래 결과표의 `Medical-NLA` 행은 최종 목표를 명시하기
위한 행이며 성공한 checkpoint가 생기기 전에는 `미측정`으로 둔다. 실패 checkpoint를 최종
방법의 결과로 바꿔 쓰지 않는다.

### 3.2 Faithfulness Evaluation Framework

#### 3.2.1 RQ1: Perturbation-based explanation responsiveness

원본 사례 $X_i$와 한 임상 근거를 변경한 $X'_i$를 같은 target model에 넣는다. 각 조건에서
P0 activation을 추출하고 CoT, Vanilla NLA, Medical-NLA explanation을 생성한다. fact ablation,
demographic swap, irrelevant distractor, negation flip, severity reversal, temporal shift는
`Right Diagnoses, Decorative Reasoning`의 M-block 정의를 유지한다.

설명이 변경된 사실을 등록했는지 $U_Z$, final answer가 바뀌었는지 $U_Y$로 표시한다.

\[
\mathrm{Registration}=P(U_Z), \qquad
\mathrm{EDR}=P(\neg U_Z\land\neg U_Y)
\]

EDR(Explanation-Decoupling Rate)은 CDR의 `chain`을 CoT와 NLA를 포괄하는 `explanation`으로
확장한 이름이다. 낮을수록 좋다. 원 논문의 CDR 수치를 EDR이라고 단순히 이름만 바꿔 재사용하지
않고, 공개 수치는 CDR로 보존하며 우리 explanation 출력은 같은 case pair에서 다시 측정한다.

Answer Flip Rate, FCS, ECR, CHS, DFG는 perturbation population과 backbone 행동을 설명하는
보조 metric이다. 같은 Gemma와 같은 case pair라면 answer-side 값은 explanation method마다
반복하지 않고 context panel에 한 번만 보고한다.

RQ1은 $A\to A'$ 변경의 등록 여부를 보지만, 출력 전체의 $B,C,D,E$가 정확한지는 보장하지
않는다. 그 빈칸이 RQ2다.

#### 3.2.2 RQ2: Sentence-chunk clinical factuality

CoT와 NLA를 동일한 deterministic sentence splitter로 나누고, method와 문체를 가린
style-blind judge가 각 chunk를 `correct`, `error`, `uncertain`으로 판정한다. 번호가 있는
reasoning step을 전제하지 않기 위해 `Better Accuracies, Worse Reasoning`의 sentence-chunk
control을 채택한다.

\[
\mathrm{ChunkError}=\frac{N_{error}}{N_{correct}+N_{error}}, \qquad
\mathrm{UncertainRate}=\frac{N_{uncertain}}{N_{correct}+N_{error}+N_{uncertain}}
\]

`Chunks/case`와 empty/non-clinical output rate를 함께 보고한다. 아무 말도 하지 않아 오류율을
낮추는 trivial solution을 막기 위해서다. 이 metric은 발화한 내용의 사실성을 보지만 중요한
finding을 모두 포함했는지와 activation source를 직접 증명하지는 않는다.

#### 3.2.3 RQ3: Direct activation dependence

DDXPlus에서 진단은 같지만 finding set이 다른 두 환자 $i,j$를 짝짓는다.

\[
D_i=D_j, \qquad F_i\neq F_j
\]

두 환자의 P0/HS32 activation을 얻고 decoder prompt, temperature, max token과 semantic mapper를
고정한 채 reader 입력 activation만 바꾼다.

\[
Z_i^{own}=G(h_i), \qquad Z_i^{shuffle}=G(h_j)
\]

환자 $i$의 reference에 대해 own/shuffled finding F1과 그 차이를 계산한다.

\[
\Delta_{activation}=F1(\widehat F(h_i),F_i)-F1(\widehat F(h_j),F_i)
\]

주 분석은 환자 난이도를 상쇄하는 symmetric 2x2 pair score다.

\[
S_{matched}=\frac{S(G(h_i),F_i)+S(G(h_j),F_j)}{2}
\]

\[
S_{crossed}=\frac{S(G(h_i),F_j)+S(G(h_j),F_i)}{2}, \qquad
\Delta_{pair}=S_{matched}-S_{crossed}
\]

Own F1과 pair gap이 함께 높아야 정확하면서 patient-specific한 reader다. 진단 category cluster
bootstrap 95% CI가 0보다 큰지도 보고해 일부 질환군만 효과를 만드는지 확인한다.

Linear probe는 91개 finding probability에 validation에서 동결한 threshold를 적용해 동일한
F1을 계산한다. SAE는 validated feature-to-finding mapping을 확보한 경우에만 같은 표에 넣는다.
Vanilla/Medical NLA의 free text는 frozen semantic mapper로 evidence ID와 value에 매핑한다.

## 4. Experimental Setup

### 4.1 Data and populations

- **RQ1:** `Right Diagnoses, Decorative Reasoning`의 공개 4-dataset medical QA population과
  M-block perturbation을 재사용한다. 핵심 직접 비교는 동일 Gemma에서 CoT, Vanilla NLA,
  Medical-NLA를 생성한 결과다.
- **RQ2:** 공개 medical QA/clinical vignette에서 동일 question과 answer reference를 사용한다.
  published Qwen 값은 출처 표기로 유지하고, 직접 비교는 동일 Gemma 출력에 동일 splitter와
  judge를 적용한다.
- **RQ3:** DDXPlus validation/locked population에서 P0/HS32 own activation과 same-diagnosis
  different-finding donor를 사용한다. Locked test는 validation에서 method, threshold와 mapper를
  동결한 뒤 한 번만 평가한다.

### 4.2 Baselines

| Method | Backbone/site | 출력 | RQ1 | RQ2 | RQ3 | 상태 |
|---|---|---|---:|---:|---:|---|
| Gemma CoT | Gemma-3-12B | visible rationale | O | O | X | 실행 필요 |
| Linear probe | Gemma P0/HS32 | 91-label probabilities | △ | X | O | 일부 값 존재 |
| SAE | Gemma P0/HS32 | sparse feature IDs | X | X | O* | mapping 미구축 |
| Qwen Vanilla NLA | Qwen2.5-7B L20 | free-text activation report | O | O | O† | 공개 AV/AR 존재, 우리 task 재실행 필요 |
| Gemma Vanilla NLA | Gemma-3-12B L32 | free-text activation report | O | O | O | DDXPlus locked 평가 완료 |
| Medical-NLA | Gemma-3-12B L32 | medical state report | O | O | O | 성공 checkpoint 미정 |

`O*`는 feature-to-clinical-concept mapping을 validation에서 동결한 경우만 가능하다. `O†`는
Qwen activation과 Qwen NLA를 함께 평가하는 별도-backbone block이다. Qwen NLA에 Gemma
activation을 넣을 수 없으므로 Gemma와 같은-backbone 인과 비교로 해석하지 않는다.

Lens, transcoder와 Patchscope는 기능 비교 및 appendix에 둔다. 현재 Patchscope는 general-domain
control은 통과했지만 DDXPlus own/donor correspondence가 0/5여서 main baseline으로 승격하지
않는다.

### 4.3 Common controls and statistics

모든 직접 비교에서 case population, decoding parameters, semantic mapper와 judge를 고정한다.
Generated method name은 judge에 제공하지 않는다. RQ1/RQ2는 같은 case의 paired difference를,
RQ3는 same-diagnosis symmetric pair와 diagnosis-category cluster bootstrap 95% CI를 사용한다.
공개 논문의 숫자는 `reported`, 이 프로젝트가 다시 계산한 값은 `ours`로 표시한다.

## 5. Experimental Results

### 5.1 RQ1: 임상 근거 변경을 설명이 등록하는가?

#### Panel A. 2026 published CoT perturbation baselines

아래 값은 `Right Diagnoses, Decorative Reasoning` Table 3의 4개 의료 benchmark 평균이다.
원 논문의 metric 이름과 숫자를 그대로 유지한다. 이는 최신 baseline의 문제 규모를 보여주는
reported block이며 우리 Gemma/NLA와 동일 실행에서 나온 숫자는 아니다.

| Model | Acc. | CDR ↓ | AFR-D | FCS | ECR | CHS ↓ | DFG ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mistral-7B | .52 | .80 | .22 | .51 | .72 | .11 | .16 |
| Qwen2.5-7B | .54 | .94 | .14 | .53 | .78 | .07 | .15 |
| Llama-3.1-8B | .51 | .72 | .17 | .50 | .87 | .11 | .11 |
| Gemma-2-9B | .67 | .75 | .06 | .52 | .90 | .08 | .25 |
| Qwen2.5-14B | .68 | .72 | .06 | .52 | .92 | .07 | .30 |
| BioMistral-7B | .38 | .96 | .01 | .50 | .00 | .06 | .22 |
| Meditron-7B | .24 | .26 | .44 | .50 | .64 | .20 | .22 |
| Med42-8B | .66 | .68 | .12 | .53 | .85 | .10 | .16 |
| OpenBioLLM-8B | .50 | .89 | .23 | .54 | .00 | .08 | .17 |
| HuatuoGPT-o1 | .55 | .80 | .10 | .53 | .93 | .10 | .09 |
| DeepSeek-R1-D | .60 | .51 | .09 | .53 | .93 | .17 | .26 |

CDR은 destructive M-block에서 `chain no-update AND answer no-flip` 비율이다. 높은 CDR은
정답을 유지했다는 장점이 아니라, chain이 변경된 임상 근거조차 등록하지 않은 decoupling을
뜻한다. AFR-D/FCS/ECR은 sensitivity diagnostic이고, CHS/DFG는 각각 위험한 answer flip과
demographic variation을 보는 보조 지표다.

#### Panel B. Same-backbone explanation comparison

| Explanation source | Registration ↑ | EDR ↓ | Empty/non-clinical ↓ | 출처/상태 |
|---|---:|---:|---:|---|
| Gemma-3-12B CoT | 미측정 | 미측정 | 미측정 | ours |
| Qwen2.5-7B released NLA L20 | 미측정 | 미측정 | 미측정 | external-backbone rerun |
| Gemma-3-12B Vanilla NLA L32 | 미측정 | 미측정 | 미측정 | ours |
| Medical-NLA L32 | 미측정 | 미측정 | 미측정 | 성공 checkpoint 이후 |

핵심 결론은 Panel B의 같은-backbone paired comparison에서 낸다. Qwen 행은 공개 NLA가 의료
perturbation에 반응하는지 보는 외부 재현이고, Gemma CoT와 Gemma NLA의 직접 효과 비교에는
사용하지 않는다.

### 5.2 RQ2: 생성 설명의 임상 문장은 정확한가?

#### Panel A. 2026 published step-level factuality baselines

아래는 `Better Accuracies, Worse Reasoning`의 reported robustness table이다. `Step error`를
우리 마음대로 clinical-claim error로 바꾸지 않는다.

| Student | Answer acc. base | Answer acc. distilled | Step error base ↓ | Step error distilled ↓ | Uncertain base | Uncertain distilled |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-8B | 71.6 | 76.6 | 31.0 | 50.1 | 2.1 | 0.5 |
| Qwen3-14B | 74.6 | 80.8 | 31.5 | 37.9 | 0.8 | 0.5 |
| Qwen3-32B | 81.8 | 84.2 | 22.8 | 30.6 | 2.5 | 0.6 |
| Llama-3.1-8B | 66.8 | 73.6 | 31.2 | 45.5 | 0.1 | 0.4 |
| Mistral-7B | 48.6 | 67.6 | 56.5 | 49.1 | 0.1 | 14.9 |

이 표는 대부분의 capable student에서 distillation 후 answer accuracy는 오르지만 committed
reasoning-step error도 함께 증가할 수 있음을 보여준다. NLA는 numbered reasoning step이 없을
수 있으므로 직접 비교에는 아래 sentence-chunk protocol을 사용한다.

#### Panel B. Method-neutral sentence-chunk audit

| Explanation source | Chunk error ↓ | Uncertain ↓ | Chunks/case ↑ | Empty/non-clinical ↓ | 출처/상태 |
|---|---:|---:|---:|---:|---|
| Qwen3-8B Base CoT | 60.1 | 미보고 | 미보고 | 미보고 | reported control |
| Qwen3-8B Distilled CoT | 77.5 | 미보고 | 미보고 | 미보고 | reported control |
| Gemma-3-12B CoT | 미측정 | 미측정 | 미측정 | 미측정 | ours |
| Qwen2.5-7B released NLA L20 | 미측정 | 미측정 | 미측정 | 미측정 | external-backbone rerun |
| Gemma-3-12B Vanilla NLA L32 | 미측정 | 미측정 | 미측정 | 미측정 | ours |
| Medical-NLA L32 | 미측정 | 미측정 | 미측정 | 미측정 | 성공 checkpoint 이후 |

`60.1/77.5`는 원 논문의 MedQA 500문항 sentence-chunk control 값이다. 정확히는 base가
`1,546/(1,025+1,546)=60.1%`, distilled가 `1,779/(516+1,779)=77.5%`이며 uncertain chunk는
각각 48개와 56개로 분모에서 제외된다. evaluator 환경을 완전히 맞추지 않으면 우리 숫자와
절대값 SOTA 비교를 하지 않는다. 핵심 비교는 동일한 splitter와 judge로 다시 평가한 Gemma
CoT, Gemma Vanilla NLA, Medical-NLA 세 행이다.

### 5.3 RQ3: 설명은 해당 환자의 activation에 의존하는가?

#### Main paired activation table

| Method | Site | Own finding F1 ↑ | Shuffled F1 ↓ | Own-shuffled gap ↑ | Symmetric pair gap ↑ | Cluster 95% CI | 상태 |
|---|---|---:|---:|---:|---:|---:|---|
| Linear probe | Gemma HS24 | .9587 | .7938 | +.1624 | 미집계 | 미집계 | DDXPlus locked |
| Linear probe | Gemma HS32 | 미집계 | 미집계 | 미집계 | 미집계 | 미집계 | 같은-site 재집계 필요 |
| SAE | Gemma HS32 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | validated mapping 필요 |
| Qwen2.5-7B released NLA | Qwen L20 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 별도-backbone block |
| Gemma Vanilla NLA | Gemma HS32 | .0000 | .0000 | +.0000 | 미집계 | 미집계 | DDXPlus locked 10,028 rows |
| Medical-NLA | Gemma HS32 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 성공 checkpoint 이후 |

HS24 probe와 HS32 NLA를 같은-site 성능 비교로 주장하지 않는다. 현재 HS24 probe 값은
activation에 patient finding이 존재한다는 선행 positive control이고, 최종 main comparison을
위해서는 HS32 probe를 동일 donor population에서 재집계해야 한다.

Gemma Vanilla NLA의 0은 결측이 아니다. Frozen semantic mapper가 locked 10,028개 출력에서
인정한 DDXPlus ontology claim이 0개였고, 20-case private audit은 mapper miss가 아니라 generic
clinical text 생성을 원인으로 판정했다.

#### Cue-level activation counterfactual appendix

| Method | Original hit ↑ | Deletion phantom ↓ | Removal ↑ | Retained preservation ↑ | Replacement hit ↑ | Old persistence ↓ | Clean switch ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear probe, HS24 | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Qwen2.5-7B released NLA L20 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |
| Gemma Vanilla NLA L32 | .0000 | .0000 | N/A | N/A | .0000 | .0000 | N/A |
| Medical-NLA L32 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

Probe의 deletion original hit 1.0은 삭제 전 target cue를 모두 읽었다는 뜻이다. Phantom .3593은
삭제 후에도 해당 cue를 양성으로 유지한 비율이며, removal .6407은 그 보수 관계다. Clean switch는
old value를 버리고 new value만 선택한 비율이다.

### 5.4 현재 답할 수 있는 것과 없는 것

현재 확인된 사실은 다음과 같다.

1. DDXPlus probe는 P0 activation에서 환자 finding/value를 높은 정확도로 읽는다.
2. 공개 Gemma Vanilla NLA는 의료 ontology claim을 추출하지 못하고 generic clinical text를
   생성했다.
3. Clinical SFT는 의료 형식과 NLL을 학습했지만 stable patient-specific grounding을 만들지
   못했다.
4. Long-budget ranking은 deletion activation 전체를 억제하는 shortcut을 키웠고,
   specificity anchor로 그 shortcut을 막자 changed-cue 효과도 사라졌다.
5. 공개 AR는 의료 분포에서 valid reconstruction reward가 아니었고, Patchscope도 clinical
   correspondence positive control을 통과하지 못했다.

따라서 지금은 RQ3의 probe positive control과 Vanilla NLA negative control만 실제 숫자가 있고,
RQ1/RQ2의 Gemma baseline과 최종 Medical-NLA 셀은 남아 있다. 이 빈칸을 실패값으로 임의 채우지
않는다.

## 6. Discussion

이 논문이 해결하려는 문제는 CoT보다 더 좋은 정답 rationale을 생성하는 것이 아니다. CoT는
모델이 스스로 생성한 visible explanation이고, 최근 의료 연구는 이 설명이 input change,
decision cause와 clinical facts를 실제로 반영하는지 의심해야 함을 보여준다. Probe와 SAE는
hidden state를 직접 보지만 각각 closed labels와 feature interpretation에 묶인다. Medical-NLA는
두 계열 사이에서 hidden activation을 자연어로 읽되, 그 자연어를 다시 counterfactual 및
activation-swap으로 검증하는 interface를 목표로 한다.

한계도 명시한다. 첫째, 현재 target backbone과 primary extraction site는 Gemma-3-12B의 P0/HS32
하나다. 둘째, semantic mapper와 clinical judge에 의존한다. 셋째, activation에서 정보가
decodable하다는 사실은 backbone이 그 정보를 final prediction에 인과적으로 사용했다는 뜻이
아니다. 넷째, 공개 Qwen NLA는 다른 backbone/site이므로 외부 재현 baseline이지 same-backbone
ablation이 아니다.

## 7. Conclusion

> 의료 LLM의 정답률이 높아져도 visible CoT가 실제 판단 근거와 내부 상태를 faithful하게
> 설명한다고 보장할 수 없다. Medical-NLA는 pre-generation hidden activation을 환자별 임상
> state report로 언어화하고, 임상 변경에 대한 반응성, 발화 내용의 사실성, own-vs-shuffled
> activation dependence를 함께 검증한다. 세 조건을 모두 만족할 때만 이를 faithful clinical
> activation reader라고 부른다.

## 8. 다음 실행 순서

1. RQ1 공개 perturbation population에서 Gemma CoT와 Gemma Vanilla NLA를 동일 case pair로
   생성하고 Registration/EDR을 측정한다.
2. RQ2 동일 Gemma 출력에 frozen sentence splitter와 style-blind clinical judge를 적용한다.
3. RQ3 HS32 linear probe를 NLA와 동일 same-diagnosis donor population에서 재집계한다.
4. 공개 Qwen2.5-7B L20 NLA를 같은 의료 task에 별도-backbone external block으로 재실행한다.
5. 성공한 Medical-NLA checkpoint가 나온 뒤 같은 frozen protocol로 세 표의 마지막 행을 한 번에
   채운다.

## 교수님께 확인할 사항

1. 중심 주장을 reasoning improvement가 아니라 patient-specific activation verbalization으로
   고정하는가?
2. Related Work를 `Faithfulness of CoT`와 `Natural Language Autoencoders` 두 축으로 두는가?
3. 2025-2026 published 수치는 reported baseline block으로 보존하고, 직접 결론은 동일 Gemma에서
   다시 측정한 CoT/Vanilla NLA/Medical-NLA 비교에서 내는가?
4. RQ1 perturbation responsiveness, RQ2 clinical factuality, RQ3 direct activation dependence의
   세 표를 본문 결과표로 두는가?
5. Qwen2.5-7B 공개 NLA는 중요한 외부 baseline으로 포함하되 Medical-NLA나 same-backbone
   comparison으로 표현하지 않는가?
