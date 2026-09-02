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

### 1.1 정답 정확도에서 안정성과 설명가능성으로

최근 의료 LLM은 의료 질의응답과 진단 benchmark에서 높은 정답률을 보인다. 그러나 임상적으로
신뢰할 수 있는 모델이 되려면 평균 정확도만으로는 충분하지 않다. 같은 의미를 유지하는 작은
입력 변화에 판단이 불필요하게 흔들리지 않아야 하고, 반대로 진단에 중요한 근거가 바뀌면 그
변화를 인식해 판단을 갱신해야 한다. 또한 임상의가 모델이 어떤 환자 근거를 읽었고 무엇을
무시했으며 왜 특정 진단 방향으로 기울었는지 검토할 수 있어야 한다. 본 논문은 이 두 요구를
각각 **안정성(stability)** 과 **설명가능성(explainability)** 의 문제로 본다.

[MedOmni-45°, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40864)은
의료 reasoning 모델의 안전성 축에 CoT faithfulness와 sycophancy를 포함한다.
[Trustworthy Medical Question Answering, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1398/)도
의료 QA의 신뢰성을 factuality, robustness, safety, explainability, calibration 등으로 나누며
정답률 하나로 평가할 수 없음을 정리한다. 즉 최근 평가는 “정답을 맞혔는가”를 넘어 “근거 변화에
일관되고 적절하게 반응하는가”와 “그 동작을 사람이 감사할 수 있는가”를 함께 묻기 시작했다.

Chain-of-Thought(CoT)는 원래 추론 성능을 높이는 prompting 및 학습 수단이지만, 자연어로 판단
과정을 제시한다는 이유로 모델 동작을 설명하는 **visible explanation interface**로도 활용되어
왔다. 원 논문인 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models,
NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/9d5609613524ecf4f15af0f7b31abca4-Abstract-Conference.html)은
CoT의 주목적을 복잡한 추론 성능 향상으로 두면서도, 중간 reasoning step이 모델 행동을 볼 수
있는 해석 가능한 창과 오류 경로를 디버깅할 기회를 제공한다고 설명했다. 이후
[Faithful Chain-of-Thought Reasoning, 2023](https://arxiv.org/abs/2301.13379)은 이 가능성을
명시적으로 explainability 문제로 가져와, 자연어·기호 reasoning chain을 외부 solver가 실행하게
함으로써 적어도 final answer가 공개된 chain에서 실제로 도출되도록 만들었다.

의료 분야에서도 CoT를 단순한 정답률 향상 기법을 넘어 임상의가 읽는 설명으로 사용한 선례가
있다.

1. [Large Language Models Encode Clinical Knowledge, Nature 2023](https://www.nature.com/articles/s41586-023-06291-2)은
   Med-PaLM 평가에 few-shot, CoT와 self-consistency prompting을 사용하고, 임상의가 long-form
   output에서 올바르거나 잘못된 medical comprehension, knowledge retrieval와 reasoning의
   증거를 직접 평가했다. 이 연구의 중심은 성능과 임상 답변 품질이지만, 생성된 reasoning을
   사람이 검토 가능한 대상으로 취급한 초기 의료 사례다.
2. [Diagnostic Reasoning Prompts Reveal the Potential for Large Language Model
   Interpretability in Medicine, npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01010-1)은
   진단과 함께 clinical reasoning rationale을 출력하면 임상의가 그 사실적·논리적 정확성을
   검토해 답을 감사할 수 있다고 명시했다. GPT-4 rationale 100개를 평가했을 때 논리 오류는
   오답 rationale의 65%, 정답 rationale의 18%에서 발견됐고, Figure 3은 rationale을 이용한
   임상의 검토 workflow를 제안한다.
3. [MedCoT, EMNLP 2024](https://aclanthology.org/2024.emnlp-main.962/)는 Med-VQA가 answer
   accuracy에 집중한 나머지 reasoning path와 interpretability를 간과했다고 지적했다. Initial
   Specialist가 diagnostic rationale을 생성하고 Follow-up Specialist가 이를 검증한 뒤 여러
   expert가 합의하는 명시적 reasoning-chain 구조로 accuracy와 interpretability를 함께
   개선하려 했다.
4. [The Effect of Medical Explanations from Large Language Models on Diagnostic Accuracy
   in Radiology, npj Digital Medicine 2026](https://www.nature.com/articles/s41746-026-02619-0)은
   20개 임상 사례와 101명 radiologist, 총 2,020개 평가에서 diagnosis-only, differential,
   CoT explanation 제공 조건을 비교했다. CoT는 의사가 설명의 plausibility를 검토할 수 있게
   하는 인터페이스로 사용됐고 전체 진단 정확도가 가장 높았지만, 잘못된 설명이 사용자를
   오도할 가능성도 함께 분석했다.

이 계보에서 CoT의 설명가능성은 “모델의 계산을 직접 관측한다”는 뜻이 아니라, 모델이 주장하는
판단 근거를 사람이 읽고 검토할 수 있다는 **접근 가능성**을 뜻한다. 별도 내부 분석 장치 없이
사용할 수 있다는 장점 때문에 의료 explainability에 널리 활용됐지만, 바로 이 지점에서
faithfulness 문제가 남는다.

하지만 CoT는 hidden state를 직접 관측한 값이 아니다. 출력 시점에 모델이 사후 생성한
**자기보고형 설명(self-reported explanation)** 이므로, 유창하고 의학적으로 그럴듯해도 실제
내부 상태나 decision-driving evidence와 일치하지 않을 수 있다. 따라서 CoT의 문장 품질을
확인하는 것과 모델의 안정성·설명가능성을 검증하는 것은 구분해야 한다.

### 1.2 CoT 기반 안정성·설명가능성 검증의 한계

최근 연구는 CoT가 reasoning을 잘 수행하는지에만 머물지 않고, CoT를 모델 동작의 설명으로
사용할 때 발생하는 문제를 서로 다른 방식으로 확인한다.

1. [Language Models Don't Always Say What They Think, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/ed3fea9033a80fea1376299fa7863f4a-Abstract.html)는
   CoT를 모델의 문제 해결 과정으로 해석하면 투명성과 안전성에 도움이 될 수 있다는 기대를
   직접 검증했다. 그러나 biasing feature가 answer를 바꾸어도 모델은 그 영향을 밝히지 않고,
   선택된 답을 뒷받침하는 그럴듯한 reasoning을 사후 생성할 수 있음을 보였다.
2. [Walk the Talk?, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b5ec50eb177908f21f78ed0d76ed525c-Abstract-Conference.html)는
   MedQA 임상 개념을 counterfactual하게 바꾸고 모델의 설명이 실제 decision-driving evidence를
   올바르게 밝히는지 검사했다. 그럴듯한 설명이 어떤 근거가 결정에 영향을 주었는지 잘못 말할 수
   있음을 보였다.
3. [Faithful or Just Plausible?, NeurIPS 2025 Workshop/2026 preprint](https://arxiv.org/abs/2603.13988)는
   causal ablation, positional bias, hint injection으로 의료 CoT를 검사했다. CoT step이 예측을
   인과적으로 만들지 않거나, 외부 hint를 사용하고도 이를 설명에서 밝히지 않는 현상을 보고했다.
4. [Evaluating Reasoning Faithfulness in Medical VLMs, 2025](https://arxiv.org/abs/2510.11196)는
   임상 text와 image cue를 통제해 answer accuracy와 explanation quality가 분리될 수 있음을
   확인했다. injected cue를 언급하는 것만으로 실제 grounding이 보장되지도 않았다.
5. [Better Accuracies, Worse Reasoning, 2026](https://arxiv.org/abs/2605.28301)은
   CoT distillation으로 MedQA 정답 성능과 calibration은 좋아졌지만, 같은 style-blind audit에서
   non-abstained reasoning-step 오류율이 Qwen3-8B 기준 30.6%에서 50.3%로 증가했음을 보였다.
   즉 더 높은 accuracy와 더 정확한 reasoning trace가 반대 방향으로 움직일 수 있다.
6. [Right Diagnoses, Decorative Reasoning, 2026](https://arxiv.org/abs/2608.24790)은
   14개 모델, 4개 의료 QA benchmark, 30개 임상 perturbation operator를 사용했다. 임상적으로
   의미 있는 destructive edit에서 chain이 변경을 등록하지 않고 answer도 유지한 CDR이 전체
   평균 72.9%였다. CoT corruption과 CoT prompt 제거도 accuracy를 거의 떨어뜨리지 않았다.
7. [Auditing Evidence Use in Medical LLM Diagnosis, 2026](https://arxiv.org/abs/2607.20848)은
   DDXPlus, CupCase, MedCase의 evidence subset을 통제하고 diagnostic margin을 분석했다. 높은
   진단 정확도만으로 환자 근거를 적절히 사용했는지 알 수 없음을 행동 수준에서 보였다.

이 논문들이 공통으로 말하는 것은 “CoT가 나쁘다”거나 “reasoning을 사용하면 안 된다”는 것이
아니다. 핵심은 **CoT가 유창하고 답이 맞다는 사실만으로 모델이 임상 근거 변화에 안정적으로
반응했다고 볼 수 없고, 그 CoT가 실제 model state 또는 decision process를 faithful하게
설명했다고도 볼 수 없다**는 것이다. 즉 문제는 reasoning 성능 자체보다 CoT를 안정성과
설명가능성의 대리 측정치로 그대로 믿을 수 있느냐에 있다.

### 1.3 왜 내부 상태를 관찰해야 하는가

기존 CoT audit은 input, visible chain, answer를 조작하여 안정성과 faithfulness를 행동적으로
추론한다. 이는 중요하지만 모델이 환자 사례를 읽은 직후 실제로 형성한 hidden activation을
직접 설명하지는 않는다. 이 때문에 내부 상태를 관찰하자는 접근이 필요하지만, 기존 내부
관찰 도구에도 각자의 강점과 한계가 있다.

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

### 2.1 Explanation Faithfulness

Faithfulness는 모델이 아니라 **설명의 속성**이다: 설명(CoT, self-report, rationale)이
모델의 실제 계산 과정을 반영하는가를 묻는다(Jacovi & Goldberg, ACL 2020). 선행 연구는
그 "실제 과정"의 기준을 input perturbation, hint 주입, unlearning 같은 **행동 수준**으로
조작화했다. 본 연구는 같은 질문의 기준을 hidden state 수준으로 옮긴다. CoT는 이 문헌이
다루는 지배적 설명 형식이므로 아래 검토도 CoT 연구가 중심이지만, 절의 대상은 설명
일반이며 3.2의 EDR이 CoT와 NLA readout을 같은 계약으로 평가하는 근거가 된다.

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

이 가운데 두 편은 본 논문 RQ1/RQ2가 **표를 그대로 재현하는 anchor**이므로 별도로
상술한다. 1.2가 이들의 발견을 인용했다면, 여기서는 원 표의 구조와 우리의 재현 계약을
적는다.

#### 2.1.1 Anchor 1 — Right Diagnoses, Decorative Reasoning (RQ1 표)

원 표는 14개 모델 × 4개 의료 QA benchmark에 30개 임상 perturbation operator를
적용하고, 주 지표로 **CDR**(임상적으로 의미 있는 변경을 chain이 등록하지 않고
answer도 유지한 비율; 전체 평균 72.9%)을 보고한다. Fact ablation, demographic swap,
irrelevant distractor, negation flip, severity reversal, temporal shift의 M-block
operator 정의가 공개돼 있다.

- **그대로 가져오는 것**: M-block operator 정의, case-pair 구성, CDR 정의와 공개 수치.
- **추가하는 것**: 평가 대상 설명을 CoT에서 NLA readout까지 확장한 **EDR**과
  same-backbone Gemma 행. 공개 수치는 CDR 이름으로 보존하고, 우리 explanation 출력은
  같은 case pair에서 재측정한다(3.2.1, 5.1의 계약).

#### 2.1.2 Anchor 2 — Better Accuracies, Worse Reasoning (RQ2 표)

원 표는 CoT distillation 전후로 answer accuracy/calibration과 style-blind
reasoning-step 오류율을 대조한다(Qwen3-8B 기준 step 오류율 30.6% → 50.3%). 이
논문이 재는 것은 엄밀히는 faithfulness(설명이 실제 원인을 반영하는가)가 아니라
**explanation factuality**(설명 문장 자체가 임상적으로 정확한가)다 — visible
explanation을 신뢰할 수 없다는 같은 감사 문헌의 **두 번째 축**이며, 두 축을
구분해서 쓴다.

- **그대로 가져오는 것**: style-blind sentence/step 단위 임상 사실성 audit 계약.
- **추가하는 것**: 같은 audit을 CoT뿐 아니라 Vanilla NLA와 Medical-NLA 출력에
  적용하는 method-neutral 행(3.2.2, 5.2의 계약).

Medical-NLA는 이 연구들을 “CoT보다 더 잘 reasoning하는 모델”로 대체하려는 것이 아니다.
이들이 행동 수준에서 발견한 visible self-report의 간극을 **hidden-state readout**으로 직접
측정하려는 것이다.

### 2.2 Interpreting Hidden Representations

Hidden representation을 해석하는 방법은 출력 형식으로 세 계열로 나뉜다. **닫힌
decoder** — linear probe, SAE, transcoder, logit/tuned lens — 는 사전 정의된 label/feature
공간으로만 읽으므로 검증은 정확하지만 열린 임상 서술을 만들지 못한다(각 도구의 한계는
1.3). **무학습 자연어 방법** — Patchscope, SelfIE — 는 activation을 target context에 넣어
자연어 continuation으로 읽지만 prompt·patch 위치에 민감하다. **학습형 자연어 방법** —
LatentQA(query-conditioned activation QA), Activation Oracle(범용 activation 해석기),
그리고 NLA — 이 본 논문이 확장하는 계열이다.

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

NLA가 인접 계열과 다른 점은 activation을 독립적인 자연어 bottleneck으로 만들고
reconstruction을 정보 보존 신호로 쓴다는 것이다. 본 논문은 이 NLA 인터페이스에 의료
임상성 및 patient-specific grounding constraint를 추가한다.

두 절을 합치면 지형이 하나의 행렬로 정리된다. 2.1의 검사 프로토콜은 완성돼 있으나
기준이 행동 수준이고, 2.2에서 자연어를 내는 방법은 의료·사례별 faithfulness 검증이
없으며, 검증이 정확한 방법(probe)은 자연어가 없다:

| | activation 직접 입력 | 자유 자연어 출력 | 의료 학습 | 사례별 faithfulness 검증 |
|---|:---:|:---:|:---:|:---:|
| CoT | X | O | (모델별) | X — 2.1의 결과 |
| Linear probe | O | X | O (본 연구 선행) | 제한적 (closed label) |
| SAE / Transcoder | O | X | X | X |
| Patchscope / SelfIE | O | O | X | X |
| 공개 NLA / LatentQA / AO | O | O | X | 일반-domain만 |
| **Medical-NLA (제안)** | O | O | O | **본 논문의 검증 대상** |

마지막 행의 마지막 칸은 주장이 아니라 3.2의 RQ1–RQ3가 판정할 빈칸이다. 이 교차점 —
행동 감사의 질문을 hidden-state 판독으로 측정하는 의료 방법 — 이 현재 문헌에서 비어
있는 자리다.

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

Medical-NLA는 특정 backbone 하나에 묶인 checkpoint 이름이 아니라 동일한 의료 적응 recipe다.
이를 두 공개 NLA 계열에 각각 적용한다.

- **Qwen Medical-NLA:** Qwen2.5-7B-Instruct의 P0, layer 20 last-token residual과 공개
  `nla-qwen2.5-7b-L20-av` 초기화를 사용한다.
- **Gemma Medical-NLA:** Gemma-3-12B-IT의 P0, layer 32 last-token residual과 공개
  `nla-gemma3-12b-L32-av` 초기화를 사용한다.

P0는 모델이 임상 사례를 모두 읽었지만 visible answer나 CoT를 생성하기 전이다. 따라서 두
구현 모두 answer text leakage 없이 사례를 통합한 pre-generation state를 읽는다. 각 backbone의
activation 차원과 native extraction layer가 다르므로 하나의 AV checkpoint를 공유하지 않고,
동일한 target construction, loss, sampling, counterfactual constraint와 promotion gate를 적용한
별도 checkpoint를 학습한다.

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

두 구현에서 공유하는 것과 backbone별로 달라지는 것은 다음처럼 고정한다.

| Contract item | Qwen Medical-NLA | Gemma Medical-NLA |
|---|---|---|
| Clinical cases and targets | 동일 DDXPlus train split, 동일 finding/value text target | 동일 DDXPlus train split, 동일 finding/value text target |
| Frozen source model | Qwen2.5-7B-Instruct | Gemma-3-12B-IT |
| Activation site | P0 last-token, L20 | P0 last-token, L32 |
| AV initialization | released Qwen L20 AV | released Gemma L32 AV |
| Activation corpus | Qwen으로 전체 train/counterfactual을 다시 forward | Gemma로 전체 train/counterfactual을 다시 forward |
| Optimization contract | 동일 loss 정의, sampling rule, seed, budget 원칙 | 동일 loss 정의, sampling rule, seed, budget 원칙 |
| Promotion contract | 동일 RQ1/RQ2 validation gate와 backbone 내부 RQ3 own-shuffled gate | 동일 RQ1/RQ2 validation gate와 backbone 내부 RQ3 own-shuffled gate |

즉 “같은 Medical-NLA 방법”은 weight나 activation을 공유한다는 뜻이 아니라, **같은 의료
supervision과 학습·판정 알고리즘을 각 backbone의 native NLA interface에 적용한다**는 뜻이다.

현재 위 시도들은 Gemma 계열에서 validation promotion gate를 통과하지 못했다. 공개 AR도 이
의료 분포에서 valid reconstruction instrument로 인정되지 않았다. Qwen 계열에는 같은 recipe를
적용한 의료 적응 실행 자체가 아직 없다. 따라서 아래 결과표의 Qwen/Gemma `Medical-NLA` 행은
최종 목표를 명시하기 위한 행이며 각 backbone에서 성공한 checkpoint가 생기기 전에는 `미측정`으로
둔다. 한 backbone의 실패 checkpoint나 다른 backbone의 수치를 옮겨 쓰지 않는다.

### 3.2 Faithfulness Evaluation Framework

#### 3.2.1 RQ1: Perturbation-based explanation responsiveness

원본 사례 $X_i$와 한 임상 근거를 변경한 $X'_i$를 같은 target model에 넣는다. 각 조건에서
P0 activation을 추출하고 CoT, Vanilla NLA, Medical-NLA explanation을 생성한다. fact ablation,
demographic swap, irrelevant distractor, negation flip, severity reversal, temporal shift는
`Right Diagnoses, Decorative Reasoning`의 M-block 정의를 유지한다.

설명이 변경된 사실을 등록했는지 $U_Z$, final answer가 바뀌었는지 $U_Y$로 표시한다.

\[
\mathrm{EDR}=P(\neg U_Z\land\neg U_Y)
\]

EDR(Explanation-Decoupling Rate)은 CDR의 `chain`을 CoT와 NLA를 포괄하는 `explanation`으로
확장한 이름이다. 낮을수록 좋다. 원 논문의 CDR 수치를 EDR이라고 단순히 이름만 바꿔 재사용하지
않고, CoT 행에서는 원 논문의 CDR 정의와 수치를 그대로 EDR의 CoT 특수 경우로 표시하며 우리
NLA 출력은 같은 case pair에서 다시 측정한다. 표에는 원본 문제에서의 answer accuracy와 EDR만
보고한다. Accuracy는 모델의 진단 능력에 대한 문맥이고, EDR이 설명의 변경 미등록을 측정하는
핵심값이다. NLA는 backbone answer를 바꾸지 않으므로 같은 backbone의 CoT/NLA 행은 동일한
answer accuracy를 공유한다.

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

예를 들어 환자 A와 B가 같은 진단이지만 A에만 발열이 있다고 하자. A activation을 읽은 설명은
A finding과 높은 F1을 보여야 한다. 반대로 decoder 설정은 그대로 두고 B activation을 넣으면,
A reference에 대한 F1이 낮아져야 한다. 두 조건의 차이가 activation dependence gap이다.

주 분석은 환자 난이도를 상쇄하는 symmetric 2x2 pair score지만, 본문 표에는 이를 직관적인
`Own F1`, `Shuffled F1`, `Own-Shuffled gap`으로 요약한다.

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

### 4.1 RQ별 데이터와 평가 모집단

| RQ | 근거 논문/benchmark | 원 논문 모집단 | 이 연구의 직접 비교 모집단 | 주 평가 단위 |
|---|---|---|---|---|
| RQ1: perturbation responsiveness | `Right Diagnoses, Decorative Reasoning` (2026) | MedQA, MedMCQA, PubMedQA, medical MMLU 각 200문항; M-block 13 variants | 같은 800문항에서 Qwen2.5-7B와 Gemma-3-12B 각각의 CoT/Released NLA/Medical-NLA를 실행. EDR은 실제로 fired한 M2/M4/M5 destructive edits만 사용 | 원본-변형 question pair |
| RQ2: clinical factuality | `Better Accuracies, Worse Reasoning` (2026), Appendix E sentence-chunk control | MedQA-USMLE test의 first 500 questions; Qwen3-8B Base/Distilled CoT | 같은 first-500 MedQA에서 Qwen 및 Gemma의 CoT/NLA를 동일 sentence splitter와 style-blind judge로 재평가 | question 안의 sentence chunk |
| RQ3: direct activation dependence | 이 연구의 DDXPlus benchmark | validation 4,525 originals / 4,106 same-diagnosis hard-shuffle pairs; locked test 4,543 originals / 4,121 pairs | DDXPlus P0에서 동일 진단이지만 finding set이 다른 own/donor activation pair. Qwen 계열은 L20, Gemma 계열은 L32로 backbone 내부 site를 맞춤 | symmetric patient pair |

RQ1 원 논문은 네 benchmark에서 각 200문항을 사용하지만 reference Python과 decoder별 세부 설정은
저자 요청 방식이다. Appendix F의 regex, 치환표와 seed 규칙으로 operator를 재구현하고, 정확한
sample ID를 받지 못하면 published 행은 `reported`, 우리 실행은 `re-implemented population`으로
구분한다. 원 논문의 open-model CDR 분모는 M2/M4/M5가 실제로 발화한 965개 pair다. non-firing
operator는 성공으로 세지 않고 분모에서 제외한다.

RQ2의 직접 비교는 원 논문의 전체 1,273문항 primary step audit이 아니라, 형식 차이를 줄이기 위해
사용한 **first-500 sentence-chunk control**을 따른다. 원 논문의 Qwen3-8B 수치는 Kimi-K2.6
style-blind judge 결과다. 같은 judge/version을 재현하지 못하면 published Qwen 수치와 우리 수치의
절대값 비교를 주장하지 않고, 같은 frozen judge로 채점한 Qwen 내부 및 Gemma 내부의 paired
comparison을 주 결론으로 사용한다.

RQ3는 외부 논문의 표를 옮기는 실험이 아니다. DDXPlus의 explicit finding/value annotation과
counterfactual activation을 이용해 새로 정의한 benchmark다. validation에서 donor 규칙, HS32
site, semantic mapper와 threshold를 고정하고 locked test 4,121 pair는 마지막에 한 번만 평가한다.

### 4.2 Baseline 모델과 역할

| Baseline/method | Backbone 및 site | 생성 또는 예측 대상 | RQ1 | RQ2 | RQ3 | 수치 출처/필요 작업 |
|---|---|---|---:|---:|---:|---|
| Published CoT panel | Mistral-7B, Qwen2.5-7B/14B, Llama-3.1-8B, Gemma-2-9B, BioMistral-7B, Meditron-7B, Med42-8B, OpenBioLLM-8B, HuatuoGPT-o1, DeepSeek-R1-D | visible CoT + answer | O | X | X | RQ1 원 논문 Table 3의 Acc/CDR를 reported block으로 사용 |
| Qwen3-8B Base/Distilled CoT | Qwen3-8B | sentence-chunked visible CoT | X | O | X | RQ2 원 논문 Appendix E의 first-500 reported result |
| Qwen CoT | Qwen2.5-7B-Instruct | visible rationale + answer | O | O | X | 두 공개 population에서 신규 실행; Qwen NLA의 same-backbone visible-output control |
| Gemma CoT | Gemma-3-12B-IT | visible rationale + answer | O | O | X | 두 공개 population에서 신규 실행; same-backbone visible-output control |
| Qwen Released NLA | Qwen2.5-7B-Instruct, pre-answer last-token L20, public AV/AR | free-text activation report | O | O | O | 공개 checkpoint와 Blake Masters MedQA 구현 존재; 공통 metric으로 신규 재채점 |
| Gemma Released NLA | Gemma-3-12B-IT, P0/HS32, public AV/AR | free-text activation report | O | O | O | same-backbone Vanilla NLA baseline; DDXPlus locked 생성/semantic score 완료 |
| Qwen linear probe | Qwen2.5-7B-Instruct, P0/L20 | 91 finding probabilities | X | X | O | Qwen RQ3의 closed-label positive control; 새 학습 필요 |
| Gemma linear probe | Gemma-3-12B-IT, P0/HS32 | 91 finding probabilities | X | X | O | Gemma RQ3의 closed-label positive control; HS32 재집계 필요 |
| SAE | Gemma-3-12B-IT, P0/HS32 | sparse feature activations | X | X | O* | feature-to-finding mapping을 validation에서 동결한 경우에만 포함 |
| Qwen Medical-NLA | Qwen2.5-7B-Instruct, P0/L20 | patient-specific diagnostic-state report | O | O | O | 같은 의료 적응 recipe를 Qwen AV에 적용; 성공 checkpoint 뒤 세 frozen protocol로 평가 |
| Gemma Medical-NLA | Gemma-3-12B-IT, P0/HS32 | patient-specific diagnostic-state report | O | O | O | 같은 의료 적응 recipe를 Gemma AV에 적용; 성공 checkpoint 뒤 세 frozen protocol로 평가 |

`O*`는 SAE feature가 어떤 clinical finding을 뜻하는지 held-out validation에서 먼저 판독한 경우만
가능하다는 뜻이다. Linear probe와 SAE는 자유문 explanation을 만들지 않으므로 RQ1의 EDR이나
RQ2의 sentence-chunk error 표에 억지로 넣지 않는다. 대신 RQ3에서 activation에 존재하는
patient-specific information의 closed-label ceiling/control 역할을 한다.

Qwen Released NLA는 `kitft/nla-qwen2.5-7b-L20-av/ar`를 사용한다. 기존 medical NLA preprint는
MedQA 200문항에 canonical/compact/option-shuffle 세 prompt를 적용해 answer accuracy 57.5%,
reconstruction cosine .828, heuristic alignment 5.5%, MedGemma alignment 77.7%를 보고했다. 이
값은 관련 연구의 reported result일 뿐 RQ1 EDR/RQ2 chunk error/RQ3 activation gap이 아니므로,
세 main table에 넣으려면 Qwen L20 activation과 설명을 각 protocol에서 다시 생성·채점한다.

Lens, transcoder와 Patchscope는 기능 비교와 appendix에 둔다. 현재 Patchscope는 general-domain
control은 통과했지만 DDXPlus own/donor correspondence가 0/5여서 main baseline으로 승격하지
않는다.

### 4.3 공통 모델 설정

- **Target source models:** `Qwen/Qwen2.5-7B-Instruct`와 `google/gemma-3-12b-it`.
- **Native activation sites:** answer/explanation을 생성하기 전 마지막 prompt token의 Qwen
  P0/L20 및 Gemma P0/HS32 residual activation. Backbone 내부 직접 비교는 각각 L20과 HS32로
  맞추고, 서로 다른 차원의 activation을 backbone 사이에서 교환하지 않는다.
- **Gemma CoT:** activation decoder가 아니라 동일 source model의 visible self-report baseline이다.
- **Gemma Released NLA:** `kitft/nla-gemma3-12b-L32-av`; 공개 일반-domain verbalizer다.
- **Qwen CoT/Released NLA:** 동일 Qwen source model의 visible self-report와
  `kitft/nla-qwen2.5-7b-L20-av`를 사용한 일반-domain activation report다.
- **Medical-NLA:** 각 backbone의 native activation과 공개 AV 초기화에 동일한 의료 supervision,
  target construction, patient-specific grounding constraint와 promotion gate를 적용한다.

각 backbone 안에서 CoT/Released NLA/Medical-NLA는 같은 원본·변형 question과 같은 backbone
answer를 공유한다. 따라서 RQ1/RQ2의 answer accuracy는 설명 방법별 성능이 아니라 해당
source-model의 공통 문맥값이며, 방법 간 핵심 차이는 EDR과 sentence-chunk error다.

### 4.4 Reported 결과와 신규 측정의 경계

| 항목 | 그대로 가져올 수 있는 값 | 반드시 새로 측정할 값 |
|---|---|---|
| RQ1 | 원 논문 Table 3의 11개 open/reasoning CoT `Acc`, `CDR` | Qwen/Gemma 각각의 CoT, Released NLA, Medical-NLA EDR |
| RQ2 | Qwen3-8B Base/Distilled first-500의 answer accuracy, sentence-chunk error, uncertain/chunk counts | Qwen/Gemma 각각의 CoT, Released NLA, Medical-NLA를 같은 splitter/judge로 채점한 값 |
| RQ3 | 외부 published 값 없음; 기존 HS24 probe 값은 선행 positive control | Qwen L20/Gemma HS32 probe와 각 backbone의 Released NLA/Medical-NLA own-shuffled gap |

즉 2026 논문 숫자는 빈칸을 임의로 채우는 용도가 아니라 `reported baseline`으로 보존한다. 우리
방법과의 직접 우열은 같은 case, backbone, activation site, decoder 조건 또는 최소한 같은 frozen
scorer를 사용해 신규 계산한 행에서만 주장한다.

### 4.5 Common controls and statistics

모든 직접 비교에서 case population, decoding parameters, semantic mapper와 judge를 고정한다.
Generated method name은 judge에 제공하지 않는다. RQ1/RQ2는 같은 case의 paired difference를,
RQ3는 same-diagnosis symmetric pair와 diagnosis-category cluster bootstrap 95% CI를 사용한다.
공개 논문의 숫자는 `reported`, 이 프로젝트가 다시 계산한 값은 `ours`로 표시한다.

## 5. Experimental Results

### 5.1 RQ1: 임상 근거 변경을 설명이 등록하는가?

`Right Diagnoses, Decorative Reasoning` Table 3의 4개 의료 benchmark 평균을
published CoT reference로 사용한다. 메인 표는 설명 방법을 가로로 놓고 `Acc.`와
`EDR`을 세로로 배치해, 같은 backbone의 CoT·released NLA·Medical-NLA를 읽기 쉽게
비교한다. 공개 CoT 행의 CDR은 method-neutral EDR의 CoT 특수 경우로 표시한다.

| Metric | Mistral-7B CoT (reported) | Qwen2.5-7B CoT (reported) | Gemma-2-9B CoT (reported) | DeepSeek-R1-D CoT (reported) | Qwen2.5-7B CoT (ours) | Qwen2.5-7B Released NLA L20 | Qwen2.5-7B Medical-NLA L20 | Gemma-3-12B CoT (ours) | Gemma-3-12B Released NLA L32 | Gemma-3-12B Medical-NLA L32 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Acc. ↑ | .52 | .54 | .67 | .60 | 미측정 | Qwen CoT (ours)와 공유* | Qwen CoT (ours)와 공유* | 미측정 | Gemma CoT (ours)와 공유* | Gemma CoT (ours)와 공유* |
| EDR ↓ | .80 | .94 | .75 | .51 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 | 미측정 |

`*` NLA는 정답을 다시 푸는 모델이 아니라 같은 source-model activation의 설명기다. 따라서
두 Qwen NLA의 `Acc.`는 `Qwen CoT (ours)`와 동일한 Qwen2.5-7B 실행의 answer accuracy를,
두 Gemma NLA의 `Acc.`는 `Gemma CoT (ours)`와 동일한 Gemma-3-12B 실행의 answer accuracy를
공유한다. 공개 표의 `.54`를 Qwen NLA 셀에 바로 복사하지 않고, 같은 재구현 perturbation
population에서 activation과 NLA를 생성한 실행이 확인된 뒤 공유값으로 기입한다.

#### Published CoT reference block

메인 표에서 생략한 공개 baseline까지 포함한 원 논문 Table 3 수치는 아래와
같다. 이 block은 보고된 수치의 출처 원장이며 우리 NLA와의 same-backbone 직접 비교표가
아니다.

| Model | Acc. ↑ | CDR (= CoT EDR) ↓ |
|---|---:|---:|
| Mistral-7B | .52 | .80 |
| Qwen2.5-7B | .54 | .94 |
| Llama-3.1-8B | .51 | .72 |
| Gemma-2-9B | .67 | .75 |
| Qwen2.5-14B | .68 | .72 |
| BioMistral-7B | .38 | .96 |
| Meditron-7B | .24 | .26 |
| Med42-8B | .66 | .68 |
| OpenBioLLM-8B | .50 | .89 |
| HuatuoGPT-o1 | .55 | .80 |
| DeepSeek-R1-D | .60 | .51 |

EDR은 destructive M-block에서 `explanation no-update AND answer no-flip` 비율이다. 높은 EDR은
정답을 안정적으로 유지했다는 뜻이 아니라, 임상적으로 중요한 변경을 설명과 답이 모두 무시한
decoupling을 뜻한다. 공개 CoT 행은 published result이고, Qwen/Gemma NLA 행은 같은 공개
perturbation population에서 다시 실행해야 한다. 같은 backbone에서 CoT와 NLA는 동일 target
answer를 설명하므로 `Acc.`를 공유하고, 핵심 비교값은 EDR이다.

### 5.2 RQ2: 생성 설명의 임상 문장은 정확한가?

`Better Accuracies, Worse Reasoning` Appendix E의 sentence-chunk control을 모든 자연어 설명에
공통 적용한다. 번호가 있는 CoT step과 자유 형식 NLA를 억지로 같은 reasoning-step으로 부르지
않고, 둘 다 문장 단위 clinical segment로 잘라 `correct/error/uncertain`으로 판정한다.

| Model | Answer acc. (%) ↑ | Sentence-chunk error (%) ↓ | Uncertain chunks (%) ↓ | Chunks/case |
|---|---:|---:|---:|---:|
| Qwen3-8B Base CoT (reported) | 71.6 | 60.1 | 1.83 | 5.24 |
| Qwen3-8B Distilled CoT (reported) | 76.6 | 77.5 | 2.38 | 4.70 |
| Qwen2.5-7B CoT | 미측정 | 미측정 | 미측정 | 미측정 |
| Qwen2.5-7B Released NLA L20 | 미측정 | 미측정 | 미측정 | 미측정 |
| Qwen2.5-7B Medical-NLA L20 | 미측정 | 미측정 | 미측정 | 미측정 |
| Gemma-3-12B CoT | 미측정 | 미측정 | 미측정 | 미측정 |
| Gemma-3-12B Released NLA L32 | 미측정 | 미측정 | 미측정 | 미측정 |
| Gemma-3-12B Medical-NLA L32 | 미측정 | 미측정 | 미측정 | 미측정 |

Answer accuracy `71.6/76.6`은 원 논문 Table 8의 first-500 MedQA Qwen3-8B 실행값이고,
`60.1/77.5`는 같은 first-500 control을 model-agnostic sentence splitter로 다시 자른 Appendix E
값이다. Base는 correct 1,025, error 1,546, uncertain 48개이고, distilled는 correct 516,
error 1,779, uncertain 56개다. 따라서 chunk error는 uncertain을 제외한 committed chunk 중
error 비율이고, uncertain 비율과 chunks/case는 이 공개 count에서 계산했다. Answer accuracy와
chunk error가 함께 있어야 “정답은 좋아졌지만 설명의 임상 문장은 더 부정확해지는” 현상을 볼
수 있다. NLA는 answer를 새로 생성하는 방법이 아니므로 동일 backbone의 CoT/NLA 행은 같은
answer accuracy를 공유한다. 직접 결론은 동일 splitter와 style-blind judge로 다시 측정한 Qwen
및 Gemma 내부 비교에서 낸다.

### 5.3 RQ3: 설명은 해당 환자의 activation에 의존하는가?

환자 A의 activation으로 만든 설명을 A의 finding과 비교한 것이 `Own F1`이다. 같은 진단이지만
finding이 다른 환자 B의 activation을 같은 reader에 넣고 여전히 A의 finding을 기준으로 채점한
것이 `Shuffled F1`이다. Decoder prompt와 설정은 같고 activation만 바뀐다.

| Model/reader | Site | Own F1 ↑ | Shuffled F1 ↓ | Gap [diagnosis-cluster 95% CI] ↑ |
|---|---|---:|---:|---:|
| Linear probe | Gemma HS24 | .9587 | .7938 | +.1624 [미집계] |
| Linear probe | Gemma HS32 | 미집계 | 미집계 | 미집계 |
| Linear probe | Qwen L20 | 미측정 | 미측정 | 미측정 |
| SAE | Gemma HS32 | 미측정 | 미측정 | 미측정 |
| Qwen2.5-7B Released NLA | Qwen L20 | 미측정 | 미측정 | 미측정 |
| Qwen2.5-7B Medical-NLA | Qwen L20 | 미측정 | 미측정 | 미측정 |
| Gemma-3-12B Released NLA | Gemma HS32 | .0000 | .0000 | +.0000 [미집계] |
| Gemma-3-12B Medical-NLA | Gemma HS32 | 미측정 | 미측정 | 미측정 |

해석에는 두 조건이 모두 필요하다. Own F1만 높고 gap이 0에 가까우면 모든 환자에게 같은 질환
template을 출력했을 수 있다. Gap만 높고 Own F1이 낮으면 activation에 따라 문장은 달라지지만
정확한 임상 정보를 읽은 것은 아니다. **높은 Own F1과 0보다 유의하게 큰 gap을 동시에 만족해야**
정확하면서 patient-specific한 activation reader로 해석한다. Donor를 같은 진단에서 고르는
이유는 질환 이름만 구분하는 쉬운 해를 막고 환자별 finding 차이를 실제로 읽는지 보기 위해서다.

CoT는 임의의 hidden activation을 직접 입력받는 reader가 아니므로 이 표에 넣지 않는다. CoT의
input 변화 반응은 RQ1에서 비교한다. Linear probe는 activation에 finding 정보가 실제 존재하는지
보이는 positive control이고, SAE는 feature-to-finding mapping을 validation에서 동결했을 때만
채점할 수 있다.

HS24 probe와 HS32 NLA를 같은-site 성능 비교로 주장하지 않는다. 현재 HS24 probe 값은
activation에 patient finding이 존재한다는 선행 positive control이고, 최종 main comparison을
위해서는 HS32 probe를 동일 donor population에서 재집계해야 한다.

Gemma Vanilla NLA의 0은 결측이 아니다. Frozen semantic mapper가 locked 10,028개 출력에서
인정한 DDXPlus ontology claim이 0개였고, 20-case private audit은 mapper miss가 아니라 generic
clinical text 생성을 원인으로 판정했다.

Cue deletion과 value edit의 `original hit`, `phantom`, `removal`, `retention`, `clean switch`는
이 main table의 원인을 분석하는 appendix diagnostic으로만 유지하고 별도의 본문 RQ 표로 세지
않는다.

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

따라서 지금은 RQ3의 Gemma probe positive control과 Gemma Released NLA negative control만
실제 숫자가 있다. RQ1/RQ2의 Qwen/Gemma baseline과 두 backbone의 최종 Medical-NLA 셀,
RQ3의 Qwen L20 직접 비교는 남아 있다. 이 빈칸을 실패값으로 임의 채우지 않는다.

## 6. Discussion

이 논문이 해결하려는 문제는 CoT보다 더 좋은 정답 rationale을 생성하는 것이 아니다. CoT는
모델이 스스로 생성한 visible explanation이고, 최근 의료 연구는 이 설명이 input change,
decision cause와 clinical facts를 실제로 반영하는지 의심해야 함을 보여준다. Probe와 SAE는
hidden state를 직접 보지만 각각 closed labels와 feature interpretation에 묶인다. Medical-NLA는
두 계열 사이에서 hidden activation을 자연어로 읽되, 그 자연어를 다시 counterfactual 및
activation-swap으로 검증하는 interface를 목표로 한다.

한계도 명시한다. 첫째, 두 target backbone의 native site(Qwen L20, Gemma L32)를 사용하므로
backbone 간 절대 성능 차이에는 모델 크기와 layer 차이가 함께 포함된다. 둘째, semantic mapper와
clinical judge에 의존한다. 셋째, activation에서 정보가
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
   생성하고 shared answer accuracy와 EDR을 측정한다.
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
