# Introduction — Working Draft and Evidence Map (2026-08-25)

이 문서는 실제 논문 Introduction으로 옮길 영어 초안과, 발표자가 각 문장의
근거와 주장 경계를 확인할 수 있는 한국어 메모를 함께 둔다. 인용 키는
`related_work.tex`의 키와 맞춰 최종 `.bib` 작성 때 확정한다.

## 먼저 고정할 논리

### 대전제

> A medical LLM's emitted answer and generated chain of thought need not fully
> represent the diagnostic signals decodable from its internal state. A wrong
> clinical suggestion may change the answer without becoming the dominant
> decodable diagnosis, so behavior, self-report, and activation must be measured
> separately.

이 문장은 `the model knows`, `belief`, `true reasoning`을 주장하지 않는다.
`decodable signal`은 probe가 해당 activation에서 label을 복원할 수 있다는
측정적 진술이다. 그 신호가 생성에 인과적으로 사용됐다는 뜻은 아니다.

### 가설과 연구 질문

| 가설 | 연구 질문 | 반증 또는 약화 조건 |
|---|---|---|
| **H1: internal-output dissociation** | **RQ1.** Wrong referral note가 답을 얼마나 움직이며, moved case의 gold/suggestion/other signal은 prompt landmark를 따라 어떻게 변하는가? | Moved case 대부분에서 suggestion이 내부 top-1이 되면 약화된다. |
| **H2: single-run attribution** | **RQ2.** 숨겨진 no-note 반사실 없이 wrong-note 실행 한 번만 보고 그 note가 answer movement를 일으켰는지 귀속할 수 있는가? | 내부 채널이 동일 입력을 보는 강한 LLM CoT monitor보다 낫지 않거나 heldout에서 붕괴하면 약화된다. |
| **H3: conditional correction** | **RQ3.** Decode한 내부 content를 되먹이면 답을 회복하며, 이득은 재실행·evidence·label·자연어 중 무엇에서 오는가? | 정확한 content가 generic retry/evidence-only보다 낫지 않거나 kept answer 손해를 포함한 순효과가 음수면 실용 주장이 성립하지 않는다. |

AV의 activation specificity는 별도 **M0 measurement gate**다. 이는 연구 질문의
답이 아니라 자연어 판독을 증거로 사용할 수 있는지 검사하는 선행 조건이다.

## English Introduction Draft

Clinical diagnosis rarely begins from an unframed case. A downstream clinician
may receive a referral letter that contains not only symptoms and test results
but also an upstream clinician's provisional diagnosis. Controlled human
studies show that such suggestions can narrow the differential or anchor later
judgments: Staal et al. found that both correct and incorrect referral
suggestions reduced the number of diagnoses considered, and Spaanjaars et al.
found referral-diagnosis effects in a subset of clinicians
\cite{staal2022referral,spaanjaars2015referral}. We therefore study a specific,
clinically plausible workflow in which a diagnostic model receives a case
description followed by a referral note containing a provisional diagnosis.
We do not assume that every referral contains such a suggestion or that all
medical LLM deployments share this workflow.

Medical LLMs are already known to be behaviorally vulnerable to contextual
pressure. BiasMedQA injects clinically motivated cognitive-bias statements into
USMLE questions and reports model-dependent accuracy losses
\cite{schmidgall2024biasmedqa}. MED-STRESS examines abandonment of initially
correct diagnoses under multi-turn pressure \cite{xiao2026medstress}, while
MedMisBench reports a mean accuracy decrease from 71.1\% to 38.0\% under
misleading clinical context \cite{zhou2026medmisbench}. Narrative Anchoring
further shows that diagnostic behavior can change when clinical facts are held
fixed but sociolinguistic framing is altered \cite{singh2026narrative}. These
results establish the behavioral risk. They do not reveal, for an individual
case, whether the misleading suggestion replaced the model's decodable
diagnostic state, merely perturbed answer formation, or redirected the output
to a third diagnosis.

Generated chain of thought does not close this gap by itself. Features that
causally move an answer can be omitted from a model's explanation and replaced
by a plausible rationalization \cite{turpin2023say}, and dependence on a
reasoning trace varies across models and tasks \cite{lanham2023measuring}. In
medicine, causal ablation and hint-injection experiments likewise show that
external suggestions can be incorporated without explicit acknowledgment
\cite{afolabi2026faithful}. This does not imply that CoT is devoid of useful
signal. It implies that self-report must be evaluated as one observable
channel, not treated as a complete causal record.

Recent work provides direct evidence that internal signals can exceed what a
model states. \emph{Catching Rationalization} finds that activation probes can
match or outperform monitors that observe a complete CoT in general-domain
multiple-choice settings \cite{mirtaheri2026rationalization}. In medicine,
Fraile Navarro et al. use the same released Gemma-3-12B Natural Language
Autoencoder (NLA) checkpoint and layer-32 activations to localize a triage
output-format failure \cite{frailenavarro2026internal}; Tayebi Arasteh recovers
evidence grades from hidden states when stated grades are weak
\cite{tayebi2026evidence}; and Basu et al. report strong clinical-risk probe
performance despite substantially lower output sensitivity
\cite{basu2026risk}. Our contribution is therefore not the first observation
of medical internal--output dissociation or the first medical use of NLA. The
unresolved problem is to causally attribute a referral-induced diagnostic
change, trace the competing diagnoses through answer formation, and determine
whether the decoded content can support correction.

We address this problem with complementary instruments. Cross-fitted linear
probes are our primary quantitative readout in the closed 49-diagnosis DDXPlus
space. Natural-language activation readouts provide an auxiliary,
open-vocabulary channel that can express cue and diagnosis candidates without a
fixed classifier head. This flexibility introduces a serious validity risk:
activation verbalizers may answer from their own parametric knowledge rather
than from the paired target activation \cite{li2026privileged}. We therefore
admit the activation verbalizer as a measurement only after swap, shuffled,
heldout-content, memorization, and cross-case-contamination controls. We do not
claim that it is stronger than the probe or currently suitable as a
clinician-facing explanation.

Our testbed holds patient findings fixed and varies only a referral sentence
across no-note, neutral-note, wrong-diagnosis, and correct-diagnosis arms. This
within-case design separates the generic cost of inserting a sentence from the
content-specific cost of a wrong suggestion and supplies a counterfactual
case-level label for whether the wrong note moved the answer. We measure final
behavior, rule-based CoT features, a strong LLM monitor, linear-probe
trajectories, and validated natural-language readouts. For deployment-style
attribution, each detector observes only the wrong-note run; the no-note arm is
used to define the evaluation label, not exposed as an input.

Three findings organize the paper. First, wrong referral notes substantially
reduce diagnostic accuracy in both controlled DDXPlus prompts and case-report
language, but answer movement is not equivalent to internal suggestion
dominance: among 321 causally moved DDXPlus cases, the suggestion is never the
probe top-1 diagnosis at any of six observed landmarks in 266 cases (82.9\%).
Second, this dissociation is detectable from one wrong-note run. On the
canonical silent subset, a cross-fitted probe reaches 0.9840 AUROC, compared
with 0.8302 for the natural-language readout and 0.6829 for the LLM CoT monitor.
Third, correction is conditional. Accurate decoded content can recover moved
answers, but indiscriminate re-prompting damages answers that would otherwise
remain correct, and we find no independent advantage of natural-language form
after controlling for content accuracy.

These results support three contributions. We introduce a placebo-controlled
clinical suggestion intervention with case-level counterfactual labels; show
that referral-induced answer movement and internal suggestion dominance are
distinct events; benchmark output, CoT, LLM-monitor, probe, and activation-
verbalization channels on the same single-run attribution task; validate the
activation verbalizer as a restricted measurement rather than assume fluent
text is faithful; and identify a decodability--control boundary in which
accurate internal content is useful only under a selective correction policy.
The mechanism and single-run attribution claims are restricted to DDXPlus;
the case-report corpus currently supports behavioral replication rather than
the same internal trajectory claim.

## 선행연구가 이미 한 것과 남은 공백

| 흐름 | 이미 확립된 범위 | 이 논문이 파고드는 좁은 공백 |
|---|---|---|
| 사람·의료 LLM anchoring | Referral suggestion과 misleading context가 판단·정확도를 바꿀 수 있음 | 같은 case의 placebo-controlled intervention과 `moved` 정답지, 내부 diagnosis 행방 |
| CoT faithfulness | CoT가 biasing cause를 누락하거나 합리화할 수 있음 | 동일 wrong run에서 CoT/LLM monitor와 내부 채널을 정면 비교 |
| 의료 internal-output dissociation | Triage, evidence grading, risk prediction에서 hidden-state signal이 output보다 강할 수 있음 | 최종 진단의 referral intervention, gold/suggestion/other trajectory, 사례별 causal attribution |
| Probe와 activation verbalization | 닫힌 변수 decode와 열린 자연어 readout이 가능함 | Verbalizer prior 통제 후 두 계기를 분업하고 correction까지 연결 |
| Detection-to-control | Internal signal 기반 selective reprompting 가능성과 steering 실패가 모두 보고됨 | Content accuracy와 selector를 분리한 의료 correction ladder |

따라서 신규성 문장은 다음 정도가 안전하다.

> We do not introduce medical anchoring, internal-state probing, or medical NLA
> in isolation. We connect them in a causally controlled diagnostic protocol
> that separates answer movement from internal suggestion dominance, evaluates
> single-run attribution across observable and internal channels, and tests the
> conditions under which decoded content can correct the answer.

더 직접적으로는 **새 문제 정의**를 먼저 말한다.

> Our primary novelty is the case-level causal attribution problem. A hidden
> same-case no-note run defines whether the wrong note moved the answer, while
> the detector receives only the observable wrong-note run. We then trace the
> competing diagnoses and test correction under the same causal label.

이 정의가 중요한 이유는 moved case가 suggestion-copy case와 같지 않기 때문이다.
DDXPlus의 moved 321건 중 230건(71.7%)은 suggestion이 아니라 제3 진단으로
이동한다. 따라서 answer가 suggestion과 같은지 검사하는 output-only rule로는
대부분의 인과 영향을 탐지할 수 없다.

## 원문 확인용 출처 지도

| 논문·자료 | 이 문서에서 지지하는 주장 | 원문 |
|---|---|---|
| Staal et al. (2022) | Referral suggestion이 감별진단 폭을 줄일 수 있음 | [BMC Medical Education / PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8991944/) |
| Spaanjaars et al. (2015) | Referral diagnosis가 일부 경험군의 판단에 영향 | [DOI](https://doi.org/10.1027/1015-5759/a000235) |
| BiasMedQA (2024) | 의료 인지편향 문장 주입에 따른 행동 성능 저하 | [npj Digital Medicine](https://www.nature.com/articles/s41746-024-01283-6) |
| MED-STRESS (2026) | 다중 턴 압박에서 초기 진단 포기 | [arXiv](https://arxiv.org/abs/2605.23932) |
| MedMisBench (2026) | 대규모 misleading-context 성능 저하 | [arXiv](https://arxiv.org/abs/2606.12291) |
| Narrative Anchoring (2026) | 사실을 고정한 문체 변화로 진단 행동 변화 | [arXiv](https://arxiv.org/abs/2607.27384) |
| Turpin et al. (2023) | 답을 움직인 bias가 CoT에서 누락·합리화될 수 있음 | [NeurIPS / arXiv](https://arxiv.org/abs/2305.04388) |
| Lanham et al. (2023) | CoT faithfulness가 모델·과제에 따라 달라짐 | [arXiv](https://arxiv.org/abs/2307.13702) |
| Faithful or Just Plausible? (2026) | 의료 폐쇄형 LLM의 causal ablation과 hint injection | [arXiv](https://arxiv.org/abs/2603.13988) |
| Catching Rationalization (2026) | Probe와 full-CoT monitor의 단일 실행 영향 귀속 비교 | [arXiv](https://arxiv.org/abs/2603.17199) |
| Fraile Navarro et al. (2026) | 동일 Gemma NLA를 사용한 의료 triage 내부-출력 해리 | [arXiv](https://arxiv.org/abs/2605.29889) |
| Tayebi Arasteh (2026) | Hidden-state evidence grade와 stated grade의 해리 | [arXiv](https://arxiv.org/abs/2606.29034) |
| Basu et al. (2026) | 임상 risk signal의 높은 probe 성능과 낮은 output sensitivity | [arXiv](https://arxiv.org/abs/2603.18353) |
| Patchscopes (2024) | Hidden representation을 다른 prompt 문맥에서 decode | [ICML / arXiv](https://arxiv.org/abs/2401.06102) |
| SelfIE (2024) | LLM 내부 표현의 자연어 자기해석 | [ICML / arXiv](https://arxiv.org/abs/2403.10949) |
| LatentQA (2024) | Activation에 질문해 자연어 답을 얻는 readout | [arXiv](https://arxiv.org/abs/2412.08686) |
| Natural Language Autoencoders (2026) | AV/AR natural-language bottleneck과 reconstruction 학습 | [Transformer Circuits](https://transformer-circuits.pub/2026/nla/index.html) |
| Li et al. (2026) | Verbalizer prior와 privileged-information 검증 문제 | [ICML / arXiv](https://arxiv.org/abs/2509.13316) |
| Sun et al. (2025) | Internal probe를 이용한 selective reprompting | [arXiv](https://arxiv.org/abs/2507.12379) |
| Liu (2026) | Decodability와 control 사이의 간극 | [arXiv](https://arxiv.org/abs/2605.05715) |
| Vankadaru et al. (2026) | 의료 hallucination detection과 intervention 간극 | [arXiv](https://arxiv.org/abs/2607.00158) |

## 발표와 집필에서 금지할 축약

- `The model still knows the answer.` → `The gold diagnosis remains linearly
  decodable at the measured state.`
- `CoT is useless or at chance.` → 강한 LLM monitor의 실제 AUROC를 병기한다.
- `We are the first medical NLA study.` → Fraile Navarro et al. 때문에 거짓이다.
- `AV explains why the model was wrong.` → 현재 AV는 content candidate를 읽는
  보조 측정이며, clinician-facing explanation 효용은 검증에 실패했다.
- `Probe is an upper bound.` → probe는 label supervision을 받는 주 정량 계기다.
  정보의 인과적 사용이나 모든 open-vocabulary 내용을 보장하지 않는다.
- `Correction proves natural language is special.` → label/content를 통제하면
  자연어 형식의 독립 우위는 확립되지 않았다.
