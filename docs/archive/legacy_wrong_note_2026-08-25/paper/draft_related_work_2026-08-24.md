# Related Work — 초안 v1 (2026-08-24)

영어 본문 초안. 인용은 [저자, 연도] 자리표시 — Overleaf에서 \citep로 교체.
수치 인용은 본문 결과 절 번호(§4.x)를 가리키며, 최종 조판에서 상호참조로.

---

## 2. Related Work

### 2.1 Clinical anchoring and misleading context

Diagnostic reasoning rarely begins from an unframed case. Referral letters
can carry a provisional diagnosis from an upstream clinician, and controlled
human studies show that such suggestions can narrow the differential or
anchor subsequent judgments [Spaanjaars et al., 2015; Staal et al., 2022].
This motivates our intervention as a clinically plausible *referral-mediated*
anchoring scenario. It does not imply that every referral contains a diagnosis
or that every downstream clinical model is deployed in this workflow.

Medical-LLM robustness work has already established the behavioral problem.
BiasMedQA injects seven clinically motivated cognitive-bias statements into
1,273 USMLE questions and finds model-dependent accuracy losses [Schmidgall
et al., 2024]. MED-STRESS studies abandonment of initially correct diagnoses
under escalating multi-turn pressure [Xiao et al., 2026], while MedMisBench
shows large accuracy drops under misleading clinical context [Zhou et al.,
2026]. Narrative Anchoring further holds clinical facts fixed while varying
sociolinguistic register [Singh et al., 2026]. These works make an important
point: benchmark knowledge does not guarantee resilience to context. The
average behavioral drop is therefore not our novelty.

Our question begins after that drop. We use a within-case, placebo-controlled
four-arm intervention in which the patient findings are fixed and only the
referral sentence varies. This separates the cost of inserting a sentence from
the content-specific cost of a wrong suggestion, yields a per-case
counterfactual label for whether the answer moved, and allows us to ask where
the gold, suggested, and third-diagnosis signals go inside the model. The same
behavioral effect is replicated in case-report language, but the internal
mechanism is claimed only on the controlled DDXPlus testbed.

### 2.2 Chain-of-thought faithfulness and internal-output dissociation

Generated reasoning is useful behaviorally but is not automatically a causal
record of how an answer was produced. Turpin et al. (2023) show that features
that move answers can go unmentioned while the model rationalizes the biased
answer. Lanham et al. (2023) intervene on reasoning traces and find that model
dependence on CoT varies substantially across tasks and scales. In medicine,
Afolabi et al. (2026) use causal ablation, positional perturbation, and hint
injection on closed-source assistants and find that external suggestions can
be incorporated without acknowledgment. These results motivate comparing
CoT with an internal channel; they do not justify treating CoT as devoid of
signal. Accordingly, we include both rule-based features and a strong LLM
monitor, and interpret their measured AUROC rather than assuming failure.

The closest single-run attribution study is *Catching Rationalization*
[Mirtaheri and Belkin, 2026]. In general-domain multiple-choice tasks,
pre-generation probes match a full-CoT LLM monitor and post-generation probes
outperform it. Our causal label is related but not identical. Their hint points
to a listed option, whereas in our open-diagnosis setting most moved outputs
go to a third diagnosis rather than copying the suggestion. We test whether a
single wrong-note run identifies the cases that a hidden no-note counterfactual
shows were causally moved, including a subset where output copying is blind by
construction.

Medical studies also show that activation content can exceed what the model
states. Fraile Navarro et al. (2026) use the same released Gemma-3-12B NLA and
layer-32 activations to localize a triage output-format failure. Tayebi Arasteh
(2026) recovers evidence grades from hidden states when the model's stated
grades are near chance. Basu et al. (2026) report a 0.982 clinical-risk probe
AUROC despite substantially lower output sensitivity. These are direct
precedents, not gaps we claim to fill. We differ by studying final diagnosis
under a referral-note intervention, resolving gold/suggestion/other trajectories
at prompt landmarks, and connecting case-level causal attribution to a
controlled correction ladder.

### 2.3 Reading and acting on activations

Linear probes recover prespecified variables from hidden states but establish
decodability rather than causal use [Belinkov, 2022]. Logit and tuned lenses
decode token distributions [Belrose et al., 2023], while sparse autoencoders
decompose activations into learned feature dictionaries [Cunningham et al.,
2023; Bricken et al., 2023]. These tools are appropriate for closed questions
and are our primary quantitative instrument in DDXPlus.

Natural-language readouts instead aim to express open-ended activation
content. Patchscopes, SelfIE, and LatentQA decode hidden representations into
text [Ghandeharioun et al., 2024; Chen et al., 2024; Pan et al., 2024]. Natural
Language Autoencoders jointly train an activation verbalizer and reconstructor
through a natural-language bottleneck [Fraser-Taliente et al., 2026]. NLAs can
surface unverbalized cognition and support hypothesis generation, but the
authors also report confabulation. More generally, Li et al. (2026) show that
activation-verbalization benchmarks can often be solved without target-model
internals and that descriptions may reflect the verbalizer's parametric
knowledge. We therefore do not infer faithfulness from fluent text or SFT loss.
Our AV is admitted as a measurement only after matched-vs-shuffled, swap,
heldout-content, memorization, and cross-case-contamination controls. Even then,
we report it as a complementary open-vocabulary channel, not a replacement for
the stronger supervised probe or a validated clinician-facing explanation.

Finally, decodability does not guarantee controllability. Sun et al. (2025)
use arithmetic-error probes for selective re-prompting with little disruption
to correct outputs. In contrast, medical studies find that strong internal
signals can be difficult to turn into safe steering: Basu et al. (2026),
Vankadaru et al. (2026), and Liu (2026) report gaps between detection and
activation-level correction. We test a different deployment path: leave the
source model weights and activations untouched, externalize a decoded content
candidate, and feed it back as text. The result supports only a conditional
claim. Accurate content can recover moved answers, but indiscriminate
re-prompting damages correct ones, and no independent advantage of natural
language form remains after content accuracy is controlled.

---

### 구조 메모 (본문 아님)

- **2.1 Clinical anchoring and misleading context**: 사람의 referral anchoring과
  의료 LLM의 행동 취약성을 정리한다. 이 절의 턴은 "정확도 하락은 이미
  알려졌지만, 같은 사례의 위약 대조와 내부 행방은 아직 분리되지 않았다"이다.
- **2.2 Chain-of-thought faithfulness and internal-output dissociation**: CoT가
  완전한 인과 기록이 아니라는 일반·의료 증거와, 내부 신호가 출력을 초과하는
  최근 의료 연구를 함께 둔다. 이 절의 턴은 "우리의 신규성은 내부-출력 해리의
  존재가 아니라 referral-note 개입 아래의 위치 궤적과 single-run attribution"이다.
- **2.3 Reading and acting on activations**: probe/lens/SAE에서 자연어 readout으로
  이어지는 계보, Li et al.의 privileged-information 비판, decodability-control
  gap을 정리한다. 이 절의 턴은 "probe를 주 정량 계기로, AV를 검증된 보조
  open-vocabulary 계기로 사용하고 content feedback의 조건부 효용을 시험한다"이다.
- 주의: 사용자가 가져온 "Beyond correctness…" 류 문구는 타 논문 문장이다.
  취지만 우리 문장으로 재작성했고 원문 표현은 사용하지 않는다.

### 인용 키 매핑 (bib 정리용, 08-24 추가)

- **[Wang et al., 2024] = "Safety challenges of AI in medicine in the era
  of large language models", arXiv:2409.18968 (Yu·Bitterman·Zou 등) —
  우산 인용 1순위 (사용자 확인, 08-24).**
- ~~[Nature-safety-review, 2026]~~ = Nature s41586-026-10687-1 — **제외
  (08-24): 페이월이라 본문 미확인 — 읽지 않은 논문은 인용하지 않는다.**
  기관 구독으로 전문 확보 시 재고.
- [npj-hallucination-framework, 2025] = "A framework to assess clinical
  safety and hallucination rates of LLMs for medical text summarisation",
  npj Digital Medicine, s41746-025-01670-7 (환각률 1.47%/누락 3.45%).
- [ethics-systematic-review, 2025] = "A systematic review of ethical
  considerations of large language models in healthcare and medicine",
  PMC12460403 (편향·투명성·프라이버시 최다 논점).
- [Chen et al., 2023 (drift)] = "How is ChatGPT's behavior changing over
  time?", arXiv:2307.09009.
- 보조 후보: "Beyond Multiple-Choice Accuracy" (arXiv:2410.18460),
  "Trustworthy Medical QA survey" (arXiv:2506.03659).
