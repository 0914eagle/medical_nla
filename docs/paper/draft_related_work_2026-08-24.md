# Related Work — 초안 v1 (2026-08-24)

영어 본문 초안. 인용은 [저자, 연도] 자리표시 — Overleaf에서 \citep로 교체.
수치 인용은 본문 결과 절 번호(§4.x)를 가리키며, 최종 조판에서 상호참조로.

---

## 2. Related Work

### 2.1 Explainability of medical LLMs

As large language models enter clinical decision support, accuracy alone is
not the bar they must clear. Beyond being correct, a deployed model must be
inspectable: hallucination, biased outputs, opaque provenance, and
performance drift are recognized safety concerns [Wang et al., 2024;
npj-hallucination-framework, 2025; ethics-systematic-review, 2025; Chen
et al., 2023 (drift)], and both clinicians and regulators increasingly
require an account of *why* a model answered as it did before its answers
can be trusted [Cracking-the-Clinical-Code, 2025; Why-Clinical-Reasoning-
Fails, 2026]. For diagnosis the requirement is concrete, because clinically
real context routinely moves answers: a referring physician's suspicion, a
colleague's remark, a patient's worry. In human diagnosticians this failure
has a name — anchoring, a recognized contributor to diagnostic error
[Croskerry, 2003] — and it enters LLM workflows through the same channels
[Mahajan et al., 2025].

Explainability in medical AI has so far meant two families of methods. For
predictive models, post-hoc input attribution dominates — SHAP, LIME, and
saliency maps assign importance to input features [reviews: Frontiers,
2026], with concept bottlenecks providing clinician-vocabulary
intermediates in imaging [CBM refs]. These methods presuppose a scalar
output and many perturbed reruns, and do not transfer cleanly to free-text
diagnosis. For generative LLMs, the de facto explanation is instead the
model's own chain of thought: a fluent, clinician-readable self-narration,
prominent enough that auditing the reasoning trace has been proposed as
the safety mechanism for clinical deployment [Mahajan et al., 2025].

Self-narration, however, is unfaithful precisely where an audit needs it.
In the general domain, features that demonstrably move a model's answer go
unmentioned in its reasoning [Turpin et al., 2023; Lanham et al., 2023],
and disclosure remains rare in reasoning-tuned models [Chen et al., 2025].
The medical evidence is now direct: injected clinical cognitive biases
degrade diagnostic accuracy while the accompanying explanations do not
disclose them [Schmidgall et al., 2024; medRxiv, 2025]; under causal
ablation and hint injection, chain-of-thought steps do not causally drive
the predictions of closed-source medical assistants, and injected
suggestions are absorbed without acknowledgment [Faithful-or-Plausible,
2025]; structured re-grading finds individual reasoning steps decorative —
removable without changing the answer [Clinical Reasoning Graphs, 2026].

Both available forms of explanation therefore fail at the question a
diagnostic audit must answer — *what caused this answer* — the attribution
family by construction, the self-narration family empirically. We rebuild
the clinically real intervention behind this evidence in a causally
controlled form — a placebo-controlled four-arm design whose evidence
representations are bit-identical across arms, yielding a per-case ground
truth for which answers the note actually moved — confirm on it that the
reasoning trace cannot support the proposed audit (attribution at chance;
§4.2), and then go inside the model.

### 2.2 Reading LLM internals: from probes to natural-language readouts

Instruments for reading model internals are mature. Linear probes recover
task variables from hidden states but answer only the question they were
trained on [Belinkov, 2022]; logit- and tuned-lens methods decode token
distributions from intermediate layers [Belrose et al., 2023]; sparse
autoencoders decompose activations into dictionaries of pre-learned
features [Cunningham et al., 2023; Bricken et al., 2023]. All deliver
flags or fixed concepts. A younger line verbalizes instead: Patchscopes,
SelfIE, and LatentQA prompt a model to describe hidden representations in
open-ended language [Ghandeharioun et al., 2024; Chen et al., 2024; Pan et
al., 2024], and natural-language autoencoders (NLA) train an
activation-to-text verbalizer with a reconstruction objective [Anthropic,
2026] — the instrument we build on. Verbalization invites a sharp
criticism: the description may reflect the verbalizer's own parametric
knowledge rather than the target activation [Li et al., 2026]. Our
instrument battery — counterfactual swap tracking, memorization and
shuffle controls, cross-case specificity — is designed as a direct answer
(§4.1).

In the general domain, these instruments have recently produced results
that anticipate pieces of ours. Activation probes detect the influence of
an injected hint that the chain-of-thought rationalizes away [Catching-
Rationalization, 2026]; user opinions suppress a model's learned knowledge
in late layers while the knowledge itself survives [When-Truth-Is-
Overridden, 2026]; hidden states predict reasoning errors that verbalized
confidence does not admit, though steering and self-correction
interventions built on such signals fail [Yuan et al., 2026]; and correct
answers recoverable from hidden states coexist with wrong chains [
Mehrafarin et al., 2026]. These findings are non-medical, their signals
are probes, patches, or graph distances rather than readable statements,
and they end at detection — the one attempted continuation into
correction reports failure.

Medical work has moved beyond localization. Fraile Navarro et al. (2026)
use the same Gemma-3-12B NLA checkpoint and layer-32 activations to show
that clinical content can survive a triage output-format failure and to
predict case-level flips. Tayebi Arasteh et al. (2026) likewise recover
evidence grades from hidden states when the model's verbalized grade is
near chance. These are direct convergence results, not gaps we claim to
fill. Their tasks are acuity formatting and evidence grading rather than
diagnosis; neither causally isolates a referral-note suggestion, verifies
the natural-language readout as an activation-dependent instrument, or
tests a controlled correction ladder. Probes and sparse autoencoders also
localize clinical knowledge and features [ADR-probing, 2025;
alignment-resistant probing, 2025; JMIR-SAE, 2026; EHR-SAE, 2026].

We chain these pieces, in medicine, on the causal testbed of §2.1: a
verified natural-language readout attributes which cases the note moved
from a single deployed run (§4.3, with the supervised-probe upper bound
reported alongside); a positional trajectory shows that anchoring is a
rift between a preserved internal state and the emitted answer, not an
overwriting of the state (§4.3); the readout renders that internal state
as a statement a clinician can read (§4.3); and feeding the statement back
recovers the moved cases where re-showing the evidence does not,
net-positive once paired with a precise selector (§4.4) — with the
behavioural anchoring effect replicated on real case reports whose open
diagnosis vocabulary does not admit the same fixed-class probe (§4.2).
Internal-state and correction claims on those case
reports remain separate experiments rather than consequences of the
behavioural replication.

---

### 구조 메모 (본문 아님)

- 2.1 (Explainability of medical LLMs, 08-24 개제): 4문단 — 판돈(정확도
  너머의 안전·규제 + anchoring) / 의료 설명의 두 형태(입력 기여도 SHAP·LIME
  · CoT 자기 서술과 트레이스-감사 제안) / 자기 서술의 불충실(faithfulness of
  CoT — 일반 + 의료 증거, 편향 주입 문헌은 여기의 증거로) / **턴**(두 형태
  모두 "무엇이 답을 만들었나"에 실패 — 귀속 계열은 구조적으로, 자기 서술은
  실증적으로 → 인과 재구축 + 내부로).
- 주의: 사용자가 가져온 "Beyond correctness…" 류 문구는 타 논문 문장 —
  취지만 우리 문장으로 재작성했고 원문 표현은 쓰지 않음(표절 방지).
- 2.2: 5문단 — 계기 계보 / 언어화와 Li 비판 / 일반 도메인의 조각들(신규
  이웃 2편 정면 배치) / 의료 내부 접근은 국소화까지 / **턴**(사슬로 잇기,
  §4 상호참조 5개).
- [Anonymous, 2025] = medRxiv "reasoning does not protect" (저자 확인 후
  교체). SycoEval-EM·Faithful-or-Plausible 등 임시 키는 bib 정리 시 교체.

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
