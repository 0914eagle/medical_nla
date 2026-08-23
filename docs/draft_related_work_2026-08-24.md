# Related Work — 초안 v1 (2026-08-24)

영어 본문 초안. 인용은 [저자, 연도] 자리표시 — Overleaf에서 \citep로 교체.
수치 인용은 본문 결과 절 번호(§4.x)를 가리키며, 최종 조판에서 상호참조로.

---

## 2. Related Work

### 2.1 Cognitive bias and unfaithful explanations in medical LLMs

Explanations that omit the true cause of an answer are a documented failure
mode of chain-of-thought: when a biasing feature moves a model's answer, the
generated reasoning rarely mentions it [Turpin et al., 2023; Lanham et al.,
2023], and the disclosure rate remains low even in reasoning-tuned models
[Chen et al., 2025]. In clinical practice the corresponding hazard has a
name — anchoring, a recognized contributor to diagnostic error [Croskerry,
2003] — and it enters real workflows daily through referral notes, handoffs,
and patient framing [Mahajan et al., 2025].

Medical language models inherit this vulnerability, and the literature has
measured it behaviorally. Injecting clinical cognitive biases into
USMLE-style questions degrades accuracy across model families [Schmidgall
et al., 2024], the effect persists in models trained for explicit reasoning
[Anonymous, 2025], and susceptibility extends to user-side factors such as
emphasis and misinformation in patient queries [Kim et al., 2025] and to
sycophancy in simulated clinical encounters [SycoEval-EM, 2026]. The remedy
this literature proposes is to audit the model's reasoning trace [Mahajan
et al., 2025]; whether the trace can support such an audit has, however,
not been tested against ground truth.

Faithfulness of medical explanations has also been probed directly. Under
causal ablation and hint injection, chain-of-thought steps do not causally
drive the predictions of closed-source medical assistants, and injected
suggestions are absorbed without acknowledgment [Faithful-or-Plausible,
2025]; structured re-grading finds individual reasoning steps decorative —
removable without changing the answer [Clinical Reasoning Graphs, 2026].
Because these studies target closed-source models, they are confined to
behavior by construction.

Across this literature, models are perturbed from the outside and measured
from the outside, so what is known is that suggestions degrade aggregate
accuracy. We instead rebuild the same clinically real intervention in a
causally controlled form — a placebo-controlled four-arm design whose
evidence representations are bit-identical across arms, yielding a per-case
ground truth for which answers the note actually moved — and we do not stop
at the behavioral result: we first confirm that the reasoning trace cannot
support the proposed audit (attribution at chance; §4.2), and then go
inside the model.

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

Internal access has reached medicine only as localization: probes confirm
that clinical knowledge is present in hidden states [ADR-probing, 2025;
alignment-resistant probing, 2025], and sparse autoencoders extract
medical features from clinical models [JMIR-SAE, 2026; EHR-SAE, 2026].
What knowledge lives where is mapped; why a particular case failed is not
asked.

We chain these pieces, in medicine, on the causal testbed of §2.1: a
verified natural-language readout attributes which cases the note moved
from a single deployed run (§4.4, with the supervised-probe upper bound
reported alongside); a positional trajectory shows that anchoring is a
rift between a preserved internal state and the emitted answer, not an
overwriting of the state (§4.3); the readout renders that internal state
as a statement a clinician can read (§4.3); and feeding the statement back
recovers the moved cases where re-showing the evidence does not,
net-positive once paired with a precise selector (§4.5) — including on
real case reports whose open label space admits no classifier head
(§4.6).

---

### 구조 메모 (본문 아님)

- 2.1: 4문단 — 일반→임상 훅 / 의료 편향 주입(행동) / 의료 충실성 프로브 /
  **턴**(인과 재구축 + 내부로).
- 2.2: 5문단 — 계기 계보 / 언어화와 Li 비판 / 일반 도메인의 조각들(신규
  이웃 2편 정면 배치) / 의료 내부 접근은 국소화까지 / **턴**(사슬로 잇기,
  §4 상호참조 5개).
- [Anonymous, 2025] = medRxiv "reasoning does not protect" (저자 확인 후
  교체). SycoEval-EM·Faithful-or-Plausible 등 임시 키는 bib 정리 시 교체.
