# 정독 노트 5편 (2026-08-24) — 신규성 감사의 초록 판정을 전문으로 교체

`novelty_audit_2026-08-24.md`는 egress 차단 탓에 초록·스니펫 수준이었다.
전문을 받아 읽은 결과 **판정 셋이 바뀐다**: 최근접 선행이 교체되고, 위협으로
보이던 둘이 오히려 우리 자리를 만들어 주며, RQ3의 선행 하나는 진짜다.

---

## 1. Fraile Navarro et al., "Internal Representation, Not Clinical
## Knowledge: Where Apparent LLM Triage Failures Originate" (2605.29889)

**새로운 최근접 선행이다. When Truth Is Overridden보다 가깝다.**

Macquarie University 외. 환자 목소리 트리아지 vignette 60건 × 3모델
(Gemma 3 4B/12B IT, Qwen3-8B).

**왜 가까운가 — 겹침이 불편할 만큼 크다**
- **같은 NLA 체크포인트**: "NLA evidence is limited to Gemma 3 12B IT,
  the released checkpoint available for this analysis (Fraser-Taliente
  et al., 2026)." 우리가 쓰는 바로 그것.
- **같은 층**: "We capture **L32** activations of Gemma 3 12B IT at seven
  token positions."
- **같은 형태의 결론**: 임상 내용은 서사에서 보존되는데 **결정 토큰에서
  프레임이 뒤집힌다** — MED-PRIMARY 60건 → 0건, SCA-PRIMARY 0 → 60.
  "the failure originates in the output format and not in the clinical
  representation."
- **의료 + NLA + 케이스 단위 예측**까지 있다(4.5절: source-format hidden
  state로 어느 케이스가 뒤집힐지 층별 ROC-AUC).

**그런데 저자들이 스스로 그은 선이 우리를 살린다.** Limitations의
Claim scope를 그대로 인용한다:

> "The SAE features are medical-vs-non-medical detectors, **not acuity
> probes: we claim medical-domain content is preserved on the clinical
> narrative, not that correct triage disposition is encoded.**"

**그들은 "의료 내용이 있다"를 보였고, "정답이 인코딩되어 있다"는 보이지
않았다.** 우리는 49-way 프로브로 **정답 진단이 1위**임을 82.7%에서 보인다.
주장의 강도가 다른 층위다.

그리고 Background에서 과제 자체를 우리 것과 분리해 준다:

> "for LLM evaluation the reasoning task under evaluation is an initial
> acuity disposition under sparse information, **not a final diagnosis**."

**갈리는 지점 정리**

| | 2605.29889 | 우리 |
|---|---|---|
| 과제 | 트리아지 acuity (저자 명시: 진단 아님) | **진단** |
| 원인 | 출력 **형식**(객관식 제약) | **앵커링 개입**(소견서의 의심 진단) |
| 대조 | 형식 A vs 형식 B (프롬프트가 다름) | 같은 프롬프트 ± 한 문장, **증거 활성값 비트 동일(±.000 실측)** |
| 내부 주장 | 의료 **내용**이 결정 토큰에서 강등 | **정답이 1위로 유지**(82.7%) + **용량-반응**(.007/.055/.187) |
| NLA 지위 | 7위치 서술을 LLM 판정자가 PRIMARY/PARTIAL/NO로 3분류, n=60 | **검증 배터리를 통과한 계기**(스왑 .993·암기 .000·오염 .007·무학습 답위치 대조) |
| 표본 | vignette 60건 | 1,747 + 4,995 + 1,543 |
| 교정 | 없음(형식 방향 특징의 인과 개입은 부록, 결과 미확인) | 되먹임 사다리 r3–r7 |

**그래서 §2에서 이렇게 쓴다**: 반박이 아니라 **수렴 증거**. 서로 다른 교란
(형식 제약 vs 임상적 제안)이 같은 층위의 실패로 수렴한다 — 그들이 형식에서,
우리가 앵커링에서. 그리고 **우리는 "내용이 남아 있다"를 넘어 "정답이 1위로
남아 있다"까지 간다**는 것이 우리 몫이다.

**부수 소득**: 그들이 인용하는 **Basu et al. 2026** — 임상 트리아지에서
**53pp knowledge–action gap**(Qwen 2.5 7B)에 **SAE 특징 스티어링 포함 네
가지 기전 개입이 모두 신뢰할 만한 교정에 실패**. 우리 RQ3 포지셔닝의
직접 재료다. ▢ 이 논문도 받아야 한다.

---

## 2. Tayebi Arasteh, "The strength of clinical evidence is recoverable
## from LM representations but not from their stated grades" (2606.29034)

RWTH Aachen / Stanford. 임상 주장 45,134건 → 20,611건을 4단계 근거 등급으로
정규화, 22개 오픈웨이트 모델(0.6–70B).

**핵심**: 선형 추정기가 등급을 복원(중앙 AUROC **71.8**), 모델이 **말하는**
등급은 우연 수준으로 **추정기보다 25–27%p 아래**. "Clinical LLMs thus carry
an ordered evidence-strength signal they do not express."

**의료에서 내부-발화 불일치를 정면으로 보고한 논문이 맞다.** 다만 저자
스스로 신호를 크게 약화시킨다: "The recoverable signal was **largely lexical
and did not transfer across topics or frameworks**."

**갈리는 지점**: 과제가 **근거 등급**이지 진단이 아니다 · **인과 개입이 없다**
(관찰 연구, 무엇이 답을 바꿨는지 묻지 않는다) · 계기가 선형 추정기이고
**자연어 서술이 아니다** · 케이스 단위 귀속·교정 없음 · 신호가 어휘적이라고
자인.

**함의**: "의료에서 내부-출력 불일치를 처음 보였다"는 **못 쓴다.** 우리
문장은 "**진단 과제에서, 인과 통제된 개입 아래, 케이스 단위로, 그리고
어휘적이지 않은(위치 불변 설계로 차단된) 신호로**"가 된다.

---

## 3. Sun, Stolfo & Sachan, "Probing for Arithmetic Errors in Language
## Models" (EMNLP 2025, 2507.12379)

**RQ3의 진짜 선행이다.** 3자리 덧셈 + addition-only GSM8K. 은닉 상태에서
**모델의 예측과 정답을 둘 다** 디코딩하는 프로브(정오 무관), 오류 탐지
90%+, 그리고 "probes can guide **selective re-prompting** of erroneous
reasoning steps, improving task accuracy **with minimal disruption to
correct outputs**."

**착상이 같다**: 내부에서 읽은 것을 텍스트로 되먹여 고친다.

**그런데 결정적으로 다른 것이 하나 있고, 그것이 우리 §4.4의 중심이다.**
그들은 재실행이 **"minimal disruption"**이라고 보고한다. 우리는 재실행이
**파괴적**이라고 측정했다(전체 0.814 → 0.42대, 항복률 0.293 → 0.450).
차이의 원인은 설계에 있다 — 그들은 **틀린 추론 단계만 골라** 다시 묻고,
우리는 **교란이 프롬프트에 남아 있는 채로 케이스 전체**를 다시 묻는다.
배포 상황은 후자다(소견서를 지울 수 없다). **"선별 없는 재실행은 순손해"는
그들의 결과와 모순이 아니라, 교란이 존재하는 설정으로의 확장이다.**

그 밖의 차이: 산술/GSM8K로 **의료 아님** · 외부 교란 없음(그냥 모델이 틀린
경우) · **형식 vs 내용의 통제된 분리 없음**(우리 r5/r6/4b) · 열린 어휘 문제
없음.

**함의**: §4.4를 "우리가 처음"으로 쓰면 안 된다. **"의료 앵커링에서, 교란이
남아 있는 배포 조건에서, 그리고 지렛대가 형식이 아니라 내용임을 통제로
분리해서"**까지만 주장한다.

---

## 4. Vankadaru et al., "Readable but Not Controllable: Neuron-Level
## Evidence for Medical LLM Hallucination" (2607.00158)

UC Berkeley. 4개 오픈 모델 × 의료 QA. 프로브로 환각 탐지 **AUROC
0.77–0.86**. 신호는 **분산·중복적**(수백 개 무작위 뉴런이 거의 전체 신호를
회복, 저차원 무작위 사영도 대부분 보존).

**핵심**: 16개 모델–데이터셋 조합에서 **decodability와 controllability
사이의 날카로운 간극**. 탐지를 쉽게 만드는 그 내부 구조가 **뉴런 수준
제어로는 이어지지 않는다.**

---

## 5. Ming Liu (Amazon), "Decodable but Not Corrected by Fixed
## Residual-Stream Linear Steering" (2605.05715)

의료 QA의 **Overthinking**(재샘플링하면 맞히는데 긴 CoT에서는 틀리는 안정적
행동 체제)을 대상. OT는 선형 디코딩 **71.6%** 균형 정확도(p≈10⁻¹⁶).

**그런데 고정 선형 스티어링 5계열 · 29개 설정 · n=1,273에서 전부 Δ≈0.**
교차 아키텍처(Qwen2.5-7B)·교차 도메인(MMLU-STEM)에서도 동일한 null.
원인은 **표상 얽힘**: OT 방향이 과제 핵심 계산과 **85–88% 겹침**, 비표적
스티어링은 정확도를 −12.1pp 훼손, LEACE 개념 삭제도 −3.6pp 훼손.
케이스별 probe–steering 상관은 r = **−0.002**.

긍정: 같은 프로브가 **선택적 기권**을 가능하게 한다(AUROC 0.610).

---

## 4·5번이 우리에게 만들어 주는 자리 — RQ3의 새 프레이밍

세 문장이 나란히 선다.

1. **Huang et al. (2024)**: LLM은 **외부 피드백 없이** 추론을 자기교정하지
   못한다. (5번 논문이 서론에서 인용)
2. **의료에서 활성값 경로는 막혀 있다**: 4번(뉴런 제어 실패, 16개 조합)과
   5번(선형 스티어링 실패, 29개 설정) — 둘 다 **읽히는데 안 고쳐진다**.
   5번은 원인까지 짚는다(표상 얽힘 85–88%).
3. **텍스트 경로는 열려 있다**: 3번이 산술에서 보였고, **우리가 의료
   앵커링에서 보인다**(moved 0.012 → 0.627/0.830).

**우리 판독이 하는 일은 "모델 자신의 상태를 외부 피드백으로 바꿔 주는 것"**
이다. Huang의 조건(외부 피드백)을 활성값을 건드리지 않고 충족시킨다.
그래서 §4.4의 한 문장은 이렇게 된다:

> 의료에서 내부 신호는 읽히지만 활성값 개입으로는 고쳐지지 않는다는 것이
> 두 번 보고됐다(뉴런 제어 16개 조합, 선형 스티어링 29개 설정). 원인은
> 표상 얽힘이다 — 실패 방향이 과제 계산과 85–88% 겹친다. 우리는 표상을
> 건드리지 않고 **읽은 것을 문장으로 되먹인다**. 얽힘이 막는 것은 개입
> 벡터이지 프롬프트 채널이 아니다.

**이건 그냥 좋은 위치다.** 두 편의 부정 결과가 우리 방법의 존재 이유를
대신 논증해 준다.

---

## 감사 판정의 갱신

| 조항 | 초록 판정 (08-24 오전) | 전문 판정 (08-24 오후) |
|---|---|---|
| A 행동 | 기여 아님 | **유지** — MedMisBench(10,932문항·48,889쌍, 71.1%→38.0%)로 더 확실 |
| B 내부-출력 해리 | "매우 가까운 것 있음" | **한정 후 성립** — 2605.29889는 "내용 보존"까지만, 2606.29034는 어휘적·비인과. **"정답이 1위" + 용량-반응은 우리 것** |
| C 단일 실행 탐지 | 부분 선행 | **유지** — 2605.29889 4.5절이 케이스별 flip 예측을 하나, **형식 전환**의 예측이지 인과 귀속이 아님 |
| D 자연어 서술 | "거의 비어 있음" | **한정 필요** — 같은 NLA·같은 모델·같은 층이 의료에서 이미 쓰였다(n=60, 판정자 3분류). **계기 검증 배터리는 여전히 우리 것** |
| E 되먹임 교정 | "가까운 선행 있음" | **선행 확정**(EMNLP 2025) + **의료 부정 결과 둘이 우리 자리를 만듦** |

**결론: 한 문장은 살아남는다.** 다만 §2가 반드시 해야 할 일이 생겼다 —
**2605.29889를 최근접 선행으로 정면 인용하고, 두 가지 차이를 명시**하는
것: (i) 그들의 주장은 "의료 내용 보존", 우리는 "정답 1위 유지 + 용량-반응",
(ii) 그들의 교란은 출력 형식, 우리는 임상적 제안. 이걸 숨기면 심사자가
찾아낸다 — 같은 체크포인트·같은 층·같은 모델을 썼기 때문에 더욱 그렇다.

## ▢ 남은 것

- **Basu et al. 2026** (트리아지 53pp knowledge–action gap, 4개 기전 개입
  실패) — 2605.29889가 인용. RQ3 포지셔닝에 직접 쓰인다.
- 2605.29889 부록 Q(NLA 출력 실물)·부록의 인과 개입 결과 — 페이지 미확인.
- Genadi et al., "Sycophancy Hides Linearly in the Attention Heads"
  (EACL 2026, 2601.16644).
- 2601.18939 "A Few Bad Neurons: Isolating and Surgically Correcting
  Sycophancy".
