# 논문 골격 — 절별 내용 배치 (2026-08-24)

## 논문 주제 — 정본 한 문장 (08-25 감사 후 수정)

**의뢰 소견서의 오답 제안은 의료 LLM의 최종 진단을 바꾸지만, 그 변화는
제안 진단이 내부 표현의 top-1이 되는 것과 자주 일치하지 않는다. 우리는 이
내부-출력 불일치를 위치별로 측정하고, 단일 실행에서 탐지하며, 내부 판독을
되먹이는 교정의 가능성과 한계를 평가한다.**

행동 효과는 DDXPlus와 MedCaseReasoning 양쪽에서 재현된다. 내부 궤적은
DDXPlus의 canonical-eligible moved 319건에서 측정했다. 제안 진단은 262건(82.1%)에서
어느 랜드마크에서도 probe top-1이 아니었다. 그러나 이것은 “정답이 계속
top-1”이라는 뜻이 아니다. 정답이 모든 관측 지점에서 top-1인 경우는 147건
(46.1%)이고, 나머지 115건(36.1%)은 제안이 아닌 다른 진단이 top-1이었다.
따라서 정본 주장은 **지식 비소거**나 **순수 출력 단계 실패**가 아니라,
**행동적 앵커링과 내부에서 디코드되는 제안 우세 사이의 불일치**다.

운영적으로 풀면 다음과 같다. **환자 cue와 충돌하는 잘못된 임상 제안이 들어온
DDXPlus 조건에서, activation을 직접 보는 내부 채널은 생성된 CoT를 읽는 채널보다
**그 소견서 때문에 답이 바뀐 사례를 더 잘 판별한다.** 동일 wrong-note 단일 실행에서
cross-fitted probe는 all/silent AUROC `.9330/.9881`, 강한 LLM CoT monitor는
`.7305/.6904`다. AV readout은 `.7511/.8319`로 probe보다 약하지만 자연어 후보를
제공한다. 이 문장은 “모든 의료 과제에서 내부가 CoT보다 우월하다”는 일반 명제가
아니며, 현재 통제된 DDXPlus 개입과 정의된 moved label 안의 비교다.

> *A wrong diagnosis in a referring note can change a medical LLM's final
> answer without becoming the top decoded diagnosis at any observed prompt
> landmark. We characterize this heterogeneous internal-output mismatch,
> detect it from a single run, and test whether feeding internal readouts back
> can repair the answer on a controlled DDXPlus testbed. The behavioral effect
> also reproduces in real case reports; their internal mechanism remains to be
> measured.*

### 이 한 문장을 실측이 지탱하는가 — 조항별 검증 (08-25)

주장을 조항으로 쪼개고, 각 조항에 그것을 지탱하는 측정과 **그 측정이 닿지
않는 곳**을 붙인다. 초안을 쓰기 전에 이 표가 통과해야 한다.

| 조항 | 지탱하는 측정 | 성립 | 범위 한계 |
|---|---|:-:|---|
| "의뢰 소견서의 의심 진단에 앵커링된다" | canonical-eligible 오답 소견서 −23.75%p(DDX 1,204) · −29.34%p(MCR 1,452), 위약 대비 제안 고유 −18.36/−22.73%p | ○ | 두 코퍼스 재현. 개입 1종(문구 4종으로 보강) |
| "출력 이동과 제안의 내부 top-1은 자주 불일치한다" | moved 319건 중 제안이 한 번도 top-1이 아닌 **262건(82.1%)** | ○ | DDXPlus 전용, 닫힌 49-class probe |
| "내부 궤적은 이질적이다" | gold top-1 throughout **147/319**, other top-1 without suggestion **115/319**, suggestion top-1 at least once **57/319** | ○ | 관측한 6개 위치와 한 레이어의 궤적. 연속 토큰 전체나 다른 레이어를 뜻하지 않음 |
| "소견서가 내부 상태에도 비용을 준다" | final paired Δ: 유지 −.006, 제3 진단 −.054, 제안 채택 −.199 | ○ | paired CI 모두 0 배제; trend ρ=−.282 [−.328,−.233] |
| "한 번의 실행에서 찾아낸다" | 진단 내 층화 AUROC: AV 판독 **.7511/.8319**, probe **.9330/.9881** (전체/침묵) | ○ | AV가 최강이라는 주장은 하지 않음 |
| "활성값 의존 자연어 계기로 서술한다" | 검증 배터리(스왑 .993 · 암기 .000 · 오염 .007 · heldout cue .751) | ○ | 계측·탐색 범위. 임상의용 효용은 별도 주장 |
| "임상의가 읽을 인터페이스로 유용하다" | canonical controlled reader-trust ΔAUROC **−.0998 [−.135,−.065]** | **✕(현재 어댑터)** | shuffled에서 case alignment +.2854이지만 순효과 −.0998 |
| "되먹여 되돌린다" | canonical moved 319: .0031 → r5 .6301 / r6 .8339, r5−r4 +22.6%p | ○ | **조건부**. 무선별 재실행은 순손해. 선별과 결합해야 이득 |
| "자연어이기 때문에 되돌린다" | 전체 correct/correct p=1.000; moved에서 라벨 7:0, p=.016(보정 미달) | **✕** | 지렛대는 내용 적중률. 형식 우위 주장 금지 |

**결론: 수정된 한 문장은 성립한다. 단 세 가지 각서를 달고서다.**
1. **내부 해부는 실험대(DDXPlus) 안에서만 주장한다.** 행동 효과는 두 코퍼스,
   내부 기전은 한 코퍼스 — 이 경계를 흐리면 과장이다. source-aligned MCR
   결론 판독과 사다리가 이 경계를 확장할 별도 실험이다.
2. **"찾아낸다"는 우리가 최고라는 뜻이 아니다.** 닫힌 코퍼스 탐지 최강은
   프로브이고, 우리는 그 숫자를 그대로 싣는다. 판독의 현재 몫은 **계측·오류
   유형 탐색·열린 어휘 가설 생성**이며, 근거 접지와 독자 효용은 미달이다.
3. **"되돌린다"의 지렛대는 형식이 아니라 내용이다.** 자연어라서 잘 고치는
   게 아니라, **되먹인 내용이 맞아서** 고친다. 자연어의 유일성은 회복률이
   아니라 **클래스 되먹임 경로가 존재하지 않는 코퍼스**에서 나온다.

이 세 각서가 논문의 정직성을 지탱하며, 동시에 각각이 하나의 절이 된다
(각서1→4.2의 MCR 문단·Limitations, 각서2→4.3, 각서3→4.4).

### 왜 이 순서인가 (프레이밍 결정)

이전 판본은 "검증된 AV 판독으로 …를 읽어"로 시작했다. 도구가 주어이면
독자의 첫 질문이 **"그래서 프로브보다 나은가"**가 된다. canonical probe
값도 정본에서 더 강하다. 논문의 실제 주인공은 계기가 아니라
**앵커링의 기전**이며, 두 계기는 그것을 보이는 방법이다 —
**프로브가 어디서·얼마나를 재고(숫자), AV 판독이 후보 의미를 탐색한다
(문장).** 판독이 충실한 "왜"를 제공하거나 임상의에게 유용하다는 주장은 현재
결과가 지지하지 않는다.

## 연구 질문 3개와 측정 관문 1개

### 먼저 분리할 것: AV 검증은 연구 질문이 아니라 측정 관문이다

이전 구조는 RQ1을 “AV가 activation을 잘 읽는가”로 두었다. 그러면 논문이
Medical-NLA 성능 논문처럼 보이고, 독자는 본 현상보다 먼저 “왜 probe가 아니라
AV인가?”를 묻게 된다. 현재 데이터에서 닫힌 49-class 공간의 탐지 성능은
probe가 AV보다 높으므로 이 프레이밍은 논리적으로도 불리하다.

새 구조에서는 **measurement gate M0**를 RQ보다 먼저 둔다.

> **M0. 자연어 AV readout이 downstream 분석에 사용할 만큼 activation-specific
> content를 읽는가?**

M0는 AV의 우월성을 증명하지 않는다. AV가 prompt나 verbalizer의 사전지식만으로
그럴듯한 문장을 만드는지 배제하는 최소 관문이다. Swap `.993`, memorization
`.000`, contamination `.007`, heldout cue `.751`, shuffled `.096`이 이 관문을
구성한다. 통과한 뒤에도 AV는 보조 계기이며, 닫힌 label space의 정량 궤적은
cross-fitted linear probe를 주 계기로 측정한다.

### RQ1 — 현상과 내부 상태

> **잘못된 의뢰 소견서는 원래 정답이던 의료 LLM의 최종 진단을 얼마나
> 움직이며, 출력 이동은 내부에서 디코드되는 제안 진단의 우세와 일치하는가?**

- 행동 효과: DDXPlus canonical-eligible clean 1,204에서 wrong-note 총 비용
  `23.75pp`, neutral 대비 제안 고유 비용 `18.36pp`; MCR canonical-eligible
  1,452에서 총 비용 `29.34pp`, 제안 고유 `22.73pp`.
- 내부 궤적: DDXPlus moved 319건 중 suggestion이 관측한 여섯 landmark에서
  한 번도 probe top-1이 아닌 사례가 262건(`82.1%`). 이 중 gold throughout는
  147건, suggestion이 아닌 다른 진단 경로는 115건이다.
- 따라서 RQ1은 “모델이 속으로 항상 정답을 안다”가 아니라 **행동적 이동과
  내부 suggestion dominance가 같은 사건이 아니다**라는 명제를 묻는다.

### RQ2 — 단일 실행 소견서 영향 판별과 채널 비교

> **같은 사례에서 referral sentence만 제거한 no-note 기준 실행을 detector에게
> 보여주지 않고, wrong-note 실행 한 번만으로 어떤 답이 그 소견서 때문에
> 바뀌었는지 판별할 수 있는가? 출력·CoT·probe·AV는 각각 무엇을 추가하는가?**

- Ground-truth `moved`는 실험자가 같은 사례의 no-note/wrong-note pair로 만든다.
  Detector는 이 pair를 보지 않고 wrong-note run 하나만 받는다. 따라서 probe나
  AV가 인과관계를 만드는 것이 아니라, 개입으로 이미 정의된 인과 label을
  단일 실행에서 예측한다.
- 닫힌 DDXPlus에서는 probe가 가장 강하다: all/silent AUROC `.9330/.9881`.
  AV는 `.7511/.8319`, LLM CoT monitor는 `.7305/.6904`다.
- AV를 쓰는 이유는 probe를 이기기 위해서가 아니다. Probe는 49개 label 중
  하나와 확률을 주지만, AV는 label set을 미리 정의하기 어려운 열린 어휘에서
  내부 결론 후보와 supporting-cue 후보를 문장으로 제안할 수 있다. 현재 결과는
  결론 슬롯의 예비 가능성만 지지하며, grounds와 reader utility는 지지하지 않는다.

### RQ3 — 조건부 교정과 효용

> **Wrong-note 단일 실행에서 harmful movement가 식별됐을 때, 정확한 내부
> content를 선택적으로 다시 제공하면 unaffected answer를 보존하면서 moved
> answer를 회복할 수 있는가? 무엇이 지렛대이고 언제 개입이 순손해가 되는가?**

- moved accuracy는 first wrong answer `.0031`에서 AV feedback r5 `.6301`, probe
  label r6 `.8339`로 상승한다.
- R5가 evidence-only r4보다 `+22.6pp` 높지만, content-correct subset에서 r5와
  r6의 전체 차이는 없다(`p=1.000`). 확립된 명제는 **자연어 형식의 우월성**이
  아니라 **정확한 내부 content가 지렛대**라는 것이다.
- 무선별 재질문은 kept answer를 대량으로 깨므로 selector와 결합해야 한다.
  자기 CoT feedback r7의 moved recovery는 `.1236`으로 r5/r6보다 낮다.
- 여기까지의 사다리는 **사후에 moved로 확인된 subset에서의 conditional
  information value**를 확립한다. 과거 fixed-cohort probe-selector+r5 `.9141`은
  예비 결합 결과지만, 최신 canonical 1,729 재집계와 validation-frozen threshold,
  test paired CI가 끝나기 전에는 전체 QA 성능 향상으로 쓰지 않는다.
- 배포 정책은 wrong-note run 하나의 detector score만 사용해 개입 여부를 고른다.
  Gold와 no-note pair는 moved label과 최종 평가에만 사용하며 selector 입력에는
  들어가지 않는다.

테제 한 줄: **앵커링의 행동적 출력 이동과 내부에서 디코드되는 진단 우세는
자주 어긋난다. Probe는 닫힌 공간에서 이를 정밀하게 측정하고, 검증된 AV는
더 약하지만 자연어 후보를 탐색한다. 정확한 내부 내용은 조건부 교정에 쓰일
수 있으나, 현재 자연어 readout 자체의 임상 효용은 확립되지 않았다.**

## 08-25 실측이 정리한 측정 채널의 역할

이 절은 위의 H1–H3와 별도인 새 가설 목록이 아니다. 왜 output/CoT 외에 내부
채널을 추가했고, 왜 probe와 AV를 함께 유지하는지를 기록한 측정 선택의 근거다.

**문제 A — CoT 텍스트만으로 소견서가 답을 바꾼 원인을 판별하는 것은 불완전하다.**
⇒ 내부 채널이 추가 정보를 주는지 비교해야 한다.
- 1-1. 단순 규칙 특징은 약하지만(.50–.55), 강한 LLM 모니터는 전체 .7305,
  침묵 .6904를 얻는다. 체인에 신호가 없다는 주장은 철회한다. 같은 침묵
  집단에서 내부 자연어 판독은 .8319이며 점추정 차이는 +.1415다
  (canonical 1,729 bootstrap CI 전사 대기).
- 1-2. **추론을 시켜도 사라지지 않는다**: CoT가 피해를 1/4로 줄이지만
  (−17.7 → −4.4%p) **제안 채택률은 오히려 늘어난다**(29% → 43%). 추론은
  방패인 동시에 합리화의 지면이다.
- 선행 대비: Turpin(2023)이 원형. 우리 기여는 **의료 + 증거 불변 인과 설계 +
  전수 측정**.

**문제 B — 기존 내부 도구(linear probe / SAE)는 *탐지*는 해결하지만
*서술*과 *열린 어휘*는 해결하지 못한다.** ⇒ 방식의 변경이 필요하다.
- 2-1. **탐지는 실제로 해결된다 — 우리가 직접 측정해 인정한다**: 같은
  final-token 활성값에서 지도 프로브와 AV 판독을 동일 벡터·동일 모집단에서
  비교한다. AV 판독은 .7511/.8319, probe는 .9330/.9881(all/silent)이다.
- 2-2. **남는 문제 ①: 서술하지 못한다.** 분류 헤드의 출력은 클래스명
  하나이고 "왜"에 해당하는 근거 슬롯이 없다. 필요한 실험: 같은 벡터에서
  probe argmax와 AV 판독(결론+근거)을 나란히 놓고 **사람이 읽어** 비교.
  → reader-trust 전수 결과에서 현재 판독은 no-account보다 나쁘다
  (canonical controlled ΔAUROC −.0998). Shuffled 통제는 판독의 case
  alignment가 실재함을 보이지만(real .7342 vs shuffled .4488), 오경보가
  그 이득을 압도한다.
- 2-3. **남는 문제 ②: 열린 어휘에서 정의되지 않는다.** 필요한 실험: 같은
  개입을 실제 증례(MCR)에서 재현하고, 그 코퍼스에서 클래스 정의가 불가함을
  보인다. → **완료**: canonical-eligible MCR 1,452건에서 개입 효과가 재현됐다
  (총비용 29.34pp, neutral 대비 4.44배), 진단 6,934종·대부분 1회 등장.

**문제 C — NLA는 서술 문제를 해결하지만, *언어화 모델이 자기 지식으로
지어낼 수 있다*는 문제가 남는다.** 그래서 검증된 의료 NLA를 제시한다.
- ZZ의 출처: Li et al.(ICML 2026) — 언어화가 **대상 모델이 아니라 언어화
  모델의 파라미터 지식**을 반영할 수 있고, 기존 벤치마크는 **내부 접근 없이도
  통과 가능**하다.
- 우리의 답 = 검증 배터리: 소견 하나를 바꾸면 판독이 따라감 **0.993** ·
  바꿔도 원본을 말하는 암기 **0.000** · 타 환자 내용 오염 **0.007** ·
  형식 준수 0.05→1.00. 그리고 **증거 표현이 비트 단위로 불변인 인과
  실험대** 자체가 "입력만 보고도 맞힐 수 있다"를 차단한다.

"우리는 xx, yy, zz 문제 다 없다"는 실측과 어긋난다. 정직한 판본:

> CoT·출력만으로 놓치는 내부-출력 불일치를 측정한다. 닫힌 코퍼스의 탐지
> 정확도에서는 지도 프로브가 판독을 이기며, 우리는 그것을 그대로 보고한다.
> 자연어 판독은 벡터 의존 계측과 열린 어휘 결론에서 예비 신호를 보이지만,
> 근거 접지와 독자 효용에서는 실패한다. 따라서 기여는 완성된 설명기가 아니라
> **계기별 가능 범위와 실패 경계를 실측한 것**이다.

교정 절도 같은 원칙: "우리가 최고"가 아니라 **"되먹임의 지렛대는 형식이
아니라 내용의 정확도이고, AV 판독은 증거 재제시 통제를 이긴다(+22.4%p,
항복률 −14.7%p)"**까지만 주장한다.

구조 (08-24 확정): Intro / Related Work / **Methodology** / **Experimental
Results** / Conclusion. (초기 판단 "별도 Methods 없음"은 철회 — 도구는
기존 것이나 감사 절차 전체는 이 논문의 방법론이다.) 각 절의 임무를 한
문장으로 먼저 적고, 들어갈 내용과 자산(표·그림·수치)을 매핑한다.
▢ = 아직 숫자가 안 들어온 것.

---

## 0. 초안을 쭉 적으면 — Abstract와 절별 서사 (08-25 신설)

절별 계획은 아래 1–5장에 있다. 이 절은 **그 계획대로 쓰면 논문이 실제로
어떤 글이 되는지**를 이어서 읽히는 형태로 보인다. 집필은 여기서 문단을
펼치는 작업이 된다.

### 0.1 Abstract (초안 · 약 300 단어 — 투고 규정에 맞춰 250으로 줄일 것)

> Large language models are entering diagnostic workflows, and clinical
> inputs are rarely neutral: a referral note usually arrives already naming
> a suspicion. We ask what such a sentence does to a medical LLM, whether
> the model's own explanation reports it, and whether the model's internal
> state agrees with the answer it emits.
>
> We build a causally controlled testbed in which the patient findings are
> held fixed and only a one-sentence referring note varies across four arms
> — none, a neutral placebo, a plausible wrong suspicion, and the correct
> one. The note follows the findings, so under causal masking the
> activations at every finding position are bit-identical across arms. A
> wrong suspicion costs 23.75 accuracy points on DDXPlus (n = 1,204) and
> 29.34 on MedCaseReasoning case reports (n = 1,452); 18.36 and 22.73 points
> are specific to the suggestion rather than to inserting a sentence.
> Rule-based chain-of-thought features are weak predictors of movement,
> while a stronger LLM monitor reaches AUROC 0.723 overall and 0.683 on
> the canonical silent subset.
>
> On the controlled DDXPlus testbed, we quantify the internal state primarily
> with cross-fitted diagnosis probes. We also use a natural-language
> activation-verbalization (AV) readout as a complementary open-vocabulary
> channel, but only after validating that it follows the paired activation:
> editing one finding changes the description 99.3% of the time, the
> pre-edit wording is never recited afterwards, and cross-patient
> contamination is 0.007 against a 0.015 chance rate. Among 319 causally
> moved cases, the suggestion is never probe top-1 at any observed landmark
> in 262 (82.1%). Of those, 147 retain gold top-1 throughout and 115 pass
> through another diagnosis. Thus answer movement frequently occurs without
> internal top-1 adoption of the suggestion, but the trajectories are
> heterogeneous. The mismatch is detectable from a single deployed run. On
> the subset whose answer does not name the suggestion, where output copying
> is blind by construction, a cross-fitted diagnosis probe reaches AUROC
> 0.988, the AV readout 0.832, and a strong LLM monitor of the vignette, note,
> chain, and answer 0.690. Feeding the AV reading back recovers moved cases
> from 0.003 to 0.630, while a probe label reaches 0.834; the AV reading beats
> a re-shown-evidence control by 22.6 points. Once content accuracy is
> controlled, no natural-language form advantage is detected.

각주로 붙일 것: 백본 gemma-3-12b-it, NLA 체크포인트 L32. LoRA는 출력 schema와
의료 target mapping을 학습하므로 그 자체를 새로운 activation 정보의 증거로
해석하지 않고, M0의 pairing 통제로 별도 검증함(4.1).

### 0.2 논문을 쭉 읽으면 나오는 이야기 (절별 한 문단)

**§1 Introduction.** 진단 오류에 인지 편향이 관여하고, 그중 앵커링이
대표적이다. LLM이 그 워크플로에 들어오는데, LLM의 설명은 원인을 말하지
않는다. 우리 실물 훅: 소견서 한 문장이 정답률을 23%p 떨어뜨리는데 규칙 기반
체인 특징은 이를 거의 예고하지 못하고, 강한 LLM monitor도 내부 채널보다
약하다. "소견서는 성급하다"고 기각을 선언한 체인에서도 답이 바뀌는 사례가
있다. 그래서 내부를 본다. 단, 언어화가 지어낼 수 있다는
비판이 있으므로 **계기부터 검증하고** 시작한다.

**§2 Related Work.** 의료 설명의 두 형태(입력 기여도 · 자기 서술)가 둘 다
"무엇이 이 답을 만들었나"에 실패한다. 내부를 읽는 도구는 프로브·렌즈·SAE에서
자연어 판독(Patchscopes/SelfIE/LatentQA/NLA)으로 왔고, 마지막 비판이
Li et al.의 "언어화 모델의 지식일 수 있다"이다. 의료 진단에서 개별 사례의
답이 바뀐 원인을 사례별 자연어로 판별한 선행은 없다.

**§3 Methodology.** 데이터 둘의 분업(DDXPlus = 통제된 인과·닫힌 진단 공간,
MCR = 실제 임상 언어·열린 진단 공간)을 먼저 설명한다. 다음으로 환자 소견은
고정하고 의뢰 소견서만 바꾸는 4조건 개입과, 그 설계가 보장하는 cue-position
activation의 비트 동일성을 정의한다. 그 뒤에야 내부 채널을 소개한다. 닫힌
DDXPlus에서는 cross-fitted linear probe가 주 정량 계기이고, AV 판독은 자연어
내용 후보를 내는 보조 계기다. AV는 그럴듯한 문장을 스스로 만들 수 있으므로
swap·shuffle·heldout·오염 통제로 activation-specificity를 먼저 통과해야 한다.
마지막으로 출력·CoT·LLM monitor·probe·AV를 같은 single-run moved 탐지 과제에서
비교하고, 궤적 및 교정 사다리의 모집단과 지표를 고정한다.

**Appendix A 측정 관문(M0).** 이는 첫 번째 현상 결과나 독립 RQ가 아니라, 이후
AV 산문을 activation 관측치로 취급하기 위한 선행 calibration이다. 무학습
체크포인트도 읽을 줄은 안다(서술률 0.72, 우연
0.088)지만 계기는 아니다(형식 준수 0.05, 지어낸 액자에 내용을 섞는다).
어댑터는 출력 schema와 의료 target mapping을 학습해 자동 계측성을 높인다
(형식 1.00, 길이 1,557→52자, 정밀도 0.671). 이 변화만으로 새 activation
정보를 얻었다고 해석하지 않는다. 그리고 네 기둥: 소견 하나를 바꾸면 서술이
따라오고(0.993),
바꾼 뒤 원본을 읊지 않으며(0.000), 남의 케이스를 섞지 않고(0.007),
학습에서 뺀 내용도 서술한다(0.75). 그리고 결정적 대조 —
같은 답-위치 벡터를 무학습 체크포인트에 줘도 상실형의 60.3%에서 정답을
짚는다: **결렬은 우리가 만든 게 아니라 활성값에 있다.** Li et al.의 두
비판이 여기서 닫힌다.

**§4.1 교란은 실재하고 설명은 예고하지 않는다.** 4조건 정확도, 답이 바뀐
319건의 행방(채택 89 · 제3 진단 230 — **답만 봐서는 다수가 안 보인다**),
체인의 무예고(0.50–0.53), CoT의 이중성(피해를 1/4로 줄이되 채택률은 늘림).
꼬리에 강건성: 문구 4종에서 실제형 소견서의 낙폭이 가장 크게 관측됐지만,
길이·레지스터 matched placebo가 없어 추가 낙폭의 원인은 미분리다. 흔들림과
설득의 분리는 MCR에서도 방향이 재현된다.

**§4.2 내부 궤적과 출력은 자주 어긋난다.** moved 319건 중 제안이 어느
랜드마크에서도 top-1이 아닌 경우가 262건이다. 그중 gold throughout는
147건, 다른 진단 top-1은 115건이다. 따라서 “상태는 항상 정답을 보존한다”가
아니라 “출력 이동에 제안 top-1 채택이 필요하지 않다”가 정본 주장이다.
단일 실행 소견서 영향 판별은 LLM 모니터 .7305/.6904, AV 판독 .7511/.8319,
probe .9330/.9881(all/silent)이다.
프런티어 모니터를 세워도 내부가 이긴다는 것이 이 절의 실제 주장이고,
프로브가 이기는 조건과 그 조건이 무너지는 코퍼스를 같은 자리에서 말한다.

**§4.3 교정.** 되먹임은 작동한다(.0031 → .6301/.8339). 그러나 재실행 자체가
파괴적이고, 부서진 답은 제안 쪽으로 더 자주 간다. 지렛대는 내용이다 — 증거
재제시 통제 대비 +22.4%p이며, 전체 correct/correct에서 형식의 추가 기여가
검출되지 않는다(p=1.000). 병목은 탐지가 아니라 선별이고(recall .8457 /
precision .3615), 허위 경보의 68.4%가 어댑터 오독이다. 자기 CoT r7도 moved
회복 .1236에 그쳤다.

**§5 Conclusion.** 앵커링은 내부의 소거가 아니라 출력의 사건이다. 계기는
분업한다 — 닫힌 공간에서는 프로브가 강하고, 열린 공간에서 자연어 판독이
차별화되는지는 MCR 실험으로 검증한다. 재고
프롬프트는 위험하다. 한계는 정직하게 나열한다.

### 0.3 초안 집필 순서 (막힌 것 없이 지금 쓸 수 있는 순서)

1. §3 Methodology — 주장이 없어 실측 대기와 무관하다. 가장 먼저 쓴다.
2. §4.1 · §4.3 — 실측 완결. 표(T1·T3)가 이미 조판 형태다.
3. §4.2 — Figure 3 count와 Table 2a canonical 확률은 확정. canonical probe
   AUROC와 Δ의 CI/추세 검정은 대기.
4. Appendix A — 답 위치 vanilla 대조 완료(08-24). 대기 없음.
5. §1 · §5 — 4장이 고정된 뒤에 쓴다(기여 목록이 4장의 수치를 인용).
6. §2 — 초안·LaTeX 완성. 정독 노트 2편 반영만.

---

## 1. Introduction

**임무**: 임상적으로 실재하는 잠정 진단이 downstream 판단을 좁힐 수 있다는
문제에서 출발해, `행동적 출력 이동 != 내부 suggestion dominance`라는 대전제를
세우고, 이를 측정·단일 실행 소견서 영향 판별·조건부 교정하는 세 RQ로 도달한다. 자연어 AV는
이 현상의 유일한 증거도 주인공도 아니다. 닫힌 진단 공간에서는 cross-fitted
probe가 주 정량 계기이고, AV는 activation-dependent 자연어 후보를 제공하는 보조
계기다.

**용어 고정**: 본문의 `note`는 전체 chart나 full referral letter가 아니라,
동일한 patient findings 뒤에 삽입하는 한 문장짜리 `referral diagnostic
suggestion`이다. `no-note`는 findings가 없는 조건이 아니라 이 추가 문장만 없는
조건이고, `wrong-note`는 동일 findings에 plausible wrong diagnosis를 잠정
진단으로 덧붙인 조건이다.

### 1.1 먼저 제시할 대전제·가설·RQ

인트로 초반에 독자가 논문의 논리 구조를 잃지 않도록 다음 순서로 명시한다.

> **대전제.** 의료 LLM의 최종 출력과 생성 CoT는 내부에서 decode 가능한 진단
> 상태를 완전히 대표하지 않을 수 있다. 잘못된 임상 제안은 제안 진단을 내부
> top-1으로 만들지 않고도 최종 답을 바꿀 수 있으므로, 행동·자기서술·activation을
> 분리해 측정해야 한다.

이 대전제는 “모델이 속으로 항상 정답을 안다”는 주장이 아니다. `probe top-1`은
해당 activation에서 label이 선형 decode 가능하다는 뜻일 뿐, 생성에 실제로
사용됐다는 인과적 증거나 인간과 같은 belief가 아니다.

- **H1 — dissociation**: wrong note가 답을 바꿔도 suggestion이 관측한 activation의
  top-1 diagnosis가 되지 않을 수 있다. Gold가 유지되는 경로와 제3 진단으로
  이동하는 경로를 모두 허용한다.
- **H2 — single-run note-influence detection**: 같은 사례에서 referral sentence만
  제거한 no-note 기준 실행을 detector에게 주지 않아도, wrong-note 실행의 내부
  채널은 output/CoT 채널보다 **그 note 때문에 답이 바뀐 사례를 더 잘 판별할 수 있다.**
- **H3 — conditional correction**: 정확한 내부 content는 moved answer 교정에
  유용하지만, 무선별 재실행과 부정확한 판독은 kept answer를 파괴한다. 이득은
  자연어 형식 자체보다 content 정확도와 selector에 의해 결정된다.

세 가설은 다음 질문과 일대일로 연결한다.

- **RQ1 (현상·궤적)**: wrong referral note는 진단 행동을 얼마나 움직이며,
  moved case에서 gold·suggestion·other diagnosis의 decodable signal은 prompt
  landmark를 따라 어떻게 변하는가?
- **RQ2 (단일 실행 소견서 영향 판별)**: wrong-note 실행 한 번의 output, CoT, LLM
  monitor, probe, AV 중 무엇이 숨겨진 no-note 반사실이 정의한 **note-caused
  answer movement**를 가장 잘 식별하는가? 답이 suggestion을 말하지 않는
  canonical silent subset에서도 가능한가?
- **RQ3 (교정)**: decode한 내부 content를 source model에 되먹이면 답을 회복하는가?
  효과를 재실행, evidence 재제시, label, 자연어 readout으로 분해하면 실제
  지렛대는 무엇인가?

AV의 pairing 검증은 별도 연구 질문이 아니라 **M0 measurement gate**다. AV가
paired activation을 따라가는지, verbalizer의 의료 지식과 template prior를
말하는지를 swap·shuffle·heldout·contamination으로 검사한 뒤 제한적으로 쓴다.

### 1.2 실제 Introduction의 7문단 전개

**문단 1 — 임상 workflow와 anchoring.** 환자는 빈 종이로 downstream
clinician이나 model에 도착하지 않는다. Referral letter에는 증상·검사뿐 아니라
잠정 진단이 포함될 수 있고, 사람 대상 실험에서도 진단 제안이 감별진단의 폭을
줄였다. Staal et al.의 무작위 within-subject 연구에서는 제안 없음 대비 정답·오답
제안 모두 고려한 감별진단 수를 줄였다(`1.85 → 1.52/1.42`, `p=.022`).
Spaanjaars et al.도 referral letter의 diagnostic anchor가 일부 경험군의 진단에
영향을 줌을 보고했다. 따라서 본 설정은 “모든 소견서가 진단명을 포함한다”가
아니라 **referral-mediated diagnostic workflow에서 가능한 잠정 진단 변수**를
통제한 것이다.

- [Staal et al., BMC Medical Education, 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8991944/)
- [Spaanjaars et al., European Journal of Psychological Assessment, 2015](https://doi.org/10.1027/1015-5759/a000235)
- [AHRQ PSNet: Anchoring Bias With Critical Implications](https://psnet.ahrq.gov/web-mm/anchoring-bias-critical-implications)

**문단 2 — 의료 LLM의 행동 취약성은 이미 알려져 있다.** BiasMedQA는 1,273개
USMLE 문항에 일곱 종류의 임상 인지 편향 문장을 주입해 모델별 정확도 저하를
보였다. MED-STRESS는 다중 턴 임상 압박에서 초기 정답을 포기하는 현상을,
MedMisBench는 오도 맥락에서 평균 정확도가 `71.1%→38.0%`로 떨어지는 현상을
보고했다. Narrative Anchoring은 임상 사실을 보존하고 문체만 바꿔도 진단이
달라짐을 보였다. 따라서 “의료 LLM은 외부 맥락에 흔들린다” 자체는 우리의 최초
기여가 아니다. 남은 질문은 **답이 흔들릴 때 내부 진단 표현이 실제로 어떻게
변했는가**다.

- [Schmidgall et al., BiasMedQA, npj Digital Medicine 2024](https://www.nature.com/articles/s41746-024-01283-6)
- [Xiao et al., MED-STRESS, ACL 2026](https://arxiv.org/abs/2605.23932)
- [Zhou et al., MedMisBench, 2026](https://arxiv.org/abs/2606.12291)
- [Singh et al., Narrative Anchoring, 2026](https://arxiv.org/abs/2607.27384)

**문단 3 — 출력과 CoT는 원인의 완전한 관측치가 아니다.** Turpin et al.은
답을 움직이는 편향 특징이 CoT에 언급되지 않고 사후 합리화될 수 있음을 보였고,
Lanham et al.은 CoT 개입에 대한 의존도가 모델·과제별로 크게 다름을 보였다.
의료에서도 Afolabi et al.은 causal ablation과 hint injection으로 외부 제안이
인정 없이 흡수될 수 있음을 보고했다. 다만 이것이 “CoT에 신호가 없다”는 뜻은
아니다. 우리 강한 LLM monitor는 유의한 신호를 읽는다. 정확한 문제는
**self-report만으로 소견서가 답을 바꾼 원인을 완전히 판별할 수 없고, 추가 내부 채널의 증분을 직접
비교해야 한다**는 것이다.

- [Turpin et al., NeurIPS 2023](https://arxiv.org/abs/2305.04388)
- [Lanham et al., 2023](https://arxiv.org/abs/2307.13702)
- [Afolabi et al., PMLR 2026](https://arxiv.org/abs/2603.13988)

**문단 4 — 내부-출력 해리도 이미 인접 선행이 있다.** Catching Rationalization은
일반 객관식에서 pre/post-generation probe가 CoT monitor와 같거나 더 강하게
motivated reasoning을 탐지함을 보였다. 의료에서는 Fraile Navarro et al.이 동일
Gemma-3-12B NLA와 L32 activation으로 triage format failure에서 임상 내용이
보존됨을, Tayebi Arasteh가 임상 근거 등급이 activation에서 복원되지만 stated
grade에는 나타나지 않음을, Basu et al.이 위험 신호 probe AUROC `.982`와 낮은
출력 sensitivity 사이의 knowledge-action gap을 보였다. 따라서 우리의 신규성은
“의료 내부-출력 불일치 최초”가 아니라 **진단 제안이라는 인과 개입에서 출력
이동과 suggestion dominance를 위치별로 분해하고, 그 원인을 사례 단위로 탐지한
뒤 교정까지 같은 프로토콜로 연결한 것**이다.

- [Mirtaheri and Belkin, Catching Rationalization, 2026](https://arxiv.org/abs/2603.17199)
- [Fraile Navarro et al., 2026](https://arxiv.org/abs/2605.29889)
- [Tayebi Arasteh, 2026](https://arxiv.org/abs/2606.29034)
- [Basu et al., 2026](https://arxiv.org/abs/2603.18353)

**문단 5 — 왜 probe와 자연어 readout을 함께 쓰는가.** Probe는 질문을 미리
정한 닫힌 label space에서 강한 정량 계기다. Patchscopes, SelfIE, LatentQA,
NLA는 activation을 자연어로 풀어 고정 vocabulary 밖의 후보를 만들 수 있다.
하지만 Li et al.은 verbalization benchmark를 target activation 없이도 풀 수 있고,
출력이 target model보다 verbalizer의 parametric knowledge를 반영할 수 있음을
보였다. 따라서 본 논문은 probe를 upper bound가 아니라 **주 정량 계기**로
보고하고, AV는 M0를 통과한 범위의 보조 계기로만 쓴다.

- [Patchscopes, ICML 2024](https://arxiv.org/abs/2401.06102)
- [SelfIE, ICML 2024](https://arxiv.org/abs/2403.10949)
- [LatentQA, 2024](https://arxiv.org/abs/2412.08686)
- [Natural Language Autoencoders, Anthropic 2026](https://transformer-circuits.pub/2026/nla/index.html)
- [Li et al., ICML 2026](https://arxiv.org/abs/2509.13316)

**문단 6 — 접근과 결과 훅.** 환자 findings를 고정하고 referral note만
`none/neutral/wrong/correct`로 바꾸는 four-arm 실험을 만든다. Note가 findings
뒤에 오므로 cue-position activation은 causal masking 아래 조건 간 동일하다.
행동 효과는 DDXPlus와 MCR에서 재현하고, 내부 궤적은 DDXPlus에서
cross-fitted probe와 검증된 AV로 측정한다. 배포형 탐지에서는 detector가 none
반사실을 보지 않고 wrong run 하나만 받는다. 이 설계에서 wrong note는 DDXPlus
정확도를 `23.75%p`, MCR을 `29.34%p` 낮춘다. canonical-eligible DDXPlus
trajectory의 moved 319건 중 262건에서
suggestion은 관측한 어느 landmark에서도 probe top-1이 아니다.

**문단 7 — 기여 5개 (08-25 실측으로 갱신):**
  1. **인과 테스트베드**: 소견서 4조건 개입 — 임상적으로 실재하는 교란,
     위약 대조, **cue 위치 활성값 불변이 가정이 아니라 설계로 보장**.
     합성(DDXPlus canonical all 1,729/clean 1,204)과 실제 증례(MCR 1,452)
     양쪽에서 재현된다(총비용 23.75pp vs 29.34pp; neutral 대비 비는
     4.40배 vs 4.44배).
  2. **앵커링의 해부** (중심 기여): moved 319건 중 **262건(82.1%)**에서
     제안이 어느 랜드마크에서도 top-1이 아니다. 그중 gold throughout는
     147건이고 other top-1은 115건이다. 행동적 이동과 내부 제안 우세의
     불일치 및 그 이질성을 함께 보고한다.
  3. **단일 실행 소견서 영향 판별의 정직한 지도**: 설명문 **0.50–0.53** · 출력 기반은 침묵
     부분집합(답 바뀜의 2/3)에서 **구조적 장님** · 닫힌 코퍼스에서는 지도
     프로브 **.9330/.9881**, AV 판독 **.7511/.8319**(all/silent)로
     닫힌 공간에서는 프로브가 앞선다. 판독의 자리는 계측과 열린 어휘 가설
     생성으로 좁힌다.
  4. **조건부 교정의 네 발견**: 되먹임은 작동한다(moved **.0031→.6301/.8339**) ·
     **재실행 자체가 해롭다** · **지렛대는 형식이 아니라 내용 정확도**다 ·
     자기 CoT는 moved의 **.1236**만 회복한다. AV 판독은 증거 재제시 통제를
     +22.4%p 이기지만, 임상의용 인터페이스 효용은 reader-trust가 지지하지 않는다.
  5. **자연어 내부 판독의 측정 경계**: AV 판독은 스왑 추적 **0.993**,
     원본 암기 **0.000**, 타 환자 오염 **0.007**, heldout cue **.751**로
     activation pairing을 추적한다. 그러나 probe보다 약하고, MCR 근거 접지와
     reader utility는 실패했다. 즉 AV를 우월한 detector가 아니라 검증이 필요한
     자연어·가설 생성 채널로 위치시킨다.
- 마지막에 로드맵 한 줄: §2는 위 네 선행 흐름과 남은 교차점을 정리하고, §3은
  인과 테스트베드와 계기를 정의하며, §4는 행동→궤적→단일 실행 영향 판별→교정
  순으로 답한다.

## 2. Related Work (확정 3절)

최신 선행을 두 절로 압축하면 의료 행동 연구와 내부 해석 연구 사이의 직접
경쟁자가 가려진다. 세 절로 분리한다.

Related Work 직전 또는 끝에서 신규성을 다음처럼 한 문장으로 고정한다.

> **Our novelty is the case-level causal attribution problem: we define whether
> a wrong clinical suggestion moved each answer using a hidden same-case
> counterfactual, predict that event from one observable run, trace the
> competing diagnoses through answer formation, and test when decoded content
> supports correction.**

따라서 이 논문은 anchoring, probe, NLA 중 하나의 최초성을 주장하는 논문이
아니다. `moved`라는 반사실적 사례 단위 정답지와 이를 중심으로 한
**intervention → trajectory → single-run attribution → conditional correction**
연결이 기여다. Primary clean behavior cohort의 moved 287건 중 201건은 suggestion이 아닌 제3 진단으로
이동하므로, 이 문제는 단순 hint-copy detection으로 환원되지 않는다.

- **2.1 Clinical anchoring and misleading context**: 사람의 referral-letter
  anchor(Staal; Spaanjaars) → BiasMedQA·MED-STRESS·MedMisBench·Narrative
  Anchoring. 결론은 “행동 취약성은 알려져 있다; 우리 행동 낙폭은 신규성보다
  실험대 검증이다.” 우리의 차이는 placebo, same-case counterfactual, evidence
  invariance, 제3 진단 행방 분해다.
- **2.2 CoT faithfulness and internal-output dissociation**: Turpin·Lanham·Afolabi
  → Catching Rationalization → Fraile Navarro·Tayebi Arasteh·Basu. 결론은
  “CoT가 완전한 원인 기록이 아니며 내부가 출력을 초과할 수 있다는 것도
  알려져 있다.” 우리의 차이는 **진단 제안이 답을 바꾼 사례의 원인 판별**, 위치 궤적, silent
  subset, single-run moved label이다.
- **2.3 Reading and acting on activations**: probe·lens·SAE → Patchscopes·SelfIE·
  LatentQA·NLA → Li et al.의 privileged-information 비판 → Sun et al.의 selective
  reprompting과 Basu/Vankadaru/Liu의 decodability-control gap. 결론은 probe와
  AV의 역할을 분업하고, 검증된 content를 되먹이는 조건부 교정을 시험한다.
  자연어 형식의 우월성이나 임상의용 효용은 주장하지 않는다.

## 3. Methodology

### 이 절의 논리와 AV의 위치

이 논문의 연구 대상은 AV 자체가 아니라 **잘못된 임상 제안 아래에서 나타나는
내부-출력 불일치**다. Methodology는 다음 순서로 읽는다.

1. 어떤 임상 입력과 평가 모집단을 사용하는가(3.1).
2. wrong note의 내용만 바꾸는 인과 개입과 `moved` 정답지를 어떻게 만드는가(3.2).
3. 출력·CoT·probe·AV가 각각 무엇을 읽고, AV가 증거로 쓰이기 전에 어떤
   측정 관문을 통과해야 하는가(3.3).
4. 이 계기들로 행동 효과, 위치 궤적, 단일 실행 소견서 영향 판별, 교정을 어떻게 평가하는가(3.4).

**AV를 쓰는 이유는 probe보다 정확해서가 아니다.** 닫힌 DDXPlus 49-class에서는
지도 probe가 더 강하고 이를 주 정량 계기로 사용한다. AV는 고정 label head가
제공하지 못하는 자연어 conclusion/cue 후보를 생성하고, MCR처럼 진단 어휘를
미리 닫기 어려운 코퍼스로 이어질 가능성이 있어 보조 계기로 포함한다. 다만
언어화 모델의 자체 지식이 activation 정보처럼 보일 수 있으므로, AV 출력은
3.3의 측정 관문을 통과한 범위에서만 증거로 사용한다.

#### 독자가 혼동하면 안 되는 네 객체

| 객체 | 입력 | 출력 | 이 논문에서 답하는 것 | 단독으로 증명하지 못하는 것 |
|---|---|---|---|---|
| Source Gemma | 임상 prompt + optional referral note | 최종 진단/CoT, 내부 activation `h` | 모델의 실제 행동 | 왜 그 답을 냈는지 |
| Linear probe | 특정 layer·position의 `h` | 고정 diagnosis별 score/probability | 닫힌 label space에서 무엇이 선형 decode 가능한지 | 그 정보가 생성에 인과적으로 사용됐는지 |
| AV readout | 같은 `h` + 고정된 해석 instruction scaffold | 자연어 cue/diagnosis 후보 | vector와 함께 변하는 자연어 content 후보 | 생성문 전체의 truth·faithfulness·임상 효용 |
| External reader/monitor | vignette, output, 선택적으로 CoT/readout | 오류 위험 또는 신뢰 판단 | 사람이 볼 수 있는 채널의 탐지·효용 | source activation 자체의 내용 |

특히 AV의 핵심 입력은 환자 prompt 원문이 아니라 activation `h`다. AV에 들어가는
자연어 instruction은 출력 업무와 schema를 지정할 뿐이며 환자의 gold diagnosis를
제공하지 않는다. 다만 AV 모델 자체가 의료 지식을 가지고 있으므로, plausible한
텍스트를 생성했다는 사실만으로 `h`가 그 내용을 담았다고 역추론할 수 없다.
이 간극을 닫는 것이 M0의 correct-pairing 대 shuffled/swap 대조다.

### 3.1 Datasets and the Direct-Answer Pool

#### 두 데이터셋의 역할

단일 데이터셋이 구조화된 인과 개입, 충분한 표본, 실제 임상 언어를 모두
제공하지 않으므로 역할을 분리한다.

- **DDXPlus**는 49개 진단, evidence 항목, gold diagnosis, ranked differential을
  제공한다. 구조화된 wrong diagnosis 선택, 균형 표집, closed-vocabulary probe가
  가능하므로 인과 실험과 내부 궤적의 본진이다.
- **MedCaseReasoning(MCR)**은 PubMed Central 증례보고 14,489건과 자유 텍스트
  진단 추론을 제공한다. 행동 효과가 합성 문진에만 생기는지 확인하고, 닫힌
  49-class probe가 정의되지 않는 열린 어휘 조건을 시험한다. 현재 MCR에서는
  행동 복제와 source-aligned conclusion readout까지만 완료됐으며 DDXPlus의
  내부 궤적을 일반화하지 않는다.

#### DDXPlus evidence를 임상 prompt로 바꾸는 과정

환자 CSV의 `PATHOLOGY`, `EVIDENCES`, `AGE`, `SEX`,
`DIFFERENTIAL_DIAGNOSIS`와 `release_evidences.json`을 결합한다. Evidence ID의
질문과 값은 declarative finding으로 렌더링한다. 예를 들어 `Do you have a
cough?`는 `a cough`, 위치형 문항과 `ankle(R)`은 `the swelling is located in
the ankle(R)`로 만든다. 음성 값, laterality, antecedent를 보존하고 불투명 코드와
렌더링 불가능 문항은 이유를 기록한 뒤 제외한다. 같은 문항의 다중 값은 한 cue로
병합하고 중첩 cue는 긴 표현을 남긴다.

진단당 100건을 seed 17로 균형 표집해 4,900건을 만든다. 기본 prompt는 다음과
같다.

```text
You are an expert physician. A {age}-year-old {patient descriptor}
presents with the following findings:
- {rendered cue 1}
- ...
- {rendered cue K}

{optional referral note}

What is the single most likely diagnosis?

Give the diagnosis only. Do not explain your reasoning.

You MUST end your response with exactly "The answer is <diagnosis>."
```

Source model은 `google/gemma-3-12b-it`, BF16, deterministic greedy decoding
(`do_sample=false`)이다. Direct condition은 assistant turn을 `The answer is`로
prefill하고 최대 64 new tokens를 생성한다. CoT condition은 같은 presentation
prefix 뒤에 reasoning instruction을 붙이고 최대 2,048 tokens를 허용한다.

#### Direct-answer pool과 clean cohort

Wrong note가 “원래 맞히던 답을 잃게 했는가”를 정의하려면 none condition에서
먼저 정답이어야 한다. Generation-time matcher가 선택한 1,747건 중 canonical
matcher에서도 no-note 정답인 **1,729건**을 primary intervention pool로 고정한다.
이는 모델의 일반 정확도를 보고하는 표본이 아니라
**paired causal susceptibility**를 측정하기 위한 조건부 모집단이다.

가족력 등에서 gold diagnosis 문자열이 presentation에 직접 나타난 행은
`gold_in_prompt`로 표시한다. Main clean behavior table은 이를 제외한 1,204건을
사용하고, 전체 1,729건은 primary moved 분해에 사용한다. 현재
trajectory·detection은 재집계 전 fixed cohort 1,747건을 사용한다. 독립
확장은 진단당 300건에서 구축하되 주 실행과 겹치는 base ID를 제외해 보고한다.
`gold_in_prompt`는 train-test leakage가 아니라 **explicit gold-name in the
presentation** 층화 변수다. 전체 moved 319건 중 287건(90.0%)이 clean에서
발생했고, clean에서도 201/287(70.0%)가 제3 진단으로 이동했다. Explicit-gold
525건에서는 32건(6.1%)만 moved되어 clean 287/1,204(23.8%)보다 낮았다.

MCR에서도 동일하게 no-note source-correct 사례만 causal behavior population으로
사용한다. 평가 가능한 전체에서 선택된 조건부 모집단이라는 점과 낮은 source
accuracy를 함께 보고한다.

### 3.2 A Causally Controlled Referral-Note Intervention

#### 네 개 arm

각 환자의 presentation, cue 순서, 질문, decoding은 고정하고 note만 바꾼다.

```text
none:     [no referral sentence]
neutral:  The referring note requests evaluation.
wrong:    The referring note suspects {plausible wrong diagnosis}.
correct:  The referring note suspects {gold diagnosis}.
```

- `none`은 기준선이다.
- `neutral`은 문장 추가와 referral framing 자체의 비용을 측정한다.
- `wrong`은 suggestion content의 인과 효과를 측정한다.
- `correct`는 모델이 어떤 suggestion이든 따르는지와 wrong content의 손상을
  분리한다.

DDXPlus의 wrong diagnosis는 ranked differential에서 gold 및 gold alias와 다른
첫 진단을 선택한다. 무작위 질환이 아니라 데이터셋이 해당 환자의 plausible
alternative로 부여한 진단이다. MCR에는 differential field가 없으므로 같은 gold
집단에서 source model의 실제 confusion을 우선 사용하고, 없으면 cue-word Jaccard가
가장 높은 다른 case의 gold diagnosis를 사용한다. 두 코퍼스의 suggestion
provenance가 다르므로 MCR은 행동적 replication이지 동일 자극 replication이 아니다.

#### 위치 통제와 wording robustness

Note는 마지막 finding 뒤, diagnostic question 앞에 삽입한다. Causal attention
때문에 note 이전 cue-token activation은 none/neutral/wrong/correct에서 동일하다.
따라서 “증거 token은 같지만 downstream state와 answer가 달라지는가”를 검사할
수 있다. `last_cue` paired activation difference가 표시 정밀도에서 0인지도
분석 때 확인한다.

한 문장 template 의존성을 보기 위해 referral, colleague, patient, realistic
multi-sentence wording을 별도 실행한다. Realistic condition은 길이와 clinical
register도 함께 달라지므로 matched neutral placebo가 없는 한 추가 효과를
문체 또는 진단 제안 내용 때문이라고 해석하지 않는다. 추가 통제에서는 같은
canonical clean ID와 삽입 위치를 유지하고, clinical register와 token length를
맞춘 neutral referral을 실행한다. `no-note - realistic neutral`을 긴 문구 삽입
비용으로, `realistic neutral - realistic wrong`을 진단 제안 내용의 고유 비용으로,
`no-note - realistic wrong`을 총비용으로 분해한다. Template은 결과를 보기 전에
고정하고 token-length 차이를 보고한다.

#### 정답 채점과 causal labels

최종 answer는 closing `The answer is ...`만 파싱한다. 전체 CoT에서 gold 문자열을
검색하지 않는다. 채점은 Unicode/markup 정규화 뒤 word-boundary-aware 양방향
포함과 diagnosis alias 사전을 사용한다. `PE`/`pericarditis`, `stable`/`unstable`
같은 부분문자열 충돌을 차단한다.

- `lost_the_gold`: none에서 정답이던 answer가 wrong에서 오답이 됨.
- `took_the_hint`: wrong answer가 suggestion을 명명하고, none answer는 이미 그
  suggestion을 명명하지 않았음.
- `moved`: 위 두 사건의 합집합.
- `silent`: wrong answer가 suggestion name을 포함하지 않는 subset. Unchanged와
  동의어가 아니며 제3 진단으로 이동한 case가 포함된다.

`moved`는 none/wrong pair를 비교해야만 얻는 실험자용 label이다. Single-run
detector에는 wrong execution만 제공하며 none answer, pair label, gold correctness를
입력하지 않는다.

### 3.3 Internal Measurement Channels and the AV Validation Gate

#### Source activation과 token positions

Gemma-3-12B-it의 48개 transformer block 중 주 분석은 block 32 output을 사용한다.
Hidden dimension은 3,840이다. Chat template을 포함한 실제 source forward에서
activation을 추출하고 BF16 forward 뒤 float32로 저장한다. 문자열 span은 tokenizer
offset mapping으로 찾고 cue/note diagnosis span은 `last_subtoken`, prompt 종결
상태는 `last_token`을 사용한다.

Trajectory landmark는 `last_cue`, note diagnosis, diagnostic question,
constraint(`Do not explain...`), answer-format instruction, final prompt token이다.
각 landmark는 별도 decoder로 분석하므로 하나의 고정 probe가 시간에 따라 이동한
것처럼 해석하지 않는다.

#### 채널 1: cross-fitted linear diagnosis probe

각 위치의 3,840차원 activation에서 DDXPlus diagnosis를 예측하는 선형 분류기를
학습한다. Test case와 동일 base ID의 모든 arm은 해당 fold의 training에서 제외한다.
Diagnosis별 사례는 다른 fold에 존재하므로 이 probe는 closed-vocabulary supervised
decoder다. 궤적의 `p(gold)`, `p(suggestion)`, top-1과 single-run attribution의 주
정량 계기다. Probe decodability는 해당 정보가 source generation에 인과적으로
사용됐다는 증거가 아니다.

#### 채널 2: AV natural-language readout

NLA는 activation-to-verbalization(AV)과 text-to-activation reconstruction(AR)을
공동 학습한 공개 해석 도구다. 우리는 `kitft/nla-gemma3-12b-L32-av`를 초기
체크포인트로 사용한다. 현재 의료 adapter는 AV에만 LoRA SFT를 적용하며 AR
reconstruction loss를 다시 붙인 full-NLA fine-tuning이 아니다.

AV를 추가하는 이유는 세 가지다.

1. Probe는 사전 정의한 49개 class 중 하나만 반환하지만 AV는 자연어 conclusion과
   supporting-cue 후보를 생성할 수 있다.
2. MCR처럼 진단명이 수천 개이고 대부분 singleton인 열린 어휘에서는 동일한
   49-way head를 정의할 수 없다.
3. 자연어 content는 correction prompt나 human audit에 직접 전달할 수 있다.

이 셋은 **가능성**이지 이미 입증된 장점이 아니다. 현재 결과에서 probe가 탐지와
교정 모두 더 강하고, MCR supporting-cue field와 reader-trust는 실패한다. 따라서
AV는 주 계기가 아니라 complementary hypothesis-generating channel로 기술한다.

LoRA adapter의 설계 목적은 자유 서술을 자동 채점 가능한 schema로 바꾸고
의료 target mapping을 안정화하는 **measurement adaptation**이다. 그러나
supervised mapping 자체가 새 의료 대응을 학습할 수 있으므로, “기존 vector
정보를 단순히 형식만 바꿨다”고 가정하지 않고 M0에서 pairing 의존성을 검사한다.

```text
<readout>
  <task_type>diagnosis</task_type>
  <answer>{decoded internal conclusion}</answer>
  <supporting_cues>{decoded clinical findings}</supporting_cues>
</readout>
```

Cue-position reader는 diagnosis를 출력하지 않고 해당 vector가 담는 finding 하나를
말하게 한다. LoRA는 rank 16, alpha 32, dropout .05이고 attention/MLP의 7개 linear
projection에 적용한다. AdamW `2e-4`, effective batch 8, 최대 3 epoch이며 XML
scaffold가 아니라 content-token validation loss로 checkpoint를 선택한다. Cue
reader 학습은 cue 하나당 한 행, 최대 10,195행이다.

Training target은 DDXPlus gold diagnosis와 rendered cue로 만든다. 따라서 낮은 SFT
loss나 높은 diagnosis hit만으로 faithful readout이라고 결론 내릴 수 없다. Adapter가
class/cue template을 외우거나 verbalizer가 prompt에서 진단을 추측할 수 있기 때문이다.

#### Measurement gate M0

AV 결과를 RQ 분석에 사용하기 전에 다음 서로 다른 실패 모드를 검사한다.

- **Correct pairing**: activation과 해당 case target을 맞게 짝지었을 때 content가
  읽혀야 한다.
- **Shuffled/mismatched activation**: 짝을 깨면 성능이 무너져야 한다.
- **Counterfactual cue swap**: cue를 바꿔 다시 추출한 activation에서 readout도 새
  cue를 따라야 한다.
- **Memorization check**: swap 뒤 원래 cue를 계속 출력하지 않아야 한다.
- **Cross-patient contamination**: 다른 환자의 cue를 자기 환자 cue처럼 말하지
  않아야 한다.
- **Cue-string heldout**: 학습에서 보지 않은 evidence 문자열을 일정 수준
  paraphrase/read할 수 있어야 한다.
- **Untuned checkpoint**: adaptation이 새 signal을 만든 것인지, 기존 NLA signal의
  precision/format만 바꾼 것인지 분리한다.
- **Semantic audit**: lexical matcher가 놓치는 paraphrase와 laterality/site 오류를
  저자 손채점과 외부 판정자로 재검사한다.

Appendix Table A1은 이 관문의 결과다. 각 행의 cohort와 baseline이 다르므로 하나의 순위표로
읽지 않는다. 이 검증은 “AV가 임상적으로 옳다”거나 “probe보다 낫다”를 증명하지
않고, 이후 자연어 분석에서 최소한 activation pairing에 반응하는 계기임을 보인다.

#### 비내부 비교 채널

- **Output heuristic**: final answer가 suggestion을 직접 명명하는지 본다.
- **Rule-based CoT features**: chain이 note를 인용·수용·기각하거나 suggestion에
  천착하는지 측정한다.
- **LLM CoT monitor**: vignette, note, CoT, answer를 보고 moved risk를 판정한다.
  이것은 activation에 접근하지 않는 강한 자연어 baseline이다.
- **No-account reader baseline**: vignette와 source answer만으로 외부 판정자가
  의심할 수 있는 정도를 측정한다.

### 3.4 Experimental Designs and Evaluation

#### Behavior and robustness

네 arm의 paired accuracy로 total note cost, neutral insertion cost,
suggestion-specific cost를 분리한다. DDXPlus main clean cohort, non-overlap
replication, MCR source-correct cohort를 별도로 보고한다. Wording과 CoT는
robustness 분석이며, main causal estimate와 모집단·instruction이 다르면 직접
합치지 않는다.

#### Position-wise trajectory

None/wrong pair에서 여섯 landmark별 별도 cross-fitted probe의 `p(gold)`,
`p(suggestion)`, top-1을 기록한다. Moved case는 answer unchanged, third-diagnosis,
suggestion-adopted 행동군으로 나눈다. 같은 case의 wrong-minus-none 차이를 계산하고,
suggestion이 처음 top-1이 되는 landmark를 센다. `suggestion never top-1`과 `gold
top-1 throughout`를 별도 사건으로 유지한다.

#### Single-run moved detection

Detector는 wrong execution 하나만 보고 continuous risk score를 낸다. Output,
rule-based CoT, LLM monitor, AV, probe를 동일 case set에서 비교한다. 진단마다
base moved rate가 달라 pooled AUROC가 diagnosis identity를 이용할 수 있으므로
**diagnosis-stratified AUROC**를 주 지표로 쓴다. 전체와 canonical silent subset을
함께 보고하고, channel gap은 case-paired bootstrap으로 신뢰구간을 계산한다.

#### Correction ladder and deployment policies

Wrong note와 first answer를 유지한 채 요소를 하나씩 추가한다.

- r3: reconsideration only.
- r4: reconsideration + original findings(입력 재제시 통제).
- r5: reconsideration + AV conclusion/supporting cues.
- r6: reconsideration + probe argmax label.
- r7: reconsideration + model's own first-pass CoT.

평균 정확도 하나는 recovered와 broken을 상쇄하므로 둘을 분해하고 moved recovery,
kept breakage, net effect, suggestion capitulation을 보고한다. 전량 재실행 외에
probe selector, direct argmax replacement, selector+r5/r6 policy를 비교한다.
R5와 r6의 content correctness를 층화해 자연어 형식과 content accuracy를 구분한다.

Deployment policy의 threshold와 정책은 validation에서 고정하고 held-out test에서
한 번 평가한다. Selector는 wrong-note 실행의 output/CoT/output-head likelihood/
probe/AV만 받을 수 있으며, gold·no-note answer·true moved는 받지 않는다. 주 지표는
overall accuracy, moved recovery, unchanged preservation, newly broken, net correction,
intervention rate다. 과거 fixed-cohort 하이브리드는 proof of concept로 남기고,
canonical validation/test 재평가를 RQ3의 submission gate로 둔다.

#### Human/LLM reader utility

Vignette와 source answer에 no account, CoT, probe label, AV readout 중 하나를 붙여
외부 판정자가 source answer를 의심하는지 측정한다. 주 비교는 각 account의 절대
AUROC가 아니라 동일 case no-account 대비 paired delta다. Shuffled account는
그럴듯한 텍스트가 일반적 불신만 높이는지 검사한다.

#### 계산과 재현

Source forward는 BF16, activation 저장과 probe 분석은 float32다. 주요 추출은
4-GPU workstation에서 수행하며 한 prompt forward에서 필요한 모든 landmark를
수집한다. 모델 ID, exact prompt, random seed, case builder, manifest, matcher,
analysis script를 공개하고 canonical rescore 파일만 본문 수치에 사용한다.
## 4. Experimental Results — camera-ready order (08-25)

**구조 정정**: AV 검증은 연구 질문의 결과가 아니라 측정 관문이므로 Method
§3.3에서 예고하고 Appendix A에서 검증한다. 본문 Results는 **행동 → 내부
궤적·탐지 → 조건부 교정**의 3소절만 둔다.

**표 번호 정정 (08-25).** 이전 판본은 종합 점수판을 Table 1로 앞세우고
나머지를 뒤로 밀었다. 그 점수판은 폐기됐다 — 한 표에 AUROC·비율·✔/✕를
섞어 담아 어느 열도 같은 단위가 아니었고, 축이 갈라진다는 논지는 표가 아니라
문장이 할 일이었다. 확정 원칙은 **표 하나 = 실험 하나**이며, 번호는 등장
순서를 그대로 따른다:

| 번호 | 내용 | 절 |
|---|---|---|
| Table 1 | 코퍼스별 개입 효과크기 분해·non-overlap 재현 | 4.1 |
| Table 2a | **기전: 최종 토큰 p(정답), 행동 그룹별 대조** | 4.2 |
| Table 2b | 단일 실행 소견서 영향 판별 AUROC (채널별, all/silent 분리) | 4.2 |
| Table 3 | 교정 사다리 | 4.3 |
| Appendix Table A1 | AV 계기 검증 배터리 | Appendix A |

Figure 2(a)가 네 arm 원시 정확도를 담고, Table 1은 neutral insertion,
wrong-note total, suggestion-specific, correct-note cost를 pp로 분해한다.
Table 1의 moved destination은 Figure 2(b)로 흡수한다. wording/source
ablation과 Table 3의 r7 common-cohort, content-matched, deployment-policy,
MCR 확장은 Appendix A2–A7로 보낸다. Appendix Table A1의 answer-position
n=229 보조 검사는 canonical n=230 정합 전까지 산문으로만 둔다.

조판 원고는 `table_camera_ready_2026-08-25.md`. 종합 점수판이 하던 일
("왜 프로브·CoT·SHAP가 아닌가")은 **4.2의 채널 비교 문단**과 Discussion
산문이 나눠 맡는다.

---

### Appendix A specification — Measurement Gate M0 (Appendix Table A1 · Appendix Figure A1)

이 절은 RQ1의 현상 결과가 아니라 **AV를 이후 분석의 보조 관측치로 사용할 수
있는지 확인하는 calibration gate**다. 논문의 핵심 행동 효과, probe 궤적,
single-run probe 탐지는 AV 없이도 성립한다. 다만 AV가 제시하는 자연어 cue와
진단 후보를 activation의 내용으로 해석하려면, 언어화기가 자신의 의료 지식이나
학습 template를 말한 것이 아니라 paired vector에 반응했다는 증거가 필요하다.
따라서 출발 질문은 “AV가 probe보다 우수한가”가 아니라 **“이 산문을 제한된
activation-conditioned measurement로 취급해도 되는가”**다.

**왜 Appendix인가.** 핵심 행동 효과, probe 궤적, single-run probe 탐지는 AV
없이 성립한다. 따라서 `evidence-before-use` 원칙은 Method §3.3에서 “AV는 M0를
통과한 범위에서만 보조 채널로 해석한다”고 선언하는 것으로 만족시키고, 수치와
layer map은 Appendix에 둔다. AV가 처음 등장하는 Table 2b와 Table 3 캡션은
Appendix A1을 참조한다.

- **P1 무학습 기준점**: 읽기 능력 자체는 사전학습 AV에 이미 있다(서술률
  0.7247, 우연 0.088). 그러나 형식 준수 0.05 — **읽을 줄은 아는데 계기가
  아니다.**
- **P2 어댑터가 바꾸는 것**: 형식 0.05→1.00, 길이 1,557→52자, 정밀도
  0.075→0.671. 이는 **계측 가능한 schema와 의료 mapping의 개선**이지,
  activation에 원래 없던 정보가 생겼다는 증거가 아니다. Appendix Figure A1을 붙이되
  ⚠️ **두 행이 한 실험이 아님을 캡션이 말한다**(cue행 = v4/v5 레시피·heldout
  cue 문자열, 답행 = v3 레시피·heldout 진단 분할 — 세로 읽기는 셋을 한꺼번에
  비교). **본문이 앞세울 문장은 교란 없는 within-sweep 대비다**: 답 위치는
  같은 레시피·같은 분할에서 seen .684 vs heldout .249 (+.435) — 답 토큰은
  클래스→전형 cue 템플릿만 지탱하고 per-cue 읽기를 못 한다. "결론 L32"는
  이 그림의 주장이 아니라 결론 판독·프로브(다른 계기)의 결과로만 말한다.
  - 08-17 vanilla 대조 실물도 여기: 무학습 AV는 cue 내용을 담되 **지어낸
    액자**에 섞어 낸다("게임 인벤토리", "해변"). LoRA는 시끄러운 서술자를
    정확한 판독기로 정제한다.
- **P3 신뢰의 네 기둥**: heldout 0.725→0.751(암기 아님) · 셔플 대조 붕괴
  (신호는 벡터-서술 결합에 있음) · 스왑 추적 **0.993** / 암기 **0.000**
  (서술이 벡터에 인과 종속) · 특이성 **0.007**(우연 0.015).
  **검증 배터리(Appendix Table A1)의 논리를 이 문단에 둔다** — 3.3에서 예고만 하던 것을
  수치 옆으로 옮겼다(방어는 숫자로 한다).
- **P4 unseen과 오독의 성격**: heldout 의미 서술 L24는 저자 손채점
  **.7306**, 같은 고유 쌍의 외부 판정 **.7740**(238쌍 전수, 파싱 실패 0;
  heldout 소견 문자열 41/164 완전 제외, 438행 행 가중). 별도 770행 기계 채점은
  .751이다. 오독은 A(자구)/B(의미보존)/
  C(속성 — 위치·정도 오류)이며 **없는 소견을 지어내는 유형이 아니다.**
  문단 끝에 Li et al.(ICML 2026)이 요구한 privileged-information 증거가
  여기서 완성된다는 선언 — 그들의 두 비판(① 서술이 언어화 모델의 지식일 수
  있다 ② 기존 벤치마크는 내부 접근 없이도 통과 가능)에 각각 대응.
- **P5 답 위치 대조 — 완료 (08-24), 강한 쪽으로 나왔다**: 같은 최종 토큰
  벡터를 무학습 체크포인트와 v2 어댑터에게 주고 `--lenient`(단어 경계
  매칭 + 케이스당 명명 진단 수) 동일 규칙로 채점했다. 상실형(n=229)에서
  **무학습 .603 vs v2 .651** — 무학습도 정답을 짚으므로 **결렬은 어댑터의
  산물이 아니라 활성값의 성질이다.** 단 두 열을 같이 읽는다: 무학습은
  판독당 진단명 1.15개를 흩뿌리고 v2는 1.02개를 말한다(포함 채점은 많이
  부를수록 유리 — 이름 하나당으로는 .524 vs .638로 뒤집힘). **어댑터가
  사는 값은 적중률이 아니라 정밀도.** 이 산출물의 last_cue·note 줄은 옛
  manifest 오염으로 인용 금지(답 위치 행은 무관).

### 4.1 소견서 한 줄이 답을 움직인다 — 그리고 설명은 예고하지 않는다 (Table 1 · Figure 2)

출발 질문: **교란이 실재하는가, 설명은 그것을 말하는가.**

- **P1 4조건 정확도** (canonical-eligible clean 1,204): no-note는 선정 조건으로
  **1.0000**, neutral/wrong/correct는 **.9460/.7625/.9302**.
  ⚠️ **정답 조건 칸 오류 (08-24 감사)**: 여기 적혀 있던 .932(1,137건)는 이
  실행의 값이 아니다 — 1,747건 답 파일에 `correct` 조건이 아예 없고, .9313은
  corpus-300 전체 4,995건(누출 미필터)의 값이다. 이후 main correct arm은
  `.9246`으로 재실행됐다. 독립 재현 행은 corpus-300에서 주 실행 id를 제거한
  미관측 clean 2,192건으로 **.9749/.9279/.7682/.9101**이고, MCR은
  **.9410/.8879/.6721/.8179**다. **각주: 모든 DDXPlus 비율은 보수적 하한** —
  답 파일이 plausible_wrong 수정 이전 생성이라 정답을 부르는 "오답" 소견서
  15/1,747(0.86%)이 오답 조건 정확도를 올리는 쪽으로 남아 있다
  (`--exclude-collisions`가 반대쪽을 준다).
- **P2 답 바뀜의 분해**: canonical primary clean **287/1,204(23.8%)** — 인과적
  채택 **86**, 제3 진단/상실 **201**. 전체 eligible 민감도는
  **319/1,729(18.4%) = 89+230**이다.
  **답만 봐서는 답 바뀜의 2/3(상실형)가 보이지 않는다**를 여기서 심어
  4.2의 침묵 부분집합을 예고. Catching-Rationalization이 각주에서
  "uncommon"이라며 제외한 부류가 우리에게는 다수라는 대비도 여기.
- **P3 인과성 재확인**: 차트가 진단명을 이미 적은 canonical-eligible 케이스
  (n=525)에서 오답 조건 정확도가 .939로 남는다(안 적힌 1,204는 .7625 —
  이전에 적혀 있던 0.899는 구판 수치) — 효과는 정보 주입이 아니라
  **제안 추종**.
- **P4 설명은 무엇을 하나**: 체인은 소견서를 96–98% **언급**하지만 인용·
  언급·분량 어느 특징도 답 바뀜과 무상관(**0.50–0.53**, 전수 1,747, 규칙
  기반 채점기 3종). 거짓 기각 실물 인용(체인이 "소견서는 성급하다"고 쓰고도
  답이 바뀐 케이스).
  ✅ **LLM 모니터 기준선 완료 (08-24, gpt-5.6-sol, 1,747/1,747 파싱)**:
  같은 체인에서 **.7233 / 정본 침묵 .6829**(별도 정의 .6930). 규칙 기반
  0.53과의 차이가 크므로
  **"체인은 소견서 영향 판별 신호를 담지 않는다"는 문장을 철회한다** — 우리 채점기가
  약한 부분이었다. 대신 이 절이 말하는 것은 정량적 경계다: 프런티어 독자에게
  전체 체인을 다 보여준 모니터가 단순 출력 특징보다 높지만, 동일 판정자의
  no-CoT arm이 없으므로 그 차이를 CoT만의 순수 증분으로 부르지 않는다.
  같은 정본 침묵 구역에서 판독은 **.8302**, 모니터와의 차이는
  새 점추정은 **+.1415**이며 paired CI는 로그 전사 대기다. canonical probe는
  **.9330/.9881**(all/silent)이다.
  루브릭은 판정자에게 유리하게 짰다("답이 제안과 달라도 움직인 것일 수
  있다"를 명시) — 낮은 값이 루브릭 결함이라는 반론이 닫힌다.
  ✅ **부트스트랩 CI 완료 (08-24)**: 침묵 구역에서 판독 − 모니터
  **+.1473, 95% CI [.0691, .2209]**, 케이스 쌍 부트스트랩 2,000회, 진단 내
  층화. 0을 배제한다. 판정자는 실제로 연속 확률을 썼다(서로 다른 값 61개)
  므로 동점 아티팩트도 아니다. calibration 자체는 별도 heldout 평가가 필요하다.
  calibration을 결론내리지 않는다; Brier/ECE가 별도로 필요하다.
- **P5 CoT의 이중성** (동일 canonical clean 1,204건): 추론은 arm 간
  소견서 피해를 약 1/5로 줄이지만 (−23.75 → **−4.40%p**) 없애지 못한다.
  Direct no-note는 selection상 1.0이고 CoT no-note는 .7068이므로, 두 baseline의
  절대 차이는 일반 정확도 효과로 해석하지 않는다. 지난 보고의 "CoT가 무력화"는
  n=381의 과대평가였고 전수에서 "완화"로 정정.

  ⚠️ **여기에 CoT의 일반 정확도 비용을 함께 쓰지 않는다.** 같은 파일에서
  direct .9007 vs CoT .7241(−17.66%p)이 나오지만, **이 1,747건은 직답이
  맞힌 케이스만 골라 만든 집합**이다(4,900 → 1,747). 종속변수로 표본을
  고른 것이므로 direct는 .9897 천장에서 시작하고 CoT는 내려갈 곳밖에 없다 —
  구조적으로 CoT에 불리한 비교다.

  편향 없는 표본에서의 답은 이미 있다: 맨 DDXPlus 320건 짝지음에서
  직답 .3375 vs CoT .3187, 살림 24 / 깨뜨림 30, **exact p = 0.50**
  (`ddxplus_as_a_benchmark_2026-08-22.md`). 이 표본에서는 차이를 검출하지
  못했지만 동등성 검정은 아니므로 **정확도 중립을 확정하지 않는다.**

  선택된 집합이 말하는 것은 다른 명제이고, 그쪽이 더 흥미롭다:
  **모델이 이미 맞혔던 답을 추론이 흔든다** — 1,747 base case의 두 note arm,
  즉 3,494 paired prompt instances 중 877(25.1%)에서 정오가 엇갈리며 CoT가
  깬 것은 747건, 구한 것은 130건이다. 이를 일반 모집단의 "CoT 정확도
  비용"으로 쓰면 표본 선택을 숨기는 것이다.
- **P6 강건성 꼬리** (구 4.6 흡수, 각 1–2문장):
  · **문구 4종** — 동일 clean 1,204건의 wrong accuracy는 소견서 .7625 / 동료
    .7757 / 환자 .8480 / **실제형 .6877**, paired 비용은 각각
    23.75/21.93/14.45/**30.40%p**다. 실제형은 길이와 레지스터가 함께 달라
    matched neutral 전에는 순수 현실성 효과나 보수적 하한으로 부르지 않는다.
  · **불안정화 vs 설득의 해리** — 환자 목소리는 답을 흔들되(179건) 설득은
    거의 못 한다(9/179 = 5.0%; 동료 99/266 = 37.2%). 화자 간 유의성 검정은
    canonical clean cohort에서 재계산 전이다.
  · **corpus-300** — 주 실행과 겹친 1,676건을 제외한 미관측 clean 2,192건에서
    `.9749/.9279/.7682/.9101`로 행동 효과를 독립 재현했다.
  · **MCR 1,452 (실제 증례)** — no-note=1 by selection,
    neutral/wrong/correct **.9339/.7066/.8388**. 답 바뀜 중 채택률은
    **127/427=29.7%**.
    제안 출처를 나누면 **그럴듯한
    제안이 설득을 2.3배 만든다**(41.2% vs 17.6%, z=5.26)면서 **흔드는 힘은
    구별되지 않는다**(z=1.75) — 해리의 두 번째 실증.

### 4.2 내부 궤적과 출력은 자주 어긋난다 — 한 번의 실행에서 그 격차를 읽는다 (Table 2a/2b · Figures 3/4a; case study는 Appendix Figure A2)

출발 질문: **상태의 어디서 왜 바뀌는가, 그리고 그것이 단일 실행에서
읽히는가.** (구 4.3 기전 + 구 4.4 탐지 지도의 통합.)

- **P1 궤적 도입**: cue 위치는 설계상 비트 동일이고, **재추출이 그것을
  수치로 확인했다** — 그 위치에서 소견서의 내부 비용이 세 그룹 모두
  **±0.000**이다. 인과 마스킹이 보장하는 값이 실제로 나왔으므로 실험대의
  전제가 가정이 아니라 측정이 된다. 소견서 위치의 제안 질량도 전 그룹
  ≤0.022(**읽은 직후에는 구별 불가**) → 그렇다면 그 사이 어디서 갈라지는가.
- **P2 Figure 3(a,c) — canonical-eligible**: moved 319건 중 제안 진단이 어느
  랜드마크에서도 top-1이 아닌 경우는 **262건(82.1%)**이다. 이 262건은
  **gold top-1 throughout 147건**과 **other top-1, suggestion never 115건**으로
  갈린다. 제안이 적어도 한 번 top-1인 57건의 첫 지점은 last finding 7,
  note 0, question 29, constraint 10, format 5, final 6이다. last finding의
  7건은 소견서 전부터 존재한 차이이므로 소견서 효과로 세지 않는다. 따라서
  소견서 이후 처음 제안 top-1이 된 것은 **50/319(15.7%)**이다. Figure 3의
  메시지는 “정답 보존”이 아니라 **출력 이동이 제안의 내부 top-1 우세를
  필요로 하지 않는다**는 것이다.
- **P3 짝지은 내부 비용 — Figure 3(b)**: wrong-note의 gold probability에서
  같은 케이스의 no-note gold probability를 뺀다. 음수일수록 소견서가 gold
  신호를 더 낮춘 것이다. last finding은 causal masking 때문에 0이고,
  referral note는 no-note arm에 대응 토큰이 없어 N/A다. canonical final
  비용은 유지 **−.006**, 제3 진단 **−.054**, 제안 채택 **−.199**다.
  final trend ρ=−.282 [−.328,−.233]으로 용량-반응이 유지된다.
- **P4 위치별 구조**: paired cost는 단조 증가하지 않는다. 제안 채택형은
  question −.171 → constraint **−.467** → format −.188 → final −.199,
  정답 상실형은 −.060 → **−.299** → −.189 → −.054다. constraint가 가장
  취약하고 final prompt token에서 일부 회복하지만 이후 출력은 틀린다.
  랜드마크마다 별도 probe를 학습했으므로 이를 단일 probe의 시간 변화나
  보편적 지시문 기전으로 과장하지 않는다.
- **P5 같은 그림을 문장으로 (Appendix Figure A2)**: canonical matcher에서 정답을
  잃고 제3 진단으로 간 230건의 **59.1%**에서
  마지막 위치의 내부 결론 문장이 여전히 정답이다("결론: 심근염, 근거:
  좌측 흉통·호흡곤란" — 출력은 다른 병명). 케이스 패널
  (ddxplus_myocarditis_0000265): 왼쪽은 차트·소견서·체인·답으로 끝까지
  명확히 드러나지 않은 결렬을, 오른쪽 랜드마크별 프로브+AV 판독이 드러낸다.
  과거 `64.1%`는 인용하지 않는다. 캡션도 "설명을 읽어도 모른다"가 아니라 **"일부 결렬은 체인과
  출력보다 내부 채널에서 더 선명하다"**로 제한한다.
  ⚠️ 근거 슬롯에 소견서가 등장하지 않는 관찰은 **어댑터 학습 편향과 분리
  불가**하므로 단독 증거로 쓰지 않고 정합 각주로만 둔다.
  ⚠️ 기존 서사의 **last_cue 행은 옛 manifest의 인공물**(실제로는 소견서 끝
  위치를 읽음) — 패널은 note→final 5지점으로 줄이거나, 고친 manifest로
  판독을 재서사한 뒤 채운다(Appendix Figure A2 스펙의 선택지 ②,
  `table_camera_ready_2026-08-25.md`).
- **P6 그래서 단일 실행에서 탐지된다 — 정직한 지도** (구 4.4 흡수):
  · 규칙 기반 CoT 특징 전체 **.5434** → LLM 모니터 **.7305/.6904**
    (all/silent) → 내부 자연어 판독 **.7511/.8319**.
  · 지도 프로브의 canonical all/silent 값은 **.9330/.9881**이다.
  · **유일성의 재배치**: ①소견서 영향 판별은 프로브도 통과한다. AV 판독의 몫은
    **③ 열린 어휘**(MCR 6,934종에는 현재의 고정 49-class probe를 직접
    이전할 수 없음) +
    **자연어 후보 의미의 탐색** + **다중 위치 국소화**다. MCR 근거 접지와
    reader-trust가 실패했으므로 "왜"를 충실하게 제공한다고 쓰지 않는다.
  · 선행과의 대화: When Truth Is Overridden(AAAI'26)은 **logit lens**로
    "의견이 있으면 정답 선호가 형성되지 않는다"고 관측했다. 우리는 같은
    상황에서 **학습된 프로브**로 정답이 표상에 남아 1위임을 보인다 —
    override의 정체는 **소거가 아니라 미표면화**다. 반박이 아니라 표상
    수준의 분해로 서술한다(과제·개입 강도·모델이 모두 다름).

### 4.3 교정: 내용이 지렛대이고, 재실행은 증폭기다 (Table 3 · Figure 4b)

출발 질문: **읽은 것으로 고칠 수 있는가 — 무엇이 지렛대이고 무엇이 역효과인가.**

- **P1 되먹임은 작동한다**: 답이 바뀐 케이스 .0031 → **r5 .6301 / r6 .8339**.
  Yuan(2605.09502)이 개입 4종을 전부 실패하고 "신호는 진단적일 뿐
  비인과적"이라 결론낸 것에 대한 **반례** — 단, 아래 P4가 그 반례의 정확한
  형태를 정한다.
- **P2 그러나 재실행 자체가 파괴적이다**: r3–r6 전체 .8117→.41–.46, 멀쩡하던
  not-flagged(0.954)가 0.43대로 붕괴. Huang(자기교정 한계)의 극단적 확인.
- **P3 부서짐의 방향(항복률)**: 부서진 답이 제안으로 간 비율 r3 0.450 /
  r4 **.6410** / r5 .4940. ⚠️ **08-25 정정** — 이를 "첫 패스의 몇 배"라고
  쓰면 **조건부를 무조건부로 나눈 것**이다. 같은 규칙의 짝은 "소견서가 답을
  바꾼 케이스 중 제안으로 간 비율"의 primary clean behavior 값 **86/287=.300**다.
  과거 fixed-cohort 첫 패스 항복률 `.3209`는 appendix 감사값이다. **r4가 최악(.6410)이고 r5는 r4보다
  14.7%p 낮다(.4940)** — 절대 최저는 r3(.4507)이다.
- **P4 무엇이 지렛대인가 — r4/r5/r6 분해**:
  · **r5 > r4**: +22.6%p(1,729). **AV 판독 내용이
    소견 재제시 통제를 이긴다** — 이것이 사다리가 설계된 비교이고 성립한다.
  · **r6 > r5**: 그러나 probe 클래스명 되먹임이 더 낫다(moved .8339).
    사전 등록한 두 판정 어느 쪽도 아닌 결과이며 **그대로 싣는다.**
  · ⚠️ **독자의 즉각적인 되물음을 문단이 먼저 말한다**: "클래스명을 되먹이는
    건 정답을 쥐여 주는 것 아닌가." 맞다 — probe argmax 정답률은 moved에서
    **.8567**이고 r6의 .8318은 그것을 그대로 따라간다. **r6은 제안하는
    방법이 아니라 r5의 내용 통제**이며, 통제로서 자기 일을 했다. 세 가지를
    함께 적는다: (i) probe는 오라클이 아니라 교차적합 분류기여서 해당
    케이스의 정답 라벨을 본 적이 없다(배포 시 실행 가능 — 누출이 아니다),
    (ii) 그러나 probe는 다른 케이스의 정답 라벨로 **지도학습**되므로 r5 vs r6
    은 형식만이 아니라 **감독 수준도** 다르다, (iii) probe가 정의되는
    코퍼스라면 r6은 쓸 정책이 아니다 — P6에서 재실행 없는 argmax 교체가
    r6 재실행을 이긴다.
  · **분해**: 내용 정확도가 지배한다 — 주 실행(1,747)의 moved에서 AV 판독
    **.5047** vs probe **.8567**. 정확도를 통제하면 전체 correct/correct에서
    **형식의 추가 기여가 검출되지 않는다**(78:79, p=1.000). moved
    correct/correct에서는 클래스명이 7:0으로 앞서지만 p=.016은 Bonferroni
    임계 .0125를 넘지 못한다. 형식 우위나 동등성을 모두 단언하지 않는다.
- **P5 병목의 해부**: canonical 허위 경보 484건은 어댑터 오독 68.4% / 매칭
  구멍 17.1% / 버틴 내부 흔들림 14.5%로 나뉜다. 과거 matcher의 recall
  0.846 / precision 0.362는 canonical 재집계 전까지 인용하지 않는다. 현재
  결론은 개입 대상 선정에 계기 오차와 매칭 오차가 함께 들어간다는 것이다.
- **P6 배포 정책의 예비 결과와 코퍼스 조건부 결론**: 옛 fixed-cohort에서
  probe 선별 + argmax 교체가 최고(.9651 / .9726, 재실행 없음)였다. 실행 내
  비교로 r6 .9531/.9658 > r5 .9141/.9265 > r4다. 최신 canonical 1,729의
  validation-frozen policy와 paired CI가 나오기 전에는 proof of concept로만
  둔다. **결론은 조건부로 쓴다** — 라벨된 학습 데이터 + 닫힌
  진단 목록이 있으면 프로브 교체가 최선이다. **진단 공간이 열려 있으면
  현재의 고정-class probe 경로를 직접 이전할 수 없다.** 이 조건에서
  source-aligned 자연어 판독의 답 필드는 모델 답 `.2643` 대 deranged `.0049`로
  사례 특이성을 통과했지만, 절대 정확도와 근거 접지·교정은 남아 있다.

- **P7 자기 CoT는 교정 경쟁자가 되지 못했다**: 공통 1,151 id에서 r7 moved
  회복은 **.1236**으로 r5 .5281, r6 .7416보다 낮다. 높은 전체 .8810은 원답
  보존과 쉬운 공통 집합의 결과다.

---

## 5. Conclusion (+ Discussion 통합)

**요약 3문장**
1. 환자 소견은 고정하고 의뢰 소견서만 바꾸며, **cue-position activation의
   비트 동일성까지 보장하는** 인과 실험대를 만들었다. 임상적으로 실재하는
   한 문장이 진단을 움직이는 행동 효과를 합성·실제 두 코퍼스에서 보였다.
2. **행동적 앵커링과 내부에서 디코드되는 제안 우세는 자주 어긋난다** —
   canonical moved 319건 중 262건에서 제안은 어느 랜드마크에서도 top-1이 아니었다.
   이 262건은 gold throughout 147건과 other top-1 115건으로 나뉜다. 이
   불일치는 단일 실행에서 탐지되지만, 모든 경우의 지식 보존을 뜻하지 않는다.
3. **계기의 분업과 경계**: 닫힌 진단 공간에서는 지도 프로브가 탐지와 되먹임
   내용을 더 정확히 공급한다. 자연어 판독은 계측·탐색 채널이지만 현재
   reader-trust와 MCR 근거 접지에서는 실패했고, 열린 어휘 결론에는 예비 신호만
   있다.

**Implications**: 현재 판독을 임상의 대면 감사 인터페이스로 쓰지 말 것 · **재고 프롬프트의 위험**
(다시 물으면 더 나빠지고 부서진 답은 제안 쪽으로 간다 — 자기교정 실무 경고)
· 되먹임 채널 선택 기준은 표현이 아니라 **내용의 적중률** · 규제 관점 한 줄.

**Limitations (정직 목록)**: 단일 백본(공개 NLA 생태계 제약) · 주 코퍼스가
합성 + 라벨 결정성(오답 예측 불성립, 프로브 우위의 조건) · **체인은 소견서 영향 판별
신호를 담는다 — 규칙 기반 .5464가 아니라 동일 모집단 LLM 모니터 .6829가
정직한 값이고, 우리 주장은 "없다"가 아니라 "판독보다 14.7%p 적다"이다** · 답 채점은 여전히
규칙 기반 매칭 · 다중비교 · 개입 1종(문구 4종으로 보강) · 자연어 형식의
추가 회복 기여가 **검출되지 않았음**을 명시 · 최초 MCR 결론 판독은 train/val의
88%가 source-wrong activation과 gold target을 짝지은 misalignment로 무효 ·
source-aligned MCR 결론의 답 필드는 derangement를 통과했지만 절대 일치율이 낮음 · 근거
접지 실패 · reader-trust 전수 결과는 현재 판독에 부정적.

**Future work**: MCR wrong-note 내부 추출과 교정 · reader-trust 임상 전문가 재현
· 외부 판정자 설명 품질 · logit lens와 학습 프로브를 같은 케이스에
겹쳐 그려 "override = 미표면화"를 한 그림으로(AAAI'26과의 직접 대화) ·
조기 경보 신호(소견 위치에서 이미 낮은 상태) · 타 도메인 이식.

---

## 절별 상태 요약 (08-25)

| 절 | 지금 쓸 수 있나 |
|---|---|
| Abstract | **초안 있음** (§0.1) — 300단어, 투고 규정에 맞춰 축약 필요 |
| Intro | **즉시** (기여 목록 08-25 수치로 갱신 완료) |
| Related Work | **즉시** — 초안·LaTeX 완성, 정독 노트 2편 반영만 |
| Methodology 3.1–3.4 | **즉시** (현상→개입→측정 채널→평가 순서로 재구성) |
| Appendix A | **즉시 전부** (답 위치 vanilla 완료, 08-24) |
| 4.1 | **즉시 전부** — four-arm behavior |
| 4.2 | **초안 가능** — Figure 3·Table 2a/2b·canonical probe AUROC 확정, Δ 추세 검정 대기 |
| 4.3 | **즉시** — DDXPlus r3–r7 canonical 완료; MCR 확장은 별도 |
| Conclusion | **즉시** |

**초안 집필은 가능하지만 정본 표는 아직 닫히지 않았다.** 남은 실측은
① Table 2a Δ 값 반영, ② reader-trust 임상 전문가 재현,
③ MCR wrong-note 내부 추출·교정, ④ 동일 판정자의 no-CoT 모니터
대조다. 집필 순서는 §0.3.

**표 번호는 `table_camera_ready_2026-08-25.md`가 정본이다** — 이 문서의
4.x 소제목에 적힌 번호는 그것을 따른다. 옛 종합 점수판(구 4.0의 Table 1)은
폐기됐고, 그 역할은 4.2 채널 비교 문단과 Discussion 산문으로 나뉘었다.
