# 논문 표 구성 v3 — 데이터 정의(D1-a~f, D2~D4) 반영

v2 대비 변경: ① 표마다 **사용 데이터 가공물 ID**를 명시 ② CoT 지표를
인용-추출에서 **개입-언급(Turpin식)/CoT-절단(Lanham식)** 으로 교체 ③
데이터셋별 측정 가능성 차이(span 주석 유무)를 표0으로 명문화 ④ 표3의
counterfactual을 1,000쌍 확장 전제로.

표기: **[완]** 확보 / **[실행 ID]** 필요 / — 해당 없음 / **N/A(구조)** =
그 방법으로는 원리적으로 측정 불가.

---

## 표0 — 데이터셋과 측정 가능성 (Method 섹션 첫 표)

**메시지**: 각 데이터셋에서 무엇을 잴 수 있는지가 gold 증거 span의 유무로
결정된다. 증거 채점은 span이 있는 데이터에서, 성능 효과는 자연 분포에서.

| 데이터 | 규모/구성 | 증거 span | 진단 공간 | construction-exact 개입 | 이 논문에서의 역할 |
|---|---|---|---|---|---|
| **D1 DDXPlus** | 케이스 프롬프트를 cue로 조립 (D1-a~f) | **조립식 → 정의상 확정** | 닫힘(26) | **가능** | 판독 검증·궤적·감사·개입 전부 |
| D2 MedQA-USMLE | 객관식 QA | 없음(문장 근사) | 열림/5지선다 | 불가 | 정확도 델타만 (이식) |
| D3 PHEE (약물) | ~5k 문장, drug/ADE **사람 주석 span** | **있음** | 판별형 | span 치환으로 가능 | 방법 일반화 (축소 프로토콜) |
| D4 DiReCT | MIMIC-IV 511 노트, 의사 주석 관찰→진단 | **있음** | 열림 | 불가(자연문) | 조건부(PhysioNet 승인) |

---

## 표1 (MAIN) — 방법 비교 매트릭스

**데이터**: D1-a(평가 풀 1058, base rate 74.2%), D1-b(heldout 438),
D1-c(개입 쌍), D1-f(감사 테이블).
**메시지**: 내부 상태에 대해 묻는 세 질문을 모든 방법에 동일하게 물으면 전
열을 채우는 방법은 하나뿐이다.

| 방법 | 접근 | ① 오답 예측 AUROC (순환성) | ② unseen 임상내용 판독 | ③ 개입 추적 / 암기 |
|---|---|---|---|---|
| answer confidence | output | 0.67–0.70 **[완]** (없음) | N/A(구조) | — |
| 26-way likelihood margin/entropy | output | **[L]** (없음) | N/A(구조) | — |
| CoT self-consistency | output | **[C1]** (없음) | — | — |
| CoT 자기설명 | output(자연어) | — | 생성은 하나 검증 필요 | **[1-A′]** 개입-언급률 |
| linear probe | 내부·닫힌라벨 | **[E1]** (준순환; 152행 예비 1.000) | **N/A(구조)** — 26지선다 출력기 | N/A(구조) |
| **logit lens** | 내부·열린토큰 | — | **[LL]** top-k 토큰 (단어 파편만) | — |
| vanilla AV (=training-free verbalization) | 내부·자연어 | — | strict 0.14 / soft 0.58 **[완]** | — |
| SAE | 내부·feature | N/A — Gemma-3-12b 공개 SAE 부재 | N/A(구조) 원자 feature, 조합 불가 | N/A |
| **Medical-NLA (ours, cue L24)** | 내부·자연어 | 불일치 0.9427 **[완]**(준순환) / **증거 인코딩률 [E2] (순환 없음)** | **A+B 73.1% [완]** | **0.993 / 0.000 [완]** |

캡션: (i) 순환성 주석 — gold와 정보가 겹치는 신호의 높은 AUROC는 예측이
아니라 채점에 가깝다. (ii) N/A(구조) 셀은 미실시가 아니라 그 방법의 출력
공간에 해당 답을 담을 칸이 없음을 뜻한다.

---

## 표2 — 설계 ablation + 내부도구 표현력 사다리

**데이터**: D1-a(v1), D1-b(v3/v4·layer·vanilla).

(A) 설계 요소별 기여 **[완]**

| 설계 | 타깃 | 추출 위치 | seen | heldout | 판정 |
|---|---|---|---:|---:|---|
| v1 naive | 진단명 | format L32 | 90.4% | **0/800** | seen-class 분류기로 붕괴 |
| v3 | cue 목록 | format L32 | 0.63 | 0.19 | 위치에 증거 없음 |
| v4 | 단일 cue | cue span L32 | 0.99 | A+B 55.7% | 판독 성립 |
| **v4 (ours)** | 단일 cue | cue span **L24** | 0.98 | **A+B 73.1%** | operating point |

(B) 표현력 사다리 — 같은 heldout 벡터(D1-b) 438개에 네 도구

| 도구 | 출력 공간 | unseen cue 결과 | 한계 |
|---|---|---|---|
| linear probe | 닫힌 라벨(26) | N/A(구조) | 물을 수 있는 것만 답함 |
| logit lens | 열린 토큰 | **[LL]** | 단어 파편, 조합 문장 불가 |
| vanilla AV | 열린 문장 | 0.14/0.58 **[완]** | 지어낸 액자에 내용 혼입(신뢰 불가) |
| **ours** | 열린 문장 | **73.1%** | 속성 해상도(C 26.0%) |

(C) layer × 위치 (Fig2의 표 버전) **[완]**: cue위치 A+B 34.0/73.1/55.7 vs
format위치 recall 0.188/0.249/0.188 (L16/24/32), format probe acc 99.2%.

---

## 표3 — 개입 기반 충실성: CoT와 판독을 같은 잣대로

**데이터**: D1-c(현행 150 swap쌍+300 retained → **1,000쌍으로 확장 권고
[CF+]**), D1-e(같은 프롬프트의 CoT 생성물).
**메시지**: 동일한 개입 아래 CoT는 원인을 말하지 않고, 판독은 벡터를 따라간다.

| 검사 (프로토콜 출처) | CoT 설명 | Medical-NLA 판독 |
|---|---:|---:|
| 개입 반영: swap된 cue를 반영/언급 (Turpin식) | **[1-A′]** | **0.993** [완] |
| 원본 잔존 = 문맥·암기 | — | **0/150** [완] |
| phantom (제거 cue 재출현) | — | ~0.003 [완] |
| retained 안정성 orig/swap/removed | — | 0.973/0.967/0.967 [완] |
| CoT 절단·오염 후 답 불변율 (Lanham식) | **[1-A′]** | 해당 없음 |
| 힌트 주입 시 미인정률 (Turpin·의료판) | **[1-B′]** | 힌트 span 판독으로 가시화 **[1-B′]** |

캡션: probe/SAE는 자연어 내용을 출력하지 않으므로 이 시험 자체가 정의되지
않는다(N/A 아님, 시험 불가).

---

## 표4 (제2 MAIN) — 오류 해부학과 해결

**데이터**: (A)(B) D1-f 감사 테이블(오답 785+정답 273), (C) **D1-d 자연 분포
split**(오답 농축 풀 사용 금지), 확장 열 D2.

(A) 오류 유형 분포 **[E2]**

| 유형 | 조작적 정의 | 비율 |
|---|---|---:|
| 인지 실패 | 핵심 gold cue의 E = 부재/왜곡 | TBD |
| distractor 과가중 | E 완비 ∧ S가 distractor 지향 | TBD |
| 통합 실패 | E 완비 ∧ S 오답 지향(distractor 아님) | TBD |
| decoding/drift | E 완비 ∧ S2=gold ∧ O≠gold | TBD |

(B) 유형 × 개입 flip rate **[E3]** — 대각 우세 + 인지실패 음성 대조

| | 증거 재주입 | verifier 재생성 | generic 재고 |
|---|---:|---:|---:|
| 인지 실패 | TBD (예측 낮음) | TBD | TBD |
| 통합/distractor | TBD (예측 높음) | TBD | TBD |
| decoding | TBD | TBD (예측 높음) | TBD |

(C) 해결 대결 — D1-d(+D2 확장 열)

| 방법 | 가중치 | 내부 | DDXPlus acc(Δ) | MedQA acc(Δ) | 원인 감사 |
|---|:--:|:--:|---:|---:|:--:|
| Base | ✕ | ✕ | **[B1]** | **[D2]** | — |
| generic CoT | ✕ | ✕ | [B1] | [D2] | ✕ |
| self-refine | ✕ | ✕ | [B1] | [D2] | ✕ |
| MedPrompt류 | ✕ | ✕ | [B1] | [D2] | ✕ |
| 출력-만 오답노트 (핵심 ablation) | ✕ | ✕ | **[E4]** | [D2] | △ |
| **ours: 내부-감사 오답노트** | ✕ | ○ | **[E4]** | [D2] | ○ |
| **ours: 유형별 표적 개입** | ✕ | ○ | **[E3]** | — | ○ |
| (참조 상한) gold LoRA SFT | ○ | ✕ | [B1-9월] | — | ✕ |

---

## 표5 (신규, 9월) — 방법 일반화: 약물 도메인

**데이터**: D3 PHEE(사람 주석 span). 축소 프로토콜 = 판독 + span 치환
counterfactual만(해부학·개입은 D1 전용).

| 항목 | DDXPlus(증상 cue) | PHEE(약물/ADE span) |
|---|---:|---:|
| unseen span 판독 A+B | 73.1% [완] | **[D3]** |
| 개입(치환) 추적 / 잔존 | 0.993 / 0.000 [완] | **[D3]** |
| 판독기 조건 | layer별 LoRA | zero-shot → 필요 시 light LoRA |

캡션: cue가 증상이 아니어도 동일 설계가 성립하는지의 시험. D4(DiReCT)는
승인 시 동일 표에 행 추가, 아니면 future work.

---

## Figure (표 변경에 맞춘 정리)

- **Fig1 (MAIN)** overview: 문제(CoT 개입 무반응·probe 닫힌 라벨) → 방법
  (cue-span 추출→검증된 판독→감사 E/S/O) → 효과(73%/암기0/acc Δ)
- **Fig2** 궤적: cue위치 역U자 vs format위치 평평 (표2-C의 그림) [완]
- **Fig3** counterfactual 도해 + 실물(GERD 케이스에 월경 cue 스왑) + CoT
  대비 막대 (표3의 그림)
- **Fig4** 해결: 방법별 acc 막대(표4-C) + 유형×개입 히트맵(표4-B)

## 실행 ID 색인

[L] 26-way likelihood · [C1] CoT self-consistency · [E1] probe 재학습+감사
테이블 · [E2] 해부학+증거기반 예측 · [E3] 표적 개입 · [E4] 오답노트 ·
[B1] 자연 split+베이스라인 · [1-A′] CoT 개입-언급/절단 · [1-B′] 힌트 주입 ·
[LL] logit lens · [CF+] counterfactual 1,000쌍 확장 · [D2] MedQA 이식 ·
[D3] PHEE 축소 프로토콜 · [J] LLM-judge 제2 평가(appendix)
