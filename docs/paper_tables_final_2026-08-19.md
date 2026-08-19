# 논문 표 설계 (확정안) — 2026-08-19

**대전제**: 믿을 수 있는 진단 설명은 모델이 말하는 것(CoT)이 아니라 검증된
방법으로 내부에서 읽어낸 것이며, 그렇게 읽은 내부 증거가 오류의 설명·예측·
교정을 더 잘 해낸다.

전제의 전반부(믿을 수 있는가) = Table 1, 후반부(더 잘 해내는가) = Table 2.
나머지는 두 표의 근거·상세.

표기: **굵게** = 확보된 값 · ▢ = 측정 예정 · ✕ = 그 방법의 출력 공간상 불가
(미실시 아님) · n.a. = 해당 과제에서 정의되지 않음 · — = 해당 없음

---

## Table 1 (MAIN) — Sources of diagnostic explanation

**메시지**: 진단 근거를 알아내는 모든 경로를 같은 잣대에 올리면, 충실하면서
동시에 케이스 고유 증거를 서술하는 것은 검증된 내부 판독뿐이다.

**행 = 설명의 출처, 열 = 설명의 속성 × 데이터셋(DDX/PHEE)**

| 설명의 출처 | 내부 | ① 오답 예측 AUROC (DDX/PHEE) | ② 충실성: 개입 반영 / 문맥 암기 (DDX/PHEE) | ③ unseen 임상증거 서술 (DDX/PHEE) | ④ 오류 원인 유형화 (DDX/PHEE) |
|---|:--:|---|---|---|---|
| Answer confidence | ✕ | **0.67–0.70** / ▢ | — | ✕ | ✕ |
| Output-distribution signal¹ | ✕ | ▢ / ▢ | — | ✕ | ✕ |
| CoT self-explanation | ✕ | ▢ / ▢ | ▢ / ▢ | ▢ (미검증 서술) | ✕ |
| Linear probe² | ○ | ▢ / n.a. | 시험 불가³ | ✕ (닫힌 라벨) | ✕ |
| Logit lens | ○ | — | 시험 불가³ | ✕ (토큰 파편) | ✕ |
| Training-free verbalization⁴ | ○ | — | — | **0.14 / 0.58** / ▢ | ✕ |
| **Ours: verified NLA readout** | ○ | ▢ / ▢ | **0.993 / 0.000** / ▢ | **73.1%** / ▢ | ▢ / ▢ |

¹ DDXPlus(닫힌 26지선다): 26-way likelihood margin·entropy / PHEE(열린 span
추출): 시퀀스 log-prob·entropy.
² probe는 닫힌 라벨 공간을 전제 → PHEE에서는 정의되지 않음(n.a.). **이 빈칸이
가설2의 시각적 증명**: 닫힌 도구는 열린 과제로 넘어가지 못하고, 자연어 판독은
넘어간다.
³ 자연어 내용을 출력하지 않으므로 개입-반영 시험 자체가 정의되지 않음.
⁴ 동일 AV 체크포인트에 어댑터 없이 같은 활성값 주입 = Patchscopes/SelfIE류
무학습 판독.

---

## Table 2 — Explanation-guided intervention (설명의 효용)

**메시지**: 믿을 수 있는 설명이 실제로 더 나은 결과를 만든다. 핵심 비교는
**출력-기반 노트 vs 내부-감사 노트**(내부 정보의 순수 기여).

**행 = 개입 방식, 데이터 = 자연 분포 split(+MedQA), 가중치 고정 조건**

| 개입 | 내부 접근 | 가중치 | DDXPlus acc (Δ) | MedQA acc (Δ) | 원인 설명 동반 |
|---|:--:|:--:|---|---|:--:|
| None (base) | ✕ | ✕ | ▢ (기준) | ▢ | — |
| Generic CoT prompting | ✕ | ✕ | ▢ | ▢ | ✕ |
| Self-refine | ✕ | ✕ | ▢ | ▢ | ✕ |
| MedPrompt-style | ✕ | ✕ | ▢ | ▢ | ✕ |
| **CoT-based error notes** | ✕ | ✕ | ▢ | ▢ | △ (미검증) |
| **Audit-based error notes (ours)** | ○ | ✕ | ▢ | ▢ | ○ |
| **Type-targeted intervention (ours)** | ○ | ✕ | ▢ | — | ○ |
| Gold LoRA SFT (참조 상한) | ✕ | ○ | ▢ | — | ✕ |

캡션: SFT는 경쟁 대상이 아니라 상한선(임상 배포에서 가중치 접근이 제한되는
조건을 서론에서 명시). 5행 vs 6행이 이 논문의 핵심 ablation.

---

## Table 3 — Design ablation (판독기가 왜 이렇게 설계되었나)

**메시지**: naive verbalization은 실패한다. 타깃과 위치를 바꿔야 판독이 선다.
Table 1 ③열의 근거.

| 설계 | 타깃 | 추출 위치 | seen | heldout | 판정 |
|---|---|---|---:|---:|---|
| v1 naive | 진단명 | format (마지막 토큰) | **90.4%** | **0/800** | seen-class 분류기로 붕괴 |
| v3 | cue 목록 | format | **0.63** | **0.19** | 위치에 증거 없음 |
| v4 | 단일 cue | cue span (L32) | **0.99** | **55.7%** | 판독 성립 |
| **v4 (ours)** | 단일 cue | cue span (**L24**) | **0.98** | **73.1%** | operating point |

(PHEE 대응 행은 9월 추가)

---

## Table 4 — Faithfulness gate (충실성 상세)

**메시지**: Table 1 ②열의 0.993/0.000이 어떤 검사들의 결과인지. CoT와 같은
개입 잣대 위에서.

| 검사 (프로토콜) | CoT self-explanation | Ours |
|---|---|---|
| 개입 반영: 교체된 cue를 반영/언급 (Turpin식) | ▢ | **0.993** |
| 원본 잔존 = 문맥 암기 | — | **0.000** |
| phantom: 제거된 cue 재출현 | — | **0.003** |
| retained 안정성 (orig/swap/removed) | — | **0.973/0.967/0.967** |
| CoT 절단·오염 후 답 불변율 (Lanham식) | ▢ | 해당 없음 |
| 힌트 주입 시 미인정률 (Turpin·의료판) | ▢ | 힌트 span 판독으로 가시화 ▢ |

---

## Table 5 — Error anatomy (오류 해부학)

**메시지**: Table 1 ④열의 실체. 오답이 내부 증거 상태로 유형화되고, 유형에
맞는 개입만 효과가 있다(= 유형 분류의 인과 검증).

(a) 유형 분포

| 유형 | 조작적 정의 (내부 신호) | 비율 |
|---|---|---:|
| 인지 실패 | 핵심 gold 증거가 판독 불가/왜곡 | ▢ |
| Distractor 과가중 | 증거 완비 + 내부 결론이 distractor 지향 | ▢ |
| 통합 실패 | 증거 완비 + 내부 결론 오답 (distractor 아님) | ▢ |
| Decoding 실패 | 증거 완비 + 내부 결론 정답 + 출력 오답 | ▢ |

(b) 유형 × 개입 flip rate — 대각선만 높아야 분류가 인과적으로 검증됨

| | 증거 재주입 | Verifier 재생성 | Generic 재고 |
|---|---:|---:|---:|
| 인지 실패 | ▢ (예측: 낮음) | ▢ | ▢ |
| 통합/distractor | ▢ (예측: 높음) | ▢ | ▢ |
| Decoding | ▢ | ▢ (예측: 높음) | ▢ |

---

## Table 6 — Clinical plausibility (별도 평가)

**메시지**: faithfulness와 plausibility는 다르다. CoT가 이 표에서 우리를
이길 수 있고, 그것이 논문의 명제다 — "그럴듯함"은 "충실함"의 증거가 아니다.

- 표본: 케이스 50–100, 평가자: 임상의(+공개 루브릭 LLM judge), 척도: 5점
- 행: CoT / training-free verbalization / ours · 열: 평균 점수, 평가자 일치도(κ)

---

## Figures

| # | 내용 | 왜 표가 아니라 그림인가 |
|---|---|---|
| Fig 1 | Overview: 문제(CoT 무반응·probe 닫힌 라벨) → 방법(cue-span 추출→검증된 판독→감사) → 효과 | 논문 전체 구조는 표로 못 담음 |
| Fig 2 | 궤적: cue 위치 A+B(34/73/56, 역U자) vs 답 위치(18.8/24.9/18.8, 평평) | **형태(shape)가 메시지** — 숫자 나열로는 안 보임 |
| Fig 3 | 판독 실물: 같은 gold cue의 layer별 4형태(원형/왜곡/반전/부재) + probe는 전부 동일 답 | "무엇이 어떤 형태로"를 독자가 이해하는 유일한 방법 |
| Fig 4 | 유형 × 개입 히트맵 (Table 5b의 그림 버전) | 대각선 패턴은 그림이 강함 — Table 5b와 택일 |

## Appendix

- A/B/C/D 채점 루브릭 전문 + 사람-judge 일치도(κ)
- 반암기 분석: gold-최근접 vs train-최근접 (layer 대조 포함)
- 데이터 가공 상세 (span 추출, construction-exact 검증, split 구성)
- 프롬프트 전문 (판독 프롬프트, CoT 지시문, 힌트 주입 템플릿, judge 루브릭)

---

## 완성 순서

| 시점 | 완성 목표 |
|---|---|
| 8월 말 초안 | **Table 1의 DDX 열 + Table 3 + Table 4의 ours 열** (대부분 확보 완료), Fig 2·3 |
| 9월 중순 | Table 1의 CoT 행·PHEE 열, Table 4의 CoT 열, Table 5, Table 6 |
| 9월 말 | Table 2 전체, Fig 1·4, appendix |
