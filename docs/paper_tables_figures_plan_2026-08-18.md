# 논문 표·Figure·정성분석 윤곽 (목요일 8/20 제출용 초안)

대상 저널: Artificial Intelligence in Medicine.
데이터셋: DDXPlus (모델: Gemma-3-12b-it + NLA L32 체크포인트).
표기: **[완료]** 수치 확보 / **[E#]** 해당 실험 실행 필요 / TBD 빈칸.

---

## 표1 — 오답 예측 신호 비교 (가설1-1, 가설2-1) — 8/25

**메시지**: output 신호는 내부가 아는 것을 회수하지 못하고(1-1), 내부를 읽는
어떤 도구든(probe 포함) 오답 탐지는 잘한다 — 탐지력은 verbalization 고유
기여가 아니다(2-1).

- 데이터셋: DDXPlus 평가 풀 1058행 (base rate 74.2% 명시 — error-enriched)
- 지표: AUROC, AP, n
- 비교: output 신호 vs 내부 신호, 순환성 여부 주석

| 신호 (source 오답 예측) | 접근 | n | AUROC | AP | 순환성 주석 |
|---|---|---:|---:|---:|---|
| source 답 confidence | output | 1058 | 0.67–0.70 **[완료]** | TBD | 없음 |
| (선택) 모델 자가보고 확신도 | output | TBD | TBD | TBD | 없음 |
| source ≠ linear probe | 내부(閉라벨) | 1058 | TBD **[E1: probe 재학습]** (152행 예비: 1.000) | TBD | probe가 gold로 학습됨 → 준순환 |
| source ≠ Medical-NLA 판독 | 내부(자연어) | 1058 | 0.9427 **[완료]** | 0.9708 | v2 타깃에 gold 포함 → 준순환 |
| 증거 인코딩률 (판독 기반) | 내부(자연어) | 785+273 | TBD **[E2]** | TBD | **없음 (답 비교 안 함)** |

캡션 포인트: 준순환 신호의 높은 AUROC는 "정답표 근사와의 대조"에 가까움(본문
Confound 분석 인용). 마지막 행이 순환 없는 진단 축의 본 결과.

---

## 표2 — Unseen 임상 내용의 자연어 판독 (가설2-2, 가설3-1) — 데이터 완비

**메시지**: 닫힌 라벨 도구는 케이스 고유 내용을 원리적으로 출력 못 하고(2-2),
naive verbalization은 분류기로 붕괴하며(3-1), 우리 설계는 unseen 문자열을
의미 수준으로 판독한다.

- 데이터셋: DDXPlus cue-string heldout (유니크 cue 164개 중 41개 학습 완전
  제외; heldout 438행 전수 수동 분류, 3 layer × 438)
- 지표: A(자구 일치) / B(패러프레이즈) / A+B(의미 판독) / C(속성 오류) / D(오독)
- 비교: v1(naive), probe, vanilla AV, Medical-NLA(L16/24/32)

| 판독기 | 위치 | seen | heldout A+B | heldout C | heldout D |
|---|---|---:|---:|---:|---:|
| v1: 진단-타깃 naive 튜닝 | format L32 | answer_hit 90.4% | **정답 0/800 (붕괴)** [완료] | — | — |
| linear probe | format L32 | acc 99.2% | **N/A — 閉라벨: unseen 내용 출력 불가(구조적)** | — | — |
| vanilla AV (무학습) | cue L24 | soft 0.58* | strict 0.142 / soft 0.582* [완료] | TBD(표본 수동라벨 옵션) | — |
| Medical-NLA (ours) | cue L16 | 0.97† | **34.0%** [완료] | 52.1% | 13.9% |
| Medical-NLA (ours) | cue L24 | 0.98† | **73.1%** [완료] | 26.0% | **0.9%** |
| Medical-NLA (ours) | cue L32 | 0.99† | **55.7%** [완료] | 35.8% | 8.4% |

\* vanilla는 형식 없는 다문단 출력이라 자동지표가 불공정(strict 과소/soft 과대)
— 캡션에 명시, 실물은 정성분석 2. † seen은 자동지표(soft), heldout만 수동.

---

## 표3 — 개입 기반 충실성 관문: 같은 잣대로 CoT vs 판독 (가설1-2', 가설3-2)

**메시지**: 동일한 counterfactual 개입(cue swap/removal) 아래에서 CoT 설명은
실제 원인과 어긋나고, 우리 판독은 벡터를 따라간다. probe/SAE는 이 시험 자체가
불가능(자연어 내용 출력이 없으므로) — 캡션에 명시.

- 데이터셋: DDXPlus 150 swap쌍 + 300 retained (construction-exact 재구성 검증)
- 지표(판독): 추적률, 원본 잔존(암기), phantom, retained 안정성
- 지표(CoT): 인용-원인 괴리율(인용 cue 제거해도 답 불변), 원인 미언급률
  (답 뒤집는 swap을 설명이 언급 안 함), 편향 힌트 미인정률(Turpin식)

| 검사 | CoT 설명 (Gemma) | Medical-NLA 판독 (ours) |
|---|---:|---:|
| 개입 추적 (swap 반영) | TBD **[실험 1-A]** | **0.993** (theme) / 0.707 full [완료] |
| 원본 잔존 = 문맥/암기 | TBD | **0/150** [완료] |
| phantom (제거 cue 재출현) | TBD | ~0.003 (수동 재검증) [완료] |
| retained 안정성 (orig/swap/removed) | — | 0.973 / 0.967 / 0.967 [완료] |
| 인용 cue 제거 시 답 불변율 (장식 설명) | TBD **[실험 1-A]** | 해당 없음 |
| 편향 힌트 미인정률 | TBD **[실험 1-B]** | 해당 없음 |

---

## 표4 — 오류 해부학과 표적 개입 (응용: 설명·진단·해결 축) — [E1~E4]

**메시지**: 감사 기록으로 오답을 4유형으로 분류하고(설명), 유형별 맞춤 개입만
효과가 있음을 보여(해결) 분류의 인과 타당성까지 검증.

- 데이터셋: DDXPlus 자연 발생 오답 785 + 정답 대조 273
- 지표: 유형 분포(%), 개입별 flip-to-correct율, held-out 정확도 델타

(A) 오류 유형 분포 **[E2]**

| 유형 | 정의 (내부 신호) | 비율 |
|---|---|---:|
| 인지 실패 | 핵심 gold cue가 cue-위치에서 판독 불가/부재 | TBD |
| distractor 과가중 | 증거 완비 + 내부결론이 distractor 쪽 | TBD |
| 통합 실패 | 증거 완비 + 내부결론 오답 (distractor 아님) | TBD |
| decoding 실패 | 내부결론 정답인데 출력 오답 | TBD |

(B) 유형 × 개입 flip rate **[E3]** — 대각선만 높아야 분류가 인과적으로 검증됨

| | 증거 재주입 | probe-verifier 재생성 | generic "재고하라" |
|---|---:|---:|---:|
| 인지 실패 | TBD (예측: 낮음 — 음성 대조) | TBD | TBD |
| 통합/distractor | TBD (예측: 높음) | TBD | TBD |
| decoding | TBD | TBD (예측: 높음) | TBD |

(C) 오답노트 시스템 프롬프트 **[E4]**: 내부-감사 노트 vs 출력-만 노트 vs 무처치
— held-out 정확도 (내부 정보의 기여 분리 ablation)

---

## Figures

- **Figure 1 (8/25, 데이터 완비)** — 궤적 그림: x=layer(16/24/32), y=heldout
  판독률. 곡선① cue-위치 A+B(34/73/56, 역U자), 곡선② format-위치
  recall(0.19/0.25/0.19, 평평), 주석: format의 probe acc 99%(클래스 신호는
  존재). 메시지: "증거는 cue 위치에 살고, 답 위치에서는 어느 깊이에서도 증거가
  아닌 결론만 남는다."
- **Figure 2** — 방법 개요: (a) 케이스 프롬프트에서 cue-span 활성값 추출 →
  동결 AV + layer별 LoRA에 주입 → 단일-cue 판독, (b) 감사 파이프라인:
  cue-위치 판독(E) + format probe(S) + 출력(O) → 4유형 분류 → 유형별 개입.
- **Figure 3** — counterfactual 설계 도해 + 실물 1쌍(GERD 케이스에 월경 cue
  스왑 → 판독이 월경을 읽음) + 요약 막대(추적/잔존/phantom).
- **Figure 4 (E2 후, 선택)** — 오류 해부학 분포 + 유형×개입 히트맵.

## 정성분석

1. **같은 cue, 세 layer** [완료]: "chest pain even at rest"가 L24에선 원문,
   L16/L32에선 "운동시 악화·안정시 완화" 서사로 반전 — 접힘의 실물.
2. **같은 벡터, vanilla vs LoRA** [완료]: 내용은 양쪽에 있으나 vanilla는
   지어낸 액자(스페인어 인용/게임 등)에 흩뿌림 — 증류의 실물.
3. **C형 속성 오류** [완료]: 위치 cue의 "iliac fossa" 끌개, black→light red —
   유일하게 남은 약점의 정밀 특정.
4. **오답 유형별 케이스 스터디** [E2 후]: 유형당 1건 — 감사 기록 원문 제시.
5. **CoT vs 판독, 같은 케이스** [실험 1-A 후]: 설명이 인용한 cue를 제거해도 답
   불변(장식) vs 판독은 개입 추적 — 표3의 실물.

## 일정 매핑

- ~8/20 (목): 이 문서 = 표 윤곽 + figure + 정성분석 제출물. Overleaf 골격 생성.
- ~8/25 (월): 표1 완성(probe 재학습 = E1 일부), Figure 1 완성(데이터 있음, 그리기만).
- ~8월 말 초안: 표2·3(판독 측) 완료 상태 + 표1 + Fig1-3 + 정성 1-3, 표3 CoT행
  (1-A/B)과 표4는 가능한 만큼; 미완이면 초안엔 설계+예비로.
- ~9월 말 최종: 표3 CoT행, 표4 (A)(B)(C) 완성.
