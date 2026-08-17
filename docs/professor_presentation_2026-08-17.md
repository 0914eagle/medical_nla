# Medical-NLA 진행 발표 자료 (2026-08-17)

전체 서사: 원래 목표 → 발견한 문제 → 문제를 어떻게 풀었나 → 지금까지의 결론 →
앞으로의 계획. 모든 수치와 판단 근거 포함.

---

## 0. 프로젝트 목표와 도구

**목표 (교수님 3축)**
1. 설명(Explanation): 모델이 맞거나 틀렸을 때 내부에서 무엇을 고려했는지
2. 진단(Diagnosis): 답이 틀릴지를 내부 신호로 예측
3. 교정(Solution): 그 내부 신호로 답을 고침

**핵심 도구 — NLA (Natural Language Autoencoder)**
- AV (Activation Verbalizer): 활성값 → 자연어
- AR (Activation Reconstructor): 자연어 → 활성값 (복원)
- 기존 기법 대비 차별점: probe는 "클래스/숫자", SAE는 "feature id"만 내지만,
  NLA는 **자연어 서술**을 낸다. 이게 NLA만의 잠재적 기여.

**세팅**
- Backbone: `google/gemma-3-12b-it` (48 layer, d_model 3840)
- NLA 체크포인트: `kitft/nla-gemma3-12b-L32-{av,ar}` (layer 32 기준)
- 데이터: DDXPlus (환자 증상 cue → 진단)

**세션 시작 시점의 상태**
- AV만 LoRA로 의료 튜닝한 "Medical-AV" 존재. 헤드라인 결과:
  - Linear probe로 layer-32 활성값에서 진단 분류 acc1 = **99.17%**
  - Medical-AV all-cue answer_hit = **86.96%**, cue_recall = **79.94%**
  - source 답 ≠ Medical-AV → source 오답률 **98.90%**, disagreement AUROC **0.9427**
  - source가 틀린 케이스에서도 Medical-AV는 **83.44%** 정답

겉보기엔 강력. 하지만 이 결과들이 **검증되지 않은 두 가지 가정** 위에 있었다.

---

## 1. 발견한 문제 — 헤드라인이 두 개의 confound 위에 있었다

**왜 의심했나.** 두 가지가 걸렸다.
- (A) probe가 이미 99%인데, "Medical-AV가 활성값을 읽는다"는 게 진짜일까,
  아니면 26개 진단 클래스를 외운 분류기일까? 둘은 in-distribution에서
  구분이 안 된다.
- (B) "불일치가 오류를 예측한다"는 게 NLA의 기여일까, 아니면 활성값을 읽는
  어떤 도구든 하는 일일까? 그리고 test셋이 오답 편향이라 순환적이지 않나?

→ 그래서 통제 실험 3개를 돌렸다.

### 실험 A — Probe 대조군 (오류 탐지가 NLA 기여인가?)

같은 활성값으로 linear probe를 만들어, "source ≠ probe" 불일치를
"source ≠ NLA" 불일치와 비교.

| 신호 | AUROC |
|---|---:|
| source ≠ NLA disagree | 0.9282 |
| source ≠ probe disagree | **1.0000** |
| (참고) source confidence baseline | 0.67–0.70 |

**결론: probe가 이겼다.** 원리적으로 당연 — probe가 99% 정확하면
"source ≠ probe"는 곧 "source ≠ 정답"이라 거의 완벽 탐지가 나온다.
→ **오류 탐지는 NLA 고유 기여가 아니다.** 활성값을 읽는 값싼 도구로도 된다.

### 실험 B — 진단-heldout OOD (Medical-AV는 판독기인가 분류기인가?)

18개 진단 클래스로만 학습, 학습에 **한 번도 없던 8개 클래스**로 시험.

| pool | answer_hit |
|---|---:|
| test_seen (학습한 18클래스, 새 환자) | 90.37% |
| test_heldout (본 적 없는 8클래스) | **0.00%** |

- heldout 800건 중 **790건(98.75%)이 학습한 18개 클래스 이름**으로 답함.
  정답 진단명을 낸 건 0/800.

**결론: 기존 Medical-AV는 seen-class 분류기였다.** 86.96%는 "판독"이 아니라
"in-distribution 분류"였다. 케이스 암기는 아님(새 환자엔 90% 일반화) — **클래스
수준 암기**: 활성값 클러스터 18개 → 외운 라벨.

### 실험 C — Vanilla 대조군 (format 위치)

LoRA를 뗀 순수 체크포인트로 같은 format-위치 활성값을 읽힘.
→ 모든 지표 0.0. 출력은 "Structured medical Q&A format signals..." 같은
**형식 서술**뿐, 진단 시도 없음.

### Part 1 종합

**죽은 주장**
- "Medical-AV가 활성값을 의미적으로 읽는다" (heldout 0%)
- "86.96%가 판독의 증거다" (in-distribution 분류의 증거였음)

**살아남은 주장**
- layer-32 활성값에 진단 정보가 선형적으로 존재한다 (probe 99.17%)
- source 오답의 다수는 **정보 부재가 아니라 디코딩 실패** — 내부엔 정답 신호가
  있는데 최종 답으로 못 나온 것

**원인 진단**: 문제는 아키텍처가 아니라 **학습 타깃**이었다. 닫힌 18-라벨
`<answer>` 타깃으로 SFT하면 분류기가 되는 게 최적해다. "읽으라"는 압력을 준 적이 없다.

---

## 2. 첫 수정 — v3 cue-first (타깃을 진단명 → cue로)

**왜 cue인가.** "읽는다"를 시험 가능하게 정의하면: *벡터 없이는 못 맞히는
내용을 출력하는가*. 진단명은 26지선다라 암기로 뚫린다(저엔트로피). 반면
**케이스별 cue 조합**은 케이스마다 달라(고엔트로피) 벡터를 읽지 않으면 못 맞힌다.
→ 타깃을 `<answer>진단명` 대신 `<observed>` cue 목록으로 바꾸고, 진단 텍스트는
아예 제거(shortcut 재유입 방지).

**결과 (layer-32 format 위치, 진단-heldout)**

| pool | cue_recall |
|---|---:|
| test_seen | 0.6251 |
| test_heldout | **0.1876** |

heldout 0.19는 v1의 암기 수준(0.31)에도 못 미침. **관문 실패.**

**해석 (예고했던 "의미 있는 실패").** 샘플을 읽어보니 출력이 주제 계열(부종/호흡곤란)은
맞히는데 케이스 고유 디테일은 틀리고, 가장 가까운 train 클래스 템플릿으로 회귀.
→ **layer-32 format 위치(=답 직전 마지막 토큰)에는 진단 클래스 신호는 강하지만,
개별 임상 근거(cue)는 자연어로 복원 가능한 형태로 보존되지 않는다.** 증거가 이미
결론으로 압축된 상태. — 이건 실패가 아니라 layer-wise 연구의 출발점이 되는 발견.

---

## 3. 원인 특정 — v4 cue-position (positive control)

**왜 이 실험인가.** v3 실패에는 두 설명이 남았다:
- (a) format 위치엔 디테일이 없다(압축) → 위치를 옮기면 읽힌다
- (b) 단일-벡터 NLA 방식 자체가 디테일을 못 꺼낸다 → 어디서도 안 됨

이 둘을 가르려면 **정보가 확실히 있는 위치**에서 시험해야 한다. 그래서 cue를
**그 단어 자신의 토큰 위치**에서 읽었다(그 자리엔 그 cue 정보가 있을 수밖에 없음).
암기 배제를 위해 **cue 문자열의 25%를 학습에서 완전히 제외**(cue-string heldout).

**결과 (L32, heldout 438건 전수 수동 분류)**

| 분류 | 비율 | 의미 |
|---|---:|---|
| A 정확 재현 | 17.8% | 자동 strict 채점과 일치 |
| B 올바른 패러프레이즈 | 37.9% | 본 적 없는 문자열을 자기 말로 |
| **A+B 의미 읽기** | **55.7%** | |
| C 계열은 맞고 디테일 오류 | 35.8% | ankle→calf 등 |
| D 완전 오류 | 8.4% | |

**결론: v3 실패는 위치 탓(a)이었다.** 정보가 있는 자리에선 단일-벡터 NLA가 unseen
케이스 고유 내용을 55.7% 읽어낸다. **프로젝트 핵심 가설의 첫 긍정 증거.**

**왜 이게 자동 strict(0.178)보다 높은가.** 모델은 본 적 없는 문자열을 그대로 인용할
수 없으니 의역한다("involuntary weight loss" → "unintentionally losing weight").
strict 매칭은 이걸 실패로 처리 → 그래서 전수 수동 판독이 필요했다.

**반(反)암기 검증**: 출력이 gold(학습에 없던 cue)에 더 가까운가, train cue에 더
가까운가? L24는 63%가 gold 쪽. 게다가 세 layer가 **완전히 같은 학습 데이터**로
서로 다른 점수(34/73/56) → 점수는 학습셋이 아니라 **입력 벡터**에서 나온다는 증거.

---

## 4. Layer 스윕 — 궤적을 그리다

**왜.** v4가 cue 디테일을 읽을 수 있음을 확인했으니, 이제 "그게 어느 layer에
얼마나 있는가"를 지도로. 같은 recipe를 layer 16/24/32에서. NLA 체크포인트는
L32용을 고정하고 layer별 LoRA만 학습(= 공유 디코더 + layer별 어댑터 구조).

**결과 (heldout 438건 전수 수동 분류, A+B 의미 읽기)**

| layer | A+B 의미 읽기 | C 디테일오류 | D 오류 |
|---|---:|---:|---:|
| L16 | 34.0% | 52.1% | 13.9% |
| **L24** | **73.1%** | 26.0% | **0.9%** |
| L32 | 55.7% | 35.8% | 8.4% |

**핵심 발견: 역U자, layer 24가 정점.** L24는 L32보다 **한 epoch 덜 학습하고도**
이겼고, 오류(D)를 거의 소멸시켰다(0.9%).

**해석 (궤적의 첫 실측):** *cue 디테일의 판독 가능성은 depth를 따라 오르다 L24에서
정점을 찍고 L32(답 직전)에서 감소한다 — 증거가 결론으로 접히는 지점이 L24~L32
사이에 있다.* logit lens·probe로는 못 만드는, 자연어 판독의 layer별 품질 곡선.

**Vanilla 대조로 LoRA의 역할을 재정의.** 같은 벡터를 LoRA 없이 읽혔더니:
- vanilla도 내용은 담는다 — 다만 "형식 해설" 포장 + 지어낸 프레임("the beach",
  "player's inventory") 섞임. 신뢰 불가한 서술자.
- vanilla는 L16/L24 벡터에서 붕괴하지 않음 → LoRA는 "좌표계 번역기"가 아님
  (residual stream 연속성).
- **→ LoRA의 진짜 기여 = 읽기 능력을 "창조"한 게 아니라, 잡음 많은 해설자를
  정밀하고 신뢰할 수 있는 판독기로 "증류(distill)"한 것.**

---

## 5. Counterfactual — faithfulness의 최종 증명

**왜 이게 필요했나.** 지금까지는 "정보가 있는 자리에서 읽더라"는 positive control.
"그래도 케이스 문맥을 외운 것 아니냐"는 반론이 남는다. → 개입(intervention)으로 증명.

**방법.** 같은 케이스에서 cue 하나만 다른 cue로 **교체(swap)**하고 그 자리를 다시
읽음. 판독이 새 cue로 바뀌면 = 벡터를 읽는 것. 옛 cue를 계속 말하면 = 문맥 암기.
또 cue를 **제거(removal)**하고 남은 cue가 그대로인지, 삭제한 cue가 유령처럼
나오는지 확인. (L24 판독기, 150 케이스, 프롬프트는 재조립+원본 대조로 정확)

**결과 (전수 150쌍 수동 재채점)**

| 지표 | 자동 | 수동 |
|---|---:|---:|
| swap이 새 cue로 이동 | 0.887 | **0.993** (T 0.707 + D 0.287) |
| 옛 cue 계속 읽음 (암기) | 0.040 | **0.000** (150건 중 0건) |
| phantom (삭제 cue 되살아남) | 0.053 | **~0.003** (16건 중 15건은 템플릿 오탐) |
| 유지 cue 안정성 (orig/swap/removed) | | 0.97 / 0.97 / 0.97 |

**결론: 인과적 faithfulness 확정.** cue 하나를 바꾸면 판독이 149/150에서 새 cue로
이동했고, 원래 cue를 계속 말한 경우는 **단 0건**. 판독기는 "케이스"가 아니라
**그 위치의 벡터**를 읽는다. probe/SAE로는 원리적으로 불가능한 증거.

**남은 유일한 약점(정확히 특정됨):** 신체 위치 속성의 해상도. 세부 오차 43건 중
28건이 온갖 신체 부위를 "iliac fossa(장골와)" 하나로 기본 응답. 위치가 아닌
내용(뺨, 이두근, 증상 서술 등)은 거의 완벽.

---

## 6. 지금까지의 종합 — "근거 있는 판독"이 4중으로 증명됨

| 증거 | 내용 |
|---|---|
| 1. OOD 일반화 (v4) | 학습에 없던 cue를 L32에서 55.7%, L24에서 73.1% 의미 수준 읽기 |
| 2. 반암기 | 출력이 train cue보다 unseen gold에 더 가까움 + layer 대조 |
| 3. Vanilla 대조 | 내용은 원래 있었고, LoRA는 그걸 정제함 |
| 4. Counterfactual | swap 추적 99.3%, 암기 0%, phantom ~0.3% |

**한 문장 요약:** *의료 LLM의 오답은 "몰라서"가 아니라 내부에 정답 신호가 있는데
출력에서 사라지는 경우가 많다. 우리는 그 내부 증거를 자연어로, 근거 있게(grounded),
학습에 없던 케이스에도 읽어내는 판독기를 만들었고, 그것이 암기·문맥의존이 아님을
개입 실험으로 증명했다. 그리고 그 판독 가능성이 layer 24에서 정점을 찍고 답 직전에
사라진다는 궤적의 첫 조각을 측정했다.*

**정직한 한계**
- 판독 라벨은 단독 채점자(나) 수동 분류
- L16 결론은 "정보 부족"과 "L32-AV에서 먼 layer라 LoRA 적응이 어려움"이 섞임(미분리)
- 신체 위치 속성 해상도 낮음
- 아직 AV-only. AR/full NLA는 미착수(의도된 선택 — 아래)

---

## 7. 왜 아직 AV만 하는가 (의도된 설계)

- 목표는 "근거 있는 자연어 설명"이고 그건 활성값→언어(AV) 방향이면 충분.
- 원래 full NLA 논리는 "AR 복원(MSE)으로 faithfulness를 검증"이었는데, 우리는 그걸
  **counterfactual 개입으로 대신 증명**했다 → 지금 AR이 필요 없다.
- AR을 지금 붙이면 위험: reconstruction만 최적화하면 "그럴듯하나 임상 무의미한"
  텍스트로 수렴(vanilla가 그 상태였음). 읽는 판독기가 먼저 있어야 AR이 의미가 생기고,
  그 판독기를 이제 막 확보했다.
- AR/full NLA는 목표가 아니라 **수단** — "출력마다 grounding을 강제"할 필요가 실제로
  생기면(post-hoc AR 일치도 → reranking → joint 학습 순서로) 그때 꺼낸다.

---

## 8. 앞으로의 계획

**1순위 — Format-position layer 스윕 (진행/대기 중)**
결론이 어느 layer에서 형성되는가. 같은 recipe를 format 위치에서 layer 16/24로.
cue 곡선(L16 34 → L24 73 → L32 56)과 겹치면 "증거→결론 접힘"의 완전한 궤적 완성.
(L32 format은 v3 결과 0.19로 이미 확보)

**2순위 — 오답노트 (교수님 "설명" 축; 검증된 판독기의 첫 소비처)**
source 모델이 틀린 케이스에서: cue 위치 판독(핵심 증거가 인코딩됐나?) + 결론
위치 판독(내부 결론이 뭐였나?) → 근거 있는 오답노트. handoff의 오답 4분류(missing
cue / distractor overweighting / late drift / decoding mismatch)를 실측으로 채움.
※ 오답이 많은 부분증거(3-cue) 세팅 필요.

**3순위 — 교정 (교수님 "solution" 축)**
`source 답 ≠ 판독`일 때 판독 내용을 주고 재고 유도. baseline은 generic "다시
생각해봐"로 두어 판독 내용의 기여를 분리.

**선택 — Attribute-resolution probe**
"iliac fossa" 약점이 벡터 탓인지(ankle vs calf를 벡터가 구분 못 함) 판독기 탓인지
(벡터엔 있는데 못 꺼냄) 분리.

**최종형 (조건부)** 위 결과들이 "grounding을 출력마다 강제해야 한다"를 요구하면
그때 AR을 학습해 full NLA로 확장, 그리고 layer-conditioned Medical-NLA
(공유 디코더 + layer별 어댑터)로.

---

## 부록 — 실험 코드/문서 위치 (repo)

- `docs/session_handoff_2026-08-01.md` — 배경
- `docs/results_2026-08-13_ood_and_probe_controls.md` — Part 1 (통제 3종) + v3 addendum
- `docs/results_2026-08-16_v4_cue_position.md` — Part 3 (원인 특정)
- `docs/results_2026-08-17_layer_sweep.md` — Part 4 (layer 곡선 + vanilla 대조)
- `docs/results_2026-08-17_counterfactual_faithfulness.md` — Part 5 (faithfulness)
- `EXPERIMENTS.md` §8–§14 — 전 실험 서버 실행 runbook
- `results_snapshot/*_hand_labeled.jsonl` — 수동 분류 라벨 (재검증 가능)
