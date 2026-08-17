# Medical-NLA 진행 발표 자료 (2026-08-17, 상세판)

이 문서는 발표에서 그대로 읽고 설명할 수 있도록 쓴 완전판이다.
순서: 배경/도구 원리 → 시작 상태 → 발견한 문제(통제 실험) → 문제를 어떻게
풀어나갔나(v3→v4→layer sweep→counterfactual) → 종합 → 왜 아직 AV만 하는가 →
앞으로의 계획 → 예상 질문 대응.

모든 수치와 "왜 그렇게 판단했는가"를 함께 적었다.

---

# 0. 배경: 우리가 원래 하려던 것과 도구의 원리

## 0.1 목표 (교수님이 준 3축)

의료 LLM을 대상으로:
1. **설명(Explanation)** — 모델이 맞거나 틀렸을 때, 내부에서 무엇을 고려했는지
   사람이 읽을 수 있게 꺼낸다.
2. **진단(Diagnosis)** — 모델의 답이 틀릴 것 같은지를 내부 신호로 예측한다.
3. **교정(Solution)** — 그 내부 신호를 이용해 모델의 답을 실제로 고친다.

## 0.2 왜 NLA인가 (기존 해석 도구와의 차이)

내부 상태(활성값)를 읽는 기존 도구들의 출력 형태:

| 도구 | 출력 | 한계 |
|---|---|---|
| Linear probe | "클래스 7, 확률 0.93" | 미리 정한 라벨 중 하나. '왜'가 없음. 새 내용 표현 불가 |
| SAE | "feature #4821 활성" | 그 feature가 뭔지 사람이 또 해석. layer마다 사전 다름 |
| Logit/Tuned lens | 중간층의 토큰 확률 | 역시 자연어 서술이 아님 |
| **NLA** | **자연어 문장** | 조합·불확실성·갈등을 미리 정한 어휘 없이 서술 가능 |

→ **probe/SAE는 "분류·감지"만, NLA만 "서술"할 수 있다.** 이게 우리 프로젝트의
잠재적 novelty이자, 교수님 3축(특히 "설명")에 유일하게 맞는 도구인 이유.

## 0.3 NLA (Natural Language Autoencoder)의 실제 작동

NLA는 두 부분으로 된 오토인코더다.

- **AV (Activation Verbalizer)**: 활성값 벡터 `h` → 자연어 설명 `z`
- **AR (Activation Reconstructor)**: 자연어 설명 `z` → 복원 활성값 `h'`
- 평가: `MSE(h, h')` (텍스트↔활성이 아니라 **원본↔복원 활성** 사이의 거리)

**AV의 활성값 주입 메커니즘 (중요, 발표에서 질문 나올 수 있음):**
- AV는 Gemma 기반 언어모델이고, 프롬프트 안에 특수 주입 토큰(문자 `㈜`,
  token id 246566) 자리가 있다.
- 그 자리의 토큰 임베딩을 **우리가 읽으려는 활성값 `h`로 교체**한다.
- 교체 전에 `h`의 norm을 sidecar에 정의된 `injection_scale`(=80000)로 재정규화한다.
  Gemma-3의 scaled embedding 때문에 residual stream norm이 크고, layer마다 다르기
  때문 — 이 재정규화가 layer별 norm 차이를 흡수한다.
- 그러면 AV가 그 벡터를 "설명하는" 텍스트를 생성한다.

## 0.4 실험 세팅

- **Backbone(source model)**: `google/gemma-3-12b-it` — 48 transformer layer,
  d_model = 3840
- **NLA 체크포인트**: `kitft/nla-gemma3-12b-L32-av` (와 `-ar`) — **layer 32**에서
  뽑은 활성값을 설명하도록 upstream에서 학습된 것
- **데이터**: DDXPlus — 환자 케이스마다 증상 cue들(예: "moderate fever",
  "productive cough")과 정답 진단(PATHOLOGY). 프롬프트 형식:
  `"A patient presents with {cue1}, {cue2}, ... What diagnosis is most likely?"`
- **두 단계 실행**: (1) Gemma에 프롬프트를 통과시켜 layer 32 활성값을 `.pt`로 저장,
  (2) 그 활성값을 AV에 주입해 텍스트 생성. (Gemma-12B + AV를 32GB에 동시에 못 올려
  두 단계로 분리)

## 0.5 세션 시작 시점의 상태 (기존 Medical-AV)

이전 세션에서 만든 것: AV 한쪽만 LoRA로 의료 튜닝한 **Medical-AV**. AR은 학습 안 함.

시작 시점의 헤드라인 수치(겉보기 매우 강력):

| 지표 | 값 |
|---|---:|
| Linear probe: layer-32 활성값 → 진단 분류 acc1 | **99.17%** (acc5 100%) |
| Medical-AV all-cue source-aligned answer_hit | **86.96%** (920/1058) |
| Medical-AV mean cue_recall | **79.94%** |
| source 답 ≠ Medical-AV → source 오답률 | **98.90%** |
| disagreement AUROC / AP | **0.9427 / 0.9708** |
| source가 **틀린** 케이스에서 Medical-AV 정답 | **83.44%** (655/785) |
| (참고) source confidence baseline AUROC | 0.67–0.70 |

특히 마지막 두 줄이 인상적이었다: "Gemma가 최종 답을 틀려도, 내부 활성값에는 정답
진단이 있어서 Medical-AV가 83% 읽어낸다"는 서사. disagreement AUROC 0.94는 source
confidence baseline(0.67~0.70)을 압도.

**그런데 이 결과들이 검증되지 않은 두 개의 가정 위에 있었다. 그것을 검증하는 것이
이번 세션의 출발점이었다.**

---

# 1. 발견한 문제 — 헤드라인이 두 개의 confound 위에 있었다

## 1.1 왜 의심했는가

두 가지 반론이 검증되지 않은 채 남아 있었다.

**Confound A — Medical-AV는 "판독기"인가 "seen-class 분류기"인가?**
linear probe가 이미 99.17%로 그 활성값에서 진단을 거의 완벽히 선형 분리한다. 그렇다면
Medical-AV가 활성값을 *읽는* 게 아니라, 26개 클래스에 대해 "활성 클러스터 → 라벨"만
외워도 86% answer_hit이 나올 수 있다. in-distribution 시험만으로는 "읽기"와 "암기 분류"가
구분되지 않는다.

**Confound B — disagreement 결과는 순환적/편향적이지 않은가?**
- test set이 **source-error-enriched**였다(train/val을 source-correct만 썼으므로
  test에 오답이 74% 몰림). base rate이 부풀려짐.
- Medical-AV는 **gold로 학습**됐다. 그러면 "source ≠ AV"는 곧 "source ≠ gold일
  확률이 높은 경우"라 정보적으로 정답 정의와 겹친다.

→ 이 둘을 각각 반증하기 위해 통제 실험 3개를 설계했다. **핵심 판단: 새 실험을
늘리기 전에, 지금 결과가 무너질 수 있는 방식을 우리가 먼저 시험한다.**

## 1.2 실험 A — Probe 대조군 (오류 탐지가 NLA 고유 기여인가?)

**방법**: 같은 활성값에서 linear probe 예측을 얻어, "source 답 ≠ probe 예측"이라는
불일치 신호를 "source 답 ≠ Medical-AV" 불일치와 **같은 잣대로** 비교. 두 신호가 모두
정의된 152행에서 paired 비교.

| 신호 | n | AUROC |
|---|---:|---:|
| source ≠ NLA disagree | 1058 | 0.9427 |
| **source ≠ probe disagree** (paired 152) | 152 | **1.0000** |
| source ≠ NLA disagree (paired 152) | 152 | 0.9282 |
| source confidence baseline | 1058 | 0.67–0.70 |

paired에서 **nla − probe AUROC = −0.0718 (probe가 이김).**

**왜 이 결과가 나오는가 (원리):** probe가 99% 정확하면 "source ≠ probe"는 거의
"source ≠ 정답"과 같아진다. 그래서 near-perfect 탐지가 구조적으로 나온다.

**결론**: **in-distribution 오류 탐지는 활성값을 읽는 어떤 강한 도구든 할 수 있는
일이고, NLA의 고유 기여가 아니다.** AUROC 0.94를 "NLA의 강점"으로 팔면 리뷰어가 정확히
이 지점을 찌른다. 우리가 먼저 확인해 무력화했다.

## 1.3 실험 B — 진단-heldout OOD (판독기 vs 분류기)

**방법**: 26개 진단 클래스를 train 18 / heldout 8로 분리(seed 17). train 클래스의
source-correct 케이스만 학습(884행)/검증(189행). 테스트는 두 pool:
- `test_seen`(727행): 학습한 18클래스의 새 환자 — in-distribution 기준
- `test_heldout`(800행): 학습에 **한 번도 없던 8클래스** — 진짜 OOD

같은 어댑터로 둘 다 읽힘. (학습은 정상: val_loss 0.0396)

| pool | answer_hit | mean_cue_recall |
|---|---:|---:|
| test_seen | **90.37%** | 0.7690 |
| test_heldout | **0.00%** | 0.3066 |

**분류기-붕괴 체크**: heldout 800건 중 **790건(98.75%)이 학습한 18개 클래스 이름**으로
답함. 정답 진단명을 낸 건 **0/800**.

붕괴가 균일하지 않다는 것도 관찰됨(임상적 인접성):
- `urti → Bronchitis` 98% (임상적으로 가까움)
- `sle → Scombroid 식중독` 96% (무의미)
- `pulmonary_neoplasm → Anemia` 100% (무의미)

**결론**: **기존 Medical-AV는 seen-class 분류기였다.** 86.96%는 "판독"이 아니라
in-distribution 분류. 정확히는 케이스 암기는 아니고(새 환자엔 90% 일반화) **클래스
수준 암기** — 활성 클러스터 18개 → 외운 라벨 + 외운 전형 cue.

## 1.4 실험 C — Vanilla(무학습) 대조군

**방법**: LoRA를 뗀 순수 체크포인트로 같은 format-위치 heldout 활성값을 읽힘.
→ 전 지표 0.0. cjk_fraction 0.0(주입 정상), 출력은 깨끗한 영어.

실제 출력 예시:
> "Structured medical Q&A format signals a clinical diagnosis response,
> establishing a structured answer about a patient presentation. The phrase
> 'Based on the symptoms described' signals a definitive clinical judgment..."

→ **형식만 서술, 임상 내용 통합 0.** (얕은 cue 반향은 간혹 있으나 진단 시도 없음)

## 1.5 Part 1 종합

같은 heldout 활성값(layer 32, format 위치)에서:

```
linear probe(seen 클래스):  99% — 정보는 선형적으로 존재
vanilla AV:                 형식 서술만, 임상 통합 0
LoRA AV(기존):              train 클래스 라벨로 collapse (분류기)
```

**죽은 주장**
- "Medical-AV가 활성값을 의미적으로 읽는다" (heldout 0%)
- "86.96% answer_hit이 판독의 증거다" (in-distribution 분류의 증거였음)

**살아남은 주장**
- layer-32 활성값에 진단 정보가 선형적으로 존재 (probe 99.17%)
- **source 오답의 다수는 정보 부재가 아니라 디코딩 실패** — 내부엔 정답 신호가
  있는데 최종 출력으로 안 나온 것 (이게 프로젝트의 핵심 흥미)
- 불일치가 오류를 탐지하긴 함 — 단 probe로도 되므로 NLA 기여는 아님

**원인 진단**: 문제는 아키텍처가 아니라 **학습 타깃**이다. 닫힌 18-라벨 `<answer>`
타깃으로 SFT하면 분류기가 되는 게 최적해다. "읽으라"는 압력을 준 적이 없다. 그리고
이건 vanilla가 원래 갖고 있던 open-vocab 생성 능력을 우리가 분류 헤드로 덮어썼다는 뜻.

---

# 2. 첫 수정 시도 — v3 cue-first (타깃 재설계)

## 2.1 왜 cue를 타깃으로 (핵심 논리)

"읽는다"를 시험 가능한 형태로 정의하면: **벡터를 보지 않고서는 맞힐 수 없는 내용을
출력한다.**

우리가 gold를 가진 정보는 두 종류:

| 정보 | 경우의 수 | 암기 뚫림? |
|---|---|---|
| 진단명 | 26지선다 (저엔트로피) | 예 — 클러스터→라벨 암기로 뚫림 (v1이 그랬음) |
| **cue 조합** | 케이스마다 다른 3~12개 (고엔트로피) | 아니오 — 읽어야만 맞힘 |

→ cue는 목적이 아니라 **"읽었음"을 채점할 수 있는 유일한 고엔트로피 정답지**.

**단, "완전히 못 외운다"는 과장이다.** 우회로가 있다: "cluster로 seen 클래스 판별 →
그 클래스의 전형 cue 뿌리기". v1이 실제로 간 길. 그래서 recall 단독이 아니라
precision + mismatched + counterfactual까지 봐야 한다(이후 4-기준 관문의 근거).

## 2.2 타깃 형식

v1 (실패한 형태):
```xml
<answer>Bronchitis</answer>   ← 18개 중 고르기 = 분류 = 암기
```
v3 (새 형태): 진단 텍스트를 **완전 제거**(shortcut 재유입 방지), cue 목록만:
```xml
<explanation><readout><observed>
- moderate fever
- sore throat
- nasal congestion
</observed></readout></explanation>
```
cue 순서는 row별 시드로 셔플(순서 암기 방지).

## 2.3 결과 (layer-32 format 위치, 진단-heldout)

| pool | cue_recall | cue_precision |
|---|---:|---:|
| test_seen | 0.6251 | 0.6962 |
| test_heldout | **0.1876** | 0.2437 |

heldout 0.19는 v1 암기 수준(0.31)에도 못 미침. **관문 실패.**

## 2.4 해석 (예고했던 "의미 있는 실패")

heldout 출력을 읽어보니: 주제 계열(부종/호흡곤란)은 맞히는데 케이스 고유 디테일은
틀리고, 가장 가까운 train 클래스 템플릿으로 회귀. 예: 폐부종(unseen) 케이스에서
`gold: swelling located ankle(R)` 인데 출력은 `calf(L)`, `thigh(R)`... 부위를 여러 개
나열하며 가끔 적중. 심지어 한 출력은 train 클래스 폐색전증(PE)의 교과서 cue 세트를 뱉음.

**결론**: **layer-32 format 위치(=답 직전 마지막 토큰)에는 진단 클래스 신호는
강하지만, 개별 임상 근거(cue)는 자연어로 복원 가능한 형태로 보존되지 않는다.** 증거가
이미 결론으로 압축된 상태. — 실패지만, "증거가 결론으로 접힌다"는 layer-wise 서사의
출발점이 되는 발견.

---

# 3. 원인 특정 — v4 cue-position (positive control)

## 3.1 왜 이 실험 (v3 실패의 두 해석 가르기)

v3 실패에는 경쟁하는 두 설명이 남았다:
- **(a) 위치 탓**: format 위치엔 디테일이 압축돼 없다 → 위치를 옮기면 읽힌다
- **(b) 메커니즘 탓**: 단일-벡터 NLA 방식으론 어디서도 디테일을 못 꺼낸다

이 둘을 가르려면 **정보가 확실히 있는 위치**에서 시험해야 한다. 그래서 cue를
**그 단어 자신의 토큰 위치**에서 읽었다 — 모델이 방금 "fever"를 읽은 자리엔 fever
정보가 있을 수밖에 없다. 이게 positive control.

## 3.2 방법 (암기 배제 설계)

- 케이스당 cue 4개까지, 각 cue를 자기 토큰 span에서 추출 (3,200케이스 → ~12,800행)
- **cue 문자열 단위 heldout**: 164개 유니크 cue 문자열 중 **41개(25%)를 학습에서
  완전히 제외**. (진단-heldout이 아님 — cue는 진단들끼리 공유돼서 진단만 빼면
  cue가 샌다. 그래서 cue-string heldout이 필수)
- train 7,515 / val 1,086 / test_seen_cue 2,122 / test_heldout_cue 438

## 3.3 결과 (L32, heldout 438건 전수 수동 분류)

자동 strict 채점은 0.178이지만, 본 적 없는 문자열은 그대로 인용할 수 없어 의역하기
때문에 strict가 실패로 처리한다. 그래서 **438건 전부를 눈으로 4범주 분류**:

| 분류 | 건수 | 비율 |
|---|---:|---:|
| A 정확 재현 | 78 | 17.8% |
| B 올바른 패러프레이즈 | 166 | 37.9% |
| **A+B 의미 읽기** | **244** | **55.7%** |
| C 계열 맞고 디테일 오류 | 157 | 35.8% |
| D 완전 오류 | 37 | 8.4% |

실제 B(패러프레이즈) 예 — 전부 학습에서 본 적 없는 문자열:
- `"pain that is increased with movement"` → `"pain that increases with movement"`
- `"had an involuntary weight loss over the last 3 months"` →
  `"been unintentionally losing weight over the past 3 months"`
- `"where is the affected region located iliac fossa(L)"` (정확 계열)

**결론: v3 실패는 위치 탓(a)이었다.** 정보가 있는 자리에선 단일-벡터 NLA가 unseen
케이스 고유 내용을 55.7% 의미 수준으로 읽는다. **probe(닫힌 라벨)로도, format 위치
readout(주제만)으로도 불가능한 — 프로젝트 핵심 가설의 첫 긍정 증거.**

## 3.4 반(反)암기 검증 (L32, 이 실험 내부)

"cue 위치라 자기 단어를 반향한 것 아니냐"에 답하기 위해: L32 출력이 gold(학습에 없던
cue)에 더 가까운가, 117개 train cue 중 최선에 더 가까운가? → **47.9%가 gold 최근접,
34.7%만 train cue 최근접.**

**이 실험(L32)만으로 서는 두 논거:**
1. **구성 불가능성**: L32 출력의 절반이 학습 타깃 어디에도 없는 gold에 최근접. 게다가
   "how bad is the itching" 같은 출력은 train 타깃에도 gold에도 없는 **제3의 신규
   문장** — 복사로는 못 만들고 합성해야만 나온다.
2. 남은 nearest-train 잔여분은 수동 분류의 C 범주(위치 계열 오차)와 일치 → 그래서
   A+B만 "읽기"로 주장.

암기 가설을 완전히 못 박는 **세 번째 논거(layer 대조)** 는 layer sweep 데이터가
있어야 성립하므로, 그 데이터를 소개한 뒤 §4.4에서 마저 다룬다.

---

# 4. Layer 스윕 — 궤적을 그리다

## 4.1 방법과 아키텍처 (발표에서 질문 나올 부분)

같은 v4 recipe를 layer 16/24/32에서. **바뀌는 것은 활성값을 어느 층에서 뽑느냐 하나뿐**
(config의 `layer:` 한 줄). NLA 체크포인트는 L32용을 **동결**하고 layer별 **LoRA만
학습**한다.

**이 구조가 곧 "공유 디코더 + layer별 어댑터"다:**
- L32-AV의 동결된 디코더 = 언어 능력·의학 어휘·서술 방식을 담은 공유 몸통
- layer별 LoRA = 그 layer의 활성값을 동결된 디코더가 알아듣는 형태로 맞춰주는 부품
- "번역기다"라는 초기 비유는 이후 vanilla 대조로 수정됨(아래 4.3)

## 4.2 결과 (heldout 438건 전수 수동 분류)

| layer | A 정확 | B 패러프레이즈 | **A+B 의미읽기** | C 디테일오류 | D 오류 |
|---|---:|---:|---:|---:|---:|
| L16 | 10.7% | 23.3% | **34.0%** | 52.1% | 13.9% |
| **L24** | 16.7% | 56.4% | **73.1%** | 26.0% | **0.9%** |
| L32 | 17.8% | 37.9% | **55.7%** | 35.8% | 8.4% |

(자동 지표도 순서 일치: heldout soft@0.5 = L16 0.607 / L24 0.813 / L32 0.699.
seen pool은 세 layer 모두 0.97~0.99 — in-distribution은 어디서나 쉽고, 갈리는 건
unseen 일반화)

**핵심 발견: 역U자, layer 24가 정점.** L24는 L32보다 **한 epoch 덜 학습(2 vs 3)하고도**
이겼고, 오류(D)를 거의 소멸시켰다(0.9% vs L32 8.4%). L24 패러프레이즈는 정밀하다:
- `"how severe is the itching"` → `"how bad/intense is the itching"`
- 복부팽만 → `"swelling of the abdomen (this is called ascites)"` (임상 해석까지 얹음)

L16은 형식은 남고 내용이 빠지는 실패("3개월간 체중감소" → "3개월간 **기침**").

**실물 3연속 — 같은 gold cue, 세 layer가 실제로 내뱉은 문장** (heldout, hand-labeled
snapshot에서 직접 발췌. 36개 cue가 세 layer에 공통 등장, 그 중 대표 3개):

*판정 기준(§3.3 재확인): **A**=문자열까지 일치 / **B**=표현만 다르고 의미 완전 보존
(패러프레이즈) / **C**=같은 계열이나 세부 속성이 틀림 / **D**=의미가 어긋남·반전.
A+B만 "읽었다"로 집계.*

**① 깔끔한 역U자 — L24가 정확, 양끝이 다른 방식으로 실패**
| layer | 실제 출력 | 판정 |
|---|---|:--:|
| gold | `pain that is increased with movement` | — |
| L16 | `pain that is increased with coughing, with an effort like lifting a weight or from walking` | C (엉뚱한 유발요인 나열) |
| **L24** | `pain that is increased with movement` | **A (완전 일치)** |
| L32 | `pain that increases with movement` | B (패러프레이즈) |

→ **L32가 왜 B인가**: gold의 `is increased with`를 `increases with`로 **문장 구조만**
바꿨다. 문자열이 다르니 A는 아니고, "움직이면 통증 증가"라는 의미는 100% 보존이라
C/D도 아니다 → B. (L16은 유발요인이 movement→coughing/lifting/walking으로 **바뀌어**
같은 계열의 세부가 틀림 → C.)

**② L16의 전형적 실패 = 형식은 남고 내용이 소실**
| layer | 실제 출력 | 판정 |
|---|---|:--:|
| gold | `how severe is the itching` | — |
| L16 | `the itching` | C (질문 형식·정도(severe) 소실, 명사만 남음) |
| **L24** | `how severe is the itching` | **A (완전 일치)** |
| L32 | `what is the level of itching` | B (패러프레이즈) |

→ **L32가 왜 B인가**: `how severe`(얼마나 심한가)를 `what is the level of`(정도가 어느
수준인가)로 **질문 프레임만** 갈아끼웠다. "가려움의 강도를 묻는다"는 내용은 동일 → B.
(반면 L16 `the itching`은 강도를 묻는 부분 자체가 **빠져** 명사만 남음 → 의미 결손이라 C.)

**③ L32의 "접힘" = 답 직전 층이 반대 결론으로 미끄러짐**
| layer | 실제 출력 | 판정 |
|---|---|:--:|
| gold | `chest pain even at rest` | — |
| L16 | `symptoms that are increased with physical exertion but alleviated while at rest` | D (운동유발 서사로 반전) |
| **L24** | `chest pain at rest` | **B (정확)** |
| L32 | `symptoms that are increased with physical exertion but alleviated with rest` | D (다시 운동유발 서사) |

→ **L24가 왜 B인가**: gold `chest pain even at rest`에서 강조어 `even`만 빠졌고 "쉬는
중에도 가슴통증"이라는 핵심 의미는 그대로 → 문자 일치 A는 아니지만 의미 보존이라 B.
(L16·L32는 "**운동하면 심해지고 쉬면 낫는다**"로 원 증거와 뜻이 뒤집혀 D — 여기서
'at rest'가 '증상 유발'이 아니라 '증상 완화' 쪽에 붙어버린 게 반전의 핵심.)

③이 궤적의 핵심을 보여준다: 원 증거는 "쉬어도 아픈 가슴통증"인데, **L16(증거 형성 전)과
L32(답 형성 후)는 둘 다 "운동하면 심해지고 쉬면 낫는다"는 임상 서사로 미끄러지고,
중간 L24만 원 증거를 그대로 읽는다.** 같은 현상의 더 극적인 예: gold
`out of breath with minimal physical effort` → L24 정확 판독, **L32는
`no shortness of breath...`로 부정(negation)까지 붙여 결론화**. 답을 만드는 층은
증거를 있는 그대로 두지 않고 진단 서사로 접는다.

**해석 (궤적의 첫 실측):** *cue 디테일의 자연어 판독 가능성은 depth를 따라 오르다
layer 24에서 정점을 찍고 layer 32(답 직전)에서 감소한다 — 증거가 결론으로 접히는
지점이 L24~L32 사이에 있다.*

## 4.3 Vanilla 대조 — LoRA의 역할 재정의

같은 벡터를 LoRA 없이 읽혔더니(무학습 기준선):

| 벡터 | 어휘 지표 | full-output soft@0.5 |
|---|---:|---:|
| L16 | 0.039 | 0.564 |
| L24 | 0.142 | 0.582 |
| L32 | 0.105 | 0.658 |

실물을 읽어야 숫자가 뭘 감췄는지 보인다:

**같은 벡터, 두 판독기 나란히 (동일 row id):**
- (증류의 전형) gold `"detached from own body or surroundings"`:
  - vanilla: "...The phrase 'feeling disconnected from themselves or their
    surroundings' sets up a definition of dissociation..." (내용 정확, 3문단 포장)
  - LoRA: `- feeling detached from their own body or surroundings` (한 줄 정제)
- (vanilla 전멸, LoRA 구조) gold `"swelling located dorsal foot(L)"`:
  - vanilla: "...location includes...**beach, river, lake**...player's tracking
    data" (완전 confabulation)
  - LoRA: `- where is the swelling located sole(L)` (부종·위치·좌측 정확, 발등↔발바닥만 틀림)

**vanilla를 layer별로 — 같은 gold cue를 세 layer의 vanilla가 각각 어떻게 읽나**
(heldout snapshot 실물, 출력 첫 문장부. vanilla 출력은 실제론 수 문단짜리 해설):

- gold `chest pain even at rest` (unstable angina):
  - L16: "...**quoted expert statement** about heart attack symptoms **in Spanish**.
    The phrase '**chest pain even at rest**' sets up a clinical classification..."
  - L24: "...case report format... The phrase 'experiences **chest pain even at r**...'"
  - L32: "...clinical definition of ACS... The phrase '**chest pain while active, now
    experiencing chest pain at rest**'..."
  - → 세 layer 모두 gold 문구가 **글자 그대로** 들어 있다. 단 L16은 "스페인어 전문가
    인용"이라는 지어낸 액자 속에, L32는 이미 "활동시→안정시 진행"이라는 임상 서사로
    풀어서. **내용은 어느 layer에나 있고, 문제는 포장이다.**
- gold `like the patient are detached from their own body or their surroundings`
  (panic attack):
  - L16: "hallucination의 정의를 서술하는 의학 기사" 액자 속 "feeling disconnected
    from their thoughts and surroundings"
  - L24: "우울·불안 증상 번호 목록" 액자 속 "feeling detached from themselves or
    their surroundings"
  - L32: "dissociation FAQ" 액자 속 "feel detached from their own body or their
    surroundings" (gold와 거의 자구 일치)
  - → 핵심 구절은 세 layer 모두 보존, **액자만 layer마다 다르게 지어냄.**
- gold `where is the swelling located ankle(R)` (pulmonary edema):
  - L16: "'pain in the **right leg**'... signals a **Japanese**/medical..." (통증으로
    왜곡 + 일본어 액자 confabulation)
  - L24: "'pain location is **rt hand** (right)'" (엉뚱한 신체부위로 표류)
  - L32: "'swelling **right leg**', '**right ankle**'" (부종·발목·우측 전부 등장)
  - → 속성(정확한 부위)은 vanilla에서 layer가 깊을수록 선명해지는 경향 — 단, 자동
    지표(ocr)로는 세 layer 다 0인 행도 많다. 내용이 해설 더미에 묻혀 있어서다.

이 layer별 실물이 보여주는 것: **vanilla의 실패는 "내용 부재"가 아니라 "신뢰 불가"다.**
같은 벡터에서 gold 문구를 품고도, 행마다 다른 액자(스페인어 인용, 일본어 문서, 챗봇
시나리오, 게임)를 지어내 그 안에 흩뿌린다. 어느 문장이 벡터에서 온 것이고 어느 문장이
지어낸 것인지 vanilla 출력만 봐서는 구분할 수 없다 — 이 구분 불가능성이 LoRA가 제거한
것이다.

**세 가지 결론:**
1. vanilla도 cue 위치에선 내용을 담는다 — 다만 "형식 해설" 포장 + 지어낸 프레임
   섞인 **신뢰 불가한 서술자**. (format 위치에선 vanilla도 0 → 위치 주장 재확인)
2. vanilla가 L16/L24 벡터에서 **붕괴하지 않는다** → LoRA는 "좌표계 번역기"가 아님.
   residual stream 연속성으로 인접 layer 좌표계가 호환됨.
3. **→ LoRA의 진짜 기여 = 읽기 능력을 "창조"한 게 아니라(체크포인트에 내재),
   잡음 많은 해설자를 정밀·신뢰 가능한 단일-cue 판독기로 "증류(distill)"한 것.**
   confabulation 제거 + 의미 정확도 73% + 오류 클래스 0.9%가 LoRA 몫.

**정직한 단서**: vanilla의 layer 순서는 자기 모국어 L32가 1등(0.658). 그래서 역U자
주장의 근거는 vanilla가 아니라 **각 layer에 동등한 어댑터를 준 통제 비교**(거기서
L24가 더 오래 학습한 L32를 이김)에 둔다.

## 4.4 반(反)암기, layer 대조로 확정 (§3.4에서 유보한 세 번째 논거)

이제 layer sweep 데이터가 있으니 §3.4의 마지막 논거를 완성한다. 각 layer의 출력이
gold(never-seen cue)에 더 가까운가, nearest train cue에 더 가까운가:

| layer | gold에 더 가까움 | train cue에 더 가까움 |
|---|---:|---:|
| L16 | 31.1% | 54.6% |
| L24 | **63.2%** | 35.6% |
| L32 | 47.9% | 34.7% |

**세 layer는 완전히 같은 학습 데이터·같은 타깃**을 쓴다. 바뀐 것은 활성값을 뽑은 층
하나(config의 `layer:`)뿐이다. 만약 점수가 학습셋 암기에서 나온다면 세 layer가 비슷해야
한다. 그런데 gold 최근접률이 **31→63→48로 갈렸고**(A+B 의미읽기율도 34→73→56으로 같은
궤적), 이 layer 의존성은 암기로는 설명되지 않는다. → 점수는 학습셋이 아니라 **입력
벡터가 어느 layer에서 왔느냐**에서 나온다. 이것이 §3.4의 구성 논거 위에 얹히는 결정적
반암기 증거다.

## 4.5 Format-position 스윕 — 궤적의 나머지 반쪽 (2026-08-17 완료)

cue 위치 곡선이 "증거가 **어디에** 읽히는가"였다면, 이건 "답이 만들어지는 자리(마지막
토큰)에는 **어느 depth에서든** 증거가 읽히는 형태로 있는가". 같은 v3 cue-first recipe,
같은 진단-heldout, 같은 채점기로 L16/L24를 신규 학습 (L32는 §2.3 결과 재사용):

| layer | format 위치 seen | format 위치 **heldout** | (비교) cue 위치 heldout A+B |
|---|---:|---:|---:|
| L16 | 0.360 | 0.188 | 34.0% |
| L24 | 0.684 | **0.249** | **73.1%** |
| L32 | 0.625 | 0.188 | 55.7% |

**헤드라인: 어느 layer도 format 위치를 구제하지 못한다.** heldout이 세 layer 모두
0.19~0.25로 평평하게 낮다. 같은 판독기 recipe가 같은 L24에서 cue 토큰은 73%를 읽고
마지막 토큰은 ~25%밖에 못 읽는다. 즉 v3의 실패는 "L32라서"가 아니었다 — **케이스 고유
증거가 자연어로 복원 가능한 형태로 답 위치에 놓이는 depth는 존재하지 않는다.**

보조 관찰 둘:
- seen 열은 L16 0.36 → L24/L32 0.63~0.68로 오른다. in-distribution의 이 점수는
  "클래스→전형 cue 템플릿" 지름길인데(heldout 붕괴가 증명), **지름길조차 depth가
  필요하다** — L16의 마지막 토큰엔 클래스 정체성도 아직 덜 형성돼 있다.
- heldout을 진단별로 쪼개면 urti만 0.69~0.72로 높고 sle/탈장은 ~0. urti의 cue 어휘가
  train 클래스(기관지염 — v1의 붕괴 짝)와 겹치기 때문 — **어휘가 겹치는 곳에서만 점수가
  나는 것은 템플릿 살포의 지문**이지 읽기가 아니다.

**궤적 지도 완성**: 임상 증거는 cue 토큰 위치에 살고(L24 정점 73%), 답 위치에는 어느
depth에서도 증거 형태로 존재하지 않으며 클래스 신호만 남는다(probe 99%). "증거가
결론으로 접힌다"는 depth의 문제이면서 동시에 **위치의 문제**다.

**실무 귀결 (오답노트 설계 확정)**: 내부 결론은 format 위치에서 **클래스 채널**(probe
또는 v2형 판독)로, 증거는 **cue 위치 L24 판독기**로 읽는다. format 토큰이 증거로 읽히는
중간 layer를 찾는 GPU 탐색은 불필요 — 이 스윕이 "없다"고 답했다.

---

# 5. Counterfactual — faithfulness의 최종 증명

## 5.1 왜 필요했나

지금까지는 "정보가 있는 자리에서 읽더라"는 positive control. "그래도 케이스 문맥을
외운 것 아니냐"는 반론이 남는다. → **개입(intervention)으로 인과적 증명.** probe/SAE가
원리적으로 못 하는 종류의 증거.

## 5.2 방법 (구성 정확성 보장)

- **swap(민감성)**: 같은 케이스에서 cue 하나만 다른 cue로 교체 → 그 자리 활성값 재추출
  → 판독이 새 cue로 바뀌면 벡터를 읽는 것, 옛 cue를 계속 말하면 문맥 암기.
- **removal(특이성)**: cue 하나 제거 → 남은 cue 위치는 그대로 읽혀야 하고, 삭제한
  cue가 유령처럼 나오면 안 됨.
- 프롬프트는 cue 목록에서 **재조립 + 원본과 일치 검증**(불일치 케이스 스킵) → 구성 오류 0.
- L24 판독기, 150 케이스.

## 5.3 결과 (전수 150쌍 수동 재채점)

| 지표 | 자동 | 수동 | 목표 |
|---|---:|---:|---|
| swap이 새 cue로 이동 | 0.887 | **0.993** (T 0.707 + D 0.287) | 높음 |
| 옛 cue 계속 읽음 (암기) | 0.040 | **0.000** (150건 중 0건) | ~0 |
| phantom (삭제 cue 되살아남) | 0.053 | **~0.003** (16건 중 15건 템플릿 오탐) | ~0 |
| 유지 cue 안정성 orig/swap/removed | | 0.973/0.967/0.967 | 안정 |

swap 예시(수동 재채점의 근거):
- `→ black stools like coal` 출력 `light red blood in stool (hematuria)`
  = GI 출혈 주제로 정확히 이동, 색/유형만 틀림 (자동은 이걸 실패로 처리) → D
- `→ detached from own body` 출력 `feeling detached from own body or surroundings`
  = 완벽 → T
- 원래 cue를 계속 읽은 경우: **0건**

**결론: 인과적 faithfulness 확정.** cue 하나를 바꾸면 판독이 149/150에서 새 cue로
이동했고, 원래 cue를 계속 말한 경우는 **단 0건**. 판독기는 "케이스"가 아니라 **그
위치의 벡터**를 읽는다. 자동의 4% 암기 신호는 원본↔교체 cue 단어 우연 겹침이었을 뿐.

phantom도 16건 중 15건이 "where is the swelling located ___" 같은 **템플릿 공유** 때문에
유지 cue를 정확히 읽었는데 잘못 잡힌 것. 진짜 유령은 1건.

## 5.4 정확히 특정된 유일한 약점

세부 오차(D) 43건 중 **28건이 신체 위치**: sole·buttock·thigh·flank·labia majora·목
옆 등 온갖 부위가 전부 **"iliac fossa(장골와)" 하나로 기본 응답**(default attractor).
위치가 아닌 내용(뺨, 이두근, 갑상연골, 증상 서술)은 거의 완벽. → "잘못 읽는다"가 아니라
"**신체 위치 속성의 해상도가 낮고 애매하면 iliac fossa로 기본 응답**"으로 좁혀짐.

---

# 6. 종합 — "근거 있는 판독"이 4중으로 증명됨

| # | 증거 | 핵심 수치 |
|---|---|---|
| 1 | OOD 일반화 (v4) | 학습에 없던 cue를 L32 55.7% / L24 73.1% 의미 수준 읽기 |
| 2 | 반(反)암기 | 출력이 train cue보다 unseen gold에 더 가까움(L24 63%) + layer 대조 |
| 3 | Vanilla 대조 | 내용은 원래 있었고, LoRA는 그걸 정제(증류)함 |
| 4 | Counterfactual | swap 추적 99.3%, 암기 0%, phantom ~0.3% |

특히 4번은 **인과적** 증거라 급이 다르다 — probe/SAE로는 원리적으로 불가능한
"개입하면 판독이 따라 움직인다"는 직접 증명.

**한 문장 요약:**
> 의료 LLM의 오답은 "몰라서"가 아니라 내부에 정답 신호가 있는데 출력에서 사라지는
> 경우가 많다. 우리는 그 내부 증거를, 근거 있게(grounded), 학습에 없던 케이스에도
> 자연어로 읽어내는 판독기를 만들었고, 그것이 암기·문맥의존이 아님을 개입 실험으로
> 증명했다. 그리고 궤적 지도를 완성했다: 판독 가능성은 cue 위치에서 layer 24에 정점을
> 찍고 답 직전에 사라지며, 답 위치에는 어느 depth에서도 증거가 아닌 결론만 남는다.

**정직한 한계 (발표에서 먼저 밝힐 것)**
- 판독 라벨은 단독 채점자(우리) 수동 분류
- L16 결론은 "정보 부족"과 "L32-AV에서 먼 layer라 LoRA 적응이 어려움"이 섞임(미분리)
- 신체 위치 속성 해상도 낮음
- 아직 AV-only (의도된 선택 — 7장)

---

# 7. 왜 아직 AV만 하는가 (의도된 설계이지, 미완이 아님)

| | AV (활성→언어) | AR (언어→활성) | full NLA |
|---|---|---|---|
| 지금까지 | 학습·검증 완료 | 손도 안 댐 | 아님 |

- 목표가 "근거 있는 자연어 설명"이라 활성값→언어(AV) 방향이면 충분.
- 원래 full NLA 논리는 "AR 복원(MSE)으로 faithfulness를 검증"이었다. 그런데 우리는
  그걸 **counterfactual 개입으로 대신 증명**했다 → 지금 AR이 필요 없다.
- AR을 지금 붙이면 위험: reconstruction만 최적화하면 "그럴듯하나 임상 무의미한"
  텍스트로 수렴한다(vanilla가 정확히 그 상태였음). 읽는 판독기가 먼저 있어야 AR이
  의미가 생기고, 그 판독기를 이제 막 확보했다.
- **AR/full NLA는 목표가 아니라 수단.** "출력마다 grounding을 강제"할 필요가 실제로
  생기면 그때 꺼낸다: post-hoc AR 일치도 → reranking → joint(SFT+복원 손실) 순서.

---

# 8. 앞으로의 계획

**~~1순위 — Format-position layer 스윕~~ (2026-08-17 완료 → §4.5)**
결과: 어느 layer도 format 위치를 구제하지 못함(heldout 0.19~0.25 평평). 궤적 figure
완성, 오답노트의 판독 지점 확정(증거=cue 위치 L24, 내부 결론=format 위치 클래스 채널).

**1순위 — 오답노트 (교수님 "설명" 축; 검증된 판독기의 첫 소비처)**
source 모델이 **틀린** 케이스에서:
- cue 위치 판독 → 핵심 증거가 인코딩됐나? (없으면 = "missing cue" 오류)
- 결론 위치 판독(1순위가 정한 layer) → 내부 결론이 뭐였나?
- → 근거 있는 오답노트: "증거 X는 layer L에 있었으나 내부 결론은 Y로 drift"

handoff의 오답 4분류(missing cue / distractor overweighting / late drift /
decoding mismatch)를 실측으로 채운다. ※ 오답이 많은 부분증거(3-cue) 세팅 필요.

**2순위 — 교정 (교수님 "solution" 축)**
`source 답 ≠ 판독`일 때 판독 내용을 주고 재고 유도. baseline은 generic "다시
생각해봐"로 두어 판독 내용의 기여를 분리.

**선택 — Attribute-resolution probe**
"iliac fossa" 약점이 벡터 탓(ankle vs calf를 벡터가 구분 못 함)인지 판독기 탓(벡터엔
있는데 못 꺼냄)인지 분리.

**최종형 (조건부)** 위 결과들이 "grounding을 출력마다 강제"를 요구하면 AR을 학습해
full NLA로, 그리고 layer-conditioned Medical-NLA(공유 디코더 + layer별 어댑터)로 확장.

---

# 9. 예상 질문 대응 (Q&A 준비)

**Q. 그거 결국 그냥 좋은 분류기 아닌가?**
A. in-distribution에선 맞다(probe도 99%). 그래서 진단-heldout으로 시험했더니 기존
Medical-AV는 0%로 붕괴했다(=분류기 확정). 새 판독기는 학습에 없던 cue를 55~73% 읽고
(v4/layer sweep), cue를 바꾸면 판독이 99% 따라가며 암기는 0%(counterfactual). 분류기는
이 세 시험을 통과할 수 없다.

**Q. 오류 탐지 AUROC 0.94가 결과 아닌가?**
A. 그건 probe로도 1.0이 나온다(실험 A). 오류 탐지는 NLA 기여가 아니라고 우리가 먼저
밝혔다. NLA의 기여는 탐지가 아니라 **근거 있는 자연어 서술**이다.

**Q. cue를 읽는 게 무슨 의미가 있나? 결국 프롬프트에 있던 단어 아닌가?**
A. cue는 최종 목표가 아니라 "벡터를 읽는지"를 채점하는 고엔트로피 supervision이다.
핵심은 (1) 학습에 없던 cue를 읽고(암기 아님), (2) cue를 바꾸면 따라가고(문맥 아님),
(3) 그 판독 가능성이 layer별로 다르다(궤적)는 것. 이건 "내부 증거를 근거 있게
서술한다"는 능력의 증명이고, 그 위에 오답노트/교정이 선다.

**Q. 왜 layer 24가 최고인가? layer 32가 NLA의 원래 층 아닌가?**
A. 맞다, 그래서 놀라운 결과다. L24는 L32보다 학습을 덜 하고도 이겼다. 해석: 개별
증거는 중간층(L24)까지 살아있고 답 직전(L32)엔 결론으로 압축돼 사라진다. 이게 궤적의
핵심.

**Q. AR 없이 full NLA라고 할 수 있나?**
A. 아직 full NLA가 아니라 AV-only가 맞다. 다만 full NLA가 하려던 일(faithfulness
검증)을 개입 실험으로 대신했다. AR은 필요가 생기면 붙일 수단이지 목표가 아니다.

---

# 부록 A. 전체 수치 한눈에

| 실험 | 지표 | 값 |
|---|---|---|
| 기존 (시작점) | probe 진단분류 | 99.17% |
| 기존 | Medical-AV answer_hit | 86.96% |
| 기존 | disagreement AUROC | 0.9427 |
| 실험A probe대조 | probe disagree AUROC (paired) | 1.0000 (vs NLA 0.9282) |
| 실험B OOD | test_seen / test_heldout answer_hit | 90.37% / 0.00% |
| 실험B | heldout이 train클래스명인 비율 | 98.75% |
| v3 cue-first | heldout cue_recall | 0.19 (실패) |
| v4 cue-position L32 | 의미읽기 A+B | 55.7% |
| layer sweep | 의미읽기 L16/L24/L32 | 34.0 / 73.1 / 55.7% |
| counterfactual L24 | swap 추적 / 암기 / phantom | 99.3% / 0% / ~0.3% |

# 부록 B. 실험 코드/문서 위치 (repo)

- `docs/session_handoff_2026-08-01.md` — 배경
- `docs/results_2026-08-13_ood_and_probe_controls.md` — Part 1 통제 3종 + v3
- `docs/results_2026-08-16_v4_cue_position.md` — Part 3 원인 특정
- `docs/results_2026-08-17_layer_sweep.md` — Part 4 layer 곡선 + vanilla 대조
- `docs/results_2026-08-17_counterfactual_faithfulness.md` — Part 5 faithfulness
- `EXPERIMENTS.md` §8–§14 — 전 실험 서버 실행 runbook
- `results_snapshot/*_hand_labeled.jsonl` — 수동 분류 라벨 (재검증 가능)
