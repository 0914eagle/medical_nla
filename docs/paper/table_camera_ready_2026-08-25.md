# Camera-ready tables, v2 (2026-08-25)

**현재 표의 정본이다.** 단, 아래 `MCR 결론 판독` subsection은 실패 원인을
보존한 감사 기록이며 표의 근거가 아니다. 그 실행의 `.052`, `.034`, `6×`
수치는 source-misaligned target에서 나온 무효 결과로 인용하지 않는다.
문서 역할과 제출 전 관문은 `README.md`를 따른다.

설계 규칙 (v1의 실패에서):
- **표 하나 = 지표 하나.** 단위가 다른 값은 같은 열에 두지 않는다.
- **열 = 조건/방법, 행 = 측정 대상.** 파생 통계(차이, 배수)는 별도 열이
  아니라 본문 또는 명시된 Δ열.
- 셀 하나에 값 하나. 슬래시로 두 값을 넣지 않는다.
- 숫자 열에 텍스트 금지. 정의되지 않는 칸은 – 와 표 각주.
- 소수 자리 통일 (비율 .xxx, pp는 정수 또는 x.x).
- **캡션이 계기를 명시한다.** T1은 자연어 판독, T3은 프로브, T3b는 채널별.
  같은 절 안에서도 어느 계기가 잰 값인지 독자가 표만 보고 알아야 한다.

---

## Table 1 — Instrument validation (§4.1)

지표: 비율 하나로 통일. 각 행에 기준값(우연 또는 무학습)을 **자기 열**로.
길이(1,557→52자)와 형식(0.05→1.00)은 단위가 달라 **본문 문장으로** 이동.

**Table 1.** Validation of the readout as a measuring instrument. Each row is
one test; Reference gives the value the test must beat (chance) or the
untuned checkpoint's score on the same items.

| Test | n | Readout | Reference | Reference type |
|---|---:|---:|---:|---|
| Swap tracking | 438 | .993 | — | (higher is better) |
| Context memorization | 438 | .000 | — | (lower is better) |
| Cross-patient contamination | 438 | .007 | .015 | chance |
| Description precision | 438 | .671 | .075 | untuned |
| Held-out cue content match (shuffle control) | 770 | .751 | .096 | shuffled pairing |
| Held-out cue content match (untuned control) | 770 | .751 | .725 | untuned |
| Conclusion at the answer position | 229 | .651 | .603 | untuned |

*Format compliance and output length are reported in the text (0.05 → 1.00;
1,557 → 52 characters): they establish that the readout is machine-scorable,
not that it is faithful. Swap tracking / memorization are the core: editing
one finding moves the description 99.3% of the time and never leaves the
original wording behind.*

**채점자 표기 (08-25 결정, 08-24 적용 확대)**: 438행 의미 채점의 L24
A+B는 .731(A/B/C/D 4등급, A+B를 성공으로)이다. Table의 .751은 별개의
770행 기계 채점이므로 둘을 섞지 않는다.
**같은 보류가 Figure 2에도 걸린다** — 층별 A+B(.340/.731/.557)는 같은
438행의 같은 채점이므로, 그림 각주에 "hand-labeled"이라고 쓰면 보류하기로
한 귀속을 단언하게 된다. 그래서 그 줄은 그림에서 뺐고, 히트맵은 기계 채점
한 자로만 그린다. **채점자 신원·인원 표기는
비워 두고, 외부 API 판정자를 확보하면 그 판정으로 대체한다** — 사람
2차 채점이나 자기 일치율로 메우지 않는다. 그때까지 본문에는 수치만 싣고
채점 주체는 서술하지 않으며, 최종 원고에서 판정자 절차로 채운다.

**답-위치 행이 이 표에서 가장 이상하게 읽히는 행이고, 그것이 요점이다
(08-24).** 다른 행과 달리 Reference가 Readout에 가깝다 — 무학습
체크포인트도 정답을 잃은 케이스의 60.3%에서 정답을 짚는다. **결렬은 우리
어댑터의 산물이 아니라 활성값의 성질이다**, 라는 뜻이고 이쪽이 더 강한
결과다. 본문은 여기에 두 번째 수치를 붙인다: vanilla는 판독 하나당
진단명을 1.15개 부르고 v2는 1.02개를 부르므로, 포함 검사로는 vanilla가
유리하고 이름 하나당으로는 .524 vs .638로 뒤집힌다. **어댑터가 사는 값은
적중률이 아니라 정밀도다** — 이 표의 나머지 행이 전부 그 이야기다.

**T1의 정본 행은 현재 전부 DDXPlus다.** MCR에서는 source-aligned
answer-position 판독까지 완료됐지만, gold/source-answer agreement와 낮은
grounding을 재는 별도 실험이므로 아래 DDXPlus 계기 검증 행에 섞지 않는다.
cue-position 계기 검증은 여전히 MCR용 어댑터·판독 실행이 필요하다.

| 행 | MCR | 경로 |
|---|:-:|---|
| 서술 정밀도 · 오염 · heldout 서술률 | ✔ | 판독 실행 후 기존 분석기 그대로 |
| 스왑 추적 · 문맥 암기 | △ | `make_span_counterfactual_rows.py`(산문용 span 치환, 미실행) — cue 축자 등장 필요·탈락률 미측정·비문 캐비앳 |
| unseen-cue 서술 | ✕→△ | MCR 분할은 케이스 분할이지 cue 문자열 heldout이 아님 — 분할 재구성 없이는 정의 안 됨 |
| 답 위치 결론 | △ | source-aligned 판독 완료; derangement 통제와 무학습 대조가 남음 |

▢ 남은 핵심은 외부 판정자의 438행 의미 재채점과 MCR cue-position 검증이다.

### ⚠️ MCR 결론 판독 실행 완료 (08-24) — 비율을 내기 전에 눈으로 본 것

821행 held-out(sft_test, base_id 필터 확인, 소견서 없는 프롬프트 0건).
**여섯 건을 읽었고, 결과는 이 칸을 단순한 서술률로 채우면 안 된다고 말한다.**

`<answer>`는 임상적 이웃까지 간다 — 무작위가 아니다:

| gold | 판독 | |
|---|---|---|
| Extraosseous peripheral calcifying odontogenic cyst | Oral fibroma | 구강 결절, 다른 병리 |
| Solid pseudopapillary epithelial neoplasm | Gastric duplication cyst | 위벽 부착 낭성 종괴 |
| Rosai-Dorfman Disease | IgG4-related disease | 조직구/섬유염증, 실제 감별 |
| proximal tibiofibular joint **osteoarthritis** | tibiofibular joint **dislocation** | **같은 관절** |
| Guillain-Barre syndrome | Guillain–Barré syndrome | **정확** |
| neurogenic **pulmonary** edema | **cerebral** edema | 장기가 틀림 |

`<supporting_cues>`는 **프롬프트에서 뽑힌 것이 아니라 생성된다**:

- 33세 케이스를 "A 50-year-old woman"으로, 68세 폐이식 케이스를
  "62-year-old man with type 2 diabetes"로 쓴다.
- 프롬프트가 "scattered calcifications"인데 근거는 "**no** calcifications".
- **진단명을 맞힌 GBS 케이스**조차 "lumbar puncture revealed **normal**
  cerebrospinal fluid"라고 쓴다 — 실제 프롬프트는 CSF 단백 76 mg/dL, 즉
  알부민세포해리를 정반대로 적었다.
- 무릎/근위경비관절 케이스의 근거가 전부 **발목**이고 좌우도 뒤집힌다.
- 상용구가 반복된다: "Laboratory studies showed a normal complete blood
  count and metabolic panel", "He had no significant medical history".

즉 어댑터가 하는 일은 **해당 전문과의 일반적 증례보고 워크업 작성**이다.
DDXPlus에서는 소견 어휘가 닫혀 있어 이 실패가 가려졌다 — 그럴듯한 DDXPlus
문장을 쓰면 자주 실제로 일치했다. **열린 산문에서 드러났다.**

**논문에 미치는 영향 셋.**
1. 이 칸은 서술률 하나로 못 채운다. 근거를 지어내고 진단명을 맞힌 판독은
   상태를 읽은 판독과 같은 점수를 받는다.
2. **caveat 1(내부 해부는 DDXPlus 한정)이 강해진다.** 이제 "프로브가 열린
   어휘에서 정의 안 됨"이라는 소극적 근거가 아니라, **판독의 근거가 실제로
   전이되지 않는다는 적극적 증거**가 있다.
3. 어댑터가 연 것은 `<answer>` 칸이지 `<supporting_cues>` 칸이 아니다.
   이전에 "NL 판독 열을 연다"고 적은 것을 이 범위로 좁힌다.

**측정 결과 (08-24, `scripts/analyze_readout_grounding.py`)**. 근거 문장을
자기 프롬프트와, **무작위로 짝지은 남의 프롬프트**와 각각 trigram 겹침으로
대조한다. 두 값의 **차이**만이 케이스 고유 정보다.

| 판독 | gap | 접지율 | 근거/행 | 태그 |
|---|---:|---:|---:|---|
| DDXPlus cue 위치 v2 (알려진 양성, 서술률 .751) | **+.094** | .161 | 1.0 | 770/770 닫힘 |
| **MCR 결론** | **+.034** | .023 | **12.6** | **444/821 절단** |
| vanilla L32 (알려진 음성) | +.000 | .000 | 1.0 | **765/770 태그 없음** |

**지표는 자기 검증을 통과했다** — 알려진 양성 > MCR > 알려진 음성 순으로
갈린다. vanilla는 스키마 자체를 못 내고, 낸 5건은 템플릿 문구를 되뇐다.

**⚠️ 두 가지를 정정한다 (같은 날, 이 표를 만든 뒤).**

1. **절대값은 해석 금지.** DDXPlus 튜닝판은 의미 채점 서술률 .751인데 이
   지표로는 접지율 .161이다. 레지스터가 바뀌면 trigram이 안 맞으므로,
   사람이 맞다고 보는 근거의 다수를 이 지표가 놓친다. **"MCR 근거의
   2.3%만 접지"를 "97.7%가 거짓"으로 읽으면 안 된다.** 상대 비교 전용.
2. **상용구 반복률 비교는 철회한다.** DDXPlus 98.6% > MCR 51.8%다.
   DDXPlus 소견은 닫힌 템플릿 어휘라 같은 문장이 여러 환자에게 실제로
   등장한다 — 두 코퍼스의 반복률은 애초에 비교 대상이 아니었다.

### 08-24 후속: 짝 맞춘 비교와, 위 관측 두 개의 철회

**짝이 맞춰졌다.** `readout_hint_final_L32_v2`는 DDXPlus·답 위치·같은 스키마
— MCR 결론 판독과 코퍼스만 다르다.

| 판독 | 근거/행 | 절단 | gap | 접지율 |
|---|---:|---:|---:|---:|
| DDXPlus 답 위치 v2 (n=3,494) | 3.0 | **0** | **+.100** | **.311** |
| MCR 결론 (n=821) | 12.6 | **444** | +.034 | .052 |

(`trajectory_L32_v2`의 gap +.035는 비교 대상이 아니다 — 6개 지점 중
format·constraint처럼 임상 내용이 없는 자리가 섞여 희석된 값이다.)

**철회 A — 근거/행 12.6은 실패가 아니다.** 학습 타깃이 평균 **10.1개**,
중앙 9개다(sft_train 10,663행). 어댑터는 배운 대로 한다. "워크업을 쓴다"는
표현은 개수에 관한 한 틀렸다.

**철회 B — 절단 54%는 모델이 아니라 우리 생성 예산이다.** 타깃이 평균
**764자**, 19%가 1,000자 초과, 최대 2,507자인데 판독을 `max_new_tokens=256`
으로 돌렸다. **타깃 자체가 예산을 넘는다.** 어댑터가 폭주한 게 아니라 우리가
말을 끊었다. `run_nla.py`에 `--max-new-tokens`를 추가하고 스크립트 기본값을
768로 올렸으며, 옛 출력은 더 작은 예산으로 만들어졌으면 건너뛰지 않고
`.bak`으로 밀어낸다. **이 실행의 어떤 비율도 재실행 전까지 쓸 수 없다.**

**살아남은 것, 그리고 이제 상한이 있다.** 타깃 근거는 프롬프트 축자 스팬이라
**타깃 접지율이 정확히 1.000**이다(≥50% 비율도 1.000). 두 가지가 따라온다:

1. **지표가 이 데이터에서 정상 작동한다.** 완벽히 접지된 텍스트에 1.000을
   준다. 지표를 버릴 이유가 없다.
2. **상한 1.000 대비 DDXPlus .311, MCR .052 — 6배 차이.** 이것이 남는 관측이고,
   절단된 행만의 문제도 아니다: 태그를 정상으로 닫은 눈검사 케이스도 근거를
   전부 지어냈다("On extraoral examination, there was no lymphadenopathy" —
   프롬프트에 없는 문장).

### 🛑 08-24 최종: `.052`는 MCR에 대한 발견이 아니었다 — 학습 데이터의 귀결

**MCR에서 소스 모델 정확도가 0.122다** (12,620건 중 1,543건). 그런데
`make_mcr_conclusion_split.py`가 파일럿 규칙 — *"gold는 모델이 그것에
도달한 곳에서만 판독 타깃이 될 수 있다"*, DDXPlus 쪽 구현은
`make_medical_nla_v2_source_aligned_splits.py` — 을 빠뜨렸다.

| split | 행 | 맞힌 것 | **틀린 타깃** |
|---|---:|---:|---:|
| sft_train | 10,663 | 1,298 | **9,365 (88%)** |
| sft_val | 1,136 | 132 | 1,004 |
| sft_test | 821 | 113 | 708 |

학습 행 열 개 중 아홉에서 활성값은 A를 담고 정답지는 B였다. 그 조건에서
손실을 낮추는 방법은 상태를 읽는 것이 아니라 **문맥으로 그럴듯한 진단을
짓는 것**이고, 눈검사가 본 것이 정확히 그것이다 — 전문과는 맞고, 진단명은
이웃이고, 근거는 그 전문과의 평균 워크업.

**DDXPlus가 이 누락을 견딘 것은 우연이다.** 49클래스에서 모델이 대부분
맞히므로 gold와 결론이 대체로 일치한다. MCR에서는 같은 누락이 아예 다른
실험이 된다.

**따라서 이 절의 앞선 표와 결론은 전부 무효다.** `.052`, gap `+.034`,
6배 비교 — 잘못된 타깃으로 학습한 어댑터를 잘린 예산에서 잰 값이다.
caveat 1을 이 숫자로 강화하려던 계획은 **철회**한다.

**고친 것**: `--answers`로 source-wrong 행을 train/val에서 제거(test에는
남긴다 — 거기서는 상태와 출력의 불일치가 결함이 아니라 측정 대상이다),
모든 행에 `source_correct` 기록, 필터 없이 만든 split과 그것으로 학습한
어댑터는 `.bak`으로 밀어내고 재구축·재학습.

### 판독 재실행의 사전 판정 기준 (08-24 18:42, 실행 전 기록)

정당한 학습셋(1,298행, 100% 정합)으로 학습한 어댑터를 예산 768로 held-out
821행에 돌리기 전에 기록한 기준이다. 실제 결과는 바로 아래 절에 보고한다.

기준선 셋 — 상한은 **1.000**(타깃이 프롬프트 축자), 같은 위치·같은 스키마의
DDXPlus는 **gap +.100 / 접지 .311**, 무효 실행이었던 옛 MCR은 **+.034 / .052**.

| 나오는 값 | 결론 | 논문에서 하는 일 |
|---|---|---|
| gap ≥ **+.08** | 열린 어휘에서 판독이 **된다** | Table 1에 MCR 행 추가, caveat 1을 "프로브만 정의 안 됨"으로 축소, 대전제 그대로 |
| gap **+.04–.08** | 되지만 약하다 | MCR 행을 싣되 DDXPlus 대비 열세를 본문에 명시 |
| gap ≤ **+.04** (옛 값과 같음) | **학습셋 결함이 원인이 아니었다** | Limitations 결과로 전환: "이 코퍼스가 줄 수 있는 지도 신호(1,543건, 진단명당 1회)로는 학습되지 않는다". §4.1의 야심을 DDXPlus로 한정 |

**옛 값과 같게 나오는 것이 실패가 아니다.** 그건 원인을 하나 배제한 것이고,
남는 설명은 과제 난이도(활성 벡터 하나에서 764자 원문 복원)와 지도 신호
부족이다. 둘 다 정량화돼 있으므로 Limitations가 변명이 아니라 측정이 된다.

**절단 수를 먼저 본다.** 로그 끝의 `readouts still cut off mid-cue`가 ~0이
아니면 gap을 읽지 않는다 — 예산이 또 모자란 것이고, 그 위에서 잰 값은
무엇이든 무효다.

**그리고 비율보다 눈검사가 먼저다.** 첫 실행에서 여섯 건을 읽어 잡은 것이
학습셋 결함이었다. 같은 세 건(odontogenic cyst / SPEN / Rosai-Dorfman)을
다시 읽어 근거가 프롬프트 문장으로 바뀌었는지 본다. 안 바뀌었는데 gap만
올랐다면 지표가 무언가 다른 것을 재고 있는 것이다.

**재학습 뒤 생긴 질문**: 살아남는 학습 행이 **1,298개**이고 MCR
진단명은 6,934종, 대부분 1회 등장 — **진단명당 예시 한 개**다. 될지 안 될지는
우리 자신의 결과가 가른다: 답 위치 **무학습 .603 대 v2 .651**, 즉 읽는 능력은
AV 체크포인트에 있고 LoRA는 형식을 입힌다. 형식만 가르치는 것이면 1,298행으로
될 수 있다. **그래서 재학습과 무학습 대조를 같이 돌린다** — Table 1의 다른
모든 행이 갖고 있는 Reference 열이 이 행에도 필요하다. 무학습이 바닥이면
"MCR 결론 판독은 우리가 감당할 수 없는 학습량을 요구한다"가 **Limitations에
적을 결과**이지 채울 구멍이 아니다.

**MCR 정확도 0.122는 그 자체로 본문에 적어야 한다.** Table 2의 MCR 행
n=1,543은 12,620건 중 모델이 맞힌 12.2%이고 — 파일럿 규칙이 거기서는 제대로
적용됐다 — 그 사실이 MCR 결과 전체의 모집단을 규정한다.

source-aligned 실행에서 근거 접지 gap은 +.025에 머물렀다. 따라서 현재 결과는
"더 학습하면 된다"는 결론이 아니라, 결론 일치와 근거 접지를 분리해 평가해야
한다는 근거다. 과거 10,663행 실행은 source-wrong target을 포함한 무효 실행이므로
학습 loss나 best epoch를 정본 결과로 인용하지 않는다.

---

### 08-24 source-aligned 재실행의 현재 결론

held-out 821행에서 판독은 gold보다 모델의 실제 답과 더 자주 일치했다. 전체
`.1389 vs .2643`, source-wrong 708행에서는 `.0692 vs .2133`이다. 이 3.1배
비대칭은 source-aligned 결론 판독과 일관되지만, 무작위 다른 케이스의 모델 답과
비교하는 derangement 통제가 남아 있어 최종 충실성 결과로 부르지 않는다.
근거 접지 gap은 `+.025`로 낮아 `<supporting_cues>` 전이는 지지되지 않는다.

---

## Table 2 — Intervention accuracy (§4.2)

행 = 코퍼스, 열 = 조건. 지표는 정확도 하나. 케이스 수·낙폭은 본문.

**Table 2.** Accuracy by arm on a cohort originally selected as source-correct
under the generation-time matcher and then rescored with the canonical
word-boundary matcher. Finding-position activations are bit-identical across
arms by construction. Canonical no-note accuracy can therefore be below 1.

| Corpus | n | No note | Neutral | Wrong | Correct |
|---|---:|---:|---:|---:|---:|
| DDXPlus | 1,220 | **.9869** | **.9377** | **.7566** | **.9246** |
| DDXPlus, 3× larger run | 3,343 | **.9800** | **.9306** | **.7670** | **.9180** |
| MedCaseReasoning | 1,543ᵉ | **.9410** | **.8879** | **.6721** | **.8179** |

ᵉ **모집단을 본문에 밝힌다 (08-24).** 1,543은 MCR 12,620건 중 소스 모델이
소견서 없이 맞힌 전부다 — **정확도 0.122**. 이 행은 "MCR의 12%"가 아니라
"모델이 맞힐 수 있는 MCR"에 대한 결과이고, 그 조건은 DDXPlus 행과 같다
(소견서로 답이 바뀌었음을 보이려면 원래 맞았어야 한다). 다만 절대 난이도는
전혀 다르며 — DDXPlus 49클래스 대 MCR 6,934개 진단, 대부분 1회 등장 —
그 차이가 §4.1 판독 실험이 MCR에서 왜 다른 문제인지를 설명한다.

08-24 감사에서 이 칸에 있던 `.932`가 다른 실행·모집단의 값임을 확인했고,
동일 fixed cohort에서 neutral/correct를 다시 실행·재채점해 `.9377/.9246`으로
교체했다.

*Under the canonical matcher, the wrong note costs 23.0 pp on the main
DDXPlus run and 26.9 pp on MedCaseReasoning. In the 3× larger DDXPlus run it
costs 21.30 pp against a 4.94 pp neutral cost: a 16.36 pp suggestion-specific
effect and a 4.31× total-cost ratio. MCR's corresponding values are 21.58 pp
and 5.06×. In the main DDXPlus run, the neutral cost is 4.92 pp and the
suggestion-specific effect is 18.11 pp (4.68× total-cost ratio). A correct note
still costs 6.23 pp, showing an intrusion cost independent of suggestion
direction.*

**두 번째 행은 손실이 아니라 소득이다 (08-24).** corpus-300은 네 조건을
자기 안에 다 갖고 있어 **자급자족하는 재현 행**이고, 전체/위약 배수가
4.31×로 같은 구조를 낸다. 정답 조건 칸이 비어 있는
동안에도 "침입 비용은 제안 방향과 무관하다"는 논증은 이 행과 MCR 행에서
이미 선다 — 정답을 부르는 소견서조차 DDXPlus에서 6.20pp(.9800→.9180),
MCR에서 12.31pp(.9410→.8179)를 깎는다. 단, corpus-300은 원 실행의
초집합이므로 **independent replication**이 아니라 3× larger run이다;
non-overlap-only 민감도 분석이 남아 있다.

**모든 DDXPlus 비율은 보수적 하한이다.** 답 파일이 `plausible_wrong`
수정(d29b754) 이전에 생성되어, 오답 소견서가 정답을 부르는 케이스가
1,747건 중 15건, 4,995건 중 44건 남아 있다(각각 0.86% · 0.88%). 그런
소견서는 사실상 정답 조건이라 오답 조건의 정확도를 **올리는** 쪽으로
작용한다. 즉 편향이 우리에게 불리하므로 전집합 수치를 그대로 싣고 각주로
밝힌다; `analyze_hint_effect.py --exclude-collisions`가 반대쪽을 준다.

**Table 2b.** Where the moved answers go.

| Corpus | Moved | To the suggestion | To a third diagnosis |
|---|---:|---:|---:|
| DDXPlus | **321** | **91** | **230** |
| MedCaseReasoning | **437** | **137** | **300** |

**Table 2c** (지면 되면; 아니면 부록). Speaker/wording variants, DDXPlus,
n = 1,747 each.

⚠️ 아래 행은 generation-time matcher 값이다. canonical 재집계 전에는
camera-ready 표에 싣지 않는다.

| Wording | Accuracy | Moved | Adopted |
|---|---:|---:|---:|
| Referral note (one line) | .814 | 324 | 95 |
| Colleague | .821 | 305 | 107 |
| Patient | .867 | 224 | 17 |
| Realistic multi-sentence note | **.745** | **445** | **236** |

**Table 2d** (2c와 한 쌍 — 같은 해리의 두 번째 조작). Suggestion source on
MedCaseReasoning: the model's own confusions vs. a cue-similar neighbour's
diagnosis (no differential field exists).

| Suggestion source | n | Wrong-arm acc. | Moved | Adopted |
|---|---:|---:|---:|---:|
| Model's own confusion | 849 | .682 | 257 | 106 |
| Nearest-neighbour diagnosis | 694 | .728 | 182 | 32 |

*Destabilisation barely differs (30.3% vs 26.2% moved, z = 1.75, n.s.);
persuasion differs 2.3× (41.2% vs 17.6% adopted, z = 5.26). The same
dissociation the patient-voice wording shows in 2c, from an unrelated
manipulation: persuasion tracks the note's properties, destabilisation
mostly does not.*

---

## Table 3 — What the note does inside (§4.3)

**08-24 신설.** 이 논문의 중심 주장이 지금까지 표가 없이 Figure 4에만
있었다. 그림은 정확히 인용되지 않고, 관성 반론을 닫는 것이 이 세 줄이므로
표가 있어야 한다. 지표 하나(최종 토큰에서 프로브가 정답에 주는 확률),
셀당 값 하나, Δ는 명시된 파생 열.

**Table 3.** Probability the cross-fit linear probe places on the gold
diagnosis at the final token, by what the model then did. "No note" reads the
same cases with the note removed; finding-position activations are identical
across the two by construction, so Δ is the note's internal cost.

| Behaviour under the wrong note | n | With the note | No note | Δ |
|---|---:|---:|---:|---:|
| Answer unchanged | **1,426** | **.980** | **.987** | **−.007** |
| Lost the gold, answered elsewhere | **230** | **.880** | **.934** | **−.055** |
| Adopted the suggestion | **91** | **.725** | **.919** | **−.195** |

*The canonical trajectory contains 321 moved cases. The suggestion is probe
top-1 at least once in 55 (17.1%); seven of those already decode to it before
the note. In 266 (82.9%) it is never top-1. That last group must not be called
"gold throughout": 151 cases keep gold top-1 at every observed landmark,
while 115 pass through another diagnosis without ever making the suggestion
top-1. At the last finding the paired cost
is zero by causal masking; the note token itself has no no-note counterpart
and its paired cost is undefined.*

*Panel (a) highlights the internal-output mismatch directly: even among cases
that emitted the suggestion, the final-token mean is
`p(gold)=.725` versus `p(suggestion)=.211` (about 3.4×). These are group means
from a 49-way probe, not per-case knowledge, model next-token probabilities, or
calibration claims. Panel (b) is non-monotonic: the paired gold
cost is largest around the constraint and partially recovers at the final
token (`adopted: −.439→−.195`; `lost: −.304→−.055`). This identifies the
instruction segment as the most vulnerable observed landmark under this L32
prompt skeleton, but each landmark uses a separately trained probe. The
recovery was insufficient for correct output; it does not establish a causal
failure to transmit the recovered signal. In panel (c), `note=0` is an observed
count and is printed explicitly; this differs from panel (b), where the note
delta is `N/A` because the no-note arm has no matching note token.*

*At the note landmark, the gold-label probe reports `p(suggestion)=.000` at
display precision in all three groups. This means the suggestion is not
decoded as a diagnosis by this probe at that point; it does not prove that
suggestion information is absent from the activation. A hint-label probe or
matched retrieval test would be required for that stronger claim.*

**계기 표기**: 이 표는 **프로브**다. 자연어 판독은 같은 방향을 독립적으로
말하지만 값이 다르다 — 상실형 최종 토큰에서 "상태가 정답을 쥠"이 프로브
.904, v2 판독 .651, 무학습 판독 .603 (세 계기 모두 같은 229건). **결렬의 존재는 두 계기가, 정밀한
해부는 프로브만 말한다**는 것을 본문이 그대로 밝힌다. 프로브는 닫힌
49클래스에 학습된 분류기이므로 **동일한 probe를 MCR에 직접 이전할 수
없다** — MCR 내부 기전은 미측정이며, 다른 open-vocabulary representation
baseline의 가능성까지 부정하지 않는다.

---

## Table 3b — Single-run attribution (§4.3)

셀당 값 하나: All / Silent를 **열 두 개**로. MCR은 숫자 열이 아니라
**적용 가능 여부 열**로 — 값이 아니라 정의의 문제라서.

**Table 3b.** Within-diagnosis AUROC for identifying moved cases from the
wrong-note run alone. All: n=1,747. Silent: n=1,641 (218 moved), restricted to
cases whose answer does not name the suggestion, where the output-copying
heuristic is blind by construction. `Task supervision` makes explicit that
the fixed-class probe and the text channels do not operate under identical
assumptions.

| Channel | Input access | Task supervision | AUROC, all | AUROC, silent |
|---|---|---|---:|---:|
| Answer names suggestion | Output text | none | **.6610** | n.a.ᵃ |
| Best rule-based CoT feature | CoT text | none | **.5464** | not reportedᶜ |
| LLM monitor | Vignette + note + CoT + answer | external LLM | **.7233** | **.6829** |
| NL activation readout (ours) | Hidden state → text | readout adapter | **.7506** | **.8302** |
| Linear diagnosis probe | Hidden state | fixed 49-class labels | **.9280** | **.9840** |

ᵃ Undefined because this feature defines the silent subset. ᵇ The same fixed
49-way probe does not directly transfer to an
open diagnosis vocabulary, although binary and retrieval baselines remain
possible. Text channels require no fixed class list in principle; their MCR
performance is not established by this table.
ᶜ The canonical ledger retains the rule feature's all-case score and its
paired gap against the readout, but not a standalone silent value; do not
reconstruct it by subtraction from rounded numbers.

**LLM 모니터 행 (08-24 실측, gpt-5.6-sol, 1,747/1,747 파싱, 실패 0).** 이
행이 §4.2의 주장을 바꾼다. 규칙 기반 특징 .53 → 프런티어 모니터 .7233이면
차이가 크므로 **"체인은 귀속 신호를 담지 않는다"는 더 이상 못 쓴다** — 우리
채점기가 약한 부분이었고, 같은 체인에서 강한 독자는 실제로 신호를 끌어낸다.
주장은 이진에서 정량으로 바뀐다. 같은 정본 침묵 구역(n=1,641)에서 모니터
.6829 대 판독 .8302(+14.7%p)다. 모니터 자체의 다른 침묵 정의(n=1,656)에서는
.6930이다. `.664 → .7233`은 강한 텍스트 모니터 전체와 단순 출력 휴리스틱의
차이지, **CoT만의 순수 증분이 아니다** — 그것은 동일 판정자의 no-CoT arm이
필요하다.
모니터는 소견서를 보므로 "답 == 제안"을 스스로 계산할 수 있고, all 열의
일부는 그것이다 — 그걸 뺀 값이 silent 열이다. 루브릭은 판정자에게 유리하게
짰다: "답이 제안과 달라도 움직인 것일 수 있다"를 명시적으로 알려주므로,
낮은 값이 루브릭 결함이라는 반론은 닫힌다.

**판정자가 실제로 확률을 썼다 (08-24).** 서로 다른 값 61개, 0.00–1.00 전
구간, 평균 0.149(실제 답 바뀜 비율 0.185). 형식만 확률이고 실질은 3단계인
경우가 아니므로 **동점으로 AUROC가 깎인 것은 아니다**. 다만 분포가 두 번째
사실을 말한다: P>0.5가 222건이고 canonical moved는 321건이다. 이 두 수치만으로
calibration이나 유병률 추정을 결론내릴 수 없다;
그 주장은 Brier/ECE 또는 calibration curve가 있어야 한다.

### ✅ 부트스트랩 CI 완료 (08-24, `bootstrap_channel_gap.py`)

케이스 단위 **쌍 부트스트랩**, 각 추출 안에서 진단 내 층화, 2,000회.
쌍으로 뽑는 이유는 두 채널이 같은 환자를 채점하므로 표본에 어떤 진단이
들어오느냐에 따라 함께 오르내리기 때문이다 — 비쌍 구간은 실제 비교보다 넓다.

| 비교 | 부분집합 | 차이 | 95% CI |
|---|---|---:|---|
| 판독 − **LLM 모니터** | 침묵 (1,641, moved 218) | **+.147** | **[+.069, +.221]** |
| 판독 − 체인 특징(최강) | 침묵 | **+.291** | **[+.230, +.354]** |
| 판독 − 출력만 | 전체 (1,747, moved 321) | **+.090** | **[+.026, +.157]** |

세 비교 모두 canonical labels에서 0을 배제한다.

**셋째 줄은 정직하게 적어야 한다** — 하한이 +2.6%p로 가장 아슬하다. 전체
집합에서 판독이 "답이 제안을 말하는가"라는 공짜 특징보다 앞서는 폭은 실재하나
크지 않다. **판독의 자리는 침묵 구역**이고, 거기서 그 공짜 특징은 정의상
AUROC 0.5다.

*침묵 구역 정의가 두 가지다.* 정본 채널 덤프는
`answer_names(wrong, hint)`로 자르고(1,641건), 모니터 라벨 파일은
`took_the_hint`로 자른다(1,656건). 15건 차이는 무소견서 arm이 이미 제안을
부른 진짜 사례이며, 모니터 AUROC는 .6829 대 .6930이다. 채널 간 비교는 모든
채널이 같은 1,641건을 볼 때만 인용한다.

*행 순서는 '내부를 안 보는 것 → 보는 것'으로, 프로브를 맨 아래로 옮겼다.
표가 주장하는 것이 순위가 아니라 **경계**이기 때문이다.*

*CoT 값은 특징 3종(0.50–0.53) 중 최댓값 하나로 통일해 셀당 값 하나 규칙을
지킨다; 특징별 값은 부록. **MCR의 출력 채널 AUROC는 완료된 개입 답에서 지금
CPU로 계산 가능(▢ 최우선); CoT 채널은 MCR CoT 실행이 필요(GPU ~1–2h,
prompt_cot는 케이스 파일에 이미 있음)** — 나오면 "AUROC, MCR (behavioural)" 열을
추가한다: 두 행동 채널 + probe "n.a." + readout "ᵈ"로 열이 완성되고, 열린
어휘에서 행동 채널도 우연 수준이면 "MCR에서는 아직 아무도 탐지 못 한다"가
어댑터 동기의 마지막 조각이 된다.*

---

## Table 4 — Correction ladder (§4.4)

**Table 4.** Second-pass accuracy with the wrong note still in place. Moved:
the canonical 321 causally moved cases. Capitulation: share of newly broken
answers landing on the suggested diagnosis (first-pass counterpart .3209).

| Rung | Appended | Overall | Moved | Capitulation |
|---|---|---:|---:|---:|
| r3 | reconsider request only | .4173 | .4548 | .4507 |
| r4 | + findings re-shown (control) | .4139 | .4050 | .6410 |
| r5 | + readout conclusion & grounds | .4098 | .6293 | .4940 |
| r6 | + probe class label | .4568 | .8318 | .5212 |

*First-pass baseline: overall .8117, moved .0031. r5 − r4 = +22.4 pp on moved;
r5 capitulation is 14.7 pp lower than r4. r3, not r5, has the lowest absolute
capitulation.*

**Table 4a-r7.** Same 1,151 IDs for every rung; r7 is evaluated separately
because it requires agreement between the direct and CoT first answers.

| Rung | Overall second pass | Moved recovery | Newly broken |
|---|---:|---:|---:|
| r3 | .4639 | .5169 | 573 |
| r4 | .4422 | .4494 | 592 |
| r5 | .4049 | .5281 | 643 |
| r6 | .4457 | **.7416** | 615 |
| **r7: own CoT** | **.8810** | **.1236** | **58** |

*The high r7 overall accuracy reflects answer preservation, not correction:
its common-ID cohort is easier (first-pass .9201; moved prevalence 7.7%), and
it recovers only 12.4% of moved cases. This is consistent with answer
entrenchment but does not by itself establish a rationalization mechanism.*

**Table 4d (예정) — 같은 사다리, MedCaseReasoning.** 어느 단이 존재하는지를
코퍼스가 정한다. 이 표의 빈칸은 미실시가 아니라 **결과**다.

| Rung | DDXPlus | MedCaseReasoning | 왜 |
|---|:-:|:-:|---|
| r3 reconsider only | .4548 | ▢ 실행 가능 | 어댑터 불필요 |
| r4 findings re-shown | .4050 | ▢ 실행 가능 | 어댑터 불필요 |
| r7 own chain | .1236ᵈ | ▢ (CoT 실행 필요) | GPU ~1–2h |
| r5 readout conclusion | .6293 | ▢ wrong-note activation 추출 필요 | source-aligned 어댑터 완료; 결론 판독은 예비 신호, 근거 접지는 실패 |
| r6 probe class label | .8318 | **n.a. (현재 설계)** | DDXPlus의 고정 49-class probe를 직접 이전할 수 없음 |

ᵈ DDXPlus r7 is the moved recovery on the 1,151-ID common cohort, not the
full-run Table 4 population.

*r6의 직접 이전 불가가 §4.4의 마지막 질문이다. DDXPlus만 보면 probe가 교정
비교에서 이기고, 독자는 자연어 채널이 잉여라고 결론지어도 좋다. MCR에서는
동일한 고정-class 채널이 없지만, 이것만으로 자연어 채널의 우위를 증명하지
않는다. source-aligned MCR 판독과 open-vocabulary baseline을 실제로 비교해야 한다.
실행: `scripts/run_mcr_ladder.sh` (기본 rungs 3 4 7; 어댑터가 나오면
`RUNGS="3 4 5 7" READOUTS=…`).*

**이 표가 확립하는 명제는 "내부를 되먹여라"이지 "자연어로 되먹여라"가
아니다 (08-25).** r5와 r6이 둘 다 r4(입력 재제시)를 이기고, 내용 정확도를
맞추면 둘 사이 차이는 0이다(4b 1행). 교정 축의 결론은 **채널 중립**이다.
자연어의 잠재적 자리는 열린 진단 어휘와 근거 제시 상황이지만, 전자는 MCR
source-aligned 실험, 후자는 외부 판정 전에는 배포 권고가 아니다(Table 5).

**r7은 가장 명백한 자기설명 경쟁자를 닫는다.** 모델 자신의 CoT를 되먹여도
moved 회복은 12.4%에 그쳤다. 다만 CoT 실행의 답이 direct 첫 답과 다른 케이스를
제외해 모집단이 쉬우므로, 반드시 같은 1,151 id로 제한한 r3–r6과만 비교한다.

**r6은 제안하는 방법이 아니라 통제다 (08-25 명시).** 표를 처음 보는 독자는
r6의 moved .830을 "probe가 이긴다"로 읽고, 곧바로 **"클래스명을 되먹이는
건 정답을 쥐여 주는 것 아닌가"**라고 되묻는다. 그 되물음은 옳고, 수치가
그대로 인정한다: probe argmax의 정답률은 moved에서 **.8567**(전체 .9588),
AV 판독 결론은 **.5047**다. r6의 .8318은 .8567을 거의 그대로
따라간다. r6이 존재하는 이유가 바로 이것이다 — r5가 r4를 이긴 것이
**문장이라서**인지 **내용이 맞아서**인지 가르려면 내용만 있고 문장이 없는
단이 필요했고, Table 4b가 그 교란을 제거한다. 답은 내용이다.

세 가지를 본문에 함께 적는다. ① probe는 오라클이 아니라 활성값만 읽는
**교차적합** 분류기이며 해당 케이스의 정답 라벨을 본 적이 없다 — 배포
시점에 실제로 실행 가능한 채널이므로 정답지 누출이 아니다. ② 그러나 probe는
다른 케이스들의 **정답 라벨로 지도학습**되고 AV 판독은 그 감독을 받지
않으므로, r5 vs r6은 형식만이 아니라 **감독 수준도** 다르다. ③ probe가
정의되는 코퍼스라면 r6은 애초에 쓸 정책이 아니다 — Table 4c에서 재실행
없는 argmax 교체(.966)가 r6 재실행(.954)보다 낫다. r6은 사다리의 통제로서
자기 일을 했고, 배포 권고에는 들어가지 않는다.

**Table 4b.** r5 vs. r6 with fed-back content accuracy held fixed in the
canonical main run. Exact McNemar tests on discordant pairs.

| Content (readout / probe) | n | r5 | r6 | r5-only : r6-only | p |
|---|---:|---:|---:|:-:|---:|
| correct / correct (moved) | 155 | .8774 | .9226 | 0 : 7 | .016 |
| wrong / correct (moved) | 120 | .3500 | .9083 | 0 : 67 | <.001 |
| wrong / wrong (moved) | 39 | .4872 | .3077 | 7 : 0 | .016 |
| correct / wrong (moved) | 7 | .7143 | .4286 | 2 : 0 | .500 |
| correct / correct (all) | 1,158 | .4914 | .4922 | 78 : 79 | 1.000 |

*Across all correct/correct cases, no additional form effect is detected
(p=1.000). Within moved cases, the bare probe label is ahead 7:0 when both
contents are correct, but p=.016 does not pass the Bonferroni threshold .0125
for four simultaneous cells. Do not claim either equivalence or a prose
advantage from this table.*

**Table 4c.** Deployment policies.

| Policy | DDXPlus | 3× replication |
|---|---:|---:|
| Keep first answer | .8117 | .824 |
| Probe selects, argmax replaces | **.9651** | **.973** |
| Probe selects, re-ask r6 | .9531 | .966 |
| Probe selects, re-ask r5 | .9141 | .927 |

---

## Table 5 — Discussion guidance (§5)

**Table 5.** When to use which instrument, from the measurements in §4.

| Setting | Instrument | Basis (§) |
|---|---|---|
| Closed label set, training labels available | Supervised probe | 4.3, 4.4 |
| Open diagnosis space | Source-aligned NL readout is a candidate; conclusion derangement and grounding remain open | 4.1, limitation |
| Clinician-facing grounds required | Do not deploy the current readout; reader-trust is negative in the interim sample | 4.1, limitation |
| Self-correction by re-asking | Neither — avoid | 4.4 |

---

## 그림이 나르는 것 (표에 안 넣는 수치)

- 절대 decoded signal + no-note 대비 paired cost + suggestion-top1 최초 지점
  → **Figure 4**. canonical 321건 중 suggestion top-1 경험 55, never 266.
  never는 gold-throughout 151과 third-diagnosis path 115로 나뉜다. note 이전
  last-finding 7건은 개입 효과가 아니라 baseline differential signal이다.
- 사례 서술(심근염 케이스) → **Figure 5**
- 서로 다른 실험을 한 heatmap에 세로 비교하지 않는다 → **Figure 2**는
  (a) cue-token/held-out cue strings와 (b) final-prompt-token/diagnosis-heldout
  sweep을 독립 패널로 표시한다.

## 남은 ▢ (표 전반)

- **canonical matcher는 확정** — DDXPlus moved/adopted/third = 321/91/230,
  MCR = 437/137/300. 남은 것은 wording·CoT 파생 표의 동일 matcher 재집계다.
- T2: corpus-300 non-overlap subset
- T3: final probability 셀 전사 완료; paired bootstrap CI/추세 검정 보강
- T3b: standalone rule-based silent 값은 원장에 없음; 필요하면 직접 재출력
- T1: 외부 판정자의 438행 의미 재채점, MCR cue-position 계기 검증 및 산문
  서술률 행
- T3b: MCR 출력 채널 AUROC(CPU 가능), MCR CoT 채널(GPU), logit lens 칸
- T4: **MCR 사다리(Table 4d)** — r3/r4는 지금 실행 가능, r7은 MCR CoT
  실행 필요, r5는 wrong-note activation 추출 필요, r6은 현재 고정-class
  설계에서 직접 이전 불가. `run_mcr_ladder.sh`

## v2 → v3에서 바뀐 것 (08-24)

- **T3 신설**: 기전(대조 곡선·never-flip)이 표 없이 그림에만 있었다.
  관성 반론을 닫는 세 줄이므로 인용 가능한 표가 필요하다.
- 구 T3(채널별 귀속) → **T3b**. 둘 다 §4.3이고, 2b/2c/2d와 같은 관례다.
- T1에 **답-위치 vanilla 행** 추가 (실행 완료).
- 캡션이 계기를 명시하도록 설계 규칙에 한 줄 추가.

## v1 대비 바뀐 것

- T1: 길이·형식 행 제거(단위 불일치 → 본문), Reference 열 신설
- T2: 파생 통계 행 제거(본문), 행=코퍼스·열=조건으로 전치, 2b/2c 분리
- T3b(구 T3): 슬래시 셀 제거(All/Silent 열 분리), MCR을 boolean 열로, CoT 셀당 한 값
- T4: 첫 패스 행을 캡션으로, 4b 열 정리
- T5: 근거 문장 대신 절 참조
