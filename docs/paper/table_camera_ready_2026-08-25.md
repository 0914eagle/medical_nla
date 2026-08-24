# Camera-ready tables, v2 (2026-08-25)

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
| Held-out description rate | 2,122 | .751 | .088 | chance |
| Unseen-cue description | 438 | .750 | .720 | untuned |
| Conclusion at the answer position | 229 | .651 | .603 | untuned |

*Format compliance and output length are reported in the text (0.05 → 1.00;
1,557 → 52 characters): they establish that the readout is machine-scorable,
not that it is faithful. Swap tracking / memorization are the core: editing
one finding moves the description 99.3% of the time and never leaves the
original wording behind.*

**채점자 표기 (08-25 결정, 08-24 적용 확대)**: unseen-cue 서술률 0.75는
1기의 438행 의미 채점(A/B/C/D 4등급, A+B를 성공으로) 결과다.
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

**T1은 현재 전 행이 DDXPlus다 — MCR 열 계획 (08-24 재검토판).**
행별 가능성은 도구가 아니라 선행 조건이 가른다. 공통 선행:
**MCR cue 위치 어댑터**(학습 여부 서버 확인 필요; `mcr_sweep_v1`은 추출만
있고 판독이 없다 — "재집계 CPU"라던 이전 계획은 오기) + 판독 실행(GPU).

| 행 | MCR | 경로 |
|---|:-:|---|
| 서술 정밀도 · 오염 · heldout 서술률 | ✔ | 판독 실행 후 기존 분석기 그대로 |
| 스왑 추적 · 문맥 암기 | △ | `make_span_counterfactual_rows.py`(산문용 span 치환, 미실행) — cue 축자 등장 필요·탈락률 미측정·비문 캐비앳 |
| unseen-cue 서술 | ✕→△ | MCR 분할은 케이스 분할이지 cue 문자열 heldout이 아님 — 분할 재구성 없이는 정의 안 됨 |
| 답 위치 결론 | ✔ | 결론 어댑터(학습 중)가 여는 칸 |

▢ 그 밖에 남은 것: shuffle-control 값, swap/memorization의 정확한 n.

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

**▢ 재학습 뒤 새로 생기는 질문**: 살아남는 학습 행이 **1,298개**이고 MCR
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

**어댑터 부족인지 상태의 성질인지는 아직 안 갈린다**: content loss 1.767,
best_epoch 1/3, 학습 10,663행. 근거 접지가 통제와 갈리지 않으면 다음 질문은
"더 학습하면 되는가"이고, 그건 별도 실행이다.

---

## Table 2 — Intervention accuracy (§4.2)

행 = 코퍼스, 열 = 조건. 지표는 정확도 하나. 케이스 수·낙폭은 본문.

**Table 2.** Accuracy by arm, on cases answered correctly with no note.
Finding-position activations are bit-identical across arms by construction.

| Corpus | n | No note | Neutral | Wrong | Correct |
|---|---:|---:|---:|---:|---:|
| DDXPlus | 1,220 | .991 | .934 | **.760** | ▢ᶜ |
| DDXPlus, 3× replication | 3,343 | .985 | .932 | **.771** | .920 |
| MedCaseReasoning | 1,543ᵉ | .981 | .926 | **.703** | .839 |

ᵉ **모집단을 본문에 밝힌다 (08-24).** 1,543은 MCR 12,620건 중 소스 모델이
소견서 없이 맞힌 전부다 — **정확도 0.122**. 이 행은 "MCR의 12%"가 아니라
"모델이 맞힐 수 있는 MCR"에 대한 결과이고, 그 조건은 DDXPlus 행과 같다
(소견서로 답이 바뀌었음을 보이려면 원래 맞았어야 한다). 다만 절대 난이도는
전혀 다르며 — DDXPlus 49클래스 대 MCR 6,934개 진단, 대부분 1회 등장 —
그 차이가 §4.1 판독 실험이 MCR에서 왜 다른 문제인지를 설명한다.

ᶜ **08-24 감사에서 걸린 오류.** 이 칸에 적혀 있던 .932는 이 행의 값이 아니다.
1,747건 실행의 답 파일 어디에도 `correct` 조건이 없고(조건 전수 스캔,
08-24), .9313은 **corpus-300의 정답 조건을 4,995건 전체(누출 미필터)에서
잰 값**이다 — 실행도 모집단 필터도 다르다. 나머지 세 칸은 모두 1,747건
실행의 clean n=1,220이 맞다(.9910 / .9344 / .7598로 재확인). 정답 조건만
같은 케이스 파일로 다시 실행한다: `scripts/run_ddxplus_correct_arm.sh`
(한 조건 1,747답, GPU 1시간 내외). 그 전까지 이 칸은 비워 둔다.

*The wrong note costs 23.1 pp on DDXPlus and 27.8 pp on MedCaseReasoning;
the neutral note costs 5.7 and 5.5 pp, so the suggestion-specific effect is
17.4 and 22.3 pp — 4.1× and 5.1× the cost of insertion alone. The 3×
replication row reproduces the ratio on an independent draw from the same
corpus: 21.5 pp against 5.3 pp, 4.1×.*

**두 번째 행은 손실이 아니라 소득이다 (08-24).** corpus-300은 네 조건을
자기 안에 다 갖고 있어 **자급자족하는 재현 행**이고, 전체/위약 배수가
4.06×로 주 실행의 4.05×를 사실상 그대로 낸다. 정답 조건 칸이 비어 있는
동안에도 "침입 비용은 제안 방향과 무관하다"는 논증은 이 행과 MCR 행에서
이미 선다 — 정답을 부르는 소견서조차 DDXPlus에서 6.6pp(.985→.920),
MCR에서 14.2pp(.981→.839)를 깎는다.

**모든 DDXPlus 비율은 보수적 하한이다.** 답 파일이 `plausible_wrong`
수정(d29b754) 이전에 생성되어, 오답 소견서가 정답을 부르는 케이스가
1,747건 중 15건, 4,995건 중 44건 남아 있다(각각 0.86% · 0.88%). 그런
소견서는 사실상 정답 조건이라 오답 조건의 정확도를 **올리는** 쪽으로
작용한다. 즉 편향이 우리에게 불리하므로 전집합 수치를 그대로 싣고 각주로
밝힌다; `analyze_hint_effect.py --exclude-collisions`가 반대쪽을 준다.

**Table 2b.** Where the moved answers go.

| Corpus | Moved | To the suggestion | To a third diagnosis |
|---|---:|---:|---:|
| DDXPlus | 324 | 95 | 229 |
| MedCaseReasoning | 441 | 138 | 303 |

**Table 2c** (지면 되면; 아니면 부록). Speaker/wording variants, DDXPlus,
n = 1,747 each.

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
| Answer unchanged | 1,423 | .980 | .987 | −.007 |
| Lost the gold, answered elsewhere | 229 | .879 | .934 | −.055 |
| Adopted the suggestion | 95 | .736 | .923 | **−.187** |

*The cost grows with the behavioural outcome, so the state is not merely
carrying an earlier answer forward: it reads the note and moves as much as it
reads. It does not move enough to be overturned. In the bottom row the probe
still puts 3.5× more mass on the gold than on the suggestion, while by
definition every one of those cases emitted the suggestion. Across all six
landmarks, 268 of the 324 moved cases (82.7%) never once read the suggestion
as top-1, against an emitted accuracy of .012 on the same cases. At the
finding positions the cost is ±.000 to three decimals, which is what causal
masking guarantees and therefore what the design must reproduce.*

**계기 표기**: 이 표는 **프로브**다. 자연어 판독은 같은 방향을 독립적으로
말하지만 값이 다르다 — 상실형 최종 토큰에서 "상태가 정답을 쥠"이 프로브
.904, v2 판독 .651, 무학습 판독 .603 (세 계기 모두 같은 229건). **결렬의 존재는 두 계기가, 정밀한
해부는 프로브만 말한다**는 것을 본문이 그대로 밝힌다. 프로브는 닫힌
49클래스에 학습된 분류기이므로 이 표는 **MCR에서 미측정이 아니라 정의
불가**다 — 그 사실이 Table 3b 마지막 열과 Table 5의 근거가 된다.

---

## Table 3b — Single-run attribution (§4.3)

셀당 값 하나: All / Silent를 **열 두 개**로. MCR은 숫자 열이 아니라
**적용 가능 여부 열**로 — 값이 아니라 정의의 문제라서.

**Table 3b.** Within-diagnosis AUROC for identifying moved cases from the
wrong-note run alone. Silent: cases whose answer differs from the suggestion
(70% of moved), where output-only signals are blind by construction. The
last column states whether the channel is definable when the diagnosis space
is open (6,934 labels, most occurring once).

| Channel | Internals | AUROC, all | AUROC, silent | Open vocab. |
|---|:-:|---:|---:|:-:|
| Chain-of-thought features | – | .53 | .53 | yes |
| Answer equals suggestion | – | .664 | –ᵃ | yes |
| LLM monitor over the chain | – | .721 | .695 | yes |
| Verified NL readout (ours) | ✓ | .755 | .842 | yes |
| Linear probe, final token | ✓ | **.924** | **.984** | noᵇ |

ᵃ Undefined on the silent subset: the feature is the subset's defining
condition. ᵇ No class set exists to train on.

**LLM 모니터 행 (08-24 실측, gpt-5.6-sol, 1,747/1,747 파싱, 실패 0).** 이
행이 §4.2의 주장을 바꾼다. 규칙 기반 특징 .53 → 프런티어 모니터 .721이면
차이가 크므로 **"체인은 귀속 신호를 담지 않는다"는 더 이상 못 쓴다** — 우리
채점기가 약한 부분이었고, 같은 체인에서 강한 독자는 실제로 신호를 끌어낸다.
주장은 이진에서 정량으로 바뀐다: 침묵 구역에서 모니터 .695 대 판독 .842
(+14.7%p) 대 프로브 .984 (+28.9%p). 그리고 체인이 **출력 위에 더해주는
몫**은 +5.7%p뿐이다 (.664 → .721, 전체 체인을 다 보여주고).
모니터는 소견서를 보므로 "답 == 제안"을 스스로 계산할 수 있고, all 열의
일부는 그것이다 — 그걸 뺀 값이 silent 열이다. 루브릭은 판정자에게 유리하게
짰다: "답이 제안과 달라도 움직인 것일 수 있다"를 명시적으로 알려주므로,
낮은 값이 루브릭 결함이라는 반론은 닫힌다.

**판정자가 실제로 확률을 썼다 (08-24).** 서로 다른 값 61개, 0.00–1.00 전
구간, 평균 0.149(실제 답 바뀜 비율 0.185). 형식만 확률이고 실질은 3단계인
경우가 아니므로 **동점으로 AUROC가 깎이지 않았고 .695는 하한이 아니라
실측**이다. 다만 분포가 두 번째 사실을 말한다: P>0.5가 222건뿐(실제 moved
324건)이고 총량은 거의 맞힌다 — **이 실패가 얼마나 자주 일어나는지는 알지만
이 환자에게 일어났는지는 못 짚는다.** 배치에서 필요한 것은 후자다.

### ✅ 부트스트랩 CI 완료 (08-24, `bootstrap_channel_gap.py`)

케이스 단위 **쌍 부트스트랩**, 각 추출 안에서 진단 내 층화, 2,000회.
쌍으로 뽑는 이유는 두 채널이 같은 환자를 채점하므로 표본에 어떤 진단이
들어오느냐에 따라 함께 오르내리기 때문이다 — 비쌍 구간은 실제 비교보다 넓다.

| 비교 | 부분집합 | 차이 | 95% CI |
|---|---|---:|---|
| 판독 − **LLM 모니터** | 침묵 (1,608, moved 217) | **+.155** | **[+.080, +.229]** |
| 판독 − 체인 특징(최강) | 침묵 | +.291 | [+.230, +.354] |
| 판독 − 출력만 | 전체 (1,747, moved 324) | +.090 | [+.026, +.157] |

**셋 다 0을 배제한다.** 첫 줄이 §4.3의 문장을 지킨다.

**셋째 줄은 정직하게 적어야 한다** — 하한이 +2.6%p로 가장 아슬하다. 전체
집합에서 판독이 "답이 제안을 말하는가"라는 공짜 특징보다 앞서는 폭은 실재하나
크지 않다. **판독의 자리는 침묵 구역**이고, 거기서 그 공짜 특징은 정의상
AUROC 0.5다.

*침묵 구역 정의가 두 가지다.* 이 덤프는 `answer_names(wrong, hint)`로
자르고(1,608건), 모니터 라벨 파일은 `took_the_hint`로 자른다(1,652건) —
후자에는 "무소견서 답이 이미 제안을 부르지 않았다"가 추가된다. 44건 차이이며
모니터 AUROC가 .6870 대 .6951로 갈린다. **비교 자체는 두 채널이 같은 케이스를
보므로 유효**하고, 본문 각주에 어느 정의인지 밝힌다.

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
the 324 causally moved cases. Capitulation: share of newly broken answers
landing on the suggested diagnosis (first-pass counterpart .293).

| Rung | Appended | Overall | Moved | Capitulation |
|---|---|---:|---:|---:|
| r3 | reconsider request only | .424 | .460 | .450 |
| r4 | + findings re-shown (control) | .417 | .398 | .644 |
| r5 | + readout conclusion & grounds | .418 | .627 | .498 |
| r6 | + probe class label | .467 | .830 | .527 |
| r7 | + the model's own chain (▢ 실행 대기) | – | – | – |

*First-pass baseline: overall .814, moved .012. r5 − r4 = +22.8 pp on moved
(+17.7 pp on the 3× replication); r5 has the lowest capitulation
(z = 6.1/10.6).*

**Table 4d (예정) — 같은 사다리, MedCaseReasoning.** 어느 단이 존재하는지를
코퍼스가 정한다. 이 표의 빈칸은 미실시가 아니라 **결과**다.

| Rung | DDXPlus | MedCaseReasoning | 왜 |
|---|:-:|:-:|---|
| r3 reconsider only | .460 | ▢ 실행 가능 | 어댑터 불필요 |
| r4 findings re-shown | .398 | ▢ 실행 가능 | 어댑터 불필요 |
| r7 own chain | ▢ | ▢ (CoT 실행 필요) | GPU ~1–2h |
| r5 readout conclusion | .627 | ▢ 어댑터 학습 중 | 결론 어댑터가 여는 칸 |
| r6 probe class label | .830 | **✕ 존재 불가** | 진단 6,934종·대부분 1회 — 되먹일 클래스 집합이 없다 |

*r6의 불가능이 §4.4의 마지막 문장이다. DDXPlus만 보면 probe가 교정 비교에서
이기고, 독자는 자연어 채널이 잉여라고 결론지어도 좋다. **그 채널만이 존재하는
코퍼스**가 그 결론의 답이며, 주장이 아니라 측정으로 보여야 한다.
실행: `scripts/run_mcr_ladder.sh` (기본 rungs 3 4 7; 어댑터가 나오면
`RUNGS="3 4 5 7" READOUTS=…`).*

**이 표가 확립하는 명제는 "내부를 되먹여라"이지 "자연어로 되먹여라"가
아니다 (08-25).** r5와 r6이 둘 다 r4(입력 재제시)를 이기고, 내용 정확도를
맞추면 둘 사이 차이는 0이다(4b 1행). 교정 축의 결론은 **채널 중립**이며,
자연어가 필요해지는 곳은 회복률이 아니라 **클래스 채널이 존재하지 않는
코퍼스**와 **근거 제시가 요구되는 상황**이다(Table 5).

**r7이 이 명제를 완성한다 (▢ 실행 대기).** 지금까지 내부 되먹임의 비교
대상은 "입력을 다시 보여주기"였고, 가장 명백한 경쟁자 — **모델 자신의 CoT를
되먹이기** — 는 측정된 적이 없다. 그것 없이는 "내부를 되먹여라"가 서지
않는다. 어댑터 불필요, DDXPlus CoT 산출물 재사용
(`make_correction_ladder_cases.py --rungs 7 --cot-answers`). CoT 실행의 답이
direct 첫 답과 다른 케이스는 제외되므로 r7의 모집단이 작다 — **같은 id로
제한한 r3–r6과만 비교하고, 표에는 그 제한된 열을 따로 싣는다.**

**r6은 제안하는 방법이 아니라 통제다 (08-25 명시).** 표를 처음 보는 독자는
r6의 moved .830을 "probe가 이긴다"로 읽고, 곧바로 **"클래스명을 되먹이는
건 정답을 쥐여 주는 것 아닌가"**라고 되묻는다. 그 되물음은 옳고, 수치가
그대로 인정한다: probe argmax의 정답률은 moved에서 **.8642**(전체 .9599),
AV 판독 결론은 **.5185**(전체 .6754)다. r6의 .830은 .8642를 거의 그대로
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

**Table 4b.** r5 vs. r6 with fed-back content accuracy held fixed. Unseen
replication set (n = 3,319), exact McNemar on discordant pairs.

| Content (readout / probe) | n | r5 | r6 | r5-only : r6-only | p |
|---|---:|---:|---:|:-:|---:|
| correct / correct | 2,189 | .514 | .511 | 144 : 137 | .72 |
| wrong / correct | 1,017 | .223 | .437 | 33 : 250 | <.001 |
| wrong / wrong | 78 | .282 | .192 | 11 : 4 | .12 |
| correct / wrong | 35 | .600 | .086 | 19 : 1 | <.001 |

*Form contributes nothing once content accuracy is matched (row 1); the two
one-sided rows reflect content accuracy, not form. Where form does show is
row 3: when both channels hand over a wrong diagnosis, prose is the safer
carrier (main run .400 vs .240, 8 : 0, p = .008) — a bare name has nothing to
check against the chart, a conclusion with its grounds does.*

**Table 4c.** Deployment policies.

| Policy | DDXPlus | 3× replication |
|---|---:|---:|
| Keep first answer | .814 | .824 |
| Probe selects, argmax replaces | **.966** | **.973** |
| Probe selects, re-ask r6 | .954 | .966 |
| Probe selects, re-ask r5 | .915 | .927 |

---

## Table 5 — Discussion guidance (§5)

**Table 5.** When to use which instrument, from the measurements in §4.

| Setting | Instrument | Basis (§) |
|---|---|---|
| Closed label set, training labels available | Supervised probe | 4.3, 4.4 |
| Open diagnosis space | NL readout | 4.2–4.3 |
| Clinician-facing grounds required | NL readout | 4.1, 4.3 |
| Self-correction by re-asking | Neither — avoid | 4.4 |

---

## 그림이 나르는 것 (표에 안 넣는 수치)

- 위치별 비용 곡선(랜드마크 6지점, 그룹 3종) → **Figure 4**. 최종 토큰의
  세 값과 never-flip 268/324는 **Table 3으로 옮겼다** — 그림은 모양을,
  표는 인용 가능한 값을 나른다.
- 사례 서술(심근염 케이스) → **Figure 5**
- layer×position → **Figure 2**

## 남은 ▢ (표 전반)

- **별칭 매칭 규칙 통일** — 채택 건수가 규칙에 따라 95 / 107 / 139로 갈린다.
  T2b·T2c·T4의 항복률이 모두 이 숫자에 매달려 있으므로, 규칙을 하나로
  정하고 세 표를 같은 규칙으로 다시 집계한다. MCR의 "답 바뀜" 정의도 같은
  통일에 딸려 있다.
- T1: shuffle-control 값, swap/memorization/specificity의 정확한 n,
  답 위치 vanilla 행, MCR 산문 서술률 행
- T3b: MCR 출력 채널 AUROC(CPU 가능), MCR CoT 채널(GPU), logit lens 칸
- T4: **MCR 사다리(Table 4d)** — r3/r4는 지금 실행 가능, r7은 MCR CoT
  실행 필요, r5는 결론 어댑터 대기, r6은 존재 불가(결과). `run_mcr_ladder.sh`
- T4: **r7(자기 설명 되먹임)** — DDXPlus·MCR 양쪽. 이 단이 없으면 4.4는
  "내부를 되먹여라"가 아니라 "뭐라도 되먹여라"까지만 주장한다

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
