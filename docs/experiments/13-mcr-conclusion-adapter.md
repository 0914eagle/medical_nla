# 13 — MCR 결론 어댑터 (열린 어휘 판독)

**질문**: 자연어 판독이 **실제 임상 산문**과 **열린 진단 공간**에서도 작동하는가.

**상태**: 🔄 2026-08-24 재학습·재실행 중. **첫 실행은 무효**였다 — 학습 행의
88%가 상태와 맞지 않는 타깃이었다.

---

## 왜 이 실험이 논문에서 가장 무거운가

AV 판독이 프로브에게 정면 대결에서 이기는 자리가 없다(탐지 .842 대 .984,
교정 .627 대 .830, 되먹임 내용 정확도 .49 대 .87, 형식 기여 p=0.720).
**판독이 유일하게 필수인 자리가 열린 어휘**이고, 그것을 보이는 실험이 이것
하나다. 이 칸이 비면 "열린 쪽에서는 AV를 써야 한다"가 논증으로만 남는다.

## 설정

- **추출**: `mcr_answerpos_L32` — MCR 프롬프트의 **답 위치**(최종 토큰), L32
- **어댑터**: `mcr_conclusion_L32_s17`. LoRA r=16 / α=32 / dropout 0.05,
  AdamW 2e-4, 3에폭, **batch 1 × grad_accum 8**, `--select-on content`, 시드 17
  - 배치가 DDXPlus(4×2)와 다른 이유: 증례보고 프롬프트가 DDXPlus 평균
    148토큰의 몇 배라 4개를 담으면 3에폭에서 OOM이 났다(3.13GiB 요구 대
    3.11GiB 여유). 1×8은 **같은 옵티마이저 스텝**에 피크가 1/4이다.
- **템플릿**: `medical_nla_v2_readout.txt` — `<answer>` + 세미콜론 `<supporting_cues>`
- **타깃**: `diagnosis_name`(결론) + `cue_targets`(근거). 판정자 불필요 —
  코퍼스가 이미 나르는 필드로 규칙 조립된다. **근거는 프롬프트 축자 스팬**이라
  접지 지표로 재면 정확히 **1.000**이다(= 상한).
- **탈락 규칙**: 활성값 없음 / 진단명·cue 없음 / **프롬프트가 정답을 그대로
  적음**(그것을 되뇌는 판독은 아무것도 보이지 못한다)

## 🛑 첫 빌드의 결함 — 파일럿 규칙 누락

DDXPlus 파이프라인에는 이 규칙만을 위한 스크립트가 있다
(`make_medical_nla_v2_source_aligned_splits.py`):

> *"train/val rows are restricted to cases where **the source model selected
> the gold label**."*

**이유**: 활성값에 무엇이 들어 있는지 아무도 모른다. 유일한 손잡이가 모델의
출력이므로, gold는 **모델이 그것에 도달한 곳에서만** 판독 타깃이 될 수 있다.
그 밖에서 gold를 내놓으라고 학습시키는 것은 **문맥으로 정답을 추측하는 법**을
가르치는 것이고, 그것이 바로 판독이 탐지해야 할 실패다.

`make_mcr_conclusion_split.py`는 답 파일을 인자로 받지도 않았다.

| split | 행 | 맞힌 것 | **틀린 타깃** |
|---|---:|---:|---:|
| sft_train | 10,663 | 1,298 | **9,365 (88%)** |
| sft_val | 1,136 | 132 | 1,004 |
| sft_test | 821 | 113 | 708 |

**MCR 소스 정확도 0.122** (12,620 중 1,543)가 원인이다. DDXPlus는 49클래스에서
모델이 대부분 맞히므로 같은 누락을 우연히 견뎠다.

## 무효 실행이 실제로 보여준 것

읽어본 여섯 건의 모양이 전부 같았다:

| gold | 판독 |
|---|---|
| Extraosseous peripheral calcifying odontogenic cyst | Oral fibroma |
| Solid pseudopapillary epithelial neoplasm | Gastric duplication cyst |
| Rosai-Dorfman Disease | IgG4-related disease |
| proximal tibiofibular joint **osteoarthritis** | tibiofibular joint **dislocation** |
| Guillain-Barre syndrome | **Guillain–Barré syndrome** ✅ |
| neurogenic **pulmonary** edema | **cerebral** edema |

**전문과는 맞고, 진단명은 이웃이고, 근거는 그 전문과의 평균 워크업.** 33세를
"50-year-old"로, "scattered calcifications"를 "**no** calcifications"로,
그리고 **진단명을 맞힌 GBS 케이스조차** CSF 단백 76 mg/dL를 "lumbar puncture
revealed **normal** cerebrospinal fluid"로 적었다.

**이것은 88% 오답 타깃 지도의 직접적 귀결이지 MCR 상태에 대한 발견이 아니다.**

## 두 번째 결함 — 생성 예산

타깃 평균 **764자**, 19%가 1,000자 초과, 최대 2,507자인데 판독을
`max_new_tokens=256`(config 기본값)으로 돌렸다. **821행 중 444행(54%)이 태그를
못 닫았다.** `run_nla.py`에 `--max-new-tokens`를 추가하고 스크립트 기본값을
768로 올렸다.

**철회한 해석**: 근거가 행당 12.6개인 것은 실패가 아니다. 학습 타깃이 평균
10.1개(중앙 9)다 — 배운 대로 하고 있었다.

## 세 번째 결함 — 재사용 가드

재실행이 `skip (exists: 128 rows)`로 건너뛰고 **"ALL DONE"을 찍었다.** 그
128행은 죽인 프로세스가 남긴 조각이고 **옛 어댑터**가 썼다. 예산만 비교하는
가드로는 못 잡는다. 지금은 셋 다 본다: **예산 / 행 수 / 파일이 어댑터보다
오래됐는가**.

## 재학습 결과 (08-24)

```
sft_train 1,298 (source-correct 1.000) · sft_val 132 · sft_test 821 (correct 113)
best_epoch 1 / 3, content 1.8209, scaffold 0.0326
epoch 3: content 2.0390  ← 나빠짐
```

**1에폭에서 정점이고 그 뒤로 나빠진다 — 에폭을 늘리는 것은 답이 아니다.**
다음 지렛대는 데이터인데 1,543건이 MCR 전부다.

*(content 1.8209는 첫 빌드의 1.767과 비교 불가다 — val이 1,136행 대부분 오답
타깃에서 132행 전부 정합으로 바뀌어 과제 자체가 다르다.)*

**source-wrong 케이스는 test에 남긴다** — 거기서는 상태와 출력의 불일치가
결함이 아니라 **측정 대상**이다.

## 결과를 읽는 기준 — 숫자가 나오기 전에 고정한다

기준선 셋: 상한 **1.000**(타깃이 축자), 같은 위치·같은 스키마의 DDXPlus
**gap +.100 / 접지 .311**, 무효 실행 **+.034 / .052**.

| gap | 결론 | 논문에서 하는 일 |
|---|---|---|
| **≥ +.08** | 열린 어휘에서 된다 | Table 1에 MCR 행, caveat 1을 "프로브만 정의 불가"로 축소 |
| **+.04–.08** | 되지만 약하다 | 싣되 DDXPlus 대비 열세 명시 |
| **≤ +.04** | 학습셋 결함이 원인이 아니었다 | **Limitations 결과**로 전환 |

**옛 값과 같게 나오는 것이 실패가 아니다.** 원인 하나를 배제한 것이고, 남는
설명 둘 — 활성 벡터 하나에서 764자 원문 복원, 지도 신호 1,543건에 진단명당
1회 — 은 **이미 정량화돼 있다.** Limitations가 변명이 아니라 측정이 된다.

**두 관문**: ① 로그의 `readouts still cut off mid-cue`가 ~0이 아니면 gap을
읽지 않는다. ② **비율보다 눈검사가 먼저다** — 같은 세 건을 다시 읽어 근거가
프롬프트 문장으로 바뀌었는지 본다. 안 바뀌었는데 gap만 올랐다면 지표가 다른
것을 재고 있는 것이다.

## 접지 지표 (`analyze_readout_grounding.py`)

근거 문장을 **자기 프롬프트**와 **무작위로 짝지은 남의 프롬프트**에 각각
word-trigram 포함으로 대조하고, **차이만**을 케이스 고유 정보로 센다. 임상
산문은 상용구를 공유하므로 자기 프롬프트 겹침 단독으로는 해석 불가다.

- 정확 일치가 아니라 trigram인 이유: 충실한 근거는 대개 패러프레이즈다.
  "bilateral alveolar infiltrates"를 "bilateral pulmonary edema"로 쓰는 것은
  진짜 읽은 것이고, 정확 일치 채점기는 그것을 조작으로 센다.
- **절대값은 해석 금지.** DDXPlus 판독은 의미 채점 서술률 .751인데 이 지표로는
  접지 .161이다. 상대 비교 전용.
- 반복률은 코퍼스 간 비교 불가 — DDXPlus 소견은 닫힌 템플릿 어휘라 같은
  문장이 여러 환자에게 실제로 등장한다.

## 두 스키마

`--readouts`에 두 형식이 다 들어온다: 결론 어댑터의 세미콜론
`<supporting_cues>`와 cue 위치 어댑터의 불릿 `<observed>`. 전자만 읽던 파서가
DDXPlus 770행을 조용히 "근거 없음"으로 버렸고, 그것이 코퍼스 차이처럼 보였다.

## 남은 것

- ▢ 판독 결과 → 접지 재측정 → 위 기준으로 판정
- ▢ Table 3b의 MCR 칸과 MCR r5는 **여전히 별도 추출**이 필요하다. 현재
  매니페스트는 소견서 없는 프롬프트다(`filter_manifest_to_split.py`가
  `prompts naming a referring note: 0`으로 보고한다).
- ▢ 답 위치 레이어 스윕([02](02-layer-sweep.md)) — MCR은 열린 어휘 코퍼스인데
  L32에서만 쟀고, 스윕이 말하는 것은 "레이어는 열린 어휘 일반화에서 갈린다"다.
