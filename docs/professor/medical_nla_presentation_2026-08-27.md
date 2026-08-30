# Medical-NLA 교수님 미팅 자료

> **2026-08-30 갱신본**<br>
> 2026-08-27 예정 미팅이 진행되지 않아, 당시 초안을 현재 실험 결과와 수치로 대체했습니다.

---

## 자료 사용 원칙

- Method와 Data를 먼저 고정한 뒤 Results를 제시합니다.
- validation, locked test, post-hoc audit을 명확히 구분합니다.
- 아직 실행하지 않은 DiReCT locked 셀은 `pending`으로 남기며 추정값을 넣지 않습니다.
- 생성형 Medical-NLA의 실패 결과도 개발 gate와 함께 그대로 보고합니다.
- DDXPlus의 closed probe/structured reader와 open-ended NLA를 같은 방법처럼 해석하지 않습니다.

---

# Part I. 연구 질문과 현재 결론

---

## Slide 1. 한 문장 요약

> 의료 LLM의 hidden activation에는 환자별 임상 finding과 값 정보가 강하게 존재하지만, 현재의 공개 NLA와 여러 생성형 fine-tuning 방법은 그 정보를 사례 특이적인 자연어 설명으로 안정적으로 변환하지 못했습니다.

### 현재 가장 강한 양성 결과

- DDXPlus locked-test finding probe micro F1: **0.9562**
- 같은 진단 내 activation shuffle 대비 finding gap: **+0.1624**
- native-value accuracy: **0.7659**
- probe-guided structured reader finding F1: **0.9587**

### 현재 가장 중요한 음성 결과

- Vanilla NLA, DDXPlus locked 10,028 readouts: ontology claim **0건**
- Full-data Medical-NLA SFT, DiReCT validation Obscomp: **0.0301 / 0.0296**
- D20 specificity-anchored objective: 3개 seed 모두 changed-gap 개선 실패
- 공개 AR matched-vs-shuffled 양성 대조 실패: gap 약 **0.0000**

### 발표 핵심

1. 정보의 **존재**는 probe와 intervention으로 확인했습니다.
2. 정보의 **자연어 변환**은 별도 문제이며 현재 방법으로 해결되지 않았습니다.
3. 다음 단계는 free-paragraph SFT 반복이 아니라 Medical Activation Oracle과 domain-adapted AR입니다.

---

## Slide 2. 연구 질문

### RQ1. Activation에 임상 정보가 존재하는가?

- 진단 category/PDD를 읽을 수 있는가?
- 환자별 finding presence와 native value를 읽을 수 있는가?
- 같은 진단 내 다른 환자 activation과 구분되는가?

### RQ2. 그 정보를 자연어로 읽을 수 있는가?

- Vanilla NLA가 환자별 finding을 말하는가?
- Medical SFT가 physician observation 또는 입력 cue에 정렬되는가?
- activation을 바꾸면 설명도 선택적으로 바뀌는가?

### RQ3. 설명은 activation에 충실한가?

- cue deletion 시 해당 claim만 사라지는가?
- untouched claim은 유지되는가?
- value edit 시 새 값으로 바뀌고 이전 값은 사라지는가?

### 성공 조건

> 유창한 의료 문장을 만드는 것만으로는 성공이 아닙니다. 사례 특이성, counterfactual specificity, seed 안정성을 동시에 만족해야 합니다.

---

## Slide 3. 8월 27일 초안 이후 달라진 점

| 당시 상태 | 현재 상태 |
|---|---|
| DDXPlus finding/value probe 계획 | validation 및 locked test 완료 |
| DDXPlus activation 추출 예정 | validation 10,006행, test 10,028행 완료 |
| Vanilla NLA 평가 미완료 | locked 10,028행 생성 및 semantic scoring 완료 |
| Medical-NLA SFT pilot 중심 | full-data SFT, counterfactual SFT, ranking, bottleneck, anchored objective까지 평가 |
| intervention 설계만 존재 | deletion/value edit locked 결과 확보 |
| AR reconstruction 가설 | 공개 AR 양성 대조 실패 확인 |
| 결과표 다수 `TBD` | DDXPlus 및 development gate 수치 대부분 확정 |

### 아직 남은 핵심 미완료

- DiReCT locked Table 1A: test-seen 72, PDD-heldout 106 source behavior 재집계
- DiReCT locked Table 1B: frozen HS24 diagnosis probe 적용
- DiReCT locked Table 2: Source CoT와 Vanilla NLA baseline
- 생성형 Medical-NLA locked 행: validation gate를 통과한 모델이 없어 아직 열지 않음

---

# Part II. Method

---

## Slide 4. Activation 위치와 공통 표현

| 위치 | 정의 | 역할 |
|---|---|---|
| CoT-P0 | clinical input과 CoT instruction을 읽고 첫 reasoning token을 생성하기 직전 | primary |
| P1 | generated reasoning과 `The answer is` marker를 읽고 진단명을 쓰기 직전 | leakage sensitivity |
| P2 | generated diagnosis의 마지막 subtoken | positive/leakage control |

### 레이어

- 후보: **HS16, HS24, HS32**
- diagnosis closed probe는 validation 기준 **HS24** 선택
- 공개 NLA/AR checkpoint는 구조상 **HS32** 사용

### 중요한 통제

- DiReCT와 DDXPlus primary activation을 모두 **CoT-P0**로 맞춤
- Direct-P0는 instruction-sensitivity control로만 사용
- P1/P2의 높은 성능을 P0 내부 표현 증거로 과대해석하지 않음

---

## Slide 5. 평가를 세 층으로 분리

| 층 | 질문 | 방법 | 해석 범위 |
|---|---|---|---|
| Closed probe | 정보가 activation에 있는가? | linear diagnosis/finding/value probe | 정보 존재 및 사례 특이성 |
| Structured reader | 선택된 상태를 결정론적으로 말로 렌더링할 수 있는가? | frozen probe + train-only lexicon | closed structured monitor |
| Open-ended NLA | activation만으로 자유 자연어 설명을 생성하는가? | Vanilla AV, Medical-NLA adapters | 자연어 readout |

### 왜 분리하는가?

- probe 성공은 자연어 decoder 성공을 보장하지 않습니다.
- structured reader 성공은 probe가 선택한 ontology를 렌더링한 결과입니다.
- open-ended NLA는 claim 선택과 문장 생성을 동시에 해결해야 합니다.

---

## Slide 6. Closed probe 검증 규약

### Finding presence

- activation: CoT-P0, HS16/24/32
- 모델: train-only multi-label linear probe
- validation에서 layer, regularization, threshold 선택
- locked test에는 선택된 설정을 한 번만 적용

### Native value

- evidence ID가 존재한다고 가정한 conditional value classification
- train에서 관찰된 multi-value evidence만 평가
- 예: `rash swollen = rated 4`처럼 evidence와 값이 함께 있는 항목

### 사례 특이성 control

- 같은 diagnosis의 다른 환자 activation을 deterministic hard shuffle
- own score에서 shuffled score를 뺀 gap 사용
- 진단 template만 읽어서는 높은 gap을 만들 수 없음

---

## Slide 7. Counterfactual 평가 규약

### Cue deletion

```text
Original activation: cue A + cue B + cue C
Deleted activation:          cue B + cue C
```

- deletion probability drop: 지운 cue의 probe probability가 얼마나 감소했는가
- original hit: 원본에서 지운 cue를 실제로 읽었는가
- deletion phantom: 삭제본에서도 그 cue를 계속 말하는가
- removal success: 원본에서 읽힌 cue가 삭제본에서 사라졌는가
- untouched retention: 삭제하지 않은 cue가 유지되는가

### Native-value edit

```text
Original: rash severity = 3
Edited:   rash severity = 5
```

- replacement hit: 새 값을 출력했는가
- old-value persistence: 이전 값을 계속 출력했는가
- clean switch: 새 값은 출력하고 이전 값은 출력하지 않았는가

> 단순 정확도보다 intervention에 대한 선택적 반응이 grounding의 더 강한 증거입니다.

---

## Slide 8. Open-ended 설명 평가

### DiReCT

- physician observation과 생성 설명을 claim 단위로 비교
- 주요 지표:
  - `Obspre`: observation precision
  - `Obsrec`: observation recall
  - `Obscomp`: gold/predicted observation set의 semantic Jaccard completeness
  - `Expcom`: matched observation에서 rationale와 diagnosis edge까지 일치한 비율
  - `Expall`: 전체 explanation chain의 일치 비율
- 진단명 언급만으로는 observation alignment를 인정하지 않음

### DDXPlus semantic mapper

1. Stage 0: claim splitting
2. Stage 1: frozen lexical alias mapping
3. Stage 2: method-blind AI semantic mapping
4. case별 evidence/value set으로 deduplicate 후 동일 scorer 적용

### Mapper validation

- G1 reader round-trip
- G2 absent-cue false mapping
- G3 cache/replay determinism
- G4 independent AI concordance

---

## Slide 9. 생성형 Medical-NLA 개발 규율

### 시도한 방법 계열

1. DiReCT-only SFT
2. DiReCT + DDXPlus common/full-data SFT
3. Counterfactual sequence SFT
4. Changed-cue ranking objective
5. Specificity-anchored ranking
6. 256-dimensional soft latent bottleneck
7. Public AR reconstruction diagnostic

### 공통 promotion 원칙

- 3개 seed 방향 일치
- category-cluster bootstrap CI가 0보다 큼
- 최소 효과 크기 충족
- changed cue뿐 아니라 retained cue specificity 통과
- validation gate 통과 전 locked test 금지
- 실패 후 checkpoint/threshold를 사후 선택하지 않음

> 잘 말하는 모델을 찾는 것이 아니라 activation-dependent한 모델을 찾습니다.

---

# Part III. Data

---

## Slide 10. 데이터 모집단

### DiReCT

| split | rows | 사용 |
|---|---:|---|
| train | 266 | probe 및 데이터 구축 |
| SFT train eligible | 248 | note에 gold label이 직접 등장한 행 제외 |
| validation | 52 | layer/hyperparameter/development gate |
| SFT validation | 50 | generation evaluation |
| test-seen | 72 | locked, pending batch |
| PDD-heldout | 106 | locked OOD, pending batch |

- canonical eligible total: **496**
- source answer rows: **496**
- P0/P1/P2 position rows: **1,488**
- HS16/24/32 tensors: **4,464**

### DDXPlus

| split/artifact | originals | activation rows |
|---|---:|---:|
| official train development | 4,655 | 4,655 original P0 |
| validation | 4,525 | 10,006 |
| locked test | 4,543 | 10,028 |

- D9a supported training pairs: **3,104**

---

## Slide 11. DDXPlus 평가 분모

| Metric family | validation | locked test | 사용 목적 |
|---|---:|---:|---|
| same-diagnosis hard-shuffle pairs | 4,106 | 4,121 | 사례 특이성 |
| native-value targets | 2,183 | 2,136 | 원래 값 판독 |
| cue-deletion pairs | 4,523 | 4,540 | 삭제 반응 |
| native-value-edit pairs | 533 | 539 | 값 변경 반응 |
| clean-switch eligible | 395 | 398 | old/new 완전 전환 |

### 숫자가 다른 이유

- 모든 환자가 multi-value evidence를 갖지는 않습니다.
- value edit은 실제로 다른 유효 값으로 바꿀 수 있는 행만 포함합니다.
- clean switch는 원본에서 old value를 읽은 사례에 조건화됩니다.
- hard shuffle은 같은 diagnosis 안에서 유효 donor가 존재해야 합니다.

---

# Part IV. Results

---

## Slide 12. Source model: Direct vs CoT exploratory 결과

> 전체 eligible 171건에 대한 exploratory 결과이며, locked 72/106 최종 표가 아닙니다.

| condition | n | parse | strict PDD | category | token F1 |
|---|---:|---:|---:|---:|---:|
| Direct | 171 | 1.0000 | 0.2105 | 0.5029 | 0.1593 |
| CoT | 171 | 1.0000 | 0.1930 | 0.5088 | 0.1850 |

- strict PDD McNemar: **p = 0.6291**
- category McNemar: **p = 1.0000**
- CoT reasoning에 answer alias가 등장한 비율: **156/171 = 0.9123**

### 해석

- CoT가 strict diagnosis accuracy를 개선하지 않았습니다.
- token overlap은 조금 높지만 category accuracy 차이는 없습니다.
- P1/P2는 문자열 leakage control이며 primary evidence는 P0입니다.

---

## Slide 13. DiReCT P0: 진단 정보는 읽히는가?

### Linear probe, validation n=52

| target | HS16 | HS24 | HS32 | majority |
|---|---:|---:|---:|---:|
| disease category Top-1 | 0.5000 | **0.5962** | 0.5192 | 0.0577 |
| canonical PDD Top-1 | 0.3846 | **0.4423** | 0.3846 | 0.0962 |

### Forced-answer likelihood baseline

| target/ranking | Top-1 | Top-5 | MRR | mean gold rank |
|---|---:|---:|---:|---:|
| category raw | 0.4808 | 0.6731 | 0.5814 | 5.02 |
| category calibrated | 0.2308 | 0.3077 | 0.3091 | 9.58 |
| PDD raw, train49 | 0.1538 | 0.5192 | 0.3250 | 7.92 |
| PDD calibrated | 0.0577 | 0.1346 | 0.1486 | 15.83 |

### 해석

- P0 activation에 진단 관련 선형 정보가 존재합니다.
- calibration은 label prior를 제거했지만 성능을 크게 낮췄습니다.
- HS24가 validation에서 category와 PDD 모두 최선이었습니다.

---

## Slide 14. DiReCT Vanilla NLA: P0를 자연어로 읽는가?

### Primary P0, validation n=52

| prompt | source answer mention | gold PDD mention | category mention |
|---|---:|---:|---:|
| default, HS32 | 0/52 | 0/52 | 0/52 |
| task-aligned, HS32 | 0/52 | 0/52 | 0/52 |

### 6-arm layer audit

- 대부분의 arm에서 세 target 모두 **0/52**
- 예외: HS16 category가 default/task-aligned에서 각각 **1/52**

### P1/P2 leakage control

| position | source | gold PDD | category | own-shuffled gap |
|---|---:|---:|---:|---:|
| P1 | 0.4912 | 0.1404 | 0.5848 | +0.4146 |
| P2 | 0.3918 | 0.0819 | 0.4854 | +0.3598 |

### 해석

- 공개 Vanilla NLA는 P0의 진단 정보를 읽지 못했습니다.
- P1/P2 성공은 reasoning/answer 문자열 노출과 일치하며 P0 reader 성공이 아닙니다.

---

## Slide 15. DDXPlus closed probe: locked test

### Finding presence

| metric | own | same-diagnosis shuffled | gap | 95% CI |
|---|---:|---:|---:|---:|
| micro F1 | **0.9562** | 0.7938 | **+0.1624** | [0.1576, 0.1672] |

### Native value

| metric | own | same-diagnosis shuffled | gap | 95% CI |
|---|---:|---:|---:|---:|
| conditional accuracy | **0.7659** | 0.5791 | **+0.1868** | [0.1650, 0.2091] |

### Validation layer sensitivity

| target | HS16 | HS24 | HS32 |
|---|---:|---:|---:|
| finding micro F1 | 0.9636 | **0.9607** | 0.9607 |
| native-value accuracy | 0.7641 | **0.7700** | 0.6990 |

### 해석

- activation에는 환자별 finding과 값 정보가 강하게 존재합니다.
- 같은 진단 내 shuffle gap이 양수이므로 diagnosis template만 읽은 결과가 아닙니다.
- HS16의 raw finding F1이 0.0029 높지만, validation own-minus-shuffled 우선 규칙으로 HS24를 frozen layer로 선택했습니다.

---

## Slide 16. DDXPlus structured reader: locked test

> Frozen probe가 선택한 state를 train-only lexicon으로 렌더링한 closed monitor입니다. Open-ended NLA가 아닙니다.

### Static readout

| metric | value |
|---|---:|
| original cases | 4,543 |
| mean emitted claims | 4.9353 |
| finding micro F1 | **0.9587** |
| shuffled F1 | 0.7938 |
| own-shuffled gap | **+0.1624** |
| native-value accuracy | **0.7654** |
| value emission coverage | 0.9995 |

### Counterfactual response

| metric | value | denominator |
|---|---:|---:|
| deletion original hit | 1.0000 | 4,540 |
| deletion phantom | 0.3593 | 4,540 |
| removal success | 0.6407 | original-hit conditional |
| untouched retention | 0.9987 | 16,105 |
| replacement hit | 0.1466 | 539 |
| old-value persistence | 0.5955 | 539 |
| clean switch | 0.0804 | 398 |

### 해석

- static finding state는 매우 잘 읽힙니다.
- deletion 반응은 부분적이고 value-edit 전환은 약합니다.
- 높은 static F1과 낮은 clean switch가 동시에 존재합니다.

---

## Slide 17. DDXPlus semantic mapper validation

| gate | metric | result | criterion |
|---|---|---:|---:|
| G1 | reader finding round-trip F1 | 1.0000 | >= 0.98 |
| G1 | reader native-value accuracy | 1.0000 | >= 0.98 |
| G2 | absent-target false map | 0/2,609 = 0.0000 | <= 0.05 |
| G3 | cache replay byte-identical | True | True |
| G3 | cold duplicate agreement | 1.0000 | report |
| G4 | evidence disagreement | 2/100 = 0.0200 | <= 0.05 |
| G4 | conditional value disagreement | 0/30 = 0.0000 | <= 0.05 |

- primary mapper: `gpt-5.6-sol`
- independent auditor: `gpt-5.4`
- validation에서 동결한 뒤 locked generation에 적용

### 주의

- G4는 두 AI judge의 concordance입니다.
- 사람 임상 타당도 검증으로 표현하지 않습니다.

---

## Slide 18. DDXPlus Vanilla NLA: locked 10,028행

### 생성 모집단

- original: **4,543**
- cue-deleted: **4,543**
- value-edited: **942**
- total readouts: **10,028**

### Frozen mapper 결과

| item | count |
|---|---:|
| lexical mappings | 0 |
| raw AI mappings | 0 |
| accepted AI mappings | 0 |
| rows with emitted ontology claim | 0/10,028 |

### Post-hoc diagnosis-stratified 20-case audit

| audit item | result |
|---|---:|
| generic clinical prose only | 20/20 |
| possible frozen-mapper miss | 0/20 |
| expected-cue paraphrase match | 0/20 |
| malformed/empty | 0/20 |
| median readout length | 683.5 characters |

> Vanilla NLA는 비어 있거나 깨진 출력이 아니라, activation과 환자에 특이적이지 않은 일반적 임상 문장을 생성했습니다.

---

## Slide 19. Medical-NLA SFT: DiReCT alignment

### Common mixed pilot population

| method | parsed | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | 0.3110 | 0.4069 | **0.2399** | 0.0657 | 0.0168 |
| Vanilla NLA | 10/50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Common SFT seed17 | 50/50 | 0.0100 | 0.0037 | 0.0034 | 0.0000 | 0.0000 |
| Common SFT seed29 | 50/50 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Common SFT seed43 | 4/50 | 0.0070 | 0.0054 | 0.0043 | 0.0000 | 0.0000 |

### Full-data population

| method | parsed | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | 0.2835 | 0.3726 | **0.2130** | 0.0650 | 0.0153 |
| Full-data SFT seed17 | 50/50 | 0.0544 | 0.0502 | 0.0301 | 0.0000 | 0.0000 |
| Full-data SFT seed29 | 50/50 | 0.0553 | 0.0388 | 0.0296 | 0.0000 | 0.0000 |

### 해석

- DDXPlus train 4,655건을 추가해도 DiReCT case-specific alignment는 Source CoT floor에 미달했습니다.
- SFT는 출력 형식을 학습했지만 activation-dependent observation을 안정적으로 복원하지 못했습니다.

---

## Slide 20. SFT 원문 census: template collapse

> AI checklist가 불안정해 최종 판정에는 사용하지 않고, 전체 50-case deterministic exact-text census를 사용했습니다.

| method | Obscomp | exact duplicate rows | unique outputs |
|---|---:|---:|---:|
| Direct-only seed17 | 0.0343 | 43/50 | 7 |
| Direct-only seed29 | 0.0047 | 47/50 | 3 |
| Direct-only seed43 | 0.0032 | 49/50 | 1 |
| Full-data seed17 | 0.0301 | 36/50 | 14 |
| Full-data seed29 | 0.0296 | 48/50 | 2 |
| Source CoT | **0.2130** | 0/50 | 50 |

### 폐기한 AI audit의 품질 문제

- initial requests: 200
- 3회 repair 후 valid: **56**, invalid: **144**
- invalid: true without quote 91, population mismatch 21, non-verbatim quote 30, JSON parse 2

### 결론

- 낮은 Obscomp는 lexical scorer의 보수성만으로 설명되지 않습니다.
- SFT output 자체가 여러 사례에서 동일한 의료 template로 붕괴했습니다.

---

## Slide 21. 생성형 개발 gate 전체 결과

| experiment | primary observed result | frozen criterion | verdict |
|---|---|---|---|
| Full-data SFT | Obscomp 0.0301 / 0.0296 | > Source CoT 0.2130 | FAIL |
| D10 ranking, 20 steps | changed +0.0005 / +0.0028 / +0.0030 | each >= 0.05, CI/specificity | FAIL |
| D14 K=5 OOF teacher | precision 0.8881 | >= 0.90 plus calibration gates | FAIL |
| D16 soft bottleneck | -0.001137 / -0.001476 / +0.001433 | each >= 0.005, CI > 0 | FAIL |
| D16 frozen-z | finding F1 -0.0009 / -0.0007 / -0.0016 | non-negative improvement | FAIL |
| D10 budget 1,552 | changed +0.5558, retained +0.5604, specificity -0.0046 | selective changed-cue gain | FAIL |
| D20 anchored 1,552 | specificity -0.0278 / -0.0255 / -0.0217 | positive each seed, CI > 0 | FAIL |
| D22 public AR | positive-control gap approximately 0 | matched > shuffled | FAIL as instrument |

### 공통 교훈

- 작은 training budget만의 문제가 아니었습니다.
- 큰 margin은 global deletion detector shortcut으로도 만들 수 있습니다.
- specificity를 loss에 넣으면 shortcut은 줄지만 목표 신호도 사라졌습니다.

---

## Slide 22. D10 budget trajectory

### Across-seed means

| step | changed-gap delta | retained-gap delta | specificity delta |
|---:|---:|---:|---:|
| 20 | +0.0019 | +0.0002 | +0.0018 |
| 194 | +0.0329 | -0.0032 | +0.0361 |
| 388 | +0.2690 | +0.3044 | -0.0354 |
| 776 | +0.0965 | +0.0428 | +0.0536 |
| 1,164 | +0.3527 | +0.4055 | -0.0527 |
| 1,552 | **+0.5558** | **+0.5604** | **-0.0046** |

### 최종 seed별 changed gap

- seed17: **-0.0177**
- seed29: **+0.5618**
- seed43: **+1.1233**

### 해석

- 삭제한 cue뿐 아니라 retained cue NLL도 거의 같은 크기로 증가했습니다.
- 모델은 `어떤 cue가 지워졌는가`가 아니라 `삭제된 activation인가`를 감지했습니다.
- specificity gate가 없었다면 잘못된 성공 판정을 내릴 수 있었습니다.

---

## Slide 23. D20: specificity를 loss에 넣은 결과

| seed | changed gap | retained gap | specificity | changed original NLL | retained original NLL |
|---:|---:|---:|---:|---:|---:|
| 17 | -0.0143 | +0.0135 | **-0.0278** | -0.0756 | -0.3342 |
| 29 | -0.0040 | +0.0215 | **-0.0255** | +0.0576 | -0.1834 |
| 43 | -0.0266 | -0.0049 | **-0.0217** | +0.0622 | -0.2263 |

### 무엇이 확인됐는가?

1. Retained anchor가 global deletion detector shortcut을 억제했습니다.
2. 그러나 changed-cue signal도 3 seed 모두 음수로 사라졌습니다.
3. retained original NLL은 개선되어 optimization 자체가 멈춘 것은 아닙니다.
4. teacher-forced gate 실패로 generation, extension, sweep을 실행하지 않았습니다.

> 기존 ranking loss의 큰 효과는 선택적 임상 정보 판독이 아니라 shortcut이었습니다.

---

## Slide 24. D22: 공개 AR reconstruction 진단

> Released `kitft/nla-gemma3-12b-L32-ar`, validation-only, arm당 n=20

| dataset | arm | own cosine | shuffled cosine | gap | 95% CI |
|---|---|---:|---:|---:|---:|
| DDXPlus | structured reader | 0.9765 | 0.9765 | -0.0000 | [-0.0002, +0.0001] |
| DiReCT | Direct-only seed17 | 0.9744 | 0.9744 | -0.0000 | [-0.0002, +0.0002] |
| DiReCT | Direct-only seed29 | 0.9772 | 0.9772 | -0.0001 | [-0.0002, +0.0001] |
| DiReCT | Direct-only seed43 | 0.9758 | 0.9757 | +0.0000 | [-0.0001, +0.0001] |
| DiReCT | Full-data seed17 | 0.9779 | 0.9780 | -0.0001 | [-0.0002, +0.0001] |
| DiReCT | Full-data seed29 | 0.9777 | 0.9777 | +0.0000 | [-0.0001, +0.0002] |
| DiReCT | Source CoT | 0.9835 | 0.9834 | +0.0001 | [-0.0000, +0.0002] |
| DiReCT | Vanilla NLA | 0.9962 | 0.9961 | +0.0001 | [+0.0000, +0.0002] |

### 해석

- 구조화 reader와 Source CoT 양성 대조도 matched activation을 구분하지 못했습니다.
- 높은 절대 cosine은 사례 정보 복원보다 공통 방향/anisotropy의 영향을 받은 것으로 보입니다.
- 이 공개 AR의 cosine을 Medical-NLA reward로 바로 사용할 수 없습니다.
- 이는 임상 정보가 없다는 증거가 아니라, **AR 측정기가 이 domain에 맞지 않는다는 증거**입니다.

---

# Part V. 종합과 다음 결정

---

## Slide 25. 현재 RQ별 답

### RQ1. Activation에 임상 정보가 존재하는가?

**예.**

- DiReCT diagnosis category/PDD linear probe가 majority를 크게 상회
- DDXPlus finding F1 0.9562, native value 0.7659
- same-diagnosis shuffled gap +0.1624 / +0.1868

### RQ2. Vanilla NLA가 그 정보를 자연어로 읽는가?

**현재 공개 checkpoint에서는 아니오.**

- DiReCT P0 target mention 거의 0
- DDXPlus locked 10,028행 frozen ontology claim 0
- 20-case audit에서 모두 generic clinical prose

### RQ3. Medical fine-tuning이 해결했는가?

**현재 시도한 surrogate 계열에서는 아니오.**

- SFT는 형식을 학습했지만 template collapse
- ranking은 deletion detector shortcut
- retained anchor는 shortcut을 막았으나 changed signal도 소거
- public AR은 medical matched-vs-shuffled 측정기로 실패

---

## Slide 26. 논문에 지금 넣을 수 있는 결과

| component | status |
|---|---|
| DDXPlus validation layer sensitivity | 확정 |
| DDXPlus locked finding/value probe | 확정 |
| DDXPlus structured reader locked | 확정 |
| DDXPlus Vanilla NLA locked | 확정 |
| DDXPlus semantic mapper V2 gates | 확정 |
| DiReCT exploratory Direct vs CoT | 확정, exploratory 표기 |
| DiReCT validation diagnosis probe | 확정 |
| DiReCT validation explanation alignment | 확정 |
| Medical-NLA development gate appendix | 확정 |
| D22 public AR diagnostic | validation diagnostic로 확정 |
| DiReCT locked 72/106 source/probe/NLA baseline | **pending** |
| Generative Medical-NLA locked row | validation 통과 모델 없음 |
| Text patching Table 4 | promotion 모델 없음, 현재 닫힘 |

### 논문의 현재 중심 주장

> Closed monitor는 의료 activation의 환자별 상태를 읽지만, 범용 NLA 또는 단순 domain fine-tuning은 이를 신뢰할 수 있는 자유 자연어 설명으로 변환하지 못한다.

---

## Slide 27. 다음 방법: Medical AO + Medical AR

### 왜 free-paragraph SFT를 더 반복하지 않는가?

- gold clinical text는 activation에 실제로 존재하지 않는 세부사항까지 포함할 수 있음
- sequence CE는 activation을 무시하고 disease template를 학습하기 쉬움
- deletion ranking은 global detector shortcut을 허용함
- 공개 AR은 이 의료 activation 분포를 구분하지 못함

### 제안 파이프라인

```text
Medical activation
      |
      v
Medical Activation Oracle
  - finding/value/uncertainty set
      |
      +--> domain Medical-AR matched-vs-shuffled 검증
      |
      v
constrained text renderer
  - claim selection과 문장 생성을 분리
```

### 단계

1. Train-only DDXPlus finding/value로 Medical Activation Oracle 학습
2. DiReCT claim ontology와 연결 가능한 공통 clinical-state schema 정의
3. Medical text-activation pair로 AR을 domain-adapt
4. Structured reader 양성 대조가 matched > shuffled를 통과하는지 확인
5. 그 이후에만 AV를 reconstruction/preference objective로 학습

---

## Slide 28. 교수님께 확인받을 결정

### 결정 1. 논문의 중심 프레이밍

- 제안: `Medical-NLA를 완성했다`가 아니라
- **`의료 activation의 정보 존재와 자연어 readout 사이의 격차를 규명했다`**로 중심 주장 설정

### 결정 2. 제출 범위

- 현재 baseline/closed-monitor 논문을 먼저 완결
- Medical AO + Medical AR은 후속 positive method로 분리할지 결정

### 결정 3. DiReCT locked batch 개봉

- Table 1A source behavior
- Table 1B HS24 probe
- Table 2 Source CoT/Vanilla NLA
- 동일 decision record/hash 아래 한 번에 수행

### 결정 4. 생성형 Medical-NLA 표기

- 주표에서는 validation gate 미통과로 제외
- 개발 실패와 shortcut 분석은 appendix/main analysis에 포함

---

## Slide 29. 결론

1. **정보는 있습니다.** DDXPlus locked probe에서 finding F1 0.9562, value accuracy 0.7659입니다.
2. **환자별 정보입니다.** 같은 진단 내 shuffle gap이 +0.1624와 +0.1868입니다.
3. **정적 readout은 가능합니다.** Structured reader finding F1은 0.9587입니다.
4. **자유 자연어 readout은 실패했습니다.** Vanilla NLA는 locked 10,028행에서 ontology claim을 하나도 내지 못했습니다.
5. **단순 SFT/ranking은 충분하지 않습니다.** Template collapse와 deletion detector shortcut이 확인됐습니다.
6. **공개 AR도 그대로는 사용할 수 없습니다.** Positive-control matched-vs-shuffled gap이 약 0입니다.
7. 다음 positive route는 **Medical Activation Oracle + domain-adapted AR + constrained renderer**입니다.

---

# Appendix

---

## Appendix A1. 주요 모델과 고정 설정

| component | setting |
|---|---|
| source model | `google/gemma-3-12b-it` |
| vanilla AV | `kitft/nla-gemma3-12b-L32-av` |
| released AR | `kitft/nla-gemma3-12b-L32-ar` |
| primary activation | CoT-P0 |
| closed probe frozen layer | HS24 |
| NLA/AR layer | HS32 |
| dtype | bfloat16 |
| semantic mapper primary | `gpt-5.6-sol` |
| semantic mapper auditor | `gpt-5.4` |

---

## Appendix A2. D9a support protocol

| item | value |
|---|---:|
| presence threshold | 0.90 |
| deletion-delta threshold | 0.00 |
| donor-margin threshold | 0.00 |
| validation positive coverage | 3,032/3,034 = 0.9993 |
| validation null false support | 112/2,964 = 0.0378 |
| supported train pairs | 3,104 |
| claims per retained case | 1 |
| unsupported policy | exclude |
| value edit included | no |

---

## Appendix A3. Counterfactual score의 해석

### 좋은 모델

- original에서 changed cue를 말함
- deletion 후 changed cue만 사라짐
- retained cue는 유지됨
- value edit 후 old 값은 사라지고 new 값이 나타남

### 나쁜 shortcut

- deletion activation이면 모든 claim NLL을 올림
- changed-gap은 커 보이지만 retained-gap도 같이 증가
- specificity = changed-gap - retained-gap이 0 또는 음수

### D10 실제 패턴

- step 1,552 changed mean: +0.5558
- retained mean: +0.5604
- specificity: -0.0046

---

## Appendix A4. D20 frozen gate와 결과

- changed-gap delta >= 0.05 in every seed
- changed-gap cluster CI > 0 in every seed
- specificity positive in every seed
- specificity cluster CI > 0 in every seed
- retained-gap non-inferiority upper bound: 0.0100
- changed/retained original NLL relative-increase upper bound: 0.1000
- teacher-forced gate 실패 시 generation/extension/sweep 금지

### 실제 결과

- changed-gap delta: -0.0143, -0.0040, -0.0266
- specificity: -0.0278, -0.0255, -0.0217
- final decision: **FAIL**

---

## Appendix A5. D22 해석 경계

### 말할 수 있는 것

- released AR은 현재 의료 CoT-P0 activation의 case-specificity 측정기로 부적합
- 절대 cosine만으로 text-grounding을 판정하면 안 됨

### 말할 수 없는 것

- activation에 의료 정보가 없음
- structured reader text가 임상적으로 틀림
- AR 기반 학습이 원리적으로 불가능함

### 다음 전제

- domain-adapted Medical-AR 자체가 matched-vs-shuffled positive control을 먼저 통과해야 함

---

## Appendix A6. 남은 locked 실행과 완료 조건

| order | job | output |
|---:|---|---|
| 1 | decision record 및 recipe hash 동결 | 접근 규율 증빙 |
| 2 | DiReCT test-seen 72 source 재집계 | Table 1A seen |
| 3 | DiReCT PDD-heldout 106 source 재집계 | Table 1A OOD |
| 4 | frozen HS24 probe 적용 | Table 1B DiReCT locked |
| 5 | Source CoT/Vanilla NLA semantic evaluation | Table 2 baseline |

### 현재 실행하지 않는 것

- validation gate 미통과 Medical-NLA의 locked generation
- DDXPlus locked threshold/layer 재선택
- D20 사후 checkpoint 선택 또는 추가 sweep
- public AR cosine을 바로 reward로 사용한 AV 학습

---

## Appendix A7. 발표 시 강조할 문장

> Probe는 activation에 정보가 있다는 것을 보여주지만, NLA가 그 정보를 읽었다는 것을 보여주지는 않습니다.

> Structured reader는 강한 closed-monitor baseline이지만 open-ended explanation model은 아닙니다.

> Vanilla NLA의 0점은 빈 출력 때문이 아니라, 환자별 frozen ontology finding을 포함하지 않는 generic clinical prose 때문입니다.

> D10의 큰 margin은 성공이 아니라 retained cue까지 억제하는 deletion-detector shortcut이었습니다.

> D20은 shortcut을 차단했지만 목표 signal도 사라졌고, 공개 AR은 positive control을 통과하지 못했습니다.

> 따라서 다음 방법은 더 많은 free-text SFT가 아니라 clinical-state selection과 language rendering의 분리입니다.
