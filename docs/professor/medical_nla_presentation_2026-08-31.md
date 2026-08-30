# Medical-NLA 교수님 미팅 자료

> **2026-08-31 갱신본**<br>
> 2026-08-27 예정 미팅이 진행되지 않아, 당시 초안을 현재 실험 결과와 수치로 대체했습니다.

---

## 자료 사용 원칙

- Method와 Data를 먼저 고정한 뒤 Results를 제시합니다.
- validation, locked test, post-hoc audit을 명확히 구분합니다.
- 아직 실행하지 않은 DiReCT locked 셀은 `pending`으로 남기며 추정값을 넣지 않습니다.
- 생성형 Medical-NLA의 실패 결과도 개발 gate와 함께 그대로 보고합니다.
- DDXPlus의 closed probe/structured reader와 open-ended NLA를 같은 방법처럼 해석하지 않습니다.

---

# Part I. Introduction

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
3. 다음 단계는 free-paragraph SFT 반복이 아니라 AR geometry audit과 Patchscopes 원인 분리, 이후 Medical Activation Oracle/domain-adapted AR입니다.

---

## Slide 2. 가설과 연구 질문

### 고정 가설

| 가설 | 내용 |
|---|---|
| H1 | CoT의 임상적 그럴듯함과 내부 상태 충실성은 같은 것이 아니다. |
| H2 | 의료 적응은 vanilla NLA보다 임상 설명을 개선할 수 있지만 SFT만으로는 충분하지 않다. |
| H3 | 개선된 설명은 독립적인 activation grounding 검증을 통과해야 한다. |

### 고정 연구 질문

| RQ | 질문 | 핵심 산출물 |
|---|---|---|
| RQ1: Clinical alignment | Medical-NLA가 CoT와 vanilla NLA보다 의사 주석의 임상 관찰과 관찰-진단 연결을 잘 복원하는가? | DiReCT Table 2 |
| RQ2: Activation grounding | 그 설명이 언어 prior가 아니라 해당 사례 activation과 finding 변화에 실제로 근거하는가? | DDXPlus Table 3, Figure 3 |
| RQ3: Causal intervention | 검증된 판독을 편집하고 AR로 복원했을 때 내부 상태와 행동이 선택적으로 변하는가? | Table 4, Figure 4 |

### 데이터셋별 검증 역할과 허용되는 표현

| 데이터/모집단 | 주 질문 | 제공하는 reference/control | 통과했을 때만 가능한 표현 |
|---|---|---|---|
| DiReCT | 생성 설명이 physician observation과 observation-rationale-diagnosis edge를 복원하는가? | physician annotation, Source CoT, seen/PDD-heldout split | clinically aligned explanation |
| DDXPlus originals | activation에 finding 존재와 native value가 표현되는가? | evidence ID, native value, same-diagnosis hard shuffle | patient-specific state is decodable |
| DDXPlus deletion/value edit | 설명 또는 probe가 하나의 cue/value 변화만 따라가는가? | matched original/deleted/edited activation family | activation-grounded readout |
| DiReCT/DDXPlus text-activation pair + AR | 생성 text가 own activation을 shuffled/mean보다 잘 복원하는가? | matched/shuffled reconstruction, positive controls | state-preserving natural-language bottleneck |
| Validation-gated intervention population | 검증된 text edit/patch가 목표 상태와 행동만 선택적으로 바꾸는가? | no patch, raw/oracle patch, target/non-target behavior | causal utility |

- DiReCT clinical alignment만으로 activation grounding을 주장하지 않습니다.
- DDXPlus의 closed finding/value 성능만으로 open-ended explanation 성공을 주장하지 않습니다.
- AR 양성 대조가 실패한 상태에서는 reconstruction 또는 causal utility를 열지 않습니다.

### RQ 이전의 Gate 0

> Probe로 activation에 진단·finding·value 정보가 decode 가능한지 확인하는 것은 세 RQ의 선행 조건입니다. 이는 Table 1의 representation audit이지 RQ1 자체가 아닙니다.

### 단계적 성공 조건

> RQ1만 통과하면 임상 설명 생성기, RQ2까지 통과해야 activation-grounded 내부 판독기, RQ3까지 통과해야 인과적으로 사용할 수 있는 자연어 bottleneck이라고 부릅니다.

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

## Slide 5. 내부 측정 도구와 평가 층을 분리

| 도구 | 실제 입력 | 출력 공간 | 측정하는 것 | 측정하지 못하는 것 |
|---|---|---|---|---|
| Linear/multi-label probe | P0 activation | 고정 diagnosis/finding/value ontology | 선형 decodability와 same-diagnosis 사례 특이성 | 새 claim·관계·자유 문장 생성 |
| Structured reader | frozen probe가 선택한 label/value | train-only canonical phrases | selected closed state의 결정론적 렌더링 | probe 없이 activation에서 claim을 발견하는 open NLA |
| Vanilla/Medical NLA | P0 activation | 자유 자연어 | activation-conditioned open-text readout | 임상 정렬·activation grounding의 자동 보장 |
| Activation reconstructor (AR) | 생성 text | reconstructed activation | own activation이 같은 진단의 다른 환자 activation보다 가까운지 측정 | text의 임상적 정확성·인과성의 자동 보장 |

### Closed와 open score를 같은 accuracy로 비교하지 않는 이유

- Probe와 structured reader는 정답 공간이 닫혀 있습니다.
- Open-ended NLA는 claim 선택과 문장 생성을 동시에 해결해야 합니다.
- AR cosine은 own activation을 shuffled/mean보다 구분하는 양성 대조를 통과해야만 해석합니다.

### 왜 분리하는가?

- probe 성공은 자연어 decoder 성공을 보장하지 않습니다.
- structured reader 성공은 probe가 선택한 ontology를 렌더링한 결과입니다.
- open-ended NLA는 claim 선택과 문장 생성을 동시에 해결해야 합니다.

---

## Slide 6. Closed probe 검증 규약

### Probe 1. DiReCT diagnosis probe

| item | protocol |
|---|---|
| input | CoT-P0 last-token activation, HS16/24/32 |
| train / validation | 266 / 52 |
| targets | canonical PDD 49-way, disease category 25-way |
| decoder | one affine linear layer + softmax |
| learning-rate grid | `3e-4`, `1e-3` |
| weight-decay grid | `0`, `1e-4`, `1e-3`, `1e-2` |
| class weighting | on/off 모두 비교 |
| training | 최대 300 epochs, patience 30, seed 17 |
| selection | validation NLL 최소, accuracy는 선택에 사용하지 않음 |
| control | majority baseline; locked test는 설정 동결 전 미접근 |

### Probe 2. DDXPlus finding/value probe

| item | protocol |
|---|---|
| input | CoT-P0 last-token activation, HS16/24/32 |
| train / validation | 4,655 / 4,525 originals |
| finding targets | train-supported 91 evidence IDs, multi-label sigmoid |
| value targets | evidence ID별 conditional native-value classifier, 6 tasks/32 values |
| support minimum | finding 20 train cases, value 10 train cases |
| learning-rate grid | `1e-3`, `3e-3` |
| weight-decay grid | `0`, `1e-3` |
| finding weighting | weighted/unweighted 비교 |
| threshold grid | `.1`, `.2`, `.3`, `.4`, `.5` |
| training | 최대 80 epochs, patience 8, batch 512, seed 17 |
| selection | own micro F1와 same-diagnosis shuffled gap을 우선해 HS24/threshold .5 동결 |

### 사례 특이성 control

1. 각 환자에 같은 diagnosis의 다른 `base_id`를 deterministic donor로 배정합니다.
2. 원래 text/label을 own activation과 donor activation에 각각 적용합니다.
3. `own score - shuffled score`를 paired gap으로 계산합니다.
4. 따라서 diagnosis template만 공통으로 읽는 probe는 높은 own score를 내더라도 gap이 작습니다.

### Probe가 증명하는 범위

- 증명: 고정 ontology의 label 정보가 activation에 선형적으로 존재함
- 증명하지 않음: 자유 자연어 설명 생성, 새로운 finding 발견, causal faithfulness

---

## Slide 7. Open-ended 설명을 어떻게 변환하고 채점하는가?

### DiReCT

- physician observation과 생성 설명을 claim 단위로 비교
- 주요 지표:
  - `Obspre`: observation precision
  - `Obsrec`: observation recall
  - `Obscomp`: gold/predicted observation set의 semantic Jaccard completeness
  - `Expcom`: matched observation에서 rationale와 diagnosis edge까지 일치한 비율
  - `Expall`: 전체 explanation chain의 일치 비율
- 진단명 언급만으로는 observation alignment를 인정하지 않음

DiReCT의 physician annotation은 `observation -> rationale -> diagnosis` graph이고, 원래 생성 CoT가 이 graph를 얼마나 복원하는지 평가하기 위해 설계되었습니다. 본 연구에서는 Source CoT뿐 아니라 NLA가 생성한 clinical explanation에도 같은 evaluator를 적용합니다. `O`를 physician gold observation set, `O_hat`을 생성 설명에서 추출한 observation set, `M`을 두 set 사이에서 의미상 대응된 observation, `m`을 observation과 그 rationale-diagnosis edge까지 모두 맞춘 항목 수라고 정의합니다.

```text
Obspre  = |M| / (|O_hat| + 1)
Obsrec  = |M| / (|O| + 1)
Obscomp = |M| / |O union O_hat|
Expcom  = m / |M|
Expall  = m / |O union O_hat|
```

- `Accdiag`: 생성 설명의 최종 diagnosis가 gold diagnosis와 일치하는가
- `Obspre`: 모델이 observation이라고 생성한 내용 중 physician observation과 대응되는 비율. 불필요한 finding을 많이 말하면 낮아집니다.
- `Obsrec`: physician observation 중 생성 설명이 회수한 비율. 중요한 finding을 누락하면 낮아집니다.
- `Obscomp`: 누락과 불필요한 finding을 동시에 벌점 주는 semantic Jaccard입니다. 현재 observation-only Medical-NLA의 주 DiReCT 지표입니다.
- `Expcom`: 이미 observation이 대응된 항목만 놓고 rationale와 linked diagnosis까지 맞았는지 측정합니다.
- `Expall`: observation 누락·추가와 rationale·diagnosis edge 오류를 모두 포함한 end-to-end explanation-chain alignment입니다.

공식 `Obspre/Obsrec`은 분모에 `+1` smoothing을 사용하므로 완전한 oracle도 정확히 1.0이 되지 않을 수 있습니다. 현재 Medical-AV SFT target은 diagnosis와 rationale를 제거한 `<observed>` finding schema이므로 `Obscomp`를 primary로 사용하고, `Expcom/Expall`은 Source CoT 비교와 향후 Full Medical-NLA를 위한 exploratory metric으로 해석합니다.

### DDXPlus semantic mapper

| stage | input/operation | output/control |
|---|---|---|
| 0. deterministic split | `<observed>` bullet 우선, 없으면 문장 분리 | 빈 출력도 0-claim case로 분모 유지 |
| 1. frozen lexical | official-train modal phrase + release metadata alias | ambiguous alias 제외, 수동 동의어 없음 |
| 2. method-blind AI | 잔여 claim + frozen 91-evidence ontology | JSON `(evidence_id, value_id/null, exact quote)` |
| aggregation | case별 evidence/value set dedupe | structured reader와 같은 metric 코드 재사용 |

### AI mapper에 주지 않은 정보

- method 이름, case ID, diagnosis, split, gold cue/value
- probe probability와 intervention type
- 다른 method의 출력

### 결정론과 provenance

- exact supporting quote가 원문에 없으면 mapping을 거부합니다.
- claim, ontology, prompt, model ID를 묶은 SHA-256 key로 cache합니다.
- alias table, prompt, mapper code commit, model ID, G1-G4 receipt를 locked scoring 전에 동결합니다.

### Mapper validation

- G1 reader round-trip
- G2 absent-cue false mapping
- G3 cache/replay determinism
- G4 independent AI concordance

> 이 계측기는 open-generator끼리 공통으로 사용하며, probe/reader의 closed score를 대체하지 않습니다.

### Slide 7의 역할

- DiReCT에서는 생성 text를 physician observation/rationale/diagnosis reference와 비교해 **RQ1 clinical alignment metric**을 만듭니다.
- DDXPlus에서는 생성 text를 frozen evidence/value ontology로 변환합니다. Semantic mapper 자체는 metric이 아니라 **text-to-label 계측기**입니다.
- 변환된 DDXPlus label set에 finding F1, value accuracy와 아래 counterfactual metric을 적용합니다.

---

## Slide 8. Counterfactual paired grounding을 어떻게 측정하는가?

### 왜 정적 정확도만으로는 부족한가?

원본 activation에서 finding을 맞혔다는 사실만으로는 모델이 해당 환자 상태를 읽었다고 결론낼 수 없습니다. 다음 shortcut도 같은 정답을 만들 수 있기 때문입니다.

- diagnosis별 전형적인 finding을 반복함
- prompt 또는 데이터셋의 평균 cue 빈도를 복원함
- 어떤 activation이 들어와도 비슷한 claim을 출력함
- 삭제본이라는 사실만 감지해 모든 claim을 함께 억제함

Counterfactual 평가는 **같은 환자에서 한 정보만 바꾸고 나머지를 고정**해, 출력이나 probe score가 바로 그 변화만 선택적으로 따라가는지 묻습니다.

```text
정보 존재:       original에서 cue A를 읽는가?
사례 특이성:     같은 진단의 다른 환자 activation으로 바꾸면 성능이 떨어지는가?
변화 추종:       cue A만 삭제하면 A만 감소하고 B/C는 유지되는가?
값 갱신:         value 3을 5로 바꾸면 old 3은 줄고 new 5는 증가하는가?
```

### Hard shuffle과 counterfactual의 역할 차이

| control | 바꾸는 것 | 배제하는 shortcut |
|---|---|---|
| same-diagnosis hard shuffle | 환자 activation 전체 | diagnosis template만으로 맞히는 경우 |
| cue deletion | 특정 cue 하나 | 해당 cue가 없어도 계속 말하는 경우 |
| native-value edit | 한 evidence의 값 | evidence 이름만 읽고 실제 값을 무시하는 경우 |
| retained-cue control | 바꾸지 않은 cue | 삭제본이면 모든 claim을 억제하는 global detector |

> 여기서 counterfactual은 readout의 activation grounding을 검증하는 paired intervention입니다. 아직 text를 편집해 source-model 행동을 바꾸는 RQ3 causal intervention은 아닙니다.

### Counterfactual activation을 만드는 방법

| arm | input text 변경 | activation 재추출 | gold/target |
|---|---|---|---|
| original | 없음 | CoT-P0 | 원래 cue set/value |
| cue deletion | 사전 선택한 cue 1개를 물리적으로 제거 | 같은 CoT prompt로 재추출 | 삭제 cue와 공통 retained cues |
| native-value edit | 하나의 categorical value를 다른 유효 native value로 교체 | 같은 CoT prompt로 재추출 | old/new value pair |

- diagnosis text나 label을 prompt에 추가하지 않습니다.
- original/deleted/value-edited family는 같은 `base_id`로 묶습니다.
- deletion은 한 번에 cue 하나만 바꿔 어떤 정보가 변했는지 식별 가능하게 합니다.
- hidden dimension을 임의로 편집하지 않고 source model을 다시 실행해 각 activation이 실제 입력에서 나온 on-manifold state가 되게 합니다.
- test에서 새 intervention 또는 threshold를 만들지 않습니다.

### Cue deletion metric

```text
Original activation: cue A + cue B + cue C
Deleted activation:          cue B + cue C
```

- deletion probability drop: 지운 cue의 probe probability가 얼마나 감소했는가
- original hit: 원본에서 지운 cue를 실제로 읽었는가
- deletion phantom: 삭제본에서도 그 cue를 계속 말하는가
- removal success: 원본에서 읽힌 cue가 삭제본에서 사라졌는가
- untouched retention: 삭제하지 않은 cue가 유지되는가

### Native-value edit metric

```text
Original: rash severity = 3
Edited:   rash severity = 5
```

- replacement hit: 새 값을 출력했는가
- old-value persistence: 이전 값을 계속 출력했는가
- clean switch: 새 값은 출력하고 이전 값은 출력하지 않았는가

> 단순 정확도보다 intervention에 대한 선택적 반응이 grounding의 더 강한 증거입니다.

### 두 병목을 분리하는 판정

- static F1이 낮음: activation state selection 또는 decoder 모두 문제일 수 있음
- static F1은 높고 phantom이 높음: 삭제 후에도 representation에 cue가 남는 표현 병목
- probe는 반응하지만 NLA가 반응하지 않음: language decoder 병목

---

## Slide 9. 생성형 Medical-NLA 개발 규율

### 용어 계약: 어디까지를 SFT-only와 Full Medical-NLA라고 부르는가?

| 방법 계열 | Clinical CE | Counterfactual/pair grounding | 검증된 medical AR reconstruction | 현재 상태와 허용 명칭 |
|---|---:|---:|---:|---|
| Vanilla NLA | No | No | 공개 general-domain pretraining만 | 공개 baseline; Medical-NLA가 아님 |
| Medical-AV SFT-only | Yes | No | No | 구현·평가 완료; clinical-format SFT ablation |
| Grounding-aware surrogate | Yes | sequence CE, ranking 또는 retained anchor | No | 구현·평가 완료; 모두 promotion gate 실패 |
| Reconstruction-capable Medical-NLA | Yes | 선택적 | Yes | domain-valid AR가 없어 아직 미구현 |
| Full Medical-NLA | Yes | Yes | Yes | 아직 실현되지 않은 최종 계약; 완료된 방법으로 표현하지 않음 |

```text
Clinical SFT:      무엇을 어떤 의료 언어로 말할 것인가?
Pair grounding:    바로 이 activation과 cue/value 변화에 해당하는가?
Reconstruction:    생성 text가 원 activation의 사례 정보를 보존하는가?
Full Medical-NLA:  세 조건을 모두 학습하고 독립 gate로 검증한 경우
```

### 이 표를 읽는 방법

이 표는 방법들의 성능 순위를 나타내는 표가 아니라, **어떤 학습 신호를 사용했고 그 결과를 어떤 이름으로 부를 수 있는지**를 구분하는 방법론적 계약입니다. 특히 의료 문장을 잘 생성하는 것, 해당 환자의 activation을 구분해 읽는 것, 생성 문장이 원 activation의 정보를 보존하는 것은 서로 다른 조건입니다.

**Clinical CE**는 activation을 입력받은 AV가 정답 임상 설명을 생성하도록 하는 token-level cross-entropy loss입니다. 이 학습은 모델에 출력 schema, 임상 finding의 표현 방식, 의료 문장의 어휘를 가르칩니다. 그러나 loss가 낮아졌다는 사실만으로 생성된 finding이 바로 그 환자의 activation에서 읽힌 것이라고 결론 내릴 수는 없습니다. 진단별로 반복되는 전형적 문장이나 학습 target의 주변 분포를 배웠을 가능성이 남기 때문입니다.

```text
activation -> AV -> "환자는 발열과 마른기침을 보인다"
                    ^ 의료적으로 올바른 형식은 Clinical CE로 학습 가능
                    ^ 이 환자 activation에 특이적인지는 별도 검증 필요
```

**Counterfactual/pair grounding**은 같은 환자에서 하나의 cue 또는 value만 바꿔 다시 계산한 activation 쌍을 사용합니다. 원본 activation에서 읽힌 cue는 삭제 activation에서 사라져야 하고, 삭제하지 않은 cue는 유지되어야 합니다. 따라서 이 항목은 단순히 설명이 의료적으로 자연스러운지가 아니라, 설명이 **바로 이 activation의 사례별 변화에 반응하는지**를 다룹니다.

- `sequence CE`: original, cue-deleted, value-edited activation마다 현재 cue set에 맞는 전체 설명을 target으로 학습
- `ranking`: 삭제된 cue의 문장이 original activation에서는 더 낮은 NLL을, deleted activation에서는 더 높은 NLL을 갖도록 학습
- `retained anchor`: 삭제하지 않은 cue의 NLL은 original/deleted activation 사이에서 유지되도록 제약

이 목적함수를 사용했다는 것과 grounding에 성공했다는 것은 다릅니다. 현재 `Grounding-aware surrogate` 계열은 grounding을 겨냥한 loss를 사용했지만, changed-cue 반응과 retained-cue specificity를 함께 요구한 promotion gate를 통과하지 못했습니다. 따라서 **grounding-aware objective를 시도했다**고는 할 수 있지만 **grounded Medical-NLA를 완성했다**고 표현하지 않습니다.

**검증된 medical AR reconstruction**은 AV가 생성한 text를 AR이 다시 activation으로 복원할 수 있는지를 뜻합니다.

```text
source activation h -> AV -> text z -> AR -> reconstructed activation h_hat
```

사례 정보가 text에 보존되었다면 `h_hat`은 같은 진단의 다른 환자 activation보다 자기 activation `h`에 더 가까워야 합니다. 따라서 높은 cosine 자체보다 `cos(h_hat, h_own) - cos(h_hat, h_same-diagnosis-shuffled)`가 양수인지가 중요합니다. 공개 general-domain AR는 D22에서 own과 shuffled가 거의 같아 positive control을 통과하지 못했습니다. 그러므로 공개 AR가 존재한다는 이유만으로 medical reconstruction 신호가 검증되었다고 간주하지 않습니다.

각 방법 계열의 해석 범위는 다음과 같습니다.

- **Vanilla NLA**: 공개 AV를 수정 없이 사용한 baseline입니다. 의료 target, pair grounding, 검증된 medical AR가 없으므로 Medical-NLA라고 부르지 않습니다.
- **Medical-AV SFT-only**: Clinical CE로 의료 finding과 출력 형식을 학습했습니다. 임상 언어 생성은 평가할 수 있지만 사례별 activation grounding은 별도 증거가 필요합니다.
- **Grounding-aware surrogate**: Clinical CE에 sequence CE, ranking 또는 retained anchor를 추가했습니다. 의료 grounding을 겨냥했지만 현재 실험에서는 promotion gate에 실패했습니다.
- **Reconstruction-capable Medical-NLA**: 의료 분포에서 own-vs-shuffled positive control을 통과한 AR를 확보하고 reconstruction objective를 사용할 수 있는 단계입니다. 현재는 domain-valid AR가 없어 미구현입니다.
- **Full Medical-NLA**: Clinical CE, pair grounding, medical AR reconstruction을 모두 사용하고 각각의 독립 gate까지 통과한 최종 성공 조건입니다. 현재 완료된 모델의 이름이 아니라 사전 정의한 목표입니다.

> 현재까지의 `full-data SFT`는 DDXPlus 4,655건을 모두 사용했다는 데이터 규모 이름입니다. `Full Medical-NLA`를 구현했다는 뜻이 아닙니다.

### 공통 출력 prompt/schema

```text
The vector above was extracted after a clinical presentation and a reasoning
instruction, immediately before the source model began its response.
Read the patient-specific clinical state represented by the vector.

Report only concrete clinical findings represented by the vector.
Do not infer a diagnosis, add background medical knowledge, or complete a
typical disease template.

<explanation><readout><observed>
- patient-specific clinical finding
</observed></readout></explanation>
```

### Common/full-data SFT recipe

| setting | value |
|---|---|
| base | `kitft/nla-gemma3-12b-L32-av` |
| activation | CoT-P0/HS32/last token |
| train | DDXPlus 4,655 + DiReCT 248 |
| validation | DDXPlus 50 + DiReCT 50 |
| source sampling | exponent alpha=.5; DiReCT case당 약 4.3회 exposure |
| optimizer | AdamW, learning rate `2e-4`, weight decay 0, gradient clipping 1.0 |
| batch | 4, gradient accumulation 2, effective 8 |
| LoRA | rank 16, alpha 32, dropout .05 |
| modules | q/k/v/o, gate/up/down projections |
| precision | frozen base bf16, trainable LoRA fp32 |
| checkpoint selection | DDXPlus/DiReCT source-macro content-token NLL |
| generation | greedy, max new tokens 512, batch 4 |

### Target construction

- DDXPlus: original presentation의 train-supported cue를 입력 순서대로 중복 제거, 최대 12개
- DiReCT: physician deduction 중 `observation_exact_in_note=true`만 사용, 최대 12개
- diagnosis와 backbone source answer는 common target에서 제거
- bullet content token loss와 XML scaffold token loss를 분리

### 실제로 시도한 방법 계열

| phase | supervision/objective | 핵심 질문 |
|---|---|---|
| Direct-only SFT | DiReCT 248 physician observation | 소량 domain target만으로 읽는가? |
| Common mixed SFT | DiReCT 248 + DDXPlus 248 | 동일 schema가 transfer를 돕는가? |
| Full-data SFT | DiReCT 248 + DDXPlus 4,655 | 데이터 양이 병목인가? |
| Counterfactual sequence SFT | original/deletion/value-edit 각각의 현재 cue set CE | intervention 예시를 직접 보여주면 반응하는가? |
| Sentence contrastive | matched text NLL < crossed text NLL | activation-target alignment를 직접 키울 수 있는가? |
| Changed-cue 1x2 ranking (D10) | one claim, original NLL < deleted NLL | 긴 문장 난이도를 제거하면 changed cue를 배우는가? |
| OOF finding-set distillation (D14) | OOF probe가 선택한 hard finding set | activation-supported target만 학습하면 되는가? |
| 256-d soft bottleneck (D16) | 3840→256→3840 + training-only auxiliary head | continuous support가 shared latent를 조직화하는가? |
| Specificity-anchored ranking (D20) | changed ranking + retained claim CE on original/deleted | global deletion-detector shortcut을 막을 수 있는가? |
| Public AR matched-vs-shuffled diagnostic (D22) | text reconstruction matched vs shuffled | 원 NLA의 AR가 medical distribution을 측정하는가? |

### 공통 promotion 원칙

- 3개 seed 방향 일치
- category-cluster bootstrap CI가 0보다 큼
- 최소 효과 크기 충족
- changed cue뿐 아니라 retained cue specificity 통과
- validation gate 통과 전 locked test 금지
- 실패 후 checkpoint/threshold를 사후 선택하지 않음

> 잘 말하는 모델을 찾는 것이 아니라 activation-dependent한 모델을 찾습니다.

### 진단 경로의 현재 상태

1. **Geometry audit A1-A5 완료**: DDXPlus positive control 실패, 모든 arm FVE < 0으로 공개 AR reward 불인정
2. **Same-layer Patchscope 완료**: short general-domain control은 통과했지만 clinical own/donor correspondence는 0/5
3. **현재 후보**: text bypass가 없는 learned medical prefix mapper

따라서 AR reward/RL과 추가 identity-Patchscope sweep은 열지 않고, 다음 학습은 별도 사전
등록한 supervised activation-language decoder로만 진행합니다.

---

## Method subsection. Experimental setup and data

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

# Part III. Experimental Results

---

## Slide 12. Results map: 질문별로 무엇을 먼저 보는가?

| 연구 단계 | 대응 논문 표 | 질문 | 메인 슬라이드 | 보조/감사 슬라이드 |
|---|---|---|---:|---|
| Gate 0 | Main Table 1 | activation에 임상정보가 실제로 존재하는가? | 13 | layer sensitivity 14, Vanilla boundary 15 |
| RQ1 | Main Table 2 | 생성 설명이 physician observation과 정렬되는가? | 16 | template-collapse audit 17 |
| RQ2 | Main Table 3A | 설명이 해당 사례 activation에 grounded되는가? | 18 | semantic-mapper audit 20 |
| RQ2 | Main Table 3B | 삭제/value edit에 선택적으로 반응하는가? | 19 | 평가 분모 11 |
| Development | Appendix gate table | 어떤 실패가 다음 Medical-NLA 설계를 만들었는가? | 21~28 | seed/gate 세부 결과 |
| RQ3 | Conditional Table 4 | 검증된 설명 편집이 내부 상태와 행동을 바꾸는가? | 아직 닫힘 | grounded readout + AR gate 필요 |

> Direct-vs-CoT 171-case pilot과 McNemar 검정은 최종 RQ 표가 아니라 source behavior 보조 통제이므로 Appendix로 이동합니다.

---

## Slide 13. Gate 0 = Main Table 1: P0에 설명할 의료 정보가 있는가?

| dataset/target | decoder와 layer | validation | locked evaluation | control |
|---|---|---:|---:|---|
| DiReCT disease category | 25-way linear, HS24 | **.5962** | pending, seen 72 | majority .0577 |
| DiReCT canonical PDD | 49-way linear, HS24 | **.4423** | pending, seen 72 | majority .0962 |
| DDXPlus finding presence | 91-label linear, HS24 | .9607 | **.9562** | shuffled .7938; gap +.1624 [.1576,.1672] |
| DDXPlus native value | 6 tasks/32 values, HS24 | .7700 | **.7659** | shuffled .5791; gap +.1868 [.1650,.2091] |

### 실험 설계

- Official train activation으로 linear probe를 학습하고 validation에서 layer/regularization을 선택했습니다.
- DDXPlus는 같은 diagnosis의 다른 환자 activation을 donor로 붙인 hard-shuffle control을 사용했습니다.
- DiReCT PDD-heldout 106은 train 49-way head에 output node가 없으므로 PDD probe가 `0`이 아니라 `N/A`입니다.

### Gate 0 판정

- DiReCT P0에서 diagnosis/category 정보가 majority보다 높게 선형 판독됩니다.
- DDXPlus finding/value는 locked test에서도 높고, own activation이 same-diagnosis donor보다 유의하게 높습니다.
- 따라서 생성형 NLA 실패를 `activation에 의료 정보가 없음`으로 설명할 수 없습니다.
- 이 표는 closed-space feasibility audit이며 open-text Medical-NLA 성공을 의미하지 않습니다.

---

## Slide 14. Main Table 1 support / Figure 2: 왜 probe는 HS24인가?

### Linear probe, validation n=52

| target | HS | majority | Top-1 | Top-5 | MRR | macro recall | val NLL |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical PDD | 16 | 0.0962 | 0.3846 | 0.6923 | 0.5294 | 0.3597 | 2.5533 |
| canonical PDD | **24** | 0.0962 | **0.4423** | **0.7692** | **0.5762** | **0.3868** | **2.0489** |
| canonical PDD | 32 | 0.0962 | 0.3846 | 0.6923 | 0.5335 | 0.2771 | 2.3784 |
| disease category | 16 | 0.0577 | 0.5000 | 0.7885 | 0.6374 | 0.4833 | 1.9679 |
| disease category | **24** | 0.0577 | **0.5962** | **0.9038** | **0.7284** | **0.5000** | **1.3961** |
| disease category | 32 | 0.0577 | 0.5192 | 0.8654 | 0.6609 | 0.4426 | 1.6869 |

### Early forced-answer behavioral baseline

| target/ranking | candidates | Top-1 | Top-5 | MRR | mean gold rank |
|---|---:|---:|---:|---:|---:|
| category raw | 25 | 0.4808 | 0.6731 | 0.5814 | 5.02 |
| category calibrated sensitivity | 25 | 0.2308 | 0.3077 | 0.3091 | 9.58 |
| PDD raw, full ontology | 61 | 0.1538 | 0.4423 | 0.3168 | 8.77 |
| PDD raw, train ontology | 49 | 0.1538 | 0.5192 | 0.3250 | 7.92 |
| PDD calibrated sensitivity | 49 | 0.0577 | 0.1346 | 0.1486 | 15.83 |

> 저장된 P0 activation을 output head로 직접 unembed한 결과가 아닙니다. CoT prompt 뒤에 `The answer is`와 각 후보 문자열을 teacher-force한 ontology-given sequence ranking입니다.

### 해석

- P0 activation에 진단 관련 선형 정보가 존재합니다.
- calibration prompt는 `Clinical case:\nN/A\n\nWhat is the most likely diagnosis?`이며 후보 문자열 prior를 차감했지만 성능을 크게 낮췄습니다.
- HS24가 validation에서 category와 PDD 모두 최선이었습니다.
- Raw PDD는 35/52에서 하나의 희귀 후보를 top-1으로 골라 candidate surface-form prior가 강했습니다.

### DDXPlus validation layer sensitivity

| target/metric | HS16 | HS24 | HS32 |
|---|---:|---:|---:|
| finding micro F1, n=4,525 | **0.9636** | 0.9607 | 0.9607 |
| finding own-shuffled gap | +0.1651 | **+0.1653** | +0.1546 |
| value accuracy, n=2,183 | 0.7641 | **0.7700** | 0.6990 |
| value own-shuffled gap | +0.1842 | **+0.1942** | +0.1205 |

- HS16의 finding F1이 0.0029 높았지만 HS24가 finding gap, value accuracy, value gap의 공동 기준에서 우세했습니다.
- HS24를 validation에서 동결한 뒤 DDXPlus locked test에서 다시 선택하지 않았습니다.
- HS32 Vanilla/Medical-AV는 공개 AV architecture의 native input이고, HS24 probe와의 차이는 표의 `input layer`에서 명시합니다.

---

## Slide 15. Main Table 1 support: 정보 존재와 Vanilla verbalization의 차이

### P0 blinded semantic audit, validation n=52

| prompt | input HS | parse | source answer | gold PDD | category |
|---|---:|---:|---:|---:|---:|
| Default | 16 | 1.0000 | 0/52 | 0/52 | 1/52 |
| Default | 24 | 1.0000 | 0/52 | 0/52 | 0/52 |
| **Default primary** | **32** | **1.0000** | **0/52** | **0/52** | **0/52** |
| Task-aligned | 16 | 1.0000 | 0/52 | 0/52 | 1/52 |
| Task-aligned | 24 | 1.0000 | 0/52 | 0/52 | 0/52 |
| Task-aligned | 32 | 1.0000 | 0/52 | 0/52 | 0/52 |

- 총 P0 readouts: 52 cases x 2 prompts x 3 layers = **312**
- local Llama-3-8B judge가 `match=true`일 때 readout의 exact supporting quote를 요구했습니다.
- Lexical default/task-aligned HS32도 source/gold/category가 모두 0이었습니다.

### P1/P2 leakage control, source-answer lexical mention

| prompt | P1 | P2 | leakage-free P1 subset |
|---|---:|---:|---:|
| Default | 0.5192 | 0.5962 | 0/5 |
| Task-aligned | 0.5577 | 0.5000 | 0/5 |

### 해석

- 공개 Vanilla NLA는 P0의 진단 정보를 읽지 못했습니다.
- P1/P2의 절대 mention은 reasoning/answer 문자열 노출과 일치하고, leakage-free P1은 0/5이므로 P0 reader 성공이 아닙니다.
- HS16/24 입력은 HS32용 decoder와 layer mismatch가 있으므로 sensitivity일 뿐 primary 성능이 아닙니다.

---

## Slide 16. RQ1 = Main Table 2: physician explanation과 정렬되는가?

> 교수님 발표에서는 현재 사용 가능한 validation 50 결과를 채웁니다. 논문 최종본은
> 같은 열로 test-seen 72와 PDD-heldout 106을 별도 panel에 보고합니다.

| method | input | extraction | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | source text | 50/50 | 0 | .2835 | .3726 | **.2130** | **.0650** | .0153 |
| Vanilla NLA | HS32 AV | 10/50 (pilot) | 0 | .0000 | .0000 | .0000 | .0000 | .0000 |
| Medical-AV, SFT only seed17 | HS32 AV | 50/50 | 0 | .0544 | .0502 | .0301 | .0000 | .0000 |
| Medical-AV, SFT only seed29 | HS32 AV | 50/50 | 0 | .0553 | .0388 | .0296 | .0000 | .0000 |
| **Medical-NLA, final** | layer/recipe 사전 동결 | pending | pending | pending | pending | pending | pending | pending |

### RQ1 판정

- DDXPlus train 4,655건을 추가한 SFT도 Source CoT의 Obscomp `.2130`에 크게 미달했습니다.
- SFT는 extraction 가능한 의료 형식을 만들었지만 observation/rationale alignment를 복원하지 못했습니다.
- Vanilla의 `10/50`은 빈 출력이 아니라 평가 가능한 observation claim을 추출한 case coverage입니다.
- `Medical-NLA, final`은 성공 방법을 위한 조건부 행이며, validation gate 뒤에만 locked 72/106을 엽니다.

---

## Slide 17. Main Table 2 support: 낮은 alignment가 scorer 문제인가?

> 전체 50-case deterministic exact-text census입니다. 불안정했던 AI checklist 결과는 최종 판정에서 제외했습니다.

| method | Obscomp | exact duplicate rows | unique outputs |
|---|---:|---:|---:|
| Direct-only seed17 | 0.0343 | 43/50 | 7 |
| Direct-only seed29 | 0.0047 | 47/50 | 3 |
| Direct-only seed43 | 0.0032 | 49/50 | 1 |
| Full-data seed17 | 0.0301 | 36/50 | 14 |
| Full-data seed29 | 0.0296 | 48/50 | 2 |
| Source CoT | **0.2130** | 0/50 | 50 |

- 낮은 Obscomp는 lexical/semantic scorer의 보수성만으로 설명되지 않습니다.
- SFT 출력 자체가 서로 다른 환자에서 동일하거나 소수의 의료 template로 붕괴했습니다.
- 따라서 RQ1은 현재 validation에서 실패이며 locked Medical-NLA 행을 열지 않았습니다.

---

## Slide 18. RQ2 = Main Table 3A: 해당 사례 activation을 읽는가?

| method class | method | input layer | finding F1 | shuffled F1 | pair gap | native-value acc |
|---|---|---:|---:|---:|---:|---:|
| closed decoder | Frozen probe | HS24 | .9562 | .7938 | +.1624 | .7659 |
| structured monitor | Probe-guided reader | HS24 | **.9587** | .7938 | +.1624 | **.7654** |
| open generator | Vanilla NLA | HS32 | .0000 | .0000 | .0000 | .0000 |
| open generator | Medical-AV, SFT only | HS32 | validation only | validation only | promotion fail | validation only |
| open generator | **Medical-NLA, final** | 사전 동결 | pending | pending | pending | pending |

### Structured reader의 정확한 의미

1. Frozen HS24 probe가 91개 finding과 지원되는 native value의 label/probability를 선택합니다.
2. 각 label을 official-train-only modal phrase lexicon으로 결정론적 bullet로 렌더링합니다.
3. prompt text, diagnosis, gold cue를 사용하지 않고 자유 생성도 하지 않습니다.

> 따라서 structured reader는 “probe가 고른 state를 말로 표시할 수 있다”는 closed-monitor 양성 대조이지, activation에서 자유 문장을 생성하는 NLA의 성공이 아닙니다.

- Pair gap은 own activation 점수에서 같은 diagnosis의 다른 환자 activation 점수를 뺀 값입니다.
- HS24는 closed probe의 validation-selected layer이고 HS32는 generative AV의 architecture-native input입니다.

---

## Slide 19. RQ2 = Main Table 3B: counterfactual change에 선택적으로 반응하는가?

| method | original hit | deletion phantom | removal | retention | replacement | old persist | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Vanilla NLA | .0000 | .0000 | N/A | N/A | .0000 | .0000 | N/A |
| Medical-AV, SFT only | validation development only | validation development only | validation development only | validation development only | validation development only | validation development only | validation development only |
| **Medical-NLA, final** | pending | pending | pending | pending | pending | pending | pending |

### Locked 분모와 해석

- Deletion: 4,540 pairs; untouched retention: 16,105 finding occurrences
- Native value edit: 539 pairs; clean-switch eligible: 398
- Static finding state는 잘 읽히지만 cue deletion 반응은 부분적이고 value clean switch는 `.0804`입니다.
- Vanilla phantom `.0000`은 성공이 아닙니다. Original hit도 `.0000`이라 removal/retention/clean-switch 분모가 없습니다.
- Final Medical-NLA는 changed cue 제거뿐 아니라 retained cue 보존과 old→new value 전환을 동시에 통과해야 합니다.

---

## Slide 20. Main Table 3 audit: open-ended Vanilla가 0인 이유

### Frozen semantic mapper validation

| gate | metric | result | criterion |
|---|---|---:|---:|
| G1 | reader finding/value round-trip | 1.0000 / 1.0000 | >= 0.98 |
| G2 | absent-target false map | 0/2,609 | <= 0.05 |
| G3 | cache replay byte-identical | True | True |
| G4 | evidence/value disagreement | 0.0200 / 0.0000 | <= 0.05 |

### Locked 10,028-row Vanilla result

| item | result |
|---|---:|
| original / deleted / edited rows | 4,543 / 4,543 / 942 |
| lexical mappings | 0 |
| AI semantic mappings | 0 |
| rows with emitted ontology claim | **0/10,028** |

### Post-hoc 20-case audit

| audit item | result |
|---|---:|
| generic clinical prose only | 20/20 |
| possible frozen-mapper miss | 0/20 |
| expected-cue paraphrase match | 0/20 |
| malformed/empty | 0/20 |

> 출력은 비어 있지 않았지만 환자의 frozen ontology finding과 연결되지 않는 일반적 임상 문장이었습니다.

---

## Slide 21. Medical-NLA는 실패 원인에 따라 어떻게 바뀌었는가?

| 단계 | 먼저 시도한 것 | 관찰된 문제 | 그래서 다음에 바꾼 것 |
|---:|---|---|---|
| 1 | Original/full-data sequence SFT | 형식은 배웠지만 환자별 observation 대신 질환 template로 붕괴 | original/deleted/value-edited target을 직접 학습 |
| 2 | Counterfactual sequence SFT | seed17 recall은 증가했지만 phantom도 약 2배, seed29에서 미재현 | 문장 생성 CE 대신 activation-target pair ranking |
| 3 | Sentence matched/crossed contrastive | 양의 NLL gap은 생겼지만 효과가 `+.001~+.005`로 매우 작음 | 한 개 changed cue만 비교하는 1x2 ranking |
| 4 | Changed-cue ranking, 20 steps | 세 seed 방향은 대체로 양수지만 최소 효과 `.05`에 크게 미달 | loss/data를 고정하고 1,552 steps까지 budget만 확대 |
| 5 | Budget calibration | changed와 retained gap이 함께 증가: deletion activation 전체를 억제하는 shortcut | retained cue CE를 넣은 specificity anchor |
| 6 | Specificity-anchored ranking | shortcut은 사라졌지만 changed-cue signal도 3 seed 모두 음수 | target/latent 구조를 OOF teacher와 bottleneck으로 변경 |
| 7 | OOF teacher / 256-d bottleneck | calibration과 3-seed 효과 모두 gate 실패 | 원 NLA의 text-to-activation AR 측정기로 회귀 |
| 8 | Public AR geometry | 의료 분포 reconstruction 측정기와 reward gate 실패 | AR를 우회하는 same-layer identity Patchscope |
| 9 | Identity Patchscope | general control은 통과했지만 clinical own/donor finding은 0/5 | text bypass가 없는 learned medical prefix mapper 후보 |

> 각 변경은 임의의 방법 sweep이 아니라, 직전 실험에서 확인한 실패 모드를 하나씩 제거하기 위한 설계 변경입니다.

---

## Slide 22. 시도 1→2: 데이터 증가만으로 부족해 counterfactual SFT로 이동

### 시도 1. Original-only sequence SFT

`CE(y_current | h_current)`로 현재 finding 문장을 생성하게 하고, DDXPlus 학습량을
248건에서 4,655건으로 늘렸습니다.

### Original-only common/full-data SFT, validation

| method | DDX cue recall | cue precision | DiReCT lexical recall | current finding | deletion phantom | removal success | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 248+248 pilot, seed29 | 0.1784 | 0.2533 | 0.0000 | 0.1499 | 0.1356 | 0.0000 | 0.0000 |
| full data, seed17 | **0.3763** | **0.3816** | 0.0216 | 0.3389 | **0.2138** | **0.4052** | **0.0244** |
| full data, seed29 | 0.3506 | 0.3758 | 0.0076 | **0.3612** | 0.2667 | 0.3232 | 0.0122 |

### 문제 1

- Full-data SFT는 DDXPlus cue recall을 높였지만 DiReCT observation alignment는 여전히 매우 낮았습니다.
- 출력 형식과 의료 문체를 배우는 것과 환자 activation을 읽는 것은 같지 않았습니다.

### 그래서 바꾼 것: original/deleted/value-edited target을 모두 CE로 학습

Counterfactual sequence SFT, validation 435 bases / 952 readouts:

| method | current recall | original target hit | deleted phantom | deletion contrast | removal success | clean switch |
|---|---:|---:|---:|---:|---:|---:|
| original-only seed17 | 0.3389 | 0.3517 | 0.2138 | 0.1379 | 0.4052 | 0.0244 |
| counterfactual seed17 | **0.5632** | **0.6345** | **0.4253** | **0.2092** | 0.3659 | **0.0488** |
| original-only seed29 | 0.3612 | 0.3770 | 0.2667 | 0.1103 | 0.3232 | 0.0122 |
| counterfactual seed29 | 0.3475 | 0.3770 | 0.2713 | 0.1057 | **0.4268** | 0.0000 |

### Value-edit 상세

- Counterfactual seed17 replacement hit: **0.0732**
- Counterfactual seed17 old-value persistence: **0.4024**
- Counterfactual seed17 clean switch: **0.0488**
- 평가 가능한 value-edit bases: **82**

### 표에서 사용한 metric과 분모

이 슬라이드의 지표는 validation 단계의 lexical diagnostic입니다. 첫 표의 DDXPlus/DiReCT cue recall·precision은 `<observed>`의 정규화된 target-term containment로 계산하고, paired grounding 표의 original/deleted/edited cue hit은 target cue의 content token 중 최소 `0.50`이 출력에 나타나면 읽은 것으로 판정합니다. Locked test의 semantic mapper score와는 구분합니다.

- `DDX cue recall`: original DDXPlus validation case마다 target cue 중 출력에서 발견된 cue의 비율을 계산한 뒤 사례 평균을 냅니다. 환자에게 있어야 할 finding을 얼마나 회수했는지 봅니다.
- `DDX cue precision`: `<observed>`에서 모델이 생성한 bullet 중 하나 이상의 target cue와 lexical match된 bullet의 비율을 사례별로 계산해 평균합니다. 질환 전형이나 unsupported finding을 많이 추가하면 낮아집니다.
- `DiReCT lexical recall`: DiReCT validation에서 physician-observation target 표현이 출력에 lexical하게 등장한 비율입니다. 의미 기반 `Obscomp`보다 엄격하고 단순한 screening 지표이며, 최종 DiReCT clinical alignment를 대신하지 않습니다.
- `current finding recall`: paired grounding population의 original, cue-deleted, value-edited 모든 readout을 각각 **그 arm에 현재 존재해야 하는 cue set**과 비교한 평균 recall입니다. Original-only DDX cue recall과 달리 intervention 이후의 현재 상태까지 포함합니다.
- `original target hit`: cue-deletion pair에서 삭제 대상으로 선택된 cue가 original activation의 출력에 나타난 비율입니다. 삭제 반응을 평가하기 전에 모델이 원래 그 cue를 읽을 수 있었는지 확인합니다.
- `deleted phantom`: 같은 target cue가 cue-deleted activation의 출력에도 남아 있는 비율입니다. 낮을수록 좋지만 original target hit이 0인 빈 모델도 phantom 0을 만들 수 있으므로 단독으로 해석하지 않습니다.
- `deletion contrast`: `original target hit - deleted phantom`입니다. 양수일수록 삭제 전후 출력 차이가 크지만, original hit 유지와 함께 봐야 합니다.
- `removal success`: original에서 target cue를 읽은 사례만 분모로 하여, deleted 출력에서는 그 cue가 사라진 비율입니다. `P(not phantom | original hit)`이므로 전체 pair 기준의 `1 - phantom`과 다릅니다.
- `replacement hit`: native-value edit 후 edited 출력에서 old와 new를 구분하는 **new-value content token**이 나타난 pair의 비율입니다.
- `old-value persistence`: value edit 후에도 edited 출력에 old-value-specific token이 남은 pair의 비율입니다. 낮을수록 좋습니다.
- `clean switch`: edited 출력에서 new value는 나타나고 old value는 나타나지 않은 pair의 비율입니다. 이 슬라이드의 validation lexical 표에서는 전체 평가 가능 value-edit pair가 분모이며, locked structured-reader 표의 original-old-hit 조건부 clean-switch 분모와 구분합니다.

```text
deletion contrast = P(old cue in original output) - P(old cue in deleted output)
removal success   = P(old cue absent after deletion | old cue read originally)
clean switch      = P(new value present AND old value absent after value edit)
```

### 문제 2와 다음 변경

- Full-data SFT는 DDXPlus finding recall과 deletion response를 개선했습니다.
- Counterfactual seed17은 deletion contrast가 0.1379에서 0.2092로 증가했지만 phantom도 0.2138에서 0.4253으로 약 2배 증가했습니다.
- Seed29에서는 contrast 개선이 재현되지 않았습니다.
- 따라서 sequence CE에 학습 가능한 신호는 있지만 changed finding을 선택적으로 말하게 하는 objective로는 불충분합니다.
- 다음 단계에서는 문장 전체를 잘 생성하는 CE 대신 `matched activation에서 해당 claim이 더 쉬운가`를 직접 최적화했습니다.

---

## Slide 23. 시도 3→4: 문장 pair ranking에서 한 개 changed cue ranking으로

### 시도 3. Sentence matched/crossed objective

```text
L = L_SFT + lambda * softplus(-(NLL_cross - NLL_matched) / T)
```

| objective | symmetric gap | category-cluster 95% CI | matched win |
|---|---:|---:|---:|
| lambda=.1 | +0.0013 | [-0.0006, +0.0033] | 0.5556 |
| lambda=1 | +0.0022 | [-0.0010, +0.0055] | 0.5778 |
| SFT=1, lambda=5 | +0.0051 | [+0.0011, +0.0099] | 0.5333 |
| SFT=0, lambda=1 | +0.0030 | [+0.0003, +0.0057] | 0.6444 |

### 문제 3

- 일부 CI는 0을 배제했지만 효과는 `+.0013~+.0051`에 불과했습니다.
- 문장마다 난이도와 길이가 달라 activation 대응 신호가 sentence NLL에 묻힐 수 있었습니다.

### 그래서 바꾼 것: 한 개 changed claim만 original/deleted activation에서 비교

Changed-cue 1x2 ranking objective, 3,104 pairs:

```text
g_changed = NLL(y_changed | h_deleted) - NLL(y_changed | h_original)
L = CE(y_changed | h_original) + softplus(-g_changed / T)
lambda = 1.0, T = 1.0, max_steps = 20, seeds = 17/29/43
specificity = changed_gap - retained_gap
```

| seed | changed delta | cluster 95% CI | retained delta | specificity | specificity 95% CI |
|---:|---:|---:|---:|---:|---:|
| 17 | +0.0005 | [-0.0006, +0.0016] | +0.0010 | -0.0005 | [-0.0020, +0.0010] |
| 29 | +0.0028 | [+0.0017, +0.0039] | -0.0000 | +0.0029 | [+0.0015, +0.0045] |
| 43 | +0.0030 | [+0.0015, +0.0048] | -0.0007 | +0.0037 | [+0.0017, +0.0059] |

- 3 seed 모두 changed delta는 양수였지만 사전 고정한 최소 효과 `0.05`보다 10배 이상 작았습니다.
- Seed17의 cluster CI와 specificity CI가 0을 포함해 promotion gate를 통과하지 못했습니다.
- 이 결과만으로 objective 실패와 20-step budget 부족을 구분할 수 없어, 다음 실험에서는 다른 조건을 고정하고 step만 늘렸습니다.

---

## Slide 24. 시도 5: budget만 20→1,552 steps로 늘렸더니 shortcut이 성장

> 직전 실험의 작은 효과가 단순한 under-training인지 확인하기 위해 데이터 3,104 pairs,
> loss, `lambda=1`, `T=1`, seeds를 고정하고 학습 step만 변경했습니다.

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

### 무엇이 잘못됐는가?

- 삭제한 cue뿐 아니라 retained cue NLL도 거의 같은 크기로 증가했습니다.
- 모델은 `어떤 cue가 지워졌는가`가 아니라 `삭제된 activation인가`를 감지했습니다.
- specificity gate가 없었다면 잘못된 성공 판정을 내릴 수 있었습니다.
- 따라서 budget 부족은 해소됐지만, objective가 허용한 가장 쉬운 해가 global deletion detector라는 것이 확인됐습니다.
- 다음 실험은 step을 더 늘리지 않고 retained cue를 원본/삭제본 모두에서 유지하도록 loss를 바꿨습니다.

---

## Slide 25. 시도 6: retained anchor로 shortcut을 막자 changed signal도 사라짐

### 직전 실패에 대한 변경

삭제본 전체를 억제하면 retained cue CE에서 손해를 보도록 다음 항을 추가했습니다.

```text
L = CE(y_changed | h_original)
  + softplus(-g_changed)
  + CE(y_retained | h_original)
  + CE(y_retained | h_deleted)

all weights = 1.0, max_steps = 1,552, seeds = 17/29/43
```

| seed | changed gap | retained gap | specificity | changed original NLL | retained original NLL |
|---:|---:|---:|---:|---:|---:|
| 17 | -0.0143 | +0.0135 | **-0.0278** | -0.0756 | -0.3342 |
| 29 | -0.0040 | +0.0215 | **-0.0255** | +0.0576 | -0.1834 |
| 43 | -0.0266 | -0.0049 | **-0.0217** | +0.0622 | -0.2263 |

### 결과와 다음 변경

1. Retained anchor가 global deletion detector shortcut을 억제했습니다.
2. 그러나 changed-cue signal도 3 seed 모두 음수로 사라졌습니다.
3. retained original NLL은 개선되어 optimization 자체가 멈춘 것은 아닙니다.
4. teacher-forced gate 실패로 generation, extension, sweep을 실행하지 않았습니다.

> 기존 ranking loss의 큰 효과는 선택적 임상 정보 판독이 아니라 shortcut이었습니다.

Loss 수준에서 shortcut을 막아도 사례별 changed cue가 학습되지 않았으므로, 다음에는
ranking 계수를 다시 조정하지 않고 target builder와 latent interface 자체를 바꿨습니다.

---

## Slide 26. 시도 7: target과 latent 구조를 바꿔도 재현되는 개선은 없었음

### 직전 실패에 대한 변경

- Free-paragraph target 대신 out-of-fold probe가 고른 hard finding set을 teacher로 사용했습니다.
- 3,840차원 activation을 256차원 latent로 강제해 decoder가 공유된 clinical state를 사용하게 했습니다.

| method | exact change | observed result | verdict |
|---|---|---|---|
| OOF finding teacher | 반대 fold probe가 선택한 K=5 hard finding set으로 distillation | precision 0.8881, required >=0.90 | FAIL |
| 256-d soft bottleneck | HS32 3,840→256→3,840 latent와 training-only finding/value auxiliary head | paired effect -0.001137 / -0.001476 / +0.001433 | FAIL |
| frozen-z evaluation | decoder를 고정한 채 learned latent의 finding/value/counterfactual state 평가 | finding F1 delta -0.0009 / -0.0007 / -0.0016 | FAIL |

- OOF teacher는 deletion activation에서 absent label을 대량 추가해 stable target builder가 되지 못했습니다.
- Bottleneck auxiliary objective는 3 seed 방향 일치와 최소 효과 `0.005`를 충족하지 못했습니다.
- 자유문장 target과 pairwise ranking만의 문제가 아니라 latent organization을 바꿔도 개선이 재현되지 않았습니다.
- 그래서 다음에는 surrogate finding loss를 더 만들지 않고 원 NLA의 핵심 측정기인 AR가 의료 분포에서 유효한지 먼저 확인했습니다.

---

## Slide 27. 시도 8: 공개 AR는 높은 cosine에도 의료 activation을 복원하지 못함

> Released `kitft/nla-gemma3-12b-L32-ar`, validation-only, arm당 n=20

### Raw cosine 진단

| positive control | own cosine | shuffled cosine | gap |
|---|---:|---:|---:|
| DDXPlus structured reader | 0.9765 | 0.9765 | 약 0 |
| DiReCT Source CoT | 0.9835 | 0.9834 | 약 0 |
| DiReCT Vanilla, report-only | 0.9962 | 0.9961 | +0.0001 |

### 평균 방향을 제거한 geometry audit

| positive control | centered gap [cluster CI] | FVE vs train mean | retrieval result |
|---|---:|---:|---|
| DDXPlus structured reader | -.0047 [-.0375,+.0261] | -119.2169 | top-1 0/20; chance보다 낮음 |
| DiReCT Source CoT | +.0304 [+.0012,+.0635] | -109.3544 | MRR만 chance보다 높음 |

### 문제 8과 다음 변경

- DDXPlus 양성 대조는 centered/retrieval도 실패했고 모든 arm의 FVE가 큰 음수였습니다.
- DiReCT 일부 rank 신호만으로는 train-mean predictor보다 나쁜 reconstruction을 구제할 수 없습니다.
- 따라서 공개 AR를 Medical-NLA reward로 사용하지 않고, AR 없는 native-layer Patchscope로 AV 변환 병목을 분리했습니다.
- 이는 임상 정보 부재가 아니라 공개 AR의 medical distribution mismatch 결과입니다.

---

## Slide 28. 시도 9: Patchscope는 일반-domain control은 읽었지만 clinical content는 못 읽음

### 왜 시도했는가?

공개 AV/AR의 domain 변환을 우회하기 위해 같은 Gemma의 source hidden state를 target
prompt의 **같은 layer**에 직접 patch했습니다. 가중치 학습은 없습니다.

### General-domain control

| prompt family | HS16→16 | HS24→24 | HS32→32 |
|---|---:|---:|---:|
| entity description keyword hit | 5/5 | 0/5 | 2/5 |
| relation-specific keyword hit | 3/5 | 3/5 | 0/5 |

- No-patch hit는 전부 0/5였으므로 short general-domain state에 대해서는 interface가 작동했습니다.

### DDXPlus clinical application, bounded n=5 smoke

| selected cell | real own finding | shuffled donor finding | result |
|---|---:|---:|---|
| entity HS16→16 | 0/5 | 0/5 | prompt 예시 문장을 설명 |
| relation HS16→16 | 0/5 | 0/5 | 일반 clinical-writing 지침 |
| relation HS24→24 | 0/5 | 0/5 | 일반 case-presentation 지침 |

### 문제 9과 다음 변경

- Vector는 continuation과 KL을 크게 바꿨지만 own patient content와 대응하지 않았습니다.
- Decoder가 target prompt의 텍스트 우회로를 사용하면 activation을 무시할 수 있었습니다.
- 그래서 추가 prompt/layer sweep을 중단하고, **activation-derived prefix만 입력으로 허용하는 learned medical prefix mapper**를 다음 후보로 둡니다.
- RQ3 causal intervention은 grounded readout과 유효한 AR가 없어 여전히 열지 않습니다.

---

# Part IV. Conclusion

---

## Slide 29. 현재 RQ별 답

### Gate 0. Activation에 임상 정보가 존재하는가?

**예. 단, 이는 RQ1의 답이 아니라 선행 representation audit입니다.**

- DiReCT diagnosis category/PDD linear probe가 majority를 크게 상회
- DDXPlus finding F1 0.9562, native value 0.7659
- same-diagnosis shuffled gap +0.1624 / +0.1868

### RQ1. Medical-NLA가 임상 설명을 더 잘 복원하는가?

**현재 validation에서는 아니오. Locked 결론은 아직 미완료입니다.**

- Source CoT Obscomp: 0.2130
- Vanilla NLA Obscomp: 0.0000
- Full-data SFT seed17/29 Obscomp: 0.0301 / 0.0296
- SFT 출력은 parse 가능한 임상 문장을 만들었지만 physician observation보다 질환 전형 template에 가까웠음

### RQ2. 생성된 설명이 해당 사례 activation에 근거하는가?

**현재 통과한 open-ended Medical-NLA가 없습니다.**

- Closed probe와 structured reader는 사례별 정보의 존재를 확인했지만 open NLA가 아님
- Vanilla NLA는 DDXPlus locked 10,028행에서 frozen ontology claim 0건
- Counterfactual SFT는 seed17에서 contrast가 올랐지만 phantom도 0.2138에서 0.4253으로 증가했고 seed29에서 미재현
- D10 budget은 deletion-detector shortcut, D20은 shortcut 차단 후 changed signal 부재
- Same-layer Patchscope는 general-domain control을 통과했지만 clinical own/donor finding은 각각 0/5

### RQ3. 검증된 설명 편집이 상태와 행동을 선택적으로 바꾸는가?

**미실행입니다. RQ2 진입 조건을 통과한 readout과 유효한 AR가 없어 Table 4를 열지 않았습니다.**

- 공개 AR는 DDXPlus structured reader의 centered/retrieval 양성 대조를 통과하지 못했고 모든 arm의 FVE가 음수
- 따라서 현재 AR cosine을 reward 또는 text-to-activation 복원 근거로 사용하지 않음
- RQ3 미실행은 인과 개입 실패가 아니라 사전 등록한 안전 gate 적용 결과임

---

## Slide 30. 발표 시점 논문 표 원장: validation은 채우고 locked는 구분

### Main Table 1 (Gate 0). P0 decodability

| dataset/target | layer와 선택 규칙 | validation | locked evaluation | control |
|---|---|---:|---:|---|
| DiReCT category | HS24, validation-selected | 0.5962 | pending, seen 72 | majority 0.0577 |
| DiReCT PDD | HS24, validation-selected | 0.4423 | pending, seen 72 | majority 0.0962 |
| DDXPlus finding | HS24, validation-selected | 0.9607 | **0.9562** | shuffled 0.7938, gap +0.1624 |
| DDXPlus native value | HS24, validation-selected | 0.7700 | **0.7659** | shuffled 0.5791, gap +0.1868 |

- HS24 closed probe와 HS32 generative AV는 같은 layer ablation이 아닙니다. Probe는 validation 선택, AV는 공개 architecture의 native interface입니다.
- 기존 Table 1A backbone behavior는 핵심 NLA 결과와 중복돼 appendix로 이동하는 안을 사용합니다.

### Main Table 2 (RQ1). DiReCT explanation alignment, validation 50

| method | input | extraction | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | source text | 50/50 | 0 | .2835 | .3726 | **.2130** | **.0650** | .0153 |
| Vanilla NLA | HS32 AV | 10/50 (pilot) | 0 | .0000 | .0000 | .0000 | .0000 | .0000 |
| Medical-AV, SFT only seed17 | HS32 AV | 50/50 | 0 | .0544 | .0502 | .0301 | .0000 | .0000 |
| Medical-AV, SFT only seed29 | HS32 AV | 50/50 | 0 | .0553 | .0388 | .0296 | .0000 | .0000 |
| **Medical-NLA, final** | layer/recipe 사전 동결 | pending | pending | pending | pending | pending | pending | pending |

- 위 숫자는 교수님 발표를 위한 **validation 결과**입니다. 논문 주표의 seen 72 / PDD-heldout 106 locked 셀은 baseline batch 뒤 별도 패널로 교체합니다.
- Vanilla 행은 같은 50-case validation pilot의 frozen output이며, `10/50`은 빈 출력 수가 아니라 평가 가능한 observation extraction coverage입니다.
- `Medical-NLA, final`은 성공 방법의 조건부 행입니다. 실제로 구별되는 checkpoint가 생기기 전에는 reconstruction/full objective라는 가상 행으로 나누지 않습니다.

### Main Table 3A (RQ2). DDXPlus static grounding, locked test

| method class | method | input layer | finding F1 | shuffled | pair gap | value acc |
|---|---|---:|---:|---:|---:|---:|
| closed decoder | Frozen probe | HS24 | .9562 | .7938 | +.1624 | .7659 |
| structured monitor | Probe-guided reader | HS24 | **.9587** | .7938 | +.1624 | **.7654** |
| open generator | Vanilla NLA | HS32 | .0000 | .0000 | .0000 | .0000 |
| open generator | Medical-AV, SFT only | HS32 | validation only | validation only | promotion fail | validation only |
| open generator | **Medical-NLA, final** | 사전 동결 | pending | pending | pending | pending |

### Main Table 3B (RQ2). DDXPlus counterfactual grounding

| method | original hit | deletion phantom | removal | retention | replacement | old persist | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Vanilla NLA | .0000 | .0000 | N/A | N/A | .0000 | .0000 | N/A |
| Medical-AV, SFT only | validation development only | validation development only | validation development only | validation development only | validation development only | validation development only | validation development only |
| **Medical-NLA, final** | pending | pending | pending | pending | pending | pending | pending |

- Vanilla의 phantom 0은 성공이 아닙니다. Original hit도 0이라 removal/retention/clean-switch 조건부 분모가 없습니다.
- Final Medical-NLA는 DiReCT clinical alignment와 DDXPlus activation grounding을 모두 통과해야 두 표의 최종 행이 됩니다.
- Table 4 text patching은 AR identity/grounding gate를 통과할 때만 엽니다.

### 논문의 현재 중심 주장

> Closed monitor는 의료 activation의 환자별 상태를 읽지만, 범용 NLA 또는 단순 domain fine-tuning은 이를 신뢰할 수 있는 자유 자연어 설명으로 변환하지 못한다.

---

## Slide 31. 현재 다음 후보: learned medical prefix mapper

### 왜 이 구조인가?

| 이전 실패 | 다음 구조가 제거하는 경로 |
|---|---|
| Free-paragraph SFT의 disease template | decoder에 patient text를 주지 않음 |
| D16 bottleneck을 decoder가 무시 | activation-derived prefix만 decoder 입력으로 허용 |
| D10 deletion detector | changed cue와 retained cue specificity를 동시에 gate |
| Public AR distribution mismatch | AR reward 없이 supervised target으로 먼저 학습 |
| Identity Patchscope의 prompt bypass | 임상 예시/질문이 없는 fixed minimal prompt 사용 |

### 제안 architecture

```text
P0 medical activation h (3,840-d)
            |
            v
small learned projector
            |
            v
K activation-derived prefix vectors
            |
            v
frozen Gemma decoder -> canonical finding/value claims
```

- 이것은 공개 AR를 이용한 원 NLA RL이 아니라 supervised activation-language decoder 계열입니다.
- Source layer, `K`, projector 크기, target contract와 training budget은 실행 전에 동결합니다.
- 첫 smoke는 DDXPlus official train 4,655와 validation만 사용하고 locked test를 읽지 않습니다.

### 성공 관문

1. Own activation이 same-diagnosis shuffled activation보다 높음, 3 seed cluster CI > 0
2. Cue deletion에서 changed claim은 감소하고 retained claim은 비열등
3. Value edit에서 replacement 증가와 old persistence 감소
4. 출력 coverage를 떨어뜨려 점수를 회피하지 않음
5. DDXPlus 통과 뒤 DiReCT validation에서 Source CoT floor에 접근 또는 사전 고정 개선폭 충족

> 성공하면 Slide 30의 `Medical-NLA, final` 행을 채우고, 실패하면 새 prompt/layer sweep 없이 supervised prefix 계열의 음성 결과로 기록합니다.

---

## Slide 32. 교수님께 확인받을 결정

### 결정 1. 논문의 중심 프레이밍

- 현재 확정 결과: **의료 activation의 정보 존재와 자연어 readout 사이의 격차**
- Positive lane: learned prefix mapper가 gate를 통과하면 `Medical-NLA, final`을 주표에 추가

### 결정 2. Table 1A의 위치

- P0 decodability를 Main Table 1로 유지
- Direct/CoT backbone 진단 정확도는 Table 2의 Accdiag와 일부 중복되므로 appendix로 이동 제안

### 결정 3. DiReCT locked batch 개봉

- Main Table 1 HS24 probe
- Table 2 Source CoT/Vanilla NLA
- 동일 decision record/hash 아래 한 번에 수행

### 결정 4. 다음 생성형 실험

- Learned medical prefix mapper를 별도 사전 등록해 validation-only smoke 실행
- `Medical-NLA, final` 행은 표 설계에 유지하되 성공 전에는 locked 숫자를 만들지 않음
- 기존 SFT/ranking/AR/Patchscope 실패와 전환 논리는 Results와 appendix에 기록

---

## Slide 33. 결론

1. **정보는 있습니다.** DDXPlus locked probe에서 finding F1 0.9562, value accuracy 0.7659입니다.
2. **환자별 정보입니다.** 같은 진단 내 shuffle gap이 +0.1624와 +0.1868입니다.
3. **정적 readout은 가능합니다.** Structured reader finding F1은 0.9587입니다.
4. **자유 자연어 readout은 실패했습니다.** Vanilla NLA는 locked 10,028행에서 ontology claim을 하나도 내지 못했습니다.
5. **단순 SFT/ranking은 충분하지 않습니다.** Template collapse와 deletion detector shortcut이 확인됐습니다.
6. **공개 AR도 그대로는 사용할 수 없습니다.** DDXPlus positive control의 centered/retrieval gate가 실패했고 모든 arm의 FVE가 음수였습니다.
7. Geometry audit와 identity Patchscope까지 종료됐고, 다음 route는 **text bypass가 없는 learned medical prefix mapper**입니다.

---

# Appendix

---

## Appendix A0. Direct-vs-CoT 171-case pilot와 McNemar

> 이 결과는 protocol freeze 전에 분석한 exploratory population입니다. 최종 72/106 confirmatory Table 1A를 대신하지 않습니다.

| condition | n | parse | strict PDD | category | token F1 |
|---|---:|---:|---:|---:|---:|
| Direct | 171 | 1.0000 | 0.2105 | 0.5029 | 0.1593 |
| CoT | 171 | 1.0000 | 0.1930 | 0.5088 | 0.1850 |

### Strict PDD paired outcomes

| Direct | CoT | n | McNemar 사용 여부 |
|---|---|---:|---|
| correct | correct | 26 | 사용하지 않음 |
| correct | wrong | 10 | Direct-only discordance |
| wrong | correct | 7 | CoT-only discordance |
| wrong | wrong | 128 | 사용하지 않음 |

- strict PDD exact McNemar: **p = 0.6291**
- category exact McNemar: **p = 1.0000**
- CoT reasoning에 answer alias가 등장한 비율: **156/171 = 0.9123**

### 왜 남기는가?

1. CoT가 Direct보다 진단 정확도가 높아서 이후 activation/readout 결과가 좋아졌다는 대안 설명을 점검합니다.
2. 같은 171 사례에 두 condition을 적용했으므로 독립 두 표본 검정이 아니라 paired McNemar 검정을 사용합니다.
3. `p=0.6291`은 두 방법이 동일하다는 증명이 아니라, 10 대 7 discordance로 차이를 주장할 근거가 부족하다는 뜻입니다.
4. 최종 논문의 중심 결과는 아니므로 본문이 아니라 보조 통제로만 보고합니다.

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
