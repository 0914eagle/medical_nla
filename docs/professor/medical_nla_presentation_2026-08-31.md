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

### 왜 이후 실험은 P0만 primary로 사용했는가?

| 위치 | activation이 이미 읽은 text | main Medical-NLA 입력으로 쓸 때의 문제 |
|---|---|---|
| P0 | clinical input + instruction만 읽음 | 답·reasoning 문자열 누출이 없어 설명해야 할 pre-response state와 일치 |
| P1 | 실제 생성 reasoning + `The answer is` marker까지 읽음 | reasoning 안에 source answer alias가 자주 이미 등장 |
| P2 | 생성된 diagnosis 문자열까지 읽음 | 정답 문자열 자체가 노출된 answer-exposed state |

- 연구 질문은 **모델이 답을 쓰기 전의 환자 상태를 자연어로 읽을 수 있는가**이므로 P0가 주 위치입니다.
- P1/P2를 주표에 넣으면 decoder가 activation의 임상 추론이 아니라 이미 노출된 문자열을 복사해도 높은 점수를 얻습니다.
- P1/P2는 실제로 실행했으며 Slide 15에서 leakage/positive control로 보고합니다. “정의만 하고 버린 위치”가 아닙니다.
- DDXPlus deletion/value-edit도 답 생성 전 상태의 선택적 변화를 묻기 때문에 CoT-P0로 고정했습니다.
- **현재 frozen position protocol에는 P3가 없습니다.** 과거 archive의 `P3` 표기는 별도 legacy experiment label이며 activation position이 아닙니다.

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
| Development | Appendix gate table | 어떤 실패가 다음 Medical-NLA 설계를 만들었는가? | 21~32 | 시도별 seed/gate 세부 결과 |
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

- Validation에서 P1/P2도 `52 cases x 2 prompts x 2 positions = 208` readouts를 생성했습니다.
- 별도 171-case exploratory audit에서도 P1/P2 L32를 각각 171행 생성했고 parse는 모두 성공했습니다.
- 그 audit의 source-answer mention은 P1 `.4912`, P2 `.3918`이었지만, reasoning에 answer alias가
  없던 P1 subset에서는 `1/15=.0667`뿐이었습니다.

### 해석

- 공개 Vanilla NLA는 P0의 진단 정보를 읽지 못했습니다.
- P1/P2의 절대 mention은 reasoning/answer 문자열 노출과 일치하고, leakage-free P1은 validation `0/5`, exploratory `1/15`이므로 P0 reader 성공이 아닙니다.
- 따라서 P1/P2는 “AV가 이미 노출된 답 문자열을 어느 정도 읽을 수 있는가”를 확인하는 positive control이며 Main Table 1–3의 독립 method row가 아닙니다.
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

### 어디에서 무엇을 측정했는가?

| item | frozen evaluation contract |
|---|---|
| dataset | DDXPlus E5 **locked test** |
| activation position | CoT-P0: 답 생성 전 clinical prompt 마지막 token |
| original population | 4,543 cases, frozen 47-diagnosis population |
| finding target | train-supported 91 evidence IDs |
| hard-shuffle control | 4,121 own/donor pairs; donor는 같은 diagnosis의 다른 환자 |
| native-value target | 6 evidence tasks / 32 value classes; all-original eligible targets 2,136 |
| value ontology coverage | train-supported native values / 전체 single-value target = .7161 |

`locked test`는 validation-only와 반대되는 단순한 데이터 이름이 아닙니다. Train에서 ontology와
head를 학습하고 validation에서 layer/threshold/hyperparameter를 고정한 뒤, test 4,543명을
**재선택 없이 한 번만** 평가했다는 뜻입니다.

### Probe를 실제로 어떻게 학습하고 고정했는가?

| component | finding-presence head | native-value head |
|---|---|---|
| train / validation / locked originals | 4,655 / 4,525 / 4,543 | 동일 |
| input | CoT-P0 HS16/24/32, train mean/std로 표준화 | 동일 |
| output | one affine map, 91 independent logits | one affine map, 6 evidence별 value-logit slices, 총 32 classes |
| train support | evidence count `>=20` | evidence-value count `>=10` |
| loss | multi-label `BCEWithLogits` | evidence-conditioned cross-entropy |
| optimizer | AdamW, batch 512 | AdamW, full batch |
| learning rate / weight decay | `.001/.003` x `0/.001` | 동일 |
| extra choice | positive weighting off/on | evidence slice 안 argmax |
| max epochs / patience / seed | 80 / 8 / 17 | 80 / 8 / 17 |

1. 각 layer에서 validation BCE/NLL로 head hyperparameter를 먼저 선택했습니다.
2. Finding threshold는 validation grid `.1/.2/.3/.4/.5`에서 micro F1→macro F1→`.5`와의
   거리 순으로 한 개를 골랐습니다.
3. Layer는 finding own-minus-shuffled gap, value own-minus-shuffled gap, 더 낮은 layer 순으로
   선택해 **HS24**로 동결했습니다.
4. 이 artifact의 weight, train normalization, label 순서, threshold를 그대로 locked test에 적용했습니다.

### Locked result와 실행 상태

| class | method | layer | finding F1 | shuffled F1 | pair gap | native-value acc | status |
|---|---|---:|---:|---:|---:|---:|---|
| closed decoder | Frozen probe | HS24 | .9562* | .7938 | +.1624 | .7659* | locked |
| structured monitor | Probe-guided reader | HS24 | **.9587†** | .7938 | +.1624 | **.7654†** | locked |
| open generator | Vanilla NLA | HS32 | .0000 | .0000 | .0000 | .0000 | locked |
| open generator | Medical-AV, SFT only | HS32 | — | — | — | — | validation gate FAIL; locked 미실행 |
| open generator | **Medical-NLA, final** | 사전 동결 | — | — | — | — | promoted checkpoint 없음 |

### 각 metric은 정확히 무엇인가?

- `finding F1`: 91개 finding의 TP/FP/FN을 모든 case-label에 합친 micro F1,
  `2TP/(2TP+FP+FN)`입니다. Diagnosis 정답률이 아닙니다.
- `shuffled F1`: own case의 prediction을 같은 diagnosis donor의 gold finding set에 채점합니다.
  질환명만 보고 전형적인 finding을 나열해도 얻는 점수를 측정합니다.
- `pair gap`: 동일한 4,121 pairs에서 `own finding F1 - shuffled F1`입니다.
  `+.1624`는 진단 공통 template를 넘는 환자별 정보가 있다는 뜻입니다.
- `native-value accuracy`: evidence ID가 존재한다고 조건을 건 뒤, train-supported native value를
  맞힌 비율입니다. 모든 4,543명에 대한 진단 accuracy가 아닙니다.
- `*`: Frozen probe의 `.9562/.7659`는 hard-shuffle과 같은 **pair-eligible subset**에서 계산한
  direct-head own score입니다.
- `†`: Structured reader의 `.9587/.7654`는 **전체 4,543 originals**와 value target 2,136에서
  canonical text를 다시 mapping한 end-to-end score입니다. 따라서 `*`와 `†`는 직접적인
  우열 비교값이 아니며, 최종 논문에서는 공통 분모 열과 all-original 열을 분리해야 합니다.

### P0 하나에서 여러 finding을 어떻게 선택하는가?

P0는 “정답 하나”가 아니라 3,840차원 환자-state vector 한 개입니다. Probe는 이 vector를
서로 다른 91개 finding score로 동시에 투영합니다.

```text
h_P0 (3,840-d) -> one affine head -> 91 logits -> 91 sigmoid probabilities
```

| target | output rule | 한 case에서 가능한 출력 수 |
|---|---|---:|
| DiReCT diagnosis/category probe | 하나의 softmax에서 top-1 | 1 |
| DDXPlus finding-presence probe | 91 sigmoid 중 frozen threshold 이상을 모두 선택 | 0~91 |
| DDXPlus native-value probe | 선택된 value-bearing evidence 안에서 value softmax argmax | evidence당 1 value |

- Finding target은 case마다 91차원 multi-hot vector이고 loss는 `BCEWithLogits`입니다.
- Threshold는 train에서 정하지 않고 validation grid `.1/.2/.3/.4/.5` 중 micro F1, macro F1
  순으로 **전체 label에 공통으로 적용할 확률 기준 `t` 하나**를 선택한 뒤 locked test 전에 동결했습니다.
- Probability 1등/2등 순서는 bullet 표시 순서에만 사용합니다. F1은 순위를 무시하고
  threshold를 넘은 predicted set과 gold finding set의 TP/FP/FN으로 계산합니다.
- 따라서 한 환자에서 fever, cough, dyspnea 세 probability가 threshold를 넘으면 세 finding을
  모두 출력하며, 그중 “1등만 정답”으로 처리하지 않습니다.

`Top-k를 고정하지 않는다`는 말은 모든 환자에게 무조건 상위 3개 또는 5개를 출력하지 않는다는
뜻입니다. 아래에서 frozen threshold를 예시로 `t=.5`라고 하면:

| case | 높은 finding probabilities | fixed top-3라면 | 실제 threshold rule |
|---|---|---|---|
| A | `.95, .88, .70, .62, .10` | 항상 3개 | `.5` 이상 **4개** 선택 |
| B | `.91, .55, .30, .20, .10` | 항상 3개 | `.5` 이상 **2개** 선택 |
| C | `.42, .35, .20, .10, .05` | 낮아도 3개 | `.5` 이상이 없어 **0개** 선택 |

DDXPlus 환자마다 실제 finding 수가 다르므로 fixed top-k는 불필요한 finding을 강제로 추가하거나
필요한 finding을 잘라낼 수 있습니다. Global threshold rule은 예측 개수가 환자별로 달라지게 하고,
그 결과의 false positive와 false negative를 micro F1이 함께 벌점 줍니다. 표의 `.5`는 동작 설명용
예시이며 실제 locked run은 validation에서 선택해 artifact에 동결한 threshold를 사용합니다.

### Structured reader의 정확한 의미

1. Frozen HS24 probe가 위 multi-label rule로 finding set과 지원되는 native value를 선택합니다.
2. Evaluation 전에 official train 4,655건만 읽어 다음 **고정 lookup 사전**을 만듭니다.
   - Finding ID별로 train의 exact cue text 빈도를 세고 가장 자주 나온 문구 하나를 저장합니다.
   - Value가 있는 finding은 `(finding ID, value ID)`별 최빈 exact 문구를 따로 저장합니다.
   - 동률이면 문자열 오름차순으로 하나를 골라 재실행 결과가 항상 같게 합니다.
3. Test-time에는 probe label을 이 사전에서 단순 치환합니다. 예를 들어 probe가
   `E_132`와 value `4`를 고르면 생성 없이 `the rash is swollen (rated 4)`를 가져옵니다.
4. 선택된 finding probability 내림차순으로 정렬해 `<observed>` 안의 bullet 목록으로 출력합니다.
5. Test prompt text, diagnosis, gold cue를 사용하지 않고 LLM decoding도 하지 않습니다.

> 따라서 2번은 “label을 자연어로 추론한다”가 아니라 **이미 고른 label의 이름표를 train-only
> 문구로 바꿔 붙인다**는 뜻입니다. Structured reader는 probe state의 closed rendering control이지,
> activation에서 claim을 발견하고 문장을 만드는 open-ended NLA가 아닙니다.
> Finding/value ID 기준으로는 probe와 structured reader가 동일하며, reader를 별도 성능 개선
> 방법으로 해석하면 안 됩니다.

### `validation only`는 무슨 뜻이었는가?

- 기존 표의 `validation only`는 점수가 아니라 **실행 상태**였습니다.
- SFT 모델은 DDXPlus validation paired pilot 435 bases / 952 readouts에서만 개발 평가했고,
  사전 등록 promotion gate를 통과하지 못해 locked test generation을 하지 않았습니다.
- 그러므로 이 셀을 0으로 쓰거나 locked 행과 비교하면 안 되며, 표에서는 `— / locked 미실행`으로
  고쳤습니다. `Medical-NLA, final`도 아직 실재하는 promoted checkpoint가 아닌 조건부 설계 행입니다.
- HS24는 closed probe의 validation-selected layer이고 HS32는 공개 generative AV의
  architecture-native input이므로 layer 차이도 method 역할과 함께 보고합니다.

---

## Slide 19. RQ2 = Main Table 3B: counterfactual change에 선택적으로 반응하는가?

### 어떤 locked counterfactual 데이터를 만들었는가?

| population/denominator | n | 생성·eligibility 규칙 |
|---|---:|---|
| original cases | 4,543 | frozen DDXPlus E5 locked originals |
| cue-deleted rows | 4,543 | 각 original에서 한 cue를 제거하고 CoT-P0 activation 재추출 |
| deletion metric eligible | 4,540 | 삭제 cue가 frozen 91-label finding ontology에 있는 pair |
| untouched finding occurrences | 16,105 | original/deleted 입력에 공통으로 남은 train-supported finding occurrences |
| value-edited rows | 942 | native value 하나를 다른 허용 값으로 바꾸고 activation 재추출 |
| value-edit metric eligible | 539 | old/new가 모두 frozen 6-task/32-class value ontology에 있는 pair |
| clean-switch denominator | 398 | original activation에서 old value를 실제로 맞힌 eligible pair |

모든 수치는 Slide 18과 같은 locked protocol을 사용합니다. Prompt를 문자열 수준에서만 고친 뒤
같은 activation을 재사용한 것이 아니라, **각 edited prompt에서 activation을 다시 추출**했습니다.

| method | original hit | deletion phantom | removal | retention | replacement | old persist | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Vanilla NLA | .0000 | .0000 | N/A | N/A | .0000 | .0000 | N/A |
| Medical-AV, SFT only | — | — | — | — | — | — | — |
| **Medical-NLA, final** | — | — | — | — | — | — | — |

### Deletion metric 정의

삭제 대상 cue를 `c`, original/deleted output의 selected finding set을 `S_orig/S_del`이라 두면:

```text
original hit     = P(c in S_orig)                                  [n=4,540]
deletion phantom = P(c in S_del)                                   [n=4,540]
removal success  = P(c not in S_del | c in S_orig)                 [conditional]
retention        = P(u in S_del | u in S_orig, u unchanged)        [16,105 occurrences]
```

- Reader는 original hit `1.0000`이어서 모든 deletion pair에서 지울 cue를 원래 읽었습니다.
- 삭제 뒤에도 `.3593`에서 cue가 남아 removal은 `.6407`이었습니다.
- Threshold 이전의 연속 확률도 평균 `.9999→.3896`, drop `+.6103`으로 반응했지만 완전 제거는 아니었습니다.
- Unchanged cue preservation `.9987`은 삭제 반응이 모든 finding을 함께 지운 결과가 아님을 확인합니다.

### Value-edit metric 정의

Old/new value를 `v_old/v_new`, edited output의 evidence-conditioned argmax를 `v_after`라 두면:

```text
replacement hit      = P(v_after = v_new)                           [n=539]
old-value persistence= P(v_after = v_old)                           [n=539]
clean switch         = P(v_after = v_new | v_before = v_old)        [n=398]
```

- Reader의 replacement `.1466`과 clean switch `.0804`는 static value accuracy `.7654`보다 훨씬 낮았습니다.
- Old persistence `.5955`는 값을 편집해도 이전 값이 약 60%에서 유지됐다는 뜻입니다.
- Value head는 evidence마다 값 하나만 argmax하므로 clean switch에서 new를 선택하면 old는 동시에 선택되지 않습니다.

### 0, N/A, `—`를 구분해야 하는 이유

- Vanilla의 original hit와 replacement가 실제 `.0000`입니다. Claim을 하나도 내지 않아 phantom도
  `.0000`이지만 이는 deletion 성공이 아닙니다.
- Vanilla removal/retention/clean-switch는 조건을 만족한 original prediction이 없어 분모 0인 `N/A`입니다.
- Medical-AV/Medical-NLA의 `—`는 0도 N/A도 아니며 validation gate FAIL 또는 checkpoint 부재로
  locked evaluation을 실행하지 않은 셀입니다.
- 따라서 RQ2를 통과하려면 changed cue/value 반응뿐 아니라 original coverage와 retained cue 보존을 동시에 충족해야 합니다.

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

## Slide 21. Medical-NLA 개발 로드맵: 매번 무엇을 고쳤는가?

| 시도 | 핵심 변경 | 직전 실패에서 제거하려던 것 | 결과 |
|---:|---|---|---|
| 1 | DDXPlus 248→4,655 original-only SFT | 데이터 부족 | 의료 형식은 습득, 사례 특이성 부족 |
| 2 | original/deleted/value-edited SFT | intervention 미노출 | seed17 recall과 phantom 동시 증가 |
| 3 | sentence matched/crossed ranking | CE의 pairwise 비교 부재 | 양의 gap이지만 `+.001~+.005` |
| 4 | 한 개 changed cue 1x2 ranking | 문장 길이·난이도 noise | 3 seed 효과가 `.05` floor 미달 |
| 5 | 20→1,552-step budget | under-training 가능성 | deletion detector shortcut 성장 |
| 6 | retained-cue specificity anchor | 삭제본 전체 억제 shortcut | shortcut과 changed signal 모두 소거 |
| 7 | OOF hard-target teacher | 자유문장 target의 불안정성 | deletion OOD calibration 실패 |
| 8 | 256-d soft bottleneck | decoder의 activation 무시 | 3 seed 모두 핵심 decodability 저하 |
| 9 | 공개 AR reconstruction | surrogate objective 의존 | medical distribution에서 FVE < 0 |
| 10 | same-layer identity Patchscope | 공개 AV/AR 변환 병목 | 일반-domain 성공, clinical own finding 0/5 |

> 다음 10장은 각각 한 시도만 다루며, `변경 → 학습/평가 → 실제 값 → 실패 진단 → 다음 변경` 순서로 읽습니다.

---

## Slide 22. 시도 1: 데이터만 늘린 original-only sequence SFT

### 무엇을 어떻게 바꿨는가?

- Input: 환자별 HS32 activation `h_current`
- Target: 그 환자에게 현재 존재하는 finding을 `<observed>` 목록으로 직렬화한 `y_current`
- Loss: token-level cross-entropy `CE(y_current | h_current)`
- DDXPlus train을 248건 pilot에서 **4,655건 full data**로 확대했습니다.

### Validation 결과

| method | DDX cue recall | cue precision | DiReCT lexical recall | current finding | deletion phantom | removal | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 248+248 pilot, seed29 | .1784 | .2533 | .0000 | .1499 | .1356 | .0000 | .0000 |
| full data, seed17 | **.3763** | **.3816** | .0216 | .3389 | .2138 | .4052 | .0244 |
| full data, seed29 | .3506 | .3758 | .0076 | **.3612** | .2667 | .3232 | .0122 |

`cue recall`은 gold finding 중 회수한 비율, `cue precision`은 생성 finding 중 gold로 지지되는
비율, `DiReCT lexical recall`은 physician observation 표현과의 직접 겹침입니다.

DiReCT 의미 기반 평가에서도 Source CoT `Obscomp=.2130`에 비해 full SFT seed17/29는
`.0301/.0296`이었습니다.

### 무엇이 문제였고 다음에 무엇을 바꿨는가?

- 데이터 증가는 DDXPlus lexical recall을 높였지만 physician observation 정렬은 거의 늘리지 못했습니다.
- CE는 출력 형식과 질환 전형 문장을 쉽게 학습하지만, 입력이 바뀌었을 때 무엇을 빼거나 고쳐야 하는지는 가르치지 않습니다.
- **다음 변경:** loss는 유지하고 같은 환자의 `original / cue-deleted / value-edited` activation-target arm을 직접 학습시켰습니다.

---

## Slide 23. 시도 2: Counterfactual sequence SFT

### 무엇을 어떻게 바꿨는가?

| arm | activation을 만드는 입력 | CE target |
|---|---|---|
| original | fever, cough, temperature 39 C | `fever; cough; temperature 39 C` |
| cue-deleted | 입력에서 cough 제거 후 activation 재추출 | `fever; temperature 39 C` |
| value-edited | temperature 39→37 C 후 재추출 | `fever; cough; temperature 37 C` |

모든 arm은 여전히 `CE(y_arm | h_arm)`입니다. 즉 deleted output의 cough 확률을 original보다
직접 낮추는 pairwise 제약은 없습니다. Validation은 435 bases / 952 readouts입니다.

### Paired grounding 결과

| method | current recall | original hit | phantom | contrast | removal | clean switch |
|---|---:|---:|---:|---:|---:|---:|
| original-only s17 | .3389 | .3517 | .2138 | .1379 | .4052 | .0244 |
| counterfactual s17 | **.5632** | **.6345** | **.4253** | **.2092** | .3659 | **.0488** |
| original-only s29 | .3612 | .3770 | .2667 | .1103 | .3232 | .0122 |
| counterfactual s29 | .3475 | .3770 | .2713 | .1057 | **.4268** | .0000 |

### Value-edit 상세, eligible 82

| metric | 의미 | counterfactual s17 |
|---|---|---:|
| replacement hit | 39→37 편집 뒤 새 값 37을 말함 | .0732 |
| old-value persistence | 편집 뒤에도 이전 값 39를 말함 | .4024 |
| clean switch | 새 값은 말하고 이전 값은 제거함 | .0488 |

`contrast = original hit - phantom`, `removal = P(cue absent after deletion | original hit)`입니다.

### 문제와 다음 변경

- Seed17은 더 많이 읽었지만 phantom도 `.2138→.4253`으로 약 2배 증가했고 removal은 악화됐습니다.
- Seed29 contrast는 `.1103→.1057`로 재현되지 않았습니다.
- **진단:** unchanged token이 대부분인 sequence CE는 “많이 말하기”와 “선택적으로 반응하기”를 구분하지 못합니다.
- **다음 변경:** matched activation에서 해당 문장의 NLL이 crossed activation보다 낮도록 pairwise ranking을 추가했습니다.

---

## Slide 24. 시도 3: Sentence matched/crossed contrastive learning

### 무엇을 어떻게 바꿨는가?

같은 disease category 안에서 환자 A의 activation과 target 문장을 짝지었습니다.

- `matched`: `NLL(y_A | h_A)`
- `crossed`: `NLL(y_A | h_B)`, B는 같은 category의 다른 환자
- 양의 gap은 자기 activation이 다른 환자 activation보다 자기 문장을 더 잘 설명한다는 뜻입니다.

```text
L = L_SFT + lambda * softplus(-(NLL_cross - NLL_matched) / T)
```

### Direct validation, 45 paired rows / 13 category clusters

| objective | symmetric cross-minus-matched | cluster 95% CI | matched win |
|---|---:|---:|---:|
| lambda=.1 | +.0013 | [-.0006,+.0033] | .5556 |
| lambda=1 | +.0022 | [-.0010,+.0055] | .5778 |
| SFT=1, lambda=5 | **+.0051** | [+.0011,+.0099] | .5333 |
| SFT=0, lambda=1 | +.0030 | [+.0003,+.0057] | **.6444** |

### 문제와 다음 변경

- 일부 CI는 0을 배제했지만 절대 효과는 `+.0013~+.0051`로 매우 작았습니다.
- 문장 전체 NLL에는 길이, 문체, target 자체의 난이도가 섞여 환자별 finding 신호를 희석합니다.
- **다음 변경:** 전체 문장 대신 deletion으로 실제 바뀐 **한 개 cue claim**만 original/deleted activation에서 비교했습니다.

---

## Slide 25. 시도 4: D9a-supported changed-cue 1x2 ranking

### 무엇을 어떻게 바꿨는가?

- OOF probe support cut `presence=.90, deletion delta=0, donor margin=0`으로 **3,104 pairs**를 고정했습니다.
- `changed claim`: 입력에서 삭제한 바로 그 finding 문장
- `retained claim`: original과 deleted 양쪽에 공통으로 남은 finding 문장

```text
g_changed = NLL(y_changed | h_deleted) - NLL(y_changed | h_original)
L = CE(y_changed | h_original) + softplus(-g_changed / T)
lambda = 1, T = 1, max_steps = 20, seeds = 17/29/43
specificity = changed_gap - retained_gap
```

### Validation 3,032 pairs

| seed | changed delta | cluster 95% CI | retained delta | specificity | specificity CI |
|---:|---:|---:|---:|---:|---:|
| 17 | +.0005 | [-.0006,+.0016] | +.0010 | -.0005 | [-.0020,+.0010] |
| 29 | +.0028 | [+.0017,+.0039] | -.0000 | +.0029 | [+.0015,+.0045] |
| 43 | +.0030 | [+.0015,+.0048] | -.0007 | +.0037 | [+.0017,+.0059] |

각 `delta`는 같은 seed의 ranking arm에서 original-only control을 뺀 paired 차이입니다.

### 문제와 다음 변경

- 세 seed의 changed delta 방향은 양수였지만 사전 고정 최소 효과 `.05`의 1/17 이하였습니다.
- Seed17은 changed와 specificity CI 모두 0을 포함했습니다.
- 20-step smoke만으로는 objective가 약한지, 단순 under-training인지 구분할 수 없습니다.
- **다음 변경:** 데이터·loss·seed·lambda/T를 모두 고정하고 step만 `20→1,552`로 늘렸습니다.

---

## Slide 26. 시도 5: D10 budget calibration, 20→1,552 steps

### 무엇을 어떻게 바꿨는가?

직전 실험에서 **학습량만** 바꿨습니다. 3,104 pairs, objective, `lambda=T=1`, seeds는 동일하며
checkpoint `{20,194,388,776,1164,1552}`를 report-only로 평가했습니다.

### Across-seed trajectory

| step | changed-gap delta | retained-gap delta | specificity |
|---:|---:|---:|---:|
| 20 | +.0019 | +.0002 | +.0018 |
| 194 | +.0329 | -.0032 | +.0361 |
| 388 | +.2690 | +.3044 | -.0354 |
| 776 | +.0965 | +.0428 | +.0536 |
| 1,164 | +.3527 | +.4055 | -.0527 |
| 1,552 | **+.5558** | **+.5604** | **-.0046** |

각 checkpoint의 값은 같은 budget으로 학습한 original-only control 대비 ranking arm의 paired delta입니다.
최종 changed delta는 seed17/29/43에서 `-.0177 / +.5618 / +1.1233`이었습니다.

### 무엇이 문제였는가?

- 학습량을 늘리자 changed gap은 커졌지만 retained gap도 거의 똑같이 커졌습니다.
- 모델은 “삭제된 cue 하나”가 아니라 **삭제본 activation 전체의 모든 문장을 어렵게 만드는 deletion detector**를 학습했습니다.
- Seed17은 최종 부호도 반대라 재현성 gate를 통과하지 못했습니다.
- **다음 변경:** deleted activation에서도 retained claim은 계속 쉬워야 한다는 CE anchor를 loss에 직접 넣었습니다.

---

## Slide 27. 시도 6: D20 specificity-anchored ranking

### 무엇을 어떻게 바꿨는가?

Global deletion detector가 retained claim까지 억제하면 직접 손해를 보도록 두 CE 항을 추가했습니다.

```text
L = CE(y_changed | h_original) + softplus(-g_changed)
  + CE(y_retained | h_original) + CE(y_retained | h_deleted)
all weights = 1, max_steps = 1,552, seeds = 17/29/43
```

### Frozen final checkpoint

| seed | changed gap | retained gap | specificity | changed orig NLL | retained orig NLL |
|---:|---:|---:|---:|---:|---:|
| 17 | -.0143 | +.0135 | **-.0278** | -.0756 | -.3342 |
| 29 | -.0040 | +.0215 | **-.0255** | +.0576 | -.1834 |
| 43 | -.0266 | -.0049 | **-.0217** | +.0622 | -.2263 |

표는 D20 anchored arm에서 같은 1,552-step control을 뺀 값입니다. Gap delta는 양수가 좋고,
original NLL delta는 양수일수록 생성 성능 저하입니다.

### 문제와 다음 변경

- Retained gap이 budget run의 `+.5604`에서 `|gap|<=.0215`로 줄어 shortcut은 차단됐습니다.
- 그러나 changed gap과 specificity가 3 seed 모두 음수였습니다.
- Retained original NLL은 크게 개선되어 optimization 실패는 아닙니다. 선택적 changed-cue 신호만 학습되지 않았습니다.
- Teacher-forced gate에서 멈추고 generation, checkpoint 선택, 추가 sweep은 하지 않았습니다.
- **다음 변경:** ranking 계수가 아니라 target 자체가 불안정한지 OOF hard-target teacher로 검사했습니다.

---

## Slide 28. 시도 7: OOF hard-finding teacher

### 무엇을 어떻게 바꿨는가?

- Free paragraph 대신 probe가 선택한 finding ID를 canonical claim으로 렌더링했습니다.
- Leakage를 막기 위해 fold 0 probe는 fold 1에서, fold 1 probe는 fold 0에서만 학습했습니다.
- 삭제 전후에 안정적인 K=5 set을 student target으로 만들려 했습니다.

### Initial K=2 calibration audit

| reader/arm | mean selected | cue precision | cue recall | cue F1 |
|---|---:|---:|---:|---:|
| OOF teacher, original | 6.0745 | .7538 | .9999 | .8595 |
| OOF teacher, deleted | **8.3590** | **.4276** | .9985 | .5988 |
| full-data probe, original | 4.7865 | .9567 | 1.0000 | .9779 |
| full-data probe, deleted | 5.6432 | .6331 | .9979 | .7747 |

- Deletion 뒤 새로 추가된 label은 case당 평균 **3.5091**개였습니다.
- 그중 deleted input에 없는 label은 **16,333/16,335 = .9999**였습니다.
- 이 실패를 완화하려고 허용된 단 한 번의 K=5 target을 평가했지만 precision은 `.8881`,
  사전 기준은 `>=.90`이었습니다.

### 문제와 다음 변경

- 삭제 activation이 OOF probe 학습분포 밖으로 이동하면서 없는 finding을 대량 추가했습니다.
- 불안정한 teacher를 student target으로 쓰면 phantom을 정답으로 굳힐 위험이 있어 target building을 중단했습니다.
- **다음 변경:** discrete teacher set 대신 3,840-d activation을 256-d continuous clinical bottleneck으로 조직했습니다.

---

## Slide 29. 시도 8: 256-d soft bottleneck과 frozen-z 평가

### 무엇을 어떻게 바꿨는가?

```text
HS32 h (3,840) -> learned z (256) -> projection (3,840) -> frozen AV decoder
                         + training-only finding/value auxiliary heads
```

Decoder가 원 activation의 임의 방향을 이용하지 못하고 압축된 공통 clinical state를 사용하게 했습니다.

### Frozen-z validation: auxiliary minus control

| seed | finding F1 delta | shuffled-gap delta | value-acc delta | deletion-drop delta | new-after-delete control→aux |
|---:|---:|---:|---:|---:|---:|
| 17 | -.0009 | -.0050 | -.0137 | -.0167 | 1.054→.514 |
| 29 | -.0007 | -.0046 | -.0096 | -.0141 | .849→.527 |
| 43 | -.0016 | -.0058 | -.0160 | -.0151 | .897→.511 |

Training-time paired auxiliary effect도 seed17/29/43에서
`-.001137 / -.001476 / +.001433`으로, 요구한 `+.005`와 3-seed 양의 부호를 충족하지 못했습니다.

### 문제와 다음 변경

- Bottleneck은 deletion 뒤 새 label 수는 줄였지만 finding F1, case-specific gap, value accuracy와 deletion response를 모두 악화시켰습니다.
- 즉 압축은 noise뿐 아니라 probe로 읽히던 의료 정보도 함께 버렸습니다.
- **다음 변경:** 새 surrogate loss 대신 원 NLA의 text→activation 측정기인 공개 AR가 의료 분포에서 유효한지 먼저 검사했습니다.

---

## Slide 30. 시도 9: 공개 AR reconstruction과 geometry audit

### 무엇을 어떻게 바꿨는가?

출력 text를 공개 `kitft/nla-gemma3-12b-L32-ar`에 넣어 activation을 복원하고,
복원 vector가 자기 환자 activation에 더 가까운지 same-diagnosis shuffled activation과 비교했습니다.
양성 대조는 사례별 finding이 확실한 DDXPlus structured reader와 DiReCT Source CoT입니다.

### Raw cosine와 centered geometry

| positive control | own/shuffled cosine | centered gap [cluster CI] | FVE vs train mean | retrieval |
|---|---|---:|---:|---|
| DDXPlus structured reader | .9765/.9765 | -.0047 [-.0375,+.0261] | **-119.2169** | top-1 0/20 |
| DiReCT Source CoT | .9835/.9834 | +.0304 [+.0012,+.0635] | **-109.3544** | top-1 .40, chance CI 포함 |
| DiReCT Vanilla, report-only | .9962/.9961 | +.0696 [+.0289,+.1111] | **-19.7012** | style/length confound 가능 |

`FVE > 0`이어야 train-mean predictor보다 나은 복원입니다. 모든 arm의 FVE는 음수였습니다.

### 문제와 다음 변경

- 높은 raw cosine은 공통 평균 방향 때문에 생겼고, DDXPlus 양성 대조조차 자기 환자를 찾지 못했습니다.
- 이는 text나 activation에 정보가 없다는 뜻이 아니라 **공개 AR가 medical distribution의 측정기로 부적합**하다는 뜻입니다.
- **다음 변경:** 공개 AV/AR 변환을 모두 우회하고 같은 Gemma layer의 hidden state를 직접 patch했습니다.

---

## Slide 31. 시도 10: Same-layer identity Patchscope

### 무엇을 어떻게 바꿨는가?

Source prompt에서 얻은 한 token hidden vector를 target prompt의 지정 token residual에 같은 layer에서
직접 덮어썼습니다. 가중치 학습과 AV/AR 변환은 없습니다. 먼저 general-domain control로 cell을 고정했습니다.

### General-domain control

| prompt family | HS16→16 | HS24→24 | HS32→32 |
|---|---:|---:|---:|
| entity-description hit | **5/5** | 0/5 | 2/5 |
| relation-specific hit | 3/5 | 3/5 | 0/5 |

No-patch hit는 전 cell에서 0/5여서 short entity/relation 정보에 대해서는 patch mechanism이 작동했습니다.

### DDXPlus clinical smoke, n=5

| selected cell | own finding | shuffled donor finding | continuation failure |
|---|---:|---:|---|
| entity HS16→16 | 0/5 | 0/5 | prompt 예시를 설명 |
| relation HS16→16 | 0/5 | 0/5 | generic clinical-writing 지침 |
| relation HS24→24 | 0/5 | 0/5 | generic case-presentation 지침 |

### 문제와 다음 변경

- Patch는 모든 continuation을 바꾸고 KL도 크게 만들었지만 own 환자 finding과 대응하지 않았습니다.
- Decoder가 target prompt의 의미를 따라가는 text bypass가 activation content보다 강했습니다.
- **다음 변경:** 임상 예시와 환자 text를 제거하고 activation-derived vectors만 prefix로 주는 learned medical prefix mapper를 설계합니다.

---

## Slide 32. 다음 후보: text bypass 없는 learned medical prefix mapper

### 직전 10개 시도에서 남은 요구조건

| 확인된 실패 | 구조적 대응 |
|---|---|
| Free-text SFT의 disease template | patient text와 clinical examples를 decoder 입력에서 제거 |
| Pair ranking의 deletion detector | changed와 retained specificity를 동시 gate |
| OOF teacher의 deletion OOD | train-supported canonical claims만 사용 |
| 256-d bottleneck의 정보 손실 | K-token prefix capacity를 validation 전에 고정하고 ablation 최소화 |
| 공개 AR distribution mismatch | 첫 단계는 AR reward 없이 supervised target만 사용 |
| Patchscope prompt bypass | activation-derived prefix + fixed minimal prompt만 허용 |

```text
P0 medical activation h (3,840-d)
      -> learned projector
      -> K prefix vectors
      -> frozen Gemma decoder
      -> canonical finding/value claims
```

### 사전 고정할 성공 관문

1. Own > same-diagnosis shuffled, 3 seed cluster CI > 0
2. Deletion에서 changed claim 감소와 retained claim 비열등 동시 충족
3. Value edit replacement 증가와 old persistence 감소
4. Claim coverage를 줄여 점수를 회피하지 않음
5. DDXPlus validation 통과 뒤에만 DiReCT validation과 locked 행 개방

> 이 실험이 통과할 때만 Slide 34의 `Medical-NLA, final` 행을 채웁니다.

---

# Part IV. Conclusion

---

## Slide 33. 현재 RQ별 답

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

## Slide 34. 발표 시점 논문 표 원장: validation은 채우고 locked는 구분

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
| open generator | Medical-AV, SFT only | HS32 | — | — | — | — |
| open generator | **Medical-NLA, final** | 사전 동결 | — | — | — | — |

- `—`는 0점이 아니라 validation promotion FAIL 또는 checkpoint 부재로 **locked evaluation을 실행하지 않은 셀**입니다.
- Frozen `.9562/.7659`는 pair-eligible direct-head 기준이고 reader `.9587/.7654`는 all-original
  end-to-end 기준입니다. 최종 논문 표에서는 Slide 18의 규약대로 공통 분모와 all-original 열을 분리합니다.

### Main Table 3B (RQ2). DDXPlus counterfactual grounding

| method | original hit | deletion phantom | removal | retention | replacement | old persist | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Vanilla NLA | .0000 | .0000 | N/A | N/A | .0000 | .0000 | N/A |
| Medical-AV, SFT only | — | — | — | — | — | — | — |
| **Medical-NLA, final** | — | — | — | — | — | — | — |

- Vanilla의 phantom 0은 성공이 아닙니다. Original hit도 0이라 removal/retention/clean-switch 조건부 분모가 없습니다.
- Medical-AV/Medical-NLA의 `—`는 validation gate 이후 locked generation을 열지 않았다는 뜻입니다.
- Final Medical-NLA는 DiReCT clinical alignment와 DDXPlus activation grounding을 모두 통과해야 두 표의 최종 행이 됩니다.
- Table 4 text patching은 AR identity/grounding gate를 통과할 때만 엽니다.

### 논문의 현재 중심 주장

> Closed monitor는 의료 activation의 환자별 상태를 읽지만, 범용 NLA 또는 단순 domain fine-tuning은 이를 신뢰할 수 있는 자유 자연어 설명으로 변환하지 못한다.

---

## Slide 35. 교수님께 확인받을 결정

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

## Slide 36. 결론

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
