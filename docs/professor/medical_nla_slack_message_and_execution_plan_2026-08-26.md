# Medical-NLA 교수님 Slack 문안과 실행 계획 (2026-08-26)

이 문서는 두 부분을 분리한다.

1. 교수님께 바로 보낼 수 있는 주제·가설 Slack 문안
2. 가설을 실제로 검증하기 위한 표·그림·실험·데이터 실행 계획

8월 19일에 확인받은 연구 방향인 `CoT의 한계 -> 닫힌 내부 도구의 한계 ->
검증된 Medical-NLA`는 유지한다. 후속 pilot에서 반증된 세부 기제와 계측
confound만 수정한다. 과거 표의 처분 근거는
[`hypothesis_disposition_2026-08-22.md`](hypothesis_disposition_2026-08-22.md)와
[`RETRACTIONS.md`](../experiments/RETRACTIONS.md)에 기록되어 있다.

---

## 1. 교수님께 보낼 Slack 문안

교수님, 기존에 확인받았던 Medical-NLA 연구 주제와 가설을 후속 pilot 결과에
맞춰 다시 정리했습니다. 연구 주제를 변경한 것은 아니며, 기존의 세 단계 논리를
유지하되 각 가설을 반증 가능한 형태로 구체화했습니다.

### 연구 주제

의료 LLM의 설명을 신뢰하려면 설명이 임상적으로 타당한 내용을 포함하는 것뿐
아니라, 그 설명이 실제 모델 내부 상태에 사례 특이적으로 근거해야 합니다.
CoT는 유용하고 그럴듯한 임상 설명을 생성할 수 있지만 모델의 실제 판단 과정을
충실하게 보고한다고 보장할 수 없습니다. 본 연구에서는 의료 LLM의 activation을
자연어로 판독하는 Medical-NLA를 만들고, 이 판독의 임상적 정렬, activation
grounding, 그리고 자연어 기반 개입 가능성을 단계적으로 검증하고자 합니다.

### 가설 1. CoT의 임상적 그럴듯함은 내부 상태 충실성을 보장하지 않는다

기존 연구들은 답에 영향을 준 힌트나 편향 요인이 CoT에 나타나지 않거나, CoT가
이미 선택된 답을 사후적으로 합리화할 수 있음을 보였습니다. 의료 도메인에서도
CoT의 인과적 충실성이 보장되지 않는다는 연구가 보고되었습니다.

- Reasoning Models Don't Always Say What They Think:
  <https://arxiv.org/pdf/2505.05410>
- Language Models Don't Always Say What They Think:
  <https://arxiv.org/pdf/2305.04388>
- Faithful or Just Plausible?:
  <https://arxiv.org/pdf/2603.13988>

최근에는 일반 도메인의 multiple-choice hint 설정에서 activation probe가 full-CoT
monitor보다 motivated reasoning을 더 잘 탐지할 수 있다는 결과도 나왔습니다.
그러나 이는 고정된 label을 예측하는 probe 연구이며, 의료 사례의 관찰·속성·관계를
열린 자연어로 읽는 문제는 다루지 않습니다.

- Catching Rationalization in the Act:
  <https://arxiv.org/abs/2603.17199>

따라서 본 연구는 동일한 source run에서 CoT와 answer-boundary activation 판독을
비교하고, Medical-NLA가 CoT에 누락되거나 왜곡된 사례 고유 관찰, 관계 또는
source-decision 정보를 추가로 복원하는지 평가합니다.

### 가설 2. 기존 내부 도구는 닫힌 탐지에는 강하지만 열린 설명을 직접 제공하지 않는다

Linear probe와 같은 내부 도구는 사전에 정의된 진단이나 속성을 매우 정확하게
탐지할 수 있고, 답을 생성하기 전에도 내부 진단 신호를 읽을 수 있습니다. 다만
질문할 label과 output head를 사전에 정해야 하며, 하나의 판독기로 학습에서
열거하지 않은 환자 고유 관찰·속성·관계를 자연어로 출력하는 과제와는 다릅니다.

Pilot에서는 같은 협심증 계열 activation에 대해 diagnosis probe는 동일한 class를
출력했지만, 자연어 판독은 layer에 따라 세부 속성 보존 여부가 달랐습니다.

```text
gold observation: chest pain even at rest
L24 readout:       chest pain at rest                              [보존]
L32 readout:       increased with exertion but alleviated by rest  [반전]
diagnosis probe:   두 위치 모두 협심증 계열
```

이 사례는 probe가 틀렸다는 뜻이 아니라, diagnosis label 하나만으로는 `at rest`와
`relieved by rest`의 차이를 표현할 수 없다는 뜻입니다. 본 연구는 별도 head를
계속 추가하지 않고 하나의 reader가 held-out 관찰과 관계를 자연어로 복원할 수
있는지를 평가합니다.

### 가설 3. 의료 SFT만으로는 faithful readout이 되지 않으며 독립적인 activation 검증이 필요하다

AV는 activation을 자연어로 출력할 수 있지만, 임상적으로 그럴듯한 문장을 생성한
것과 실제 activation을 읽은 것은 다릅니다. 실제 pilot에서도 진단명을 직접
생성하도록 SFT한 Medical-AV가 seen-class classifier처럼 작동하고 diagnosis-heldout
성능이 붕괴했습니다. 따라서 본 방법은 단순 SFT를 ablation으로 두고, 임상
supervision과 AV-AR reconstruction/contrastive grounding을 함께 사용하는 full
Medical-NLA를 평가합니다.

Medical-NLA는 다음 두 관문을 모두 통과해야 합니다.

1. 의사 주석의 observation-rationale-diagnosis 구조를 CoT와 vanilla NLA보다 잘
   보존하는가
2. matched-vs-shuffled, evidence counterfactual, activation swap, AV-AR round-trip을
   통해 판독 내용이 실제 activation에 사례 특이적으로 근거함을 보이는가

두 관문을 통과한 뒤에만 dataset-native claim을 자연어로 편집하고 AR로 activation에
되돌리는 text patching을 평가합니다. 이 단계에서는 목표 속성과 관련 likelihood가
선택적으로 변하는지, 비목표 정보가 얼마나 보존되는지를 측정합니다. Backbone
성능 향상은 사전 결론이 아니라 이 마지막 단계에서 검증할 후속 가설입니다.

### 기존 설계와의 관계

8월 19일에 확인받은 `CoT만으로는 부족함 -> 기존 내부 도구의 열린 설명 한계 ->
검증된 Medical-NLA`라는 주제는 유지했습니다. 후속 pilot에서 CoT의 단순 누락
가설, DDXPlus 오답 예측, diagnosis-target SFT에 confound가 확인되어, 설명 품질과
activation faithfulness를 분리하고 각 가설을 직접 측정하도록 표와 실험을 다시
구성했습니다.

---

## 2. 데이터셋 역할

| 데이터셋 | 원래 제공하는 정보 | 본 연구 역할 | 사용하지 않을 주장 |
|---|---|---|---|
| DiReCT | 임상 노트, physician observation, rationale, diagnosis tree | CoT 대 NLA의 주 설명 품질 평가 | Activation ground truth |
| DDXPlus | pathology, evidence ID/value, differential | Held-out cue/relation, 짝 깨기, cue 반사실, patching | 자연 임상 설명의 최종 품질 |
| MedCaseReasoning | 실제 case report 산문과 진단 reasoning | 학습하지 않은 자연 텍스트·긴 꼬리 진단 OOD | Gold evidence span이 있다는 주장 |

DiReCT는 clinical alignment를 측정하고, DDXPlus는 activation grounding을 측정한다.
둘을 섞어 하나의 `faithfulness score`로 만들지 않는다.

### DiReCT 배포본 감사 상태

2026-08-26에 PhysioNet 승인 배포본을 확인했다.

| 파일 | SHA-256 | 확인 |
|---|---|---:|
| `samples.rar` | `a35f1de8655ded767eaf5f428194ef25af0a600237f14f77ced917fefd85641e` | 일치 |
| `diagnostic_kg.rar` | `6a818a85d7b0736bad409622816b6f2e5a91fc8a147622e1a2667ffadc8f4d7f` | 일치 |

- README 기준 주석 note는 511개다.
- archive에는 디렉터리 엔트리를 포함해 585개 항목이 있다.
- KG archive에는 24개 질환 JSON과 루트 디렉터리 엔트리가 있다.
- 주석 node type은 `Input`(observation), `Cause`(rationale),
  `Intermedia`(diagnosis)이며, `deduction_assemble()`가
  `(observation, rationale, diagnosis)` 구조를 만든다.

### 제한 데이터 취급

DiReCT는 PhysioNet Restricted Health Data License 1.5.0 대상이다.

- `.rar`, 추출 JSON, 원문 note, 환자 단위 ID를 Git에 올리지 않는다.
- 다른 사용자와 접근권한을 공유하지 않는다.
- 서버에서는 사용자 전용 디렉터리와 권한 `700`을 사용한다.
- 논문에는 비식별 예시도 license와 disclosure 위험을 다시 확인한 뒤 사용한다.
- 공개 repository에는 재현 코드와 aggregate 결과만 올린다.

권장 서버 경로:

```text
/data1/heejae/restricted/direct/
  archives/
  samples/
  diagnostic_kg/
  audit/
```

---

## 3. 최종 Table 설계

### Table 1. Closed-label detection versus open natural-language readout

목적: probe보다 NLA가 무조건 정확하다고 주장하지 않고, 두 도구의 능력 경계를
보인다.

#### A. Closed diagnosis decoding

| Method | Separate task head | Seen-label Acc. | Patient-heldout Acc. | Diagnosis-heldout |
|---|---:|---:|---:|---:|
| Output-head likelihood | no | TBD | TBD | candidate ranking |
| Linear probe | yes | TBD | TBD | N/A |
| Vanilla NLA | no | TBD | TBD | TBD |
| Medical-AV, SFT only | no | TBD | TBD | TBD |
| Medical-NLA | no | TBD | TBD | TBD |

#### B. Open natural-language recovery

| Method | Held-out cue P/R | Held-out relation match | MCR source-answer fidelity | MCR gold match on source-wrong |
|---|---:|---:|---:|---:|
| CoT reasoning | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD |
| Medical-NLA | TBD | TBD | TBD | TBD |

Probe는 Panel B에서 `0`이 아니라 `N/A`다. 각 cue/relation별 별도 probe를 만들 수는
있지만, 이는 하나의 open reader와 다른 과제다. Appendix에 multi-label probe를
추가할 경우 output ontology와 head 수를 함께 보고한다.

### Table 2. DiReCT clinical explanation quality

목적: 교수님이 제안한 `정답을 얼마나 잘 맞추고 설명을 얼마나 잘하는가`를
의사 annotation 기준으로 평가한다.

| Method | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| CoT reasoning | TBD | TBD | TBD | TBD | TBD | TBD |
| Diagnosis probe | TBD | N/A | N/A | N/A | N/A | N/A |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA | TBD | TBD | TBD | TBD | TBD | TBD |

- `Accdiag`: primary discharge diagnosis 정확도
- `Obspre`: 출력 observation 중 physician observation과 대응된 비율
- `Obsrec`: physician observation 중 출력이 복원한 비율
- `Obscomp`: observation 집합의 semantic Jaccard completeness
- `Expcom`: 대응된 observation에서 rationale와 diagnosis edge까지 맞은 비율
- `Expall`: 누락·추가·관계·진단 오류를 모두 포함한 end-to-end alignment

이 표는 activation faithfulness가 아니라 `expert-reference clinical alignment`다.
CoT와 NLA 출력은 동일한 claim schema로 정규화하고 method 이름을 judge에게 숨긴다.

### Table 3. Activation grounding on controlled DDXPlus cases

| Method | Own-case F1 | Shuffled F1 | Case gap | Removed-cue deletion | Untouched retention | Round-trip FVE |
|---|---:|---:|---:|---:|---:|---:|
| CoT reasoning | TBD | TBD | TBD | TBD | TBD | N/A |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD 또는 N/A |
| Medical-NLA | TBD | TBD | TBD | TBD | TBD | TBD |

필수 통제:

1. 같은 진단·비슷한 cue 수 안에서 hard shuffle
2. mean activation 바닥
3. patient 간 activation swap
4. cue 하나 제거 후 재추출
5. 삭제하지 않은 cue 보존율
6. AV 설명만으로 AR이 activation을 복원하는 FVE

### Table 4. Text-mediated intervention

| Intervention | No-op top-1 preservation | No-op KL | Edited-value decoding | Target logit delta | Off-target KL | Target behavior rate |
|---|---:|---:|---:|---:|---:|---:|
| Plain-text prompt edit | N/A | N/A | TBD | TBD | TBD | TBD |
| Raw activation patch | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA text patch | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA text patch | TBD | TBD | TBD | TBD | TBD | TBD |

자유 산문을 임의로 고치지 않고 DDXPlus native value만 편집한다.

```text
dyspnea: present -> absent
leg swelling location: left calf -> right calf
severity: mild -> severe
```

Table 3를 통과하지 못하면 Table 4는 실행하지 않는다.

---

## 4. Figure 설계

| Figure | 내용 | Table과 겹치지 않는 역할 |
|---|---|---|
| Figure 1 | CoT, P0/P1 activation, AV, AR, 세 평가 gate의 전체 파이프라인 | 연구 논리와 위치 정의 |
| Figure 2 | DiReCT 한 사례의 physician tree / CoT / vanilla / SFT-only / full NLA 비교 | 평균 점수가 숨기는 누락·환각·관계 오류 |
| Figure 3 | 원본 대 cue-제거 반사실과 paired grounding 분포 | 설명이 case activation을 따라 변하는 과정 |
| Figure 4 | `h -> AV -> edit -> AR -> h_edit -> patch`와 target/off-target 변화 | 자연어 bottleneck을 통한 선택적 개입 과정 |

Appendix에는 layer-position heatmap, diagnosis/cue별 성능, 학습 곡선, failure
taxonomy, MCR OOD 사례, seed sensitivity를 둔다.

---

## 5. 지금 가장 시급한 작업

### E0. DiReCT 데이터·평가기 감사 -- 지금 바로

학습보다 먼저 끝내야 한다.

1. Restricted directory에 archive를 복사하고 권한을 `700`으로 고정
2. SHA-256 재확인
3. 511개 JSON의 실제 schema와 node type 집계
4. disease category, PDD, note 수, 동일 환자·동일 admission 중복 감사
5. observation/rationale/diagnosis node 수와 누락 필드 집계
6. 공식 `cal_a_json()`과 `deduction_assemble()`가 전 파일에서 동작하는지 확인
7. 공식 evaluator와 공개 baseline 수치 재현
8. patient/PDD leakage가 없는 split 후보 작성

이 단계의 산출물은 raw note가 아닌 aggregate JSON/Markdown이어야 한다.

### E0의 결정 게이트

교수님께 다음을 확인받는다.

- DiReCT를 PDD-disjoint supervised split으로 Medical-NLA 학습에도 사용할지
- DiReCT를 external-only test로 두고 DDXPlus만 학습에 사용할지

권장안은 DiReCT PDD-disjoint train/validation로 clinical supervision을 주고,
held-out PDD test와 MCR external OOD를 두는 것이다. 단 511개라 split별 표본과
disease/PDD 중복을 먼저 확인해야 최종 결정할 수 있다.

### E1. Source baseline과 activation 추출 -- E0 이후

동일 note에서 다음을 만든다.

```text
source CoT: <reasoning> ... </reasoning>
source answer: <answer> diagnosis </answer>
P0: final prompt token activation
P1: <answer> marker의 마지막 subtoken activation
P2: diagnosis 생성 후 activation, positive control only
```

L16/L24/L32를 추출하되 validation에서 primary layer를 선택하고 test에는 한 번만
적용한다. CoT reasoning에 final diagnosis alias가 먼저 등장한 행은 P1
source-decision 분석에서 제외한다.

### E2 이후

```text
E0 dataset/evaluator audit
 -> E1 source CoT + P0/P1 activation
 -> E2 output-head/probe/vanilla baseline
 -> E3 SFT-only vs full Medical-NLA
 -> E4 DiReCT Table 2
 -> E5 DDXPlus Table 3
 -> E6 text patching, E5 통과 시에만
 -> E7 MCR external OOD
```

## 6. 다음 미팅에서 확인받을 질문

1. 위 세 가설의 표현이 기존 컨펌 범위를 유지하는가
2. DiReCT를 supervised PDD-disjoint split으로 사용할 것인가
3. `clinical alignment`와 `activation grounding`을 별도 gate로 두는 데 동의하는가
4. CoT 대 NLA 주 위치를 P1 answer boundary로 두는가
5. Table 3 통과 전에는 patching을 하지 않는 중단 기준에 동의하는가

더 상세한 metric 산식과 학습 목적식은
[`medical_nla_evaluation_confirmation_2026-08-26.md`](medical_nla_evaluation_confirmation_2026-08-26.md)에 있다.
