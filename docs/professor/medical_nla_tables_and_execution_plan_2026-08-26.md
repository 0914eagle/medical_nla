# Medical-NLA 표·그림과 실행 계획 (2026-08-26)

이 문서는 가설을 실제로 검증하기 위한 표·그림·실험·데이터 실행 계획만 기록한다.
교수님께 전달할 메시지는 저장하지 않는다.

8월 19일에 확인받은 연구 방향인 `CoT의 한계 -> 닫힌 내부 도구의 한계 ->
검증된 Medical-NLA`는 유지한다. 후속 pilot에서 반증된 세부 기제와 계측
confound를 반영해 평가 방법을 갱신했다. 과거 표의 처분 근거는 archive의
[`hypothesis_disposition_2026-08-22.md`](../archive/legacy_wrong_note_2026-08-25/professor/hypothesis_disposition_2026-08-22.md)와
[`RETRACTIONS.md`](../archive/legacy_wrong_note_2026-08-25/experiments/RETRACTIONS.md)에 기록되어 있다.
현재 논문 정본은 [`../paper/README.md`](../paper/README.md)다.

## 1. 데이터셋 역할

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

Aggregate audit와 공식 공개 `data_list.csv` 대조에서 다음을 확인했다.

- sample JSON은 511/511 모두 유효하고, 각 파일에 진단 root가 하나씩 있다.
- 공식 data list는 25 disease category와 61 PDD를 포함한다. 경로만으로 PDD를
  추정한 최초 감사의 62개는 3-depth 경로에서 annotation root를 대신 사용해 생긴
  값이므로 정본 PDD 수로 인용하지 않는다.
- PDD별 표본 수는 1--41개, 중앙값 5개로 불균형하다. 따라서 단순 row-random
  split이나 PDD별 균등 성능 주장은 부적절하다.
- 최종 canonical manifest에는 469개 환자 그룹이 있다. 환자 ID를 파싱하지 못한 4행은
  primary split에서 제외했다. 14개 환자 그룹(37행)은 둘 이상의 resolved PDD에 걸치고,
  1개 환자 그룹(4행)은 둘 이상의 disease category에 걸친다. 따라서 patient-disjoint
  split과 함께, 같은 환자가 연결한 PDD들을 하나의 connected component로 묶었다.
- 완전 동일 JSON/input-text 중복은 한 그룹, 두 행이다. 결정론적으로 한 행만 남기고
  duplicate copy 한 행을 primary split에서 제외했다.
- 폴더 PDD와 annotation root가 다른 파일은 43개다. 공백·개행·복수형을 공식 PDD
  vocabulary로 정규화한 뒤 501/511행의 canonical PDD를 해결했다. 남은 10행은 모두
  `Acute Coronary Syndrome / STEMI / NSTE-ACS`의 의미 충돌이므로 자동 보정하지 않고
  primary split에서 제외했다. 43건 전체 제외가 아니라 이 10건 제외가 정본 규칙이다.
- restricted KG archive에는 `Gastritis`가 빠져 24개지만 공식 GitHub KG에는
  `Gastritis.json`을 포함한 25개가 있다. 그러나 공통 24개 중 canonical JSON hash가
  일치한 것은 7개뿐이고 17개는 내용이 달라 두 release를 섞지 않는다. 주 설명 평가는
  sample annotation만으로 수행하고, KG가 필요한 별도 실험에서는 restricted 24개만
  사용해 Gastritis를 제외하거나 KG 버전을 별도 조건으로 명시한다.
- data list에서 73개 note가 amended로 표시된다. 그러나 restricted release는 55개
  category/PDD 디렉터리 그룹과 그룹별 행 수를 유지하면서 511개 파일명을 전부
  바꿨다(basename overlap 0). 따라서 공개 data list의 amendment flag는 restricted
  개별 note에 경로로 조인할 수 없다. Content-based 조인을 별도로 검증하기 전에는
  73이라는 aggregate만 기록하고 row-level sensitivity 분석에 사용하지 않는다.

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

## 2. 최종 Table 설계

### Table 1. Backbone behavior and internal readout capability

목적: Direct/CoT의 실제 행동과 P0 내부 판독을 분리하고, probe보다 NLA가 무조건
정확하다고 주장하지 않으면서 두 도구의 능력 경계를 보인다.

#### A. Backbone behavior on identical case IDs

| Method | n | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled | TBD | TBD | TBD | TBD | TBD |
| Source CoT | TBD | TBD | TBD | TBD | TBD |

#### B. CoT-P0 internal readout on identical activations

| Method | Coverage | Seen-PDD gold | Held-out-PDD gold | Category gold | Source-decision fidelity | Open evidence | Trained head | Eval ontology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Output-head candidate score | TBD | TBD | TBD | TBD | TBD | N/A | no | yes |
| Linear PDD probe | TBD | TBD | N/A | TBD | TBD | N/A | yes | yes |
| Vanilla NLA, default prompt | TBD | TBD | TBD | TBD | TBD | TBD | no | no |
| Vanilla NLA, task-aligned prompt | TBD | TBD | TBD | TBD | TBD | TBD | no | no |
| Medical-NLA | TBD | TBD | TBD | TBD | TBD | TBD | no | train text only |

Output-head candidate score는 사전등록 PDD 문자열의 길이 정규화 sequence likelihood다.
Probe는 held-out PDD output node와 open evidence output이 없으므로 해당 칸은 `N/A`다.
Source-decision fidelity와 physician-gold alignment는 같은 점수로 합치지 않는다.

### Table 2. DiReCT clinical explanation quality

목적: 교수님이 제안한 `정답을 얼마나 잘 맞추고 설명을 얼마나 잘하는가`를
의사 annotation 기준으로 평가한다.

| Method | n | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoT reasoning | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, SFT only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

- `Accdiag`: 공식 코드 `acc_diag`; canonical PDD 문자열 exact match
- `Obspre`: 공식 코드 `comp_pre = matched / (predicted + 1)`
- `Obsrec`: 공식 코드 `comp_re = matched / (gold + 1)`
- `Obscomp`: observation 집합의 semantic Jaccard completeness
- `Expcom`: 대응된 observation에서 rationale와 diagnosis edge까지 맞은 비율
- `Expall`: 누락·추가·관계·진단 오류를 모두 포함한 end-to-end alignment

이 표는 activation faithfulness가 아니라 `expert-reference clinical alignment`다.
CoT와 NLA 출력은 동일한 claim schema로 정규화하고 method 이름, 원 note, gold annotation을
extractor에게 숨긴다. 실패 행은 제거하지 않고 failure로 세며 coverage를 함께 보고한다.
Full objective 행은 실제 objective 구현 후에만 유지한다.
공식 평가는 Llama-3-8B가 observation/rationale 의미 대응을 `Yes`로 판정한 뒤
`statistics.py`가 집계한다. Greedy observation matching의 순서 의존성, exact `Yes`
판정, 누락 파일을 0으로 처리하는 동작은 공식 재현과 별도의 민감도 분석으로 감사한다.

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

## 3. Figure 설계

| Figure | 내용 | Table과 겹치지 않는 역할 |
|---|---|---|
| Figure 1 | CoT, P0/P1/P2 activation, AV, AR, 세 평가 gate의 전체 파이프라인 | 연구 논리와 위치 정의 |
| Figure 2 | DiReCT 한 사례의 physician tree / CoT / vanilla / SFT-only / full NLA 비교 | 평균 점수가 숨기는 누락·환각·관계 오류 |
| Figure 3 | 원본 대 cue-제거 반사실과 paired grounding 분포 | 설명이 case activation을 따라 변하는 과정 |
| Figure 4 | `h -> AV -> edit -> AR -> h_edit -> patch`와 target/off-target 변화 | 자연어 bottleneck을 통한 선택적 개입 과정 |

Appendix에는 layer-position heatmap, diagnosis/cue별 성능, 학습 곡선, failure
taxonomy, MCR OOD 사례, seed sensitivity를 둔다.

---

## 4. 지금 가장 시급한 작업

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

현재 schema·중복·환자·PDD 감사와 canonical split manifest 생성은 완료되었다.
511행 중 label conflict 10행, unparsed patient 4행, duplicate copy 1행을 제외한 496행을
사용한다. seed 17의 pilot split은 train 263 / val-seen 62 / test-seen 71 /
test-PDD-heldout 100행이며 모든 split이 patient-disjoint다. Held-out PDD는 `HFrEF`,
`HFpEF`, `NSTEMI`, `Low-risk PE`, `Non-Allergic Asthma`다. 남은 E0 작업은 공식
loader/evaluator 재현과 evaluator version·prompt hash 고정이다.

이 split은 파이프라인을 여는 pilot 정본이지 최종 일반화 근거 하나로 고정하지 않는다.
Held-out 100행이 심폐계에 치우치고 `Non-Allergic Asthma`는 3행뿐이므로, 최종 결과는
connected-component 단위의 복수 seed 또는 group K-fold와 PDD별 macro 결과로 재확인한다.

공식 evaluator smoke test 코드는 준비되었다.

- `make_direct_oracle_predictions.py`: 공식 `cal_a_json()`과
  `deduction_assemble()`로 gold-oracle prediction 10건을 만든다.
- `run_direct_official_evaluator.py`: 원본 evaluator의 GPU 2 하드코딩을 제거하되 greedy
  matching과 exact `Yes` 규칙은 유지하고 raw judge 응답과 실패를 private audit으로 남긴다.
- `score_direct_official_eval.py`: 공식 `statistics.py`의 `+1` denominator와 누락 0점
  처리를 재현하며, unsmoothed observation P/R은 별도 민감도 값으로만 낸다.
- `run_direct_official_smoke.sh`: 위 세 단계를 10건 oracle에서 순서대로 실행한다.

Llama 접근 승인 전에는 `PREPARE_ONLY=1`로 oracle schema까지만 검사하고, 승인 후 로컬
Meta-Llama-3-8B-Instruct native weights로 평가한다. Oracle smoke가 observation/rationale
matching과 chain 진단 비교에서 예상 상한을 내지 못하면 실제 CoT/NLA 평가는 시작하지 않는다.

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

HS16/HS24/HS32를 추출하되 공개 AV/AR와 호환되는 HS32를 primary로 고정하고 test에는 한 번만
적용한다. 설명 품질의 주 NLA 입력은 CoT 생성 전 P0다. P1은 reasoning 이후 trajectory
분석이며, CoT reasoning에 final diagnosis alias가 먼저 등장한 행은 P1
source-decision 분석에서 제외한다. 10행 smoke에서 이 누출이 8행이었으므로 P1 전체를
CoT와의 주 비교에 사용하는 설계는 폐기한다.

현재 test-seen 71행과 PDD-heldout 100행은 이미 이 위치 결정과 vanilla AV 점검에
사용됐으므로 exploratory pilot이다. 이후 표의 최종 수치는 새 confirmatory split 또는
nested patient/PDD-group protocol을 먼저 동결하고, 그 output을 보기 전 table schema와
analysis code를 고정한 뒤 산출한다.

### E2 이후

```text
E0 dataset/evaluator audit
 -> E1 source CoT + P0/P1/P2 activation
 -> E2 output-head/probe/vanilla baseline
 -> E3 SFT-only, full objective는 구현된 경우에만 추가
 -> E4 DiReCT Table 2
 -> E5 DDXPlus Table 3
 -> E6 text patching, E5 통과 시에만
 -> E7 MCR external OOD
```

## 5. 다음 미팅에서 확인받을 질문

1. 위 세 가설의 표현이 기존 컨펌 범위를 유지하는가
2. DiReCT를 supervised PDD-disjoint split으로 사용할 것인가
3. `clinical alignment`와 `activation grounding`을 별도 gate로 두는 데 동의하는가
4. CoT 대 NLA 주 위치를 생성 전 P0로 두고, P1은 trajectory 보조 분석으로 두는가
5. Table 3 통과 전에는 patching을 하지 않는 중단 기준에 동의하는가

더 상세한 metric 산식과 학습 목적식은
[`medical_nla_evaluation_confirmation_2026-08-26.md`](medical_nla_evaluation_confirmation_2026-08-26.md)에 있다.
