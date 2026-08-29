# 논문 표 완결 계획

## 질문

현재까지 확정된 양성·음성 결과로 어떤 표를 본문에 남기고, 각 셀을 어떤
artifact로 채우며, 남은 계산을 최소화할 것인가?

이 문서는 성공할 생성형 Medical-NLA를 가정해 빈 행을 만드는 계획이 아니다. 검증
관문을 통과한 결과만 본문에 넣고, 탈락한 방법은 validation 개발 결과로 보고한다.

## 표 구성 원칙

1. 본문 표는 **representation/behavior**, **DiReCT clinical alignment**,
   **DDXPlus grounding**의 세 개로 닫는다.
2. 생성형 NLA가 validation gate를 통과하지 못하면 locked-test 행을 만들지 않는다.
   `TBD`나 0으로 채우지 않고 그 행 자체를 제외한다.
3. Probe와 structured reader를 open-ended NLA처럼 부르지 않는다. 각각
   `closed decoder`와 `structured monitor`로 명시한다.
4. D10/D14/D16 같은 방법 개발 결과는 동일 metric이 아니므로 clinical alignment나
   grounding 주표에 억지로 합치지 않고 별도 validation ablation 표로 둔다.
5. Table 3 grounding gate를 통과한 free-generating NLA가 없으므로 기존 text-patching
   Table 4는 실행·게재하지 않는다. 이는 누락이 아니라 사전 등록된 gate의 결과다.
6. Validation과 locked test를 같은 열에서 혼합하지 않는다. Method selection 수치는
   appendix/development table, 동결 후 한 번 계산한 값은 main result table에 둔다.

## Main Table 1 — Internal-state availability and readout boundary

Table 1은 backbone 행동, closed probe, vanilla open readout의 역할을 분리한다. 한 행의
accuracy가 서로 다른 의미를 갖지 않도록 세 panel로 구성한다.

### Panel A. Backbone diagnostic behavior

동결 split의 동일 case ID에서 기존 source output을 재집계한다. 새 backbone generation은
필요 없다.

| Generation | Pool | n | Parse | Strict PDD | Disease category |
|---|---|---:|---:|---:|---:|
| Direct, answer-prefilled | test seen | 72 | pending reindex | pending | pending |
| Source CoT | test seen | 72 | pending reindex | pending | pending |
| Direct, answer-prefilled | PDD-heldout | 106 | pending reindex | pending | pending |
| Source CoT | PDD-heldout | 106 | pending reindex | pending | pending |

이미 계산된 171-case exploratory 참고값은 Direct/CoT strict PDD
`.2105/.1930`, category `.5029/.5088`이다. 이 값은 새 72/106 두 pool의 셀에
복사하지 않는다. 재집계 결과와 대조하는 audit 값으로만 사용한다.

`Official semantic diagnosis` 열은 필수 열에서 뺀다. 공식 DiReCT evaluator의 주 목적은
explanation alignment이고, strict/category와 별도로 source diagnosis semantic judge를
다시 돌리는 것이 핵심 주장에 필요하지 않기 때문이다. 실행 시간이 남으면 appendix
sensitivity로만 추가한다.

### Panel B. Closed P0 decodability

| Dataset / target | Decoder | Validation | Locked test | Control |
|---|---|---:|---:|---:|
| DiReCT disease category | HS24 linear, 25-way | `.5962` | test-seen pending | label-shuffle pending |
| DiReCT canonical PDD | HS24 linear, 49 train labels | `.4423` | test-seen pending | label-shuffle pending |
| DDXPlus finding presence | HS24 multi-label, 91 IDs | `.9607` | `.9562` | shuffled `.7938`; gap `+.1624 [.1576,.1672]` |
| DDXPlus native value | HS24 conditional, 6 tasks/32 values | `.7700` | `.7659` | shuffled `.5791`; gap `+.1868 [.1650,.2091]` |

Canonical PDD는 train ontology에 없는 PDD-heldout node에 대해 top-1이 정의되지 않으므로
PDD-heldout 셀을 만들지 않는다. Disease category는 실제 train-supported category 여부를
감사한 뒤 정의되는 행만 보고한다. Source-decision probe는 논문의 필수 결론에 필요하지
않고 ontology 추가 결정이 남아 있으므로 본문 표에서 제외한다.

Layer sensitivity는 Table 1의 열을 늘리지 않고 Figure 2 또는 appendix에 다음 확정값을
그대로 사용한다.

| Target | HS16 | HS24 | HS32 |
|---|---:|---:|---:|
| DiReCT category top-1 | `.5000` | `.5962` | `.5192` |
| DiReCT PDD top-1 | `.3846` | `.4423` | `.3846` |
| DDXPlus finding micro F1 | `.9636` | `.9607` | `.9607` |
| DDXPlus value accuracy | `.7641` | `.7700` | `.6990` |

### Panel C. Open vanilla AV boundary

Validation 52 cases × 2 prompts × 3 layers의 semantic audit 312행을 짧은 boundary
panel 또는 본문 문장으로 보고한다.

| Readout | source answer | gold PDD | category |
|---|---:|---:|---:|
| Vanilla AV, default HS32 | `0/52` | `0/52` | `0/52` |
| Vanilla AV, task-aligned HS32 | `0/52` | `0/52` | `0/52` |

이는 observation grounding 0이 아니라 **명시적 diagnosis readout 0**이다. Exact readout
quote를 요구한 single Llama-3-8B judge 결과임을 caption에 쓴다. HS16 category의
`1/52`는 appendix layer sensitivity로 둔다.

## Main Table 2 — DiReCT clinical explanation alignment

### Locked-test 본문 표

72 test-seen과 106 PDD-heldout을 두 panel로 보고한다. 지금 확정적으로 실행할 행은
Source CoT와 Vanilla NLA 두 개다.

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | pending | pending | pending | pending | pending | pending | pending |
| Vanilla NLA | pending | pending | pending | pending | pending | pending | pending |
| Selected generative Medical-NLA | gate 통과 시에만 생성 |  |  |  |  |  |  |

마지막 행은 D10 1,552-step 결과가 teacher-forced D5 gate를 통과하고, 이어지는 generation
validation에서 `Obscomp > .2130`을 통과할 때만 materialize한다. 실패하면 빈칸을 남기지
않고 행을 삭제한다. D10이 DDXPlus one-claim objective라는 이유만으로 곧바로 DiReCT
locked test를 읽지 않는다.

### Appendix development table

생성형 방법 실패는 숨기지 않고 validation 표로 보고한다. 서로 다른 extractor run의
Source CoT 값이 조금 다르므로 아래 두 block을 합치지 않는다.

**Common 248+248 SFT, same 50 Direct cases**

| Method | Obs. rows | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | `.3110` | `.4069` | `.2399` | `.0657` | `.0168` |
| Vanilla NLA | 10/50 | `0` | `0` | `0` | `0` | `0` |
| Common SFT seed 17 | 50/50 | `.0100` | `.0037` | `.0034` | `0` | `0` |
| Common SFT seed 29 | 50/50 | `0` | `0` | `0` | `0` | `0` |
| Common SFT seed 43 | 50/50 | `.0070` | `.0054` | `.0043` | `0` | `0` |

**Full-data canonical-target SFT, same 50 Direct cases**

| Method | Obs. rows | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | `.2835` | `.3726` | `.2130` | `.0650` | `.0153` |
| Full-data SFT seed 17 | 50/50 | `.0544` | `.0502` | `.0301` | `0` | `0` |
| Full-data SFT seed 29 | 50/50 | `.0553` | `.0388` | `.0296` | `0` | `0` |

이 표의 결론은 “Medical-NLA가 0점”이 아니라, 형식 생성은 성공했지만 physician
observation과 사례별로 정렬되지 않았다는 것이다. 따라서 seed를 골라 locked test로
보내지 않았다.

## Main Table 3 — DDXPlus activation grounding

이 표는 이미 locked test 숫자로 거의 완결됐다. Probe와 structured monitor를 나란히
두되 동일한 종류의 모델이라고 해석하지 않는다.

### Panel A. Static availability and case specificity

| Method | Finding F1 | Same-diagnosis shuffled | Pair gap | Native-value accuracy |
|---|---:|---:|---:|---:|
| Frozen closed probe | `.9562` | `.7938` | `+.1624 [.1576,.1672]` | `.7659` |
| Probe-guided structured monitor | `.9587` | `.7938` | `+.1624` | `.7654` |

Structured monitor의 mean emitted claims는 `4.9353`, native-value emission coverage는
`.9995`다. Prompt text는 prediction construction에 사용하지 않았다.

### Panel B. Counterfactual response

| Method | Deletion phantom | Removal success | Untouched retention | Replacement hit | Old persistence | Clean switch |
|---|---:|---:|---:|---:|---:|---:|
| Probe-guided structured monitor | `.3593` | `.6407` | `.9987` | `.1466` | `.5955` | `.0804` |

분모는 deletion `4,540`, value edit `539`, clean-switch eligible `398`이다. 이 결과는
정적 finding은 강하게 읽히지만 삭제 state가 완전히 사라지지 않고 native value update는
약하다는 양면 결론을 만든다.

Free-generating Vanilla/SFT/D10 행은 validation generation grounding을 통과한 경우에만
추가한다. 현재까지 통과한 행이 없으므로 locked-test `TBD` 행을 만들지 않는다.

## Appendix Table — Generative method development gates

서로 다른 실패를 하나의 accuracy로 합치지 않고 각 방법의 사전 지정 gate를 숫자로
보고한다.

| Method | Primary validation statistic | Frozen requirement | Result |
|---|---|---|---|
| Full-data SFT | DiReCT Obscomp | `>.2130` | `.0301/.0296`, fail |
| D10 1×2, 20 steps | ranking-control changed-gap delta, seeds 17/29/43 | each `>=.05`, CI > 0, specificity | `+.0005/+.0028/+.0030`, fail |
| D14 K=5 OOF teacher | original cue precision | `>=.90` plus six calibration gates | `.8881`, fail |
| D16 soft bottleneck | proposed-control Direct alignment delta | each `>=.005`, CI > 0 | `-.001137/-.001476/+.001433`, fail |
| D16 frozen-z | auxiliary-control finding F1 | positive across seeds | `-.0009/-.0007/-.0016`, fail |
| D10 budget calibration | final step 1,552 D5 gate | same D5 gate, no extension | running/pending |

D16 frozen-z의 own-shuffled gap delta는 `-.0050/-.0046/-.0058`, value accuracy delta는
`-.0137/-.0096/-.0160`, deletion-drop delta는 `-.0167/-.0141/-.0151`이었다.

이 표는 여섯 objective를 같은 metric으로 순위화하지 않는다. 각 objective가 자기
promotion gate를 통과했는지만 보인다. D10 budget run은 기존 결과를 본 뒤 승인된
post-hoc exploratory calibration이라고 명시한다.

## 삭제할 기존 표와 그림

### Text patching Table 4

삭제한다. Free-generating NLA가 grounding gate를 통과하지 않았으므로 identity patch,
edit patch, behavioral utility를 수행하면 사전 등록 순서를 위반한다. `not run because the
grounding prerequisite failed`를 Results와 Limitations에 한 문장으로 기록한다.

### Figure 4 text bottleneck intervention

같은 이유로 삭제한다. 성공하지 않은 미래 방법을 전제로 빈 pipeline 그림을 남기지 않는다.

### 유지할 그림

1. Figure 1: 전체 evaluation pipeline과 두 gate.
2. Figure 2: HS16/24/32 probe layer sensitivity.
3. Figure 3: DDXPlus original/deletion/value-edit paired response 분포. 평균만 반복하지
   않고 deletion probability drop, phantom, untouched retention 분포를 시각화한다.

## 가장 빠른 셀 완결 순서

### D10과 독립적으로 즉시 실행

1. 기존 496 source outputs을 frozen split으로 재집계해 Table 1A의 72/106 Direct/CoT
   strict/category 셀을 채운다. 새 GPU generation은 없다.
2. 이미 고정한 HS24 DiReCT probe를 test-seen 72에 한 번 적용하고 label-shuffle control을
   계산한다. PDD-heldout은 정의되지 않은 셀을 억지로 채우지 않는다.
3. Source CoT와 Vanilla NLA만 frozen 72/106에서 공통 extractor/evaluator로 평가해
   Table 2 baseline을 닫는다. Vanilla output이 없는 case만 생성한다.
4. Table 3은 현재 locked-test artifact로 즉시 원고에 반영한다.

### D10 budget run 종료 후 분기

1. Step 1,552 D5 gate 실패: Appendix gate 행을 확정하고 generative Medical-NLA의
   locked-test 행은 만들지 않는다. 본문은 structured monitor 양성 결과와 생성형 objective
   음성 결과로 닫는다.
2. Teacher-forced D5 gate 통과: validation generation을 한 번 수행한다.
3. Generation에서 `Obscomp <= .2130`: locked test를 읽지 않고 실패 ablation으로 닫는다.
4. Generation에서 `Obscomp > .2130`: 해당 단일 frozen method만 Table 2와 Table 3의
   locked test에 한 번 보낸다.

어느 분기에서도 1,552 step 이후 epoch/lambda/temperature를 자동 탐색하지 않는다.

## 완료 정의

다음 네 조건이면 숫자 표는 완료다.

- Table 1A frozen 72/106 재집계 완료.
- Table 1B DiReCT test-seen closed probe와 label-shuffle 완료.
- Table 2 Source CoT/Vanilla frozen baseline 완료.
- D10 step 1,552의 최종 분기 판정 완료.

Table 3은 이미 완료됐고 patching table은 gate 실패로 제거됐다. 따라서 D10 결과를 기다리는
동안에도 나머지 논문 표를 병렬로 채울 수 있다.

## 판정

현재 상태는 **표 구조 제안 / 사람 확인 대기**다. 승인되면
`docs/paper/tables_and_figures.md`를 이 구조로 교체하고, Table 1A → Table 1B →
Table 2 baseline 순서로 실행 스크립트를 고정한다. 승인 전에는 canonical paper table을
바꾸거나 locked-test 추가 method를 실행하지 않는다.
