# Medical-NLA 성공 시 논문 표 편입 계획

## 질문

Medical-NLA가 사전 지정한 validation 관문을 모두 통과한다고 가정할 때, 현재 논문 표를
어떻게 바꾸고 각 셀을 어떤 모집단과 통계 단위로 채울 것인가?

이 문서는 성공을 선언하지 않는다. 성공했을 때 표를 사후적으로 다시 설계하지 않도록
미리 정한 **조건부 편입안**이다. 실패 시 출구는
[`2026-08-29-paper-table-completion.md`](2026-08-29-paper-table-completion.md)를 따른다.

## 성공의 정확한 정의

최종 Medical-NLA 행은 다음 조건을 모두 만족한 뒤에만 materialize한다.

1. D10 1x2 ranking의 step 1,552 최종 checkpoint가 seeds 17/29/43 모두에서 동결 D5
   관문을 통과한다. 중간 checkpoint는 dose-response 보고용이며 모델 선택에 쓰지 않는다.
2. DDXPlus validation generation에서 changed-cue 반응, original hit, deletion phantom,
   retained-cue specificity를 포함한 동결 generation 관문을 통과한다.
3. DiReCT validation에서 같은 extractor와 official evaluator를 사용해
   `Obscomp > .2130`과 `Expcom > .0650`을 모두 만족한다.
4. 학습 데이터, objective, decoding prompt, layer, checkpoint 선택 규칙을 test 접근 전에
   하나의 recipe로 동결한다.
5. 세 seed를 모두 평가하고 평균과 seed 간 표준편차를 보고한다. validation에서 가장 좋은
   seed 하나를 골라 test에 보내지 않는다.

DiReCT 248행 adaptation을 final recipe에 포함한다면 epoch와 stopping rule은
`val_seen`의 gold-label-absent 50행에서만 정하고, DDXPlus locked test와 DiReCT 두 test
pool을 본 뒤에는 바꾸지 않는다.

## 모집단과 역할

### DiReCT

Raw 511행 중 충돌·ID 실패·중복 15행을 제외한 496행이 논문의 logical population이다.
Population SHA-256은
`7d0a89a880fa868959099b7146c369cccaac5e7701d7ce5d8f01356ecfb68894`다.

| Split | notes | patient groups | PDDs | categories | gold label in note | 역할 |
|---|---:|---:|---:|---:|---:|---|
| train | 266 | 244 | 49 | 25 | 18 | probe 학습과 Medical-NLA adaptation |
| val_seen | 52 | 47 | 24 | 18 | layer, epoch, promotion gate |
| test_seen | 72 | 64 | 25 | 21 | seen-PDD clinical alignment |
| test_pdd_heldout | 106 | 103 | 12 | 10 | unseen-PDD clinical alignment |

- Generative SFT의 primary 학습/validation은 gold label이 note에 직접 노출된 행을 제외한
  `248/50`을 사용한다. Closed probe의 기존 `266/52` 결과와 분모를 혼동하지 않는다.
- Test primary는 72행과 106행을 모두 포함한다. Gold-label exposure `3/72`, `5/106`은
  사후 제외하지 않고, 각각을 제외한 결과를 sensitivity로 병기한다.
- 72행과 106행을 합친 178행 pooled score는 보조 요약일 뿐 주 결과가 아니다.
- 같은 환자의 여러 note를 독립 표본으로 보지 않는다. CI와 paired difference는
  `patient_group` cluster bootstrap으로 계산한다.
- 기존 backbone output이 496행에 존재하므로 `dataset-level untouched test`가 아니라
  **locked downstream method evaluation**이라고 부른다.

### DDXPlus

| Split/artifact | original cases | intervention rows/targets | 역할 |
|---|---:|---:|---|
| official train development | 4,655 | 4,655 CoT-P0 rows | ontology와 probe 학습 |
| D9a supported training pairs | 3,104 | original/deleted selected-cue pairs | ranking objective 학습 |
| validation | 4,525 | 10,006 total activation rows | layer, threshold, promotion gate |
| locked test | 4,543 | 10,028 total activation rows | 최종 grounding 평가 |

Locked test의 metric별 실제 분모는 서로 다르다.

| Metric family | validation | locked test |
|---|---:|---:|
| same-diagnosis hard-shuffle pairs | 4,106 | 4,121 |
| native-value targets | 2,183 | 2,136 |
| cue-deletion pairs | 4,523 | 4,540 |
| native-value-edit pairs | 533 | 539 |
| clean-switch eligible | 395 | 398 |

따라서 Table 3과 Table 4에 단일 `n=4,543`을 모든 열의 분모처럼 쓰지 않는다. 각 열 또는
caption에 eligible denominator를 적고, paired CI는 `base_id` 단위 bootstrap으로 계산한다.
Hard shuffle donor는 같은 diagnosis 안에서만 고정한다.

### MCR

MCR은 natural-text external OOD 역할이다. 현재 모집단과 freeze protocol이 확정되지
않았으므로 기존 네 표의 빈 열로 넣지 않는다. Final recipe 동결 전에 MCR protocol이
확정되면 별도 external-OOD 표를 만들고, 그렇지 않으면 future work로 남긴다. MCR 결과로
DiReCT/DDXPlus method나 threshold를 다시 선택하지 않는다.

## 성공 시 Main Table 1

### Backbone behavior and representation availability

Medical-NLA가 성공해도 Table 1의 역할은 바뀌지 않는다. 이 표는 최종 생성 설명의 성능이
아니라 **backbone 행동과 P0에 정보가 존재하는지**를 보여준다.

#### Panel A. Backbone diagnostic behavior

72 seen-PDD와 106 PDD-heldout을 같은 열 구조의 별도 panel로 둔다.

| Generation | Parse coverage | Strict PDD | Disease category |
|---|---:|---:|---:|
| Direct, answer-prefilled | frozen split 재집계 | frozen split 재집계 | frozen split 재집계 |
| Source CoT | frozen split 재집계 | frozen split 재집계 | frozen split 재집계 |

#### Panel B. Closed P0 decodability

| Dataset / target | Frozen decoder | Validation | Locked evaluation | Control |
|---|---|---:|---:|---:|
| DiReCT category | HS24, 25-way linear | .5962 | test-seen 72에서 계산 | label shuffle |
| DiReCT canonical PDD | HS24, 49 train labels | .4423 | test-seen 72에서 계산 | label shuffle |
| DDXPlus finding | HS24, 91-label multi-label | .9607 | .9562 | shuffled .7938; gap +.1624 |
| DDXPlus native value | HS24, 6 tasks/32 values | .7700 | .7659 | shuffled .5791; gap +.1868 |

PDD-heldout에는 train ontology에 없는 output node가 있으므로 canonical-PDD probe cell은
`N/A`다. 0으로 쓰지 않는다. HS16/24/32 비교는 Figure 2 또는 appendix에 두고 주표에
`Layer` 열을 반복하지 않는다.

#### Panel C. Open vanilla boundary

Validation 52행의 vanilla AV diagnosis semantic audit를 boundary로 짧게 남긴다.
Default/task-aligned HS32의 source answer, gold PDD, category는 각각 모두 `0/52`였다.
이는 clinical observation alignment가 0이라는 뜻이 아니라, 명시적 diagnosis readout이
실패했다는 뜻이다.

**기존 표 대비 변경:** 최종 Medical-NLA 행을 Table 1에 추가하지 않는다. 성공한
Medical-NLA는 Table 2와 Table 3의 대상이고, Table 1은 정보 가용성과 vanilla 경계를
고정하는 표다.

## 성공 시 Main Table 2

### DiReCT clinical explanation alignment

Panel A는 test_seen `72 notes/64 groups`, Panel B는 PDD-heldout `106 notes/103 groups`다.
두 panel에서 다음 행과 열을 동일하게 사용한다.

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | locked evaluation |  |  |  |  |  |  |
| Vanilla NLA | locked evaluation |  |  |  |  |  |  |
| Medical-NLA, SFT only | 3-seed mean +/- SD |  |  |  |  |  |  |
| Medical-NLA, final | 3-seed mean +/- SD |  |  |  |  |  |  |

- `SFT only`는 동일한 mixed-data target과 학습량을 쓰되 ranking/grounding objective만 뺀
  사전 고정 ablation이다.
- `final`은 성공 관문을 통과한 하나의 recipe다. `reconstruction`과 `full objective`를
  실제로 서로 다른 동결 checkpoint로 평가하지 않았다면 두 개의 가상 행으로 나누지 않는다.
- Deterministic Source CoT/Vanilla는 case bootstrap CI를, 3-seed 방법은 seed 평균 +/- SD와
  case-cluster paired CI를 함께 보고한다.
- 모든 방법에 동일한 claim extractor와 official DiReCT evaluator를 적용한다. Extraction
  실패는 분모에서 제거하지 않고 coverage와 0점 처리 규칙을 유지한다.
- 공식 `+1` smoothing이 적용된 `Obspre/Obsrec`을 주표에 쓰고 unsmoothed 값은 appendix로
  보낸다.

**기존 표 대비 변경:** `Medical-NLA, reconstruction`과 `Medical-NLA, full objective`라는
가정 행을 없애고, 재현 가능한 `SFT only`와 `final` 두 행만 남긴다. Pool 열은 만들지 않고
72/106을 panel title과 caption에 적는다.

## 성공 시 Main Table 3

### DDXPlus activation grounding

Validation 수치는 method selection 및 appendix에만 사용한다. Main Table 3은 frozen
locked-test `4,543` original cases와 대응 intervention arms를 사용한다.

#### Panel A. Static grounding and case specificity

| Method class | Method | Parse coverage | Finding F1 | Shuffled F1 | Pair gap (95% CI) | Native-value accuracy |
|---|---|---:|---:|---:|---:|---:|
| closed decoder | Frozen probe | 1.0 | .9562 | .7938 | +.1624 [.1576,.1672] | .7659 |
| structured monitor | Probe-guided reader | 1.0 | .9587 | .7938 | +.1624 | .7654 |
| open generator | Vanilla NLA | locked evaluation |  |  |  |  |
| open generator | Medical-NLA, SFT only | 3-seed mean +/- SD |  |  |  |  |
| open generator | Medical-NLA, final | 3-seed mean +/- SD |  |  |  |  |

Probe와 structured monitor는 generative Medical-NLA의 경쟁 모델이 아니라 representation
upper baseline과 deterministic rendering control이다. 같은 표에 두되 `Method class`로
역할을 명시한다.

#### Panel B. Counterfactual response

| Method | Original hit | Deletion phantom | Removal success | Untouched retention | Replacement hit | Old persistence | Clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| Vanilla NLA | locked evaluation |  |  |  |  |  |  |
| Medical-NLA, SFT only | 3-seed mean +/- SD |  |  |  |  |  |  |
| Medical-NLA, final | 3-seed mean +/- SD |  |  |  |  |  |  |

Deletion `n=4,540`, value edit `n=539`, clean switch `n=398`을 caption에 고정한다. Final
Medical-NLA의 향상은 finding recall만이 아니라 phantom 비증가, untouched retention,
value update를 동시에 봐야 한다.

#### Panel C. Round-trip faithfulness

AR round-trip을 실제로 동결해 평가했다면 다음 panel을 추가한다.

| Method | Identity FVE/cosine | Matched FVE | Same-diagnosis shuffled FVE | FVE gap |
|---|---:|---:|---:|---:|
| Vanilla NLA + frozen AR | locked evaluation |  |  |  |
| Medical-NLA, SFT only + frozen AR | 3-seed mean +/- SD |  |  |  |
| Medical-NLA, final + frozen AR | 3-seed mean +/- SD |  |  |  |

AR가 구현되지 않았거나 identity gate를 통과하지 못하면 Panel C를 비워 두지 않고 제거한다.
Open-text output의 source-decision fidelity는 별도 ontology가 동결되지 않은 한 main panel에
넣지 않는다.

**기존 표 대비 변경:** `CoT` 행을 제거하고 closed probe/structured monitor control을
명시적으로 추가한다. `Hard shuffle`을 모호한 단일 열로 두지 않고 own score, shuffled
score, paired gap으로 분리한다. Counterfactual metric도 deletion과 value edit을 나눈다.

## 성공 시 Main Table 4

### Text bottleneck intervention and utility

Medical-NLA의 생성 grounding 성공만으로 Table 4를 자동 실행하지 않는다. Table 3C의
AR identity preservation과 matched-over-shuffled round-trip gate까지 통과했을 때만 연다.
모집단은 DDXPlus locked-test의 eligible intervention subset이며 DiReCT 178행과 섞지 않는다.

#### Panel A. Identity and target selectivity

| Intervention | Identity preservation | Edited-value decoding | Target logit delta | Off-target KL | Eligible coverage |
|---|---:|---:|---:|---:|---:|
| Raw activation no-op/control | locked evaluation | N/A | 0 reference | 0 reference | 1.0 |
| Vanilla NLA round-trip | locked evaluation |  |  |  |  |
| Medical-NLA final round-trip | locked evaluation |  |  |  |  |
| Oracle counterfactual activation | locked evaluation |  |  |  |  |

#### Panel B. Behavioral utility

| Policy | Overall accuracy | Wrong-to-right | Right-to-wrong | Net correction | Intervention rate |
|---|---:|---:|---:|---:|---:|
| No intervention | frozen baseline | 0 | 0 | 0 | 0 |
| Patch all | locked evaluation |  |  |  | 1.0 |
| Probe-gated | locked evaluation |  |  |  |  |
| Medical-NLA-gated | locked evaluation |  |  |  |  |
| Oracle-gated | locked evaluation |  |  |  |  |

Oracle는 attainable method가 아니라 상한이다. Medical-NLA-gated policy는 gold diagnosis나
gold changed cue를 사용하지 않는다. Identity patch가 원 답과 비목표 logits를 보존하지
못하면 edit utility를 해석하지 않고 Table 4 전체를 appendix failure로 이동한다.

**기존 표 대비 변경:** 성공 시 Table 4를 유지하되 `Identity preservation` 하나로 뭉치지
않고 no-op identity, target selectivity, off-target drift, 최종 행동 변화를 순서대로
검증한다. AR gate 실패 시에는 성공한 AV 결과가 있더라도 Table 4를 본문에 싣지 않는다.

## 선택과 test 접근 규칙

1. DDXPlus validation 4,525와 DiReCT validation 50/52만 method selection에 사용한다.
2. D10 중간 checkpoint 6개는 trajectory 설명용이다. Step 1,552 외 checkpoint를 final로
   고르지 않는다.
3. Final recipe 동결 후 seeds 17/29/43을 모두 DiReCT 72/106과 DDXPlus 4,543에 적용한다.
4. Locked-test seed 평균이 낮아도 seed, threshold, prompt, claim 수를 다시 고르지 않는다.
5. DiReCT exact-label-exposed sensitivity와 source-correct/source-wrong subgroup은 primary
   결과를 대체하지 않는다.
6. DDXPlus test는 probe/structured reader에 이미 사용됐으므로 새로운 dataset-level
   confirmatory set이라고 부르지 않는다. Final generative method에 대한 locked evaluation로
   표현한다.

## 가장 빠른 표 완결 순서

Medical-NLA 성공 여부와 독립적으로 다음 셀을 먼저 닫는다.

1. 기존 source outputs을 frozen split으로 재집계해 Table 1A 72/106을 채운다.
2. HS24 DiReCT probe를 test-seen 72에 한 번 적용해 Table 1B를 닫는다.
3. Source CoT와 Vanilla NLA를 72/106에서 평가해 Table 2 baseline을 닫는다.
4. 이미 확정된 DDXPlus probe/structured reader locked 수치를 Table 3에 반영한다.

D10 성공 후에는 다음 순서를 고정한다.

1. 동일 final recipe와 SFT-only ablation의 3-seed validation generation.
2. D5와 Gate C 최종 확인 및 recipe hash 기록.
3. DiReCT test_seen 72와 PDD-heldout 106 평가.
4. DDXPlus locked test 4,543 original/intervention 평가.
5. Table 3 round-trip gate 통과 시에만 Table 4 patching과 utility 평가.
6. 마지막에만 paper table을 채우며 test 결과로 행·열·threshold를 바꾸지 않는다.

## 기존 네 표에서 실제로 바뀌는 것

| 기존 구성 | 성공 시 구성 |
|---|---|
| Table 1에 서로 다른 readout을 한 accuracy 표로 혼합 | backbone, closed probe, vanilla boundary를 panel로 분리; final Medical-NLA는 넣지 않음 |
| Table 2의 SFT/reconstruction/full 가상 3행 | 실제 동결된 SFT-only와 final 2행으로 축소 |
| Table 2에 pool/n 반복 | 72/64 groups와 106/103 groups를 별도 panel title에 고정 |
| Table 3에 CoT, probe 없는 generative 행만 배치 | closed probe, structured monitor, vanilla, SFT-only, final의 역할을 구분 |
| Table 3의 단일 hard-shuffle/edited response 열 | static specificity, deletion, value edit, round-trip을 별도 panel로 분리 |
| Table 4를 Medical-NLA 성공과 동시에 실행 | AR identity/round-trip까지 통과해야만 조건부 실행 |
| best checkpoint/seed 한 개의 단일 점수 | 3 seeds 모두 평가, mean +/- SD와 cluster-aware paired CI 보고 |

## 판정

현재 상태는 **성공 조건부 표 구조 제안 / 사람 확인 대기**다. 이 문서는 Medical-NLA의
성공 또는 locked-test 실행을 승인하지 않는다. 사람이 이 구조를 승인하면 성공 분기에서
[`docs/paper/tables_and_figures.md`](../../paper/tables_and_figures.md)를 이 규격으로
갱신하고, 실패 분기에서는 기존 paper-table completion 문서를 따른다.
