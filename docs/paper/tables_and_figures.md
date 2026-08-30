# 논문 표와 그림 설계

빈 칸은 아직 실험하지 않은 값이다. 과거 소견서 pilot 수치로 채우지 않는다.

## Table 1. Backbone behavior and P0 representation audit

목적은 backbone의 실제 답과 Medical-NLA가 설명하려는 정보의 P0 decodability를 분리하는
것이다. Open-text NLA를 closed-label probe와 같은 accuracy 표에 넣지 않는다.

### Panel A. Backbone diagnostic behavior on identical case IDs

Seen PDD 72행과 held-out PDD 106행은 같은 열 구조의 두 패널로 보고한다.

| Generation | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | batch 단계 | | | |
| Source CoT | batch 단계 | | | |

이 패널은 새 generation 없이 기존 496 출력의 **frozen split 재집계(CPU)**로
채운다. 다만 locked label을 집계하므로 분석 접근 기록을 남기고, D19/D21 최종 판정과
baseline-only recipe 동결 후 Table 1B locked 열, Table 2 baseline과 **한 번에 일괄
실행**한다.

### Panel B1. DiReCT CoT-P0 diagnosis decodability audit

| Target | Decoder | Output space | Validation | Test seen | Test PDD-OOD | Required control |
|---|---|---|---:|---:|---:|---|
| Gold disease category | Linear probe | 25-way | .5962 | TBD | N/A | label shuffle |
| Gold canonical PDD | Linear probe | 49-way train labels | .4423 | TBD | N/A | label shuffle |

Source-decision probe 행은 본문에서 제외한다 — 논문의 필수 결론에 필요하지
않고 source-answer ontology 동결 결정이 남아 있기 때문이다(실행 계획 문서의
확정 사항). 되살리려면 별도 사람 결정이 필요하다.

### Panel B2. DDXPlus CoT-P0 finding/value decodability audit

| Target | Decoder | Output space | Validation | Locked test | Required control |
|---|---|---|---:|---:|---|
| Finding presence | Multi-label probe | 91 frozen evidence IDs | .9607 | **.9562** | same-diagnosis shuffled .7938; gap +.1624 [.1576,.1672] |
| Finding value | Conditional probe | 6 evidence tasks / 32 native values | .7700 | **.7659** | same-diagnosis shuffled .5791; gap +.1868 [.1650,.2091] |

`N/A`는 0점이 아니라 closed probe에 unseen output node가 없어 과제가 정의되지 않았다는 뜻이다.
Finding/value head는 diagnosis별로 따로 만들지 않는다. Gold diagnosis와 source decision도
같은 target으로 합치지 않는다.

Validation layer sensitivity는 주표에 `Layer` 열을 반복하지 않고 Figure 2와 아래 보조표에
HS16/24/32를 모두 보고한다.

| Target | HS16 Top-1 | HS24 Top-1 | HS32 Top-1 | Majority |
|---|---:|---:|---:|---:|
| Disease category | .5000 | **.5962** | .5192 | .0577 |
| Canonical PDD | .3846 | **.4423** | .3846 | .0962 |
| Finding presence, micro F1 | .9636 | **.9607** | .9607 | N/A |
| Finding value, conditional accuracy | .7641 | **.7700** | .6990 | N/A |

각 target은 validation에서 선택된 index 하나로 locked test를 한 번 평가하고, Table 1B
caption에 `category=HS24`, `PDD=HS24`, `finding/value=HS24`처럼 mapping을 명시한다.
Finding/value HS24는 validation의 own-minus-shuffled 우선 규칙으로 고정했다. Finding HS16과
HS24 gap 차이는 .0002에 불과하므로 HS24가 압도적으로 우세하다고 해석하지 않는다. 이 값은
test 결과가 아니며, AV/AR 호환 때문에 HS32로 고정한 Medical-NLA primary index를
바꾸는 근거로 사용하지 않는다. Table 1B의 probe는 설정을 동결한 뒤 locked test에서 한 번
계산한다.

Locked test의 cue deletion은 target probability를 평균 `+.6103` 낮추고 original-hit 조건
removal success `.6407`을 보였다. 반면 native value edit은 replacement hit `.1466`, old-value
persistence `.5955`, clean switch `.0804`였다. 따라서 정적 value decodability는 통과했지만
value counterfactual faithfulness는 실패했다. 이 probe 결과는 Table 3의 자연어 NLA 행을
대신하지 않는다.

Validation의 matched raw early forced-answer 결과는 category 25-way
`.4808/.6731/.5814`, PDD 49-way `.1538/.5192/.3250`(top-1/top-5/MRR)이었다. PDD는
corpus 빈도 1인 한 후보가 35/52 top-1이어서 후보 문자열 prior가 강했다. 사전 고정한
content-free prompt를 차감하면 category top-1이 `.2308`, PDD top-1이 `.0577`로 더
악화되고 다른 소수 후보로 다시 붕괴했다. 따라서 Table 1에는 raw matched 값을 무학습
행동 기준선으로 보고하되 이 prior 제한을 캡션에 적고, calibrated 값은 appendix
sensitivity로 둔다. 이 결과는 likelihood가 `나쁘다`는 일반 명제가 아니라 이 고정
completion과 label surface form으로 만든 ranking이 안정적인 내부 판독이 아니라는
진단이다.

동일 validation의 vanilla AV 결과는 Table 1에 섞지 않고 Results의 open-text baseline으로
보고한다. Default/task-aligned prompt와 HS16/24/32의 312 readout에서 primary HS32의 source
answer, gold PDD, category semantic match는 모두 0/52였다. 이는 diagnosis target의 명시적
복원 실패이며 observation 품질이나 activation grounding 점수가 아니다.

Blinded semantic audit도 312/312행을 판정했고 exact readout quote를 요구했다. Primary
default/HS32/P0의 source answer, gold PDD, category match는 모두 0/52였으며 task-aligned
HS32도 동일했다. HS16에서 category 1/52만 두 prompt에 관찰됐다. 따라서 약칭·동의어를
허용해도 P0 진단 target 복원이 개선되지 않았다는 validation 진단은 유지된다. 이는 열린
observation/rationale 점수나 activation grounding 점수가 아니므로 Table 2·3을 대신하지
않는다. Table 1의 semantic diagnostic 열은 exact readout quote를 요구한 local
Llama-3-8B 판정으로 확정하며, 표 머리말에 `LLM-as-a-judge`임을 명시한다. 사람 검증 점수로
부르지 않고 single-judge 한계는 limitations에 기록한다.

## Table 2. Clinical explanation alignment on DiReCT

Seen PDD 72행과 held-out PDD 106행은 아래 열 구조의 두 패널로 보고한다.

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | batch 단계 | | | | | | |
| Vanilla NLA | batch 단계 | | | | | | |

- Source CoT는 기존 496 출력의 평가만 필요하다. Vanilla NLA는 기존 출력과 겹치지
  않는 사례만 생성하되, **기존 출력의 생성 설정(prompt/decoding)이 동결 recipe와
  일치할 때만 재사용**하고 다르면 178건 전부 재생성한다.
- 두 baseline 행은 D19/D21 최종 판정과 recipe hash 동결 후 Table 1A/1B와 함께
  **한 번에 일괄 계산**한다(locked-test 규율). `TBD` 대신 빈 칸으로 두는 이유:
  이 행들은 반드시 채워질 예정 셀이다.
- 이번 frozen recipe에는 validation-promoted checkpoint가 없으므로 Medical-NLA 행을
  생성하거나 채점하지 않는다. 별도 사전 등록 방법이 향후 promotion gate를 통과하는
  경우에만 별도의 locked 접근과 행 추가를 결정한다. 가상 행(`reconstruction`,
  `full objective`)은 만들지 않는다.

- `Accdiag`: 생성한 세부 진단과 의사 주석 진단의 의미 일치
- `Obspre`: 생성 관찰 중 의사 observation과 일치하는 정도
- `Obsrec`: 의사 observation 중 생성 설명이 회수한 정도
- `Obscomp`: 필요한 observation 구성요소의 coverage
- `Expcom`: observation에 연결한 rationale의 일치
- `Expall`: 전체 explanation chain의 일치

공식 `Obspre`와 `Obsrec`에는 `+1` denominator smoothing이 있다. 주표는 공식값을
사용하고 unsmoothed precision/recall은 민감도 분석으로만 둔다. 이 표는 clinical
alignment를 측정하며 activation faithfulness를 단독으로 증명하지 않는다.
모든 method에 동일한 claim extractor를 적용하고, extraction 실패는 분모에서 제거하지 않는다.
`full objective` 행은 AR reconstruction 또는 preference/RL objective가 코드로 구현되고
검증됐을 때만 유지한다. 현재 `train_medical_nla_lora.py`는 SFT-only다.

## Table 3. Activation grounding on DDXPlus

이 표는 locked-test artifact로 이미 확정된 두 행을 갖는다. Probe와 structured
monitor는 free-generating NLA의 경쟁 모델이 아니라 각각 representation upper
baseline과 deterministic rendering control이며, `Method class` 열로 역할을 구분한다.

### Panel A. Static grounding and case specificity (locked test, n=4,543)

| Method class | Method | Finding F1 | Same-diagnosis shuffled | Pair gap (95% CI) | Native-value accuracy |
|---|---|---:|---:|---:|---:|
| closed decoder | Frozen probe | .9562 | .7938 | +.1624 [.1576, .1672] | .7659 |
| structured monitor | Probe-guided reader | .9587 | .7938 | +.1624 | .7654 |
| open generator | Vanilla NLA | .0000 | .0000 | +.0000 | .0000 |

Structured monitor의 mean emitted claims는 4.9353, native-value emission coverage는
.9995이며, prompt text는 prediction 구성에 사용하지 않았다.

### Panel B. Counterfactual response (deletion n=4,540, value edit n=539, clean switch n=398)

| Method class | Method | Deletion phantom | Removal success | Untouched retention | Replacement hit | Old persistence | Clean switch |
|---|---|---:|---:|---:|---:|---:|---:|
| structured monitor | Probe-guided reader | .3593 | .6407 | .9987 | .1466 | .5955 | .0804 |
| open generator | Vanilla NLA | .0000 | N/A | N/A | .0000 | .0000 | N/A (n=0) |

양면 결론: 정적 finding은 강하게 읽히지만(패널 A), 삭제된 state가 표현에서 완전히
사라지지 않고(phantom .3593) native value update는 약하다(clean switch .0804).

Vanilla NLA는 sealed locked readout 10,028행 전부에서 frozen 91-evidence
ontology claim을 하나도 방출하지 않았다. Lexical mapping `0`, method-blind
`gpt-5.6-sol` raw/accepted semantic mapping `0/0`, mapped row `0/10,028`이었고
finding/value 지표는 모두 0이었다. Deletion phantom `.0000`은 성공이 아니다:
original hit도 `.0000`이므로 removal success와 untouched retention은 조건부
분모가 없어 `N/A`다. Value edit도 replacement `.0000`, old persistence
`.0000`, clean-switch 분모 `0`으로 `N/A`다. Generic evaluator가 쓴 runtime
`summary.md` 제목은 structured-reader renderer를 재사용하지만 canonical method
class는 이 표의 `open generator`다.

- Free-generating 행: Vanilla는 sealed generation + mapper V2 receipt 뒤 위
  locked all-zero 결과로 확정했다(branch-independent baseline). Medical-NLA
  `SFT only`/`final` 행은
  validation generation grounding gate 통과가 조건이었으나 **D10 budget
  calibration(1,552 steps)이 frozen gate FAIL로 종료**되어 현재 materialize
  근거가 없다 — 행을 만들지 않는다.
- Hard shuffle은 같은 진단·비슷한 finding 수의 다른 사례 activation과 text 짝을
  바꾼다. Round-trip FVE(AR)는 별도 Panel C로, AR identity gate 통과 시에만 연다.
  공개 AR가 extraction index 32용이므로 round-trip과 patching은 HS32에서만
  보고하며, HS16/HS24에 같은 AR를 적용한 값은 decoder distribution shift가 섞여
  주표에 넣지 않는다.

## Appendix Table. Generative method development gates (validation)

서로 다른 실패를 하나의 accuracy로 합치지 않고, 각 방법이 자기 사전 지정 gate를
통과했는지만 보인다. 전부 validation 수치이며 locked test가 아니다.

| Method | Primary validation statistic | Frozen requirement | Result |
|---|---|---|---|
| Full-data SFT | DiReCT Obscomp | > .2130 | .0301/.0296, fail |
| D10 1x2, 20 steps | ranking−control changed-gap delta, seeds 17/29/43 | each ≥ .05, CI > 0, specificity | +.0005/+.0028/+.0030, fail |
| D14 K=5 OOF teacher | original cue precision | ≥ .90 + 6 calibration gates | .8881, fail |
| D16 soft bottleneck | proposed−control DiReCT alignment delta | each ≥ .005, CI > 0 | −.001137/−.001476/+.001433, fail |
| D16 frozen-z | auxiliary−control finding F1 | positive across seeds | −.0009/−.0007/−.0016, fail |
| D10 budget calibration (1,552 steps) | final-step D5 gate | 동일 D5, 연장 없음 | changed-gap −.0177/+.5618/+1.1233, specificity −.0442/+.0345/−.0040, fail |
| D20 specificity-anchored 2×2 (1,552 steps) | same-seed paired delta, seeds 17/29/43 | changed ≥ .05 + specificity CI > 0 + 동결 비열등 4종 | changed −.0143/−.0040/−.0266, specificity −.0278/−.0255/−.0217, retained-gap +.0135/+.0215/−.0049, fail |

D16 frozen-z의 보조 delta: own-shuffled gap −.0050/−.0046/−.0058, value accuracy
−.0137/−.0096/−.0160, deletion drop −.0167/−.0141/−.0151. D10 budget run은 기존
결과를 본 뒤 사람이 승인한 post-hoc exploratory calibration임을 명시한다.
Budget run의 해석: across-seed mean changed-gap은 `.0019→.5558`로 상승했으나
retained-gap이 `.0002→.5604`로 동반 상승하고 specificity는 `−.0046`으로
평탄했다 — margin 성장은 changed cue의 선택적 반영이 아니라 deleted-activation
detector 퇴화 해였다(RunPod A100-SXM4-80GB, 하드웨어·버전은 수치 원장 §8.4).
D20 해석: retained CE anchor가 detector를 차단했고(retained-gap delta 전 구간
|값| ≤ .0225, budget run의 +.5604와 대비) 그 상태에서 changed-gap 신호가 어느
dose에서도 나타나지 않았다 — budget run의 성장이 전부 편법이었다는 독립 확인.
Retained original NLL은 −.13~−.33으로 개선돼 최적화 자체는 정상이었다.

## Methods 기록. D9a cue support protocol

- Support rule: presence AND deletion delta AND same-fold/same-diagnosis
  cue-absent donor margin. 동결 cut `P=.90, D=0, M=0`.
- Validation coverage 3,032/3,034 = .9993, null false support 112/2,964 = .0378,
  Wilson 95% CI [.0315, .0453]. Train retained pairs 3,104/4,655.
- Cross-fitting `crc32(base_id) % 2`, out-of-fold scoring. Artifact SHA256는
  `08-ddxplus-d9a-selected-cue.md`에 고정.

## Table 4. Text patching and behavioral utility

이 표는 Table 3의 grounding 관문을 통과한 방법만 평가한다.

### Panel A. Identity preservation and target selectivity

| Intervention | Identity preservation | Edited-value decoding | Target logit delta | Off-target KL |
|---|---:|---:|---:|---:|
| Raw activation patch | TBD | TBD | TBD | TBD |
| Vanilla NLA round-trip | TBD | TBD | TBD | TBD |
| Medical-NLA round-trip | TBD | TBD | TBD | TBD |
| Oracle counterfactual activation | TBD | TBD | TBD | TBD |

### Panel B. Final behavioral utility

| Policy | Overall accuracy | Wrong-to-right | Right-to-wrong | Net correction | Intervention rate |
|---|---:|---:|---:|---:|---:|
| No intervention | TBD | TBD | TBD | 0 | 0 |
| Patch all | TBD | TBD | TBD | TBD | 1.0 |
| Probe-gated | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA-gated | TBD | TBD | TBD | TBD | TBD |
| Oracle-gated | TBD | TBD | TBD | TBD | TBD |

먼저 아무 내용도 바꾸지 않은 identity patch가 원 답과 비목표 logits를 보존해야 한다.
그 뒤 DDXPlus가 정의한 evidence value만 편집한다. 임의의 의학 문장을 만들지 않는다.

## Figure 1. 전체 파이프라인

DiReCT note -> Gemma source run -> P0/P1/P2 activations -> CoT/vanilla NLA/Medical-NLA
-> clinical alignment와 activation grounding의 두 관문 -> 조건부 text patching을 한 장에
표현한다. P0가 주 입력이고 P2가 leakage control임을 구분한다.

## 공통 population caption

주표는 confirmatory protocol에서 동결한 test-seen 72행과 PDD-heldout 106행을 사용하고
두 pool을 분리해 보고한다. 178행 pooled 값은 보조 요약으로만 둔다. Parse 또는 claim
extraction 실패는 행을 삭제하지 않고 failure로 처리하며 coverage를 함께 보고한다.
`source-correct`와 `source-wrong`은 subgroup 분석이지 primary eligibility 조건이 아니다.
모든 paired CI와 유의성 검정은 동일 `patient_group`을 함께 resample하는 cluster bootstrap
또는 cluster-aware paired test를 사용한다. 특히 heldout 106행은 103 patient groups이므로
106행을 서로 독립이라고 가정하지 않는다.

## Figure 2. P0 decodability와 layer sensitivity

Validation `val_seen=52`에서 HS16/HS24/HS32별 probe 성능을 target별 heatmap 또는 grouped
point plot으로 제시한다. 축은 category, canonical PDD, finding presence,
finding value 네 target이다(source decision은 본문 제외 결정에 따라 그림에서도
제외). Majority와 shuffled-label control을
함께 표시한다. 이 그림은 layer를 고르는 근거와 target별 decodability를 보여주며, 선택된
layer 하나만 보고하는 Table 1B를 보완한다. Locked-test 성능과 섞지 않는다.

## Figure 3. 사례 특이적 grounding

같은 진단의 두 사례에서 activation-text 짝을 유지하거나 바꾸고, evidence 하나를
삭제했을 때 판독 항목이 어떻게 변하는지 paired plot으로 제시한다. Table 3의 평균값과
겹치지 않게 개별 변화 분포와 대표 반사실을 보여준다.

## Figure 4. Text bottleneck intervention

Table 3 통과 후에만 포함한다. 자연어 판독에서 데이터셋 고유 attribute를 편집하고
AR로 activation을 복원한 뒤 target attribute, target diagnosis logit, off-target drift,
최종 답 변화를 순서대로 표시한다. 실패하면 본문이 아니라 limitation/appendix로 이동한다.

## Appendix Figure S1. DiReCT 사례별 설명 비교

공개가 허용된 합성 예시 또는 라이선스 검토를 마친 예시에서 physician deduction tree,
source CoT, vanilla NLA, SFT-only가 어떤 observation과 관계를 복원하거나 환각하는지
보여준다. Full objective는 실제 구현 후에만 추가한다. 사례 그림은 Table 2의 평균을
대체하지 않으며 제한 데이터 원문을 그대로 노출하지 않는다.
