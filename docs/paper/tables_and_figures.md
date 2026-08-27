# 논문 표와 그림 설계

빈 칸은 아직 실험하지 않은 값이다. 과거 소견서 pilot 수치로 채우지 않는다.

## Table 1. Backbone behavior and P0 representation audit

목적은 backbone의 실제 답과 Medical-NLA가 설명하려는 정보의 P0 decodability를 분리하는
것이다. Open-text NLA를 closed-label probe와 같은 accuracy 표에 넣지 않는다.

### Panel A. Backbone diagnostic behavior on identical case IDs

Seen PDD 72행과 held-out PDD 106행은 같은 열 구조의 두 패널로 보고한다.

| Generation | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|
| Direct, answer-prefilled | TBD | TBD | TBD | TBD |
| Source CoT | TBD | TBD | TBD | TBD |

### Panel B. CoT-P0 decodability audit

| Target | Decoder | Output space | Test seen | Test OOD | Required control |
|---|---|---|---:|---:|---|
| Gold disease category | Linear probe | 25-way | TBD | N/A | label shuffle |
| Gold canonical PDD | Linear probe | 49-way train labels | TBD | N/A | label shuffle |
| Source decision | Linear probe | frozen source-answer ontology | TBD | TBD | answer shuffle |
| Finding presence | Multi-label probe | frozen evidence IDs | TBD | TBD | same-diagnosis hard shuffle |
| Finding value | Conditional probe | frozen native values | TBD | TBD | within-finding value shuffle |

`N/A`는 0점이 아니라 closed probe에 unseen output node가 없어 과제가 정의되지 않았다는 뜻이다.
Finding/value head는 diagnosis별로 따로 만들지 않는다. Gold diagnosis와 source decision도
같은 target으로 합치지 않는다.

Validation layer sensitivity는 주표에 `Layer` 열을 반복하지 않고 Figure 2와 아래 보조표에
HS16/24/32를 모두 보고한다.

| Target | HS16 Top-1 | HS24 Top-1 | HS32 Top-1 | Majority |
|---|---:|---:|---:|---:|
| Disease category | .5000 | **.5962** | .5192 | .0577 |
| Canonical PDD | .3846 | **.4423** | .3846 | .0962 |
| Source decision | TBD | TBD | TBD | TBD |
| Finding presence | TBD | TBD | TBD | TBD |
| Finding value | TBD | TBD | TBD | TBD |

각 target은 validation에서 선택된 index 하나로 locked test를 한 번 평가하고, Table 1B
caption에 `category=HS24`, `PDD=HS24`처럼 mapping을 명시한다. 아직 실행하지 않은 target의
index는 미리 HS24로 간주하지 않는다. 이 값은 test 결과가 아니며, AV/AR 호환 때문에 HS32로 고정한 Medical-NLA primary index를
바꾸는 근거로 사용하지 않는다. Table 1B의 probe는 설정을 동결한 뒤 locked test에서 한 번
계산한다.

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
| Source CoT | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

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

### Panel A. Claim grounding and pair specificity

| Method | Finding F1 | Value accuracy | Source-decision fidelity | Hard shuffle | Pair gap |
|---|---:|---:|---:|---:|---:|
| CoT | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD |

### Panel B. Counterfactual response and reconstruction

| Method | Edited-finding response | Untouched retention | Matched FVE | Shuffled FVE | FVE gap |
|---|---:|---:|---:|---:|---:|
| Vanilla NLA | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | TBD | TBD | N/A | N/A | N/A |
| Medical-NLA, reconstruction | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | TBD | TBD | TBD | TBD | TBD |

Hard shuffle은 같은 진단·비슷한 finding 수의 다른 사례 activation과 text 짝을 바꾼다.
Finding deletion/value edit은 하나의 native evidence만 바꾸고 해당 claim과 나머지 finding의
변화를 함께 본다. Round-trip FVE는 판독 text를 AR로 되돌린 activation이 원 activation
분산을 얼마나 설명하는지 본다.
공개 AR가 extraction index 32용이므로 주 round-trip과 patching은 HS32에서만 보고한다.
HS16/HS24에 같은 AR를 적용한 값은 decoder distribution shift가 섞여 주표에 넣지 않는다.

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
point plot으로 제시한다. 현재 category와 canonical PDD를 채우고, source decision, finding
presence, finding value는 실행 후 같은 축에 추가한다. Majority와 shuffled-label control을
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
