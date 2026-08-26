# 논문 표와 그림 설계

빈 칸은 아직 실험하지 않은 값이다. 과거 소견서 pilot 수치로 채우지 않는다.

## Table 1. Backbone behavior and internal readout capability

목적은 `probe보다 NLA가 더 정확하다`가 아니라 서로 답하는 질문이 다름을 보이는 것이다.
서로 다른 분모와 출력 공간을 한 숫자로 합치지 않기 위해 두 panel로 나눈다.

### Panel A. Backbone diagnostic behavior on identical case IDs

| Method | Pool | n | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled | Seen PDD | 72 |  |  |  |  |
| Direct, answer-prefilled | Held-out PDD | 106 |  |  |  |  |
| Source CoT | Seen PDD | 72 |  |  |  |  |
| Source CoT | Held-out PDD | 106 |  |  |  |  |

### Panel B. CoT-P0 internal readout on identical activations

| Method | Coverage | Seen-PDD gold | Held-out-PDD gold | Category gold | Source-decision fidelity | Open evidence |
|---|---:|---:|---:|---:|---:|---:|
| CoT-P0 early forced-answer candidate likelihood |  |  |  |  |  | N/A |
| Linear PDD probe |  |  | N/A |  |  | N/A |
| Vanilla NLA, default prompt |  |  |  |  |  |  |
| Vanilla NLA, task-aligned prompt |  |  |  |  |  |  |
| Medical-NLA |  |  |  |  |  |  |

`N/A`는 0점이 아니라 해당 출력 공간이 정의되지 않았다는 뜻이다. 특히 supervised PDD
probe는 train에 없던 PDD를 출력할 수 없으므로 PDD-heldout을 zero-shot accuracy처럼
보고하지 않는다. Category probe는 held-out PDD의 category가 train에 있을 때 별도로
평가한다. Early forced-answer baseline은 CoT-P0 prompt 뒤에 `The answer is`를 붙이고 각
사전등록 PDD 문자열을 teacher-force해 길이 정규화 sequence log-likelihood로 순위를
매긴다. 이는 raw next-token logit이나 저장된 P0 벡터의 직접 unembedding이 아니라,
reasoning을 생략하고 답을 강제한 backbone 행동 기준선이다. 별도 head를 학습하지 않지만
후보 ontology를 제공받으므로 열린 생성 기준선도 아니다. Panel A와 B는 같은 case IDs를
쓰더라도 질문이 다르므로 한 평균으로 합치지 않는다.
학습 head와 평가 ontology 여부는 별도 열로 두지 않고 캡션에 적는다. Probe는 supervised
closed-label classifier, early forced-answer likelihood는 supplied-ontology ranking,
NLA는 open-text generation이다.

Validation에서 probe layer를 고를 때는 별도 보조표에 HS16/24/32의 top-1, top-5, MRR,
macro recall, NLL을 모두 보고한다. 현재 52행 validation에서는 HS24가 PDD top-1 .4423,
category top-1 .5962로 가장 높다. 이 값은 test 결과가 아니며, AV/AR 호환 때문에 HS32로
고정한 Medical-NLA primary index를 바꾸는 근거로 사용하지 않는다. 주 Table 1의 probe
행은 설정을 동결한 뒤 locked test에서 한 번 계산하며, PDD-heldout에는 train label space
밖 PDD가 있으므로 category probe만 해석 가능한 경우를 분리한다.

Validation의 matched raw early forced-answer 결과는 category 25-way
`.4808/.6731/.5814`, PDD 49-way `.1538/.5192/.3250`(top-1/top-5/MRR)이었다. PDD는
corpus 빈도 1인 한 후보가 35/52 top-1이어서 후보 문자열 prior가 강했다. 사전 고정한
content-free prompt를 차감하면 category top-1이 `.2308`, PDD top-1이 `.0577`로 더
악화되고 다른 소수 후보로 다시 붕괴했다. 따라서 Table 1에는 raw matched 값을 무학습
행동 기준선으로 보고하되 이 prior 제한을 캡션에 적고, calibrated 값은 appendix
sensitivity로 둔다. 이 결과는 likelihood가 `나쁘다`는 일반 명제가 아니라 이 고정
completion과 label surface form으로 만든 ranking이 안정적인 내부 판독이 아니라는
진단이다.

동일 validation에서 HS32용 vanilla AV decoder는 default와 task-aligned prompt 모두
HS16/24/32 P0 입력에서 source answer, gold PDD, category literal mention 및 own-donor
source gap이 0이었다. 이 결과는 HS24 probe가 가장 높았던 결과와 함께 제시해, 내부 정보
부재와 자연어 decoder 실패를 구분한다. HS16/24 AV 값은 decoder가 HS32에서 학습됐기
때문에 appendix sensitivity로만 두며 Table 1의 primary vanilla NLA 행은 HS32를 사용한다.

Blinded semantic audit도 312/312행을 판정했고 exact readout quote를 요구했다. Primary
default/HS32/P0의 source answer, gold PDD, category match는 모두 0/52였으며 task-aligned
HS32도 동일했다. HS16에서 category 1/52만 두 prompt에 관찰됐다. 따라서 약칭·동의어를
허용해도 P0 진단 target 복원이 개선되지 않았다는 validation 진단은 유지된다. 이는 열린
observation/rationale 점수나 activation grounding 점수가 아니므로 Table 2·3을 대신하지
않는다. Table 1의 semantic diagnostic 열은 exact readout quote를 요구한 local
Llama-3-8B 판정으로 확정하며, 표 머리말에 `LLM-as-a-judge`임을 명시한다. 사람 검증 점수로
부르지 않고 single-judge 한계는 limitations에 기록한다.

## Table 2. Clinical explanation alignment on DiReCT

| Method | Pool | n | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | Seen PDD | 72 |  |  |  |  |  |  |  |
| Source CoT | Held-out PDD | 106 |  |  |  |  |  |  |  |
| Vanilla NLA | Seen PDD | 72 |  |  |  |  |  |  |  |
| Vanilla NLA | Held-out PDD | 106 |  |  |  |  |  |  |  |
| Medical-NLA, SFT only | Seen PDD | 72 |  |  |  |  |  |  |  |
| Medical-NLA, SFT only | Held-out PDD | 106 |  |  |  |  |  |  |  |
| Medical-NLA, full objective | Seen PDD | 72 |  |  |  |  |  |  |  |
| Medical-NLA, full objective | Held-out PDD | 106 |  |  |  |  |  |  |  |

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

| Method | Own pair | Hard shuffle | Pair gap | Cue deletion | Untouched retention | Round-trip FVE |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla NLA |  |  |  |  |  |  |
| Medical-NLA, SFT only |  |  |  |  |  |  |
| Medical-NLA, full objective |  |  |  |  |  |  |

Hard shuffle은 같은 진단·비슷한 길이의 다른 사례 activation과 text 짝을 바꾼다.
진단명이나 문체만 맞혀서 얻는 점수를 제거하기 위해서다. Cue deletion은 prompt에서
한 evidence를 제거한 뒤 그 속성만 판독에서 감소하는지 본다. Untouched retention은
나머지 evidence가 유지되는지 측정한다. Round-trip FVE는 판독 text를 AR로 되돌린
activation이 원 activation 분산을 얼마나 설명하는지 본다.
공개 AR가 extraction index 32용이므로 주 round-trip과 patching은 HS32에서만 보고한다.
HS16/HS24에 같은 AR를 적용한 값은 decoder distribution shift가 섞여 주표에 넣지 않는다.

## Table 4. Text patching and behavioral utility

이 표는 Table 3의 grounding 관문을 통과한 방법만 평가한다.

| Method | No-op preservation | Edited attribute | Target logit delta | Off-target KL | Diagnostic change |
|---|---:|---:|---:|---:|---:|
| Original activation |  | N/A | 0 | 0 | 0 |
| Decode-encode identity |  | N/A |  |  |  |
| CoT text edit baseline |  |  |  |  |  |
| Medical-NLA text edit |  |  |  |  |  |
| Oracle activation patch |  |  |  |  |  |

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

## Figure 2. DiReCT 사례별 설명 비교

공개가 허용된 합성 예시 또는 라이선스 검토를 마친 예시에서 physician deduction tree,
source CoT, vanilla NLA, SFT-only가 어떤 observation과 관계를 복원하거나 환각하는지
보여준다. Full objective는 실제 구현 후에만 같은 panel에 추가한다. 정량 결과는 Table 2에
두고 Figure 2는 오류 유형을 설명한다.

## Figure 3. 사례 특이적 grounding

같은 진단의 두 사례에서 activation-text 짝을 유지하거나 바꾸고, evidence 하나를
삭제했을 때 판독 항목이 어떻게 변하는지 paired plot으로 제시한다. Table 3의 평균값과
겹치지 않게 개별 변화 분포와 대표 반사실을 보여준다.

## Figure 4. Text bottleneck intervention

Table 3 통과 후에만 포함한다. 자연어 판독에서 데이터셋 고유 attribute를 편집하고
AR로 activation을 복원한 뒤 target attribute, target diagnosis logit, off-target drift,
최종 답 변화를 순서대로 표시한다. 실패하면 본문이 아니라 limitation/appendix로 이동한다.
