# 논문 표와 그림 설계

빈 칸은 아직 실험하지 않은 값이다. 과거 소견서 pilot 수치로 채우지 않는다.

## Table 1. Backbone behavior and internal readout capability

목적은 `probe보다 NLA가 더 정확하다`가 아니라 서로 답하는 질문이 다름을 보이는 것이다.
서로 다른 분모와 출력 공간을 한 숫자로 합치지 않기 위해 두 panel로 나눈다.

### Panel A. Backbone diagnostic behavior on identical case IDs

| Method | n | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled |  |  |  |  |  |
| Source CoT |  |  |  |  |  |

### Panel B. CoT-P0 internal readout on identical activations

| Method | Coverage | Seen-PDD gold | Held-out-PDD gold | Category gold | Source-decision fidelity | Open evidence | Trained task head | Eval ontology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Output-head candidate score |  |  |  |  |  | N/A | No | Yes |
| Linear PDD probe |  |  | N/A |  |  | N/A | Yes | Yes |
| Vanilla NLA, default prompt |  |  |  |  |  |  | No | No |
| Vanilla NLA, task-aligned prompt |  |  |  |  |  |  | No | No |
| Medical-NLA |  |  |  |  |  |  | No | train text only |

`N/A`는 0점이 아니라 해당 출력 공간이 정의되지 않았다는 뜻이다. 특히 supervised PDD
probe는 train에 없던 PDD를 출력할 수 없으므로 PDD-heldout을 zero-shot accuracy처럼
보고하지 않는다. Category probe는 held-out PDD의 category가 train에 있을 때 별도로
평가한다. Output-head candidate score는 P0 뒤에 각 사전등록 PDD 문자열을 teacher-force해
길이 정규화 sequence log-likelihood로 순위를 매긴다. 별도 head를 학습하지 않지만 후보
ontology를 제공받으므로 열린 생성 기준선이 아니다. Panel A와 B는 같은 case IDs를
쓰더라도 질문이 다르므로 한 평균으로 합치지 않는다.

## Table 2. Clinical explanation alignment on DiReCT

| Method | n | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Source CoT |  |  |  |  |  |  |  |  |
| Vanilla NLA |  |  |  |  |  |  |  |  |
| Medical-NLA, SFT only |  |  |  |  |  |  |  |  |
| Medical-NLA, full objective |  |  |  |  |  |  |  |  |

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

주표는 confirmatory protocol에서 동결한 동일 case ID만 사용한다. Parse 또는 claim
extraction 실패는 행을 삭제하지 않고 failure로 처리하며 coverage를 함께 보고한다.
`source-correct`와 `source-wrong`은 subgroup 분석이지 primary eligibility 조건이 아니다.

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
