# 논문 표와 그림 설계

빈 칸은 아직 실험하지 않은 값이다. 과거 소견서 pilot 수치로 채우지 않는다.

## Table 1. Closed-label detection and open-text readout

목적은 `probe보다 NLA가 더 정확하다`가 아니라 서로 답하는 질문이 다름을 보이는 것이다.

| Method | Position | PDD/category accuracy | Held-out PDD | Open evidence text | Task labels |
|---|---|---:|---:|---:|---:|
| Output head | P0 |  |  | No | No |
| Linear probe | P0 |  |  | N/A | Yes |
| Source CoT | output |  |  |  | No |
| Vanilla NLA | P0 |  |  |  | No medical labels |
| Medical-NLA | P0 |  |  |  | train only |

`N/A`는 0점이 아니라 probe 출력 공간에 자유 자연어 설명이 정의되지 않았다는 뜻이다.
Closed-label 열에서는 probe가 upper bound 역할을 한다.

## Table 2. Clinical explanation alignment on DiReCT

| Method | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT |  |  |  |  |  |  |
| Vanilla NLA |  |  |  |  |  |  |
| Medical-NLA, SFT only |  |  |  |  |  |  |
| Medical-NLA, reconstruction only |  |  |  |  |  |  |
| Medical-NLA, full |  |  |  |  |  |  |

- `Accdiag`: 생성한 세부 진단과 의사 주석 진단의 의미 일치
- `Obspre`: 생성 관찰 중 의사 observation과 일치하는 정도
- `Obsrec`: 의사 observation 중 생성 설명이 회수한 정도
- `Obscomp`: 필요한 observation 구성요소의 coverage
- `Expcom`: observation에 연결한 rationale의 일치
- `Expall`: 전체 explanation chain의 일치

공식 `Obspre`와 `Obsrec`에는 `+1` denominator smoothing이 있다. 주표는 공식값을
사용하고 unsmoothed precision/recall은 민감도 분석으로만 둔다. 이 표는 clinical
alignment를 측정하며 activation faithfulness를 단독으로 증명하지 않는다.

## Table 3. Activation grounding on DDXPlus

| Method | Own pair | Hard shuffle | Pair gap | Cue deletion | Untouched retention | Round-trip FVE |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla NLA |  |  |  |  |  |  |
| Medical-NLA, SFT only |  |  |  |  |  |  |
| Medical-NLA, reconstruction only |  |  |  |  |  |  |
| Medical-NLA, full |  |  |  |  |  |  |

Hard shuffle은 같은 진단·비슷한 길이의 다른 사례 activation과 text 짝을 바꾼다.
진단명이나 문체만 맞혀서 얻는 점수를 제거하기 위해서다. Cue deletion은 prompt에서
한 evidence를 제거한 뒤 그 속성만 판독에서 감소하는지 본다. Untouched retention은
나머지 evidence가 유지되는지 측정한다. Round-trip FVE는 판독 text를 AR로 되돌린
activation이 원 activation 분산을 얼마나 설명하는지 본다.

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

## Figure 2. DiReCT 사례별 설명 비교

공개가 허용된 합성 예시 또는 라이선스 검토를 마친 예시에서 physician deduction tree,
source CoT, vanilla NLA, SFT-only, full Medical-NLA가 어떤 observation과 관계를 복원하거나
환각하는지 보여준다. 정량 결과는 Table 2에 두고 Figure 2는 오류 유형을 설명한다.

## Figure 3. 사례 특이적 grounding

같은 진단의 두 사례에서 activation-text 짝을 유지하거나 바꾸고, evidence 하나를
삭제했을 때 판독 항목이 어떻게 변하는지 paired plot으로 제시한다. Table 3의 평균값과
겹치지 않게 개별 변화 분포와 대표 반사실을 보여준다.

## Figure 4. Text bottleneck intervention

Table 3 통과 후에만 포함한다. 자연어 판독에서 데이터셋 고유 attribute를 편집하고
AR로 activation을 복원한 뒤 target attribute, target diagnosis logit, off-target drift,
최종 답 변화를 순서대로 표시한다. 실패하면 본문이 아니라 limitation/appendix로 이동한다.
