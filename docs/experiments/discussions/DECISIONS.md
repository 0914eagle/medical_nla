# Frozen decision ledger

이 파일에는 사람이 승인한 결정만 기록한다. 상세한 논거와 토의는 각 주제 문서와
archive를 참조한다. 새 데이터 없이 기존 결정을 재론하지 않는다.

| # | 결정 | 핵심 근거 |
|---:|---|---|
| D1 | Capacity sweep 대신 objective를 수정한다. | original-only, CF sequence, sentence contrastive가 같은 병목을 보임 |
| D2 | Changed-claim paired ranking이 primary이며 GRPO는 사전 조건 전 금지한다. | verbosity 오염 위험 |
| D3 | DDXPlus cue는 candidate pool이며 cross-fitted support 검증 없이 전체 cue SFT를 하지 않는다. | prompt reconstruction shortcut |
| D4 | Gate A primary layer는 HS32로 유지한다. | validation finding `.9607`, value `.6990`; public AV index와 일치 |
| D5 | Smoke 승격은 seeds 17/29/43, cluster CI, effect floor, hit/phantom/specificity를 모두 만족해야 한다. | seed 분산 및 deletion-detector 퇴화 통제 |
| D6 | 전체 sequence CE 확장을 중단한다. | CF contrast seed 미재현, phantom `.2138→.4253` |
| D7 | Gate C는 source CoT 자기설명 `Obscomp .2130 / Expcom .0650`을 넘어야 한다. | 최소 논문 기준 |
| D8 | Value gate는 Phase 1에서 비악화 항목으로만 둔다. | validation n=82로 판정력 부족 |
| D9 | D9a cut `.90/0/0`을 통과한 selected changed-cue 3,104 pairs만 사용한다. | validation coverage `.9993`, false support `.0378` |
| D10 | 첫 ranking smoke는 1 claim x 2 activations, lambda=T=1.0, seeds 17/29/43이다. | deleted-state target 발명 방지 |
| D11 | Support는 presence AND deletion delta AND same-diagnosis donor margin이다. | approved protocol SHA `a968a63f...` |
| D12 | D10 1x2 smoke 실패를 확정하고 budget/lambda/step sweep을 금지한다. | changed gap `.0005/.0028/.0030` vs floor `.05` |
| D13 | Structured reader는 open NLA가 아니라 control/upper baseline이다. | test finding F1 `.9587`, phantom `.3593`, clean switch `.0804` |
| D14 | 다음 learned method는 training-only OOF probe teacher를 쓰는 set-to-text NLA다. Inference는 raw HS32→단일 decoder다. | activation-conditioned target |
| D15 | K=2 teacher는 폐기하고 K=5를 단 한 번 평가한다. Threshold `.5`와 hyperparameter를 유지하며 실패 시 추가 K/threshold sweep을 금지한다. K=5는 gate FAIL로 종료했다. | K=5 precision `.8881`; deleted mean gap `18.10%`; 추가 sweep 금지 |
| D16 | Soft auxiliary bottleneck one-shot을 승인한다. `d_z=256`, train-only source-balanced PCA, validation cosine `.95`, original OOF soft BCE, approved D9a selected-cue paired margin, `248+248` gradient parity, 8+8/20-step seeds 17/29/43, control-first paired-delta gate를 사용한다. | 사람이 2026-08-29 승인; 실패 시 `d_z`/lambda/step/threshold sweep 금지 |

## D15 calibration gate

| criterion | gate |
|---|---:|
| original cue precision | `>= .90` |
| original cue recall | `>= .98` |
| full-data 대비 original mean claims 차이 | `<= 10%` |
| OOF/full original set Jaccard mean | `>= .90` |
| full-data 대비 deleted mean claims 차이 | `<= 10%` |
| full-data 대비 deleted phantom 절대차 | `<= .05` |
| fold별 original precision | 모두 `>= .85` |

일곱 조건은 AND다. K=5 실패 시 hard-set target을 만들지 않는다.

## D16 soft auxiliary bottleneck

- Architecture: `h32 -> PCA-initialized P_down -> z[256] -> P_up -> AV injection`.
  우회 경로는 없고 projector는 inference에 남으며 91-way auxiliary head만 제거한다.
- PCA fit: DDXPlus official train original `4,655`와 DiReCT train `248`, source
  weight `.5/.5`. DDXPlus/DiReCT validation source별 reconstruction cosine mean이
  모두 `.95` 이상이어야 한다.
- Loss: DiReCT language SFT + D9a original K=5 OOF soft BCE + approved `3,104`
  pair selected-cue original/deleted softplus margin. Deleted absolute target은 금지한다.
- Lambda: seed-17 initialization에서 Direct `248` + SHA-ordered D9a `248` pairs의
  `dL/dz` row-L2 RMS parity로 한 번 계산해 모든 seed에 공유한다.
- Smoke: seed별 동일 order로 optimizer step당 Direct 8 + D9a 8 pairs, 20 steps.
  Control은 동일 architecture/order에서 auxiliary coefficient만 0이다.
- Control 3 seeds 이후 floor JSON을 먼저 고정한다:
  `max(2 * control-gap range, .005)`. Proposed는 그 뒤에만 실행한다.
- 통과: 각 seed `proposed-control >= floor`, paired category-cluster CI > 0,
  세 seed 부호 일치. Gate C `Obscomp > .2130`은 별도 절대 출구다.
- 어느 hard gate든 실패하면 이 branch를 종료하고 hyperparameter sweep을 하지 않는다.

### D16 결과

- PCA validation gate: PASS. DDXPlus/DiReCT mean reconstruction cosine은
  `.999997/.999983`이었다.
- Frozen effect floor: `.005` (control gap range `.001524`).
- Proposed-control Direct alignment delta, seeds 17/29/43:
  `-.001137/-.001476/+.001433`.
- 세 category-cluster CI가 모두 0을 포함했고 어느 seed도 floor를 넘지 못했다.
- Primary three-seed gate: **FAIL**. D16은 종료하며 승인된 hyperparameter를 sweep하지
  않는다. Locked test는 읽지 않았다.
- Frozen-z에서도 auxiliary-control finding F1은 `-.0009/-.0007/-.0016`,
  own-shuffled gap은 `-.0050/-.0046/-.0058`, value accuracy는
  `-.0137/-.0096/-.0160`, deletion drop은 `-.0167/-.0141/-.0151`이었다.
- 실패가 decoder 사용에만 국한되지 않고 `z`의 정보/반응성에도 있으므로 full
  generation과 Gate C는 실행하지 않는다. 삭제 후 새 label 감소는 필요한 반응성도
  함께 낮아져 promotion 근거로 사용하지 않는다.
