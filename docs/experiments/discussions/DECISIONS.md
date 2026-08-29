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
