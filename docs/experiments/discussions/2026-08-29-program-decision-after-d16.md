# D16 이후 프로그램 결정

## 질문

사전 등록된 gate 아래에서 생성형 objective 여섯 개가 연속 실패했다. 이제
결정할 것은 다음 objective가 아니라 **프로그램의 출구**다: 전략 문서가
사전 승인해 둔 정직한 결론 조항을 발동할 것인가, 아니면 남은 방법 계열에
한 번 더 베팅할 것인가.

## 현재까지의 근거

실패한 생성형 objective (전부 사전 등록 gate, 사후 sweep 없음):

| # | objective | 판정 | 기록 |
|---:|---|---|---|
| 1 | Original-only common SFT | DiReCT Obscomp `.0301`, alignment gap `+.0051` | 03 문서 |
| 2 | Counterfactual sequence SFT | seed 미재현, phantom 2배 | D6 |
| 3 | Sentence-level contrastive | gap `+.0013~.0030`, baseline 미달 | D6 |
| 4 | 1x2 paired ranking (D10) | changed-gap `<= .0030` vs floor `.05` | D12 |
| 5 | Hard-set OOF distillation (D14) | K=5 calibration gate FAIL | D15 |
| 6 | Soft auxiliary bottleneck (D16) | 3-seed gate FAIL, frozen-z 전 지표 하락 | D16 |

성공한 것:

| 결과 | 수치 | 기록 |
|---|---|---|
| Structured monitor (probe + deterministic verbalizer), locked test | finding F1 `.9587`, own-shuffled `+.1624`, retained `.9987` | D13 |
| 두 병목 분해 | decoder 병목(정적 노출) + 표현 병목(deletion phantom `.3593`) | D13 |
| Cross-fit support 방법론 | false-support `.0378 [.0315,.0453]`, coverage `.9993` | D9/D11 |

전략 문서의 사전 승인 조항 (2026-08-29, 결과들 이전에 작성됨):

> "만약 최종 Medical-NLA가 grounding gate를 계속 통과하지 못하면, probe +
> deterministic verbalizer가 더 정직한 실용 baseline이라는 결론도 허용해야
> 한다."

## 선택지

### (A) 정직한 결론 조항 발동 — Claude 권고안

DDXPlus grounding arm은 여기서 결론 낸다:

1. **양성 결과**: 검증된 structured monitor — locked test 재현, own-shuffled
   gap으로 사례 특이성 입증, cross-fit support 방법론.
2. **음성 결과**: free-generating decoder는 선형으로 decode 가능한(F1 `.96`)
   정보를 여섯 가지 objective 어느 것으로도 자연어 생성에 쓰지 못했다 —
   전부 사전 등록 gate 아래에서.
3. **표현 한계**: deletion 후 ~36% 잔존 인코딩은 방법이 아니라 표현의
   ceiling이다.

남은 실행은 승인 완료된 독립 작업(decision-relevance 측정, DiReCT 쪽 분석)과
논문 작성으로 전환한다. 사전 등록-연속 기각-양성 대조군 구조는 그 자체로
방법론적 기여다.

### (B) Offline preference (H)에 마지막 베팅

기제가 다르다는 근거는 있다: gradient가 행동을 발견할 필요 없이 샘플 중
선택만 하면 되고, 기존 SFT decoder의 recall `.35~.56`은 grounded 출력이
가끔은 나온다는 뜻이다. 그러나 전제 조건이 있다 — **Discussion 11의 교훈
3(smoke budget에서 primary metric이 움직일 수 있음을 사전 증명)을 먼저
해결해야 한다.** 이것 없이는 일곱 번째 자동 탈락이 된다.

### (C) Full set decoder (J)

새 architecture 전면 구현. 비용이 가장 크고, D16의 frozen-z 결과(표현
수준에서도 개선 실패)가 이 방향의 기대 수익을 낮췄다.

## 권고

**(A)를 primary로, (B)는 sensitivity 문제를 풀 수 있을 때만 optional.**
근거: 여섯 실패는 서로 다른 지점(target, loss, architecture)을 겨냥했고
모두 같은 결론에 도달했다. 전략 문서가 이 상황을 예견해 출구를 미리
승인해 뒀으며, 그 출구는 빈손이 아니다 — 검증된 monitor, 사전 등록된
음성 결과, 표현 ceiling 계측은 *Artificial Intelligence in Medicine*에
제출 가능한 이야기 구조다.

## 판정

현재 상태: **discussion / 사람 결정 대기**. 이 결정은 에이전트 합의로
닫을 수 없다 — 연구 방향의 출구 선언이므로 희재의 결정이 필요하다.

## Discussion 2 — 사람 방향 지시: budget-calibration full run (2026-08-29)

희재가 "20 step이 아니라 길게 한 번 돌려보자"를 지시했다. 이는 Discussion
11(soft-aux 문서)의 교훈 3 — 두 smoke가 방법과 함께 budget을 기각했을
가능성 — 에 대한 직접 검증이며, 사람 승인에 의한 D12 예외다(규칙 4:
동결 규칙 변경은 사람 승인 필요). 단 "그냥 오래 돌리기"가 아니라 **변수
하나(step 수)만 바꾸는 사전 등록 실험**으로 고정한다.

### 사전 등록 (승인 대상)

- **대상 objective: D10 1x2 ranking, 코드·데이터·hyperparameter 완전 동일.**
  선택 근거: 여섯 실패 중 유일하게 방향 신호가 있었다(seeds 29/43
  changed-gap CI 0 배제, +.0028/+.0030). D16 soft-aux는 frozen-z에서
  방향 자체가 음(-)이므로 scaling 후보가 아니다.
- **바꾸는 것: `--max-steps`만.** 3,104쌍, grad_accum 4 기준 776 step =
  1 epoch. **2 epochs = 1,552 steps**를 상한으로 사전 등록.
- **Dose-response 측정**: checkpoint {20, 194, 388, 776, 1164, 1552}에서
  teacher-forced changed-gap/retained-gap/specificity만 평가(생성 없음,
  저비용). 이 궤적이 budget 질문의 답이다 — margin이 step에 따라 자라면
  budget이 병목이었던 것, `.003` 수준에서 평탄하면 objective가 약한 것.
- **판정은 최종 step에서만**: D5 그대로 (seed 3개 부호 일치 + cluster CI
  0 배제 + δ_min `.05` + hit 유지 + phantom 비증가 + specificity). 중간
  checkpoint는 report-only이며, 궤적이 어떻든 1,552 step 이후 연장은 없다.
- **두 arm 동일 budget**: original-only control continuation도 1,552 steps,
  같은 seed(17/29/43)·pair order. 총 6 runs.
- **비용**: smoke(20 steps) 대비 run당 ~78배. 카드 0/1에서 2 runs 병렬 시
  대략 1.5~2일 wall clock (smoke 실측 시간 x78로 갱신할 것).
- **결과 해석의 사전 약속**: (a) 통과 → Phase 1 계속. (b) 실패 + 궤적
  평탄 → budget 면책 소멸, 남은 기제는 offline preference(H) 하나.
  (c) 실패 + 궤적 상승 중 → "budget이 병목"이 확정되나 연장은 별도 사람
  결정(자동 연장 없음).

### 남은 방법 지도 (희재 질문 "이제 진짜 다른 방법이 없나"에 대한 답)

| 후보 | 기제가 다른 이유 | 상태 |
|---|---|---|
| ① Budget-calibration full run (이 문서) | 유일하게 안 바꿔본 변수가 step 수 | 지금 승인 대상 |
| ② Offline preference (H) | gradient가 행동을 "발견"할 필요 없음 — decoder가 가끔 내는 grounded 출력(recall .35~.56)을 선택·증폭 | ① 결과 후 후보 |
| ③ Full set decoder (J) | 생성 순서 noise 제거, 선택-발화 구조 분리 | 비용 최대, frozen-z 결과로 기대 수익 하락 |

①②③이 모두 소진되면 그때의 "생성형은 안 된다"는 완결된 음성 결과이고,
정직한 결론 조항(선택지 A)이 남는다. 역순 진행은 비합리적이다 — ①이
가장 싸고, ②는 ①의 budget 답을 알아야 smoke 설계가 가능하다.

### 사람 승인 및 실행 계약

희재가 2026-08-29에 **2 epochs / 1,552 steps** 실행을 승인했다. 이 실험은
초기 D12의 confirmatory 결과가 아니라, 그 결과를 본 뒤 승인한
**post-hoc exploratory budget calibration**으로 보고한다.

구현 계약:

- control/ranking 모두 seed 17/29/43, 같은 초기화와 pair order를 사용한다.
- 중간 checkpoint `{20,194,388,776,1164,1552}`는 report-only다.
- 중간 결과와 무관하게 1,552 step까지 실행하며 자동 조기 종료하지 않는다.
- 판정은 step 1,552에서 기존 D5 gate로 한 번만 수행한다.
- 1,552 이후 epoch, lambda, temperature를 자동 탐색하지 않는다.
- 장시간 실행의 checkpoint에는 optimizer, RNG, epoch/row cursor를 기록한다.
- DDXPlus validation만 읽고 locked test는 읽지 않는다.

실행 구현:

- trainer: `scripts/train_ddxplus_d10_1x2.py`
- 125번 4-GPU queue: `scripts/run_ddxplus_d10_budget_4gpu_125.sh`
- trajectory 집계: `scripts/summarize_ddxplus_d10_budget_trajectory.py`

현재 상태: **사람 승인 완료 / 구현 및 실행 준비**.

### 실행 환경 수정 (2026-08-29, 사람 지시)

Budget-calibration run은 lab 4090 대신 **RunPod A100/H100**에서 실행한다.
근거: 12B bf16은 4090(24GB) 한 장에 안 올라가 2-card sharding + PCIe 통신
+ gradient checkpointing이 강제되는 반면, A100 80GB 한 장이면 전체가 단일
device에 올라간다(run당 3~5배 기대).

- **데이터 반출 범위: DDXPlus 파생물만.** D10 파이프라인은 DiReCT 의존이
  없음을 코드로 확인했다(`train_ddxplus_d10_1x2.py`,
  `make_ddxplus_d10_validation_pairs.py`, `evaluate_ddxplus_d10_specificity.py`
  전부 DDXPlus pair만 사용). DDXPlus는 공개 데이터셋이므로 파생 activation의
  클라우드 반출에 제약이 없다. **DiReCT 원문·파생물은 어떤 형태로도
  RunPod에 올리지 않는다** — repo에는 원래 없고(git 제외 규칙), 전송
  tarball에도 포함 금지.
- 반출 목록: train 3,104쌍 + validation 3,032쌍의 activation tensor
  (~15,000여 개, 총 1GB 미만), pair/protocol JSONL, init adapter. Base
  model은 HF 공개 checkpoint를 pod에서 직접 받는다.
- Pair JSONL의 activation 경로가 절대경로이므로, pod에 **동일 경로 구조**
  (`/data1/heejae/medical_nla/data/...`)를 재현해 코드 수정 없이 돌린다.
- 6 runs 전부 같은 pod 하드웨어에서 실행하고 하드웨어를 리포트에 기록한다.
  실행 전 20-step smoke 1회로 wall-clock을 실측해 비용 항목을 갱신한다.
- DiReCT가 필요한 후속 단계(Gate C, Phase 2 adaptation — 248행 규모의 경량
  작업)는 통과 시 lab 서버로 돌아와 수행한다.

RunPod 실행 시 `run_ddxplus_d10_budget_4gpu_125.sh`는 125 전용 가드와 4-GPU
queue를 가정하므로, pod의 단일 A100/H100용 변형 wrapper가 필요하다(동일
trainer·인자, GPU 배치만 다름). 전체 6 runs는 같은 pod 하드웨어에서
실행한다는 조건은 유지된다.

## Discussion 3 — D10 budget calibration 최종 판정 (Claude, 2026-08-30)

**[판정] Frozen gate FAIL 확정, 연장 없음.** Step 1,552에서 changed-gap
delta는 seed별 `-.0177 / +.5618 / +1.1233`으로 부호 불일치(seed 17은 CI가
음수쪽으로 0 배제), specificity delta는 `-.0442 / +.0345 / -.0040`으로 전부
gate 미달이다. Runner의 사전 등록 gate 출력 그대로 FAIL이며 자동 연장은
승인되지 않았다.

**[해석] 이 실패는 시리즈 전체에서 가장 정보량이 많다 — budget 질문이
확정적으로 닫혔다.**

1. **Raw margin은 budget에 반응했다.** Across-seed mean changed-gap이
   `.0019 → .5558`로 상승했으므로 "20-step budget이 margin을 못 키웠다"는
   면책은 이제 성립한다... 그러나:
2. **자란 것은 퇴화 해다.** Retained-gap mean이 `.0002 → .5604`로 changed-
   gap과 **정확히 같이** 자랐고 specificity mean은 `-.0046`으로 평탄하다.
   모델은 changed cue를 선택적으로 잊은 것이 아니라 **deleted-activation
   detector**를 학습했다 — R10이 예견하고 specificity gate를 추가했던 바로
   그 퇴화 해다. Gate가 없었다면 seeds 29/43은 floor의 10~20배로 통과했을
   것이다.
3. **구조적 결론**: `g = NLL(y|h_del) - NLL(y|h_orig)`는 h_del을 전역으로
   억제하는 어떤 기제로도 최대화되며, budget을 늘리면 그 가장 쉬운 기제
   (detector)가 자란다. 따라서 이것은 사전 약속 분기 (c)("궤적 상승 =
   budget 병목")가 아니라 실질적으로 **(b)의 강화판**이다: 상승한 것은
   confound이고, 연장은 confound만 더 키운다. **연장을 권고하지 않는다.**
4. **Seed 불안정은 full budget에서도 지속** (-.02 vs +.56 vs +1.12) —
   프로그램 전체의 진단(objective가 행동을 규정하지 못함)과 정합.
5. **측정 감도 질문도 닫혔다**: 이 지표는 이 budget에서 크게 움직일 수
   있음이 증명됐다. 움직이지 않은 것은 specificity — 즉 문제는 계측이
   아니라 objective다.

**[제안] 이 trajectory 자체를 appendix figure로.** Changed-gap과
retained-gap이 함께 오르고 specificity가 평탄한 dose-response 곡선은
"퇴화 해의 성장"을 시각적으로 증명하는, specificity 통제의 필요성에 대한
교과서적 그림이다.

**[결정 원장 제안 — 사람 승인 대기]**

- D19: D10 budget calibration (1,552 steps, RunPod A100-SXM4-80GB) frozen
  gate FAIL. Changed-gap은 budget에 반응했으나(mean `.0019→.5558`)
  retained-gap이 동반 상승(mean `.0002→.5604`), specificity 평탄
  (`-.0046`) — deletion-detector 퇴화 해 확정. 연장 금지. Budget 면책
  소멸: 생성형 1x2 ranking 계열 종료.

**[열린 사람 결정] 프로그램 출구.** 사전 약속에 따라 남은 선택지는 둘이다.

- **(A) 정직한 결론 조항 발동 (Claude 권고, 이전보다 강화됨)**: 생성형
  objective 7종이 사전 등록 gate에서 전부 실패했고, 마지막 실패는 budget
  면책까지 소거했다. Structured monitor 양성 결과 + 음성 결과 7건 + 표현
  ceiling + 이 dose-response가 논문의 뼈대다.
- **(B) Offline preference (H) 마지막 베팅**: 기제는 여전히 다르다
  (gradient가 아니라 샘플 선택). 단, reward에 specificity-형 통제가 반드시
  들어가야 한다는 것을 이번 결과가 증명했다 — changed-gap류 단독 reward는
  detector를 뽑는다. Vanilla 채점용으로 구축 중인 semantic mapper가 H의
  후보 채점기로 재사용될 수 있다는 점은 (B)의 비용을 낮춘다.

이 결정은 에이전트가 닫을 수 없다. 어느 쪽이든 **D10 decision record와
recipe hash 동결이 지금 가능**해졌으므로 DiReCT locked batch(Table 1A→1B→2)
는 즉시 열린다.
