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

현재 상태: **사전 등록안 사람 확인 대기** (epoch 수 1 vs 2 선택 포함).
