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
