# 남은 작업 실행 런북

투고 게이트에 남은 항목을 **의존 순서**로 적는다. 날짜를 붙이지 않은 이유는
이것이 스냅샷이 아니라 항목이 끝날 때마다 줄어드는 살아 있는 목록이기
때문이다. 수치는 여기 적지 않는다 — 나온 값은
`RESULTS_CANONICAL_2026-08-24.md`에 스크립트·입력 파일과 함께 먼저 기록한 뒤
표 문서로 옮긴다.

## 0. 먼저 — 디스크에 물어본다

    source scripts/env.sh
    bash scripts/preflight_remaining_work.sh

여덟 항목은 전부 **문서**를 보고 잡은 일정이다. corpus-300이 재채점됐다고
문서는 말하지만, 그것을 말할 수 있는 것은 파일뿐이다. 이 스크립트는 단계마다
READY 또는 없는 경로를 정확히 찍고, corpus-300 provenance 모순에는 별도
판정을 낸다(재채점 파일이 `src/answer_matching.py`보다 새로운가).

## 1–5. GPU도 판정자도 필요 없는 것들

    bash scripts/run_remaining_cpu_work.sh          # 전부
    STEPS="2" bash scripts/run_remaining_cpu_work.sh # 한 단계만

| # | 하는 일 | 왜 지금 |
|---|---|---|
| 1 | corpus-300 provenance + non-overlap 3,319 | Table 2 안에서 main 행과 c300 행이 다른 매처일 수 있는 유일한 내부 모순 |
| 2 | MCR 판독 derangement | **GPU 항목의 게이트.** `.2643`이 사례 특이적이 아니면 MCR 내부 분기 전체가 무의미해진다 |
| 3 | wording 4종 + CoT canonical 재채점 | 이 행들은 아직 생성 시점 매처를 달고 있다 |
| 4 | Figure 5 `64.1%` 재집계 | 분석기는 이미 canonical, 낡은 것은 그것이 group by 하는 사다리 파일이다 |
| 5 | ~~reader-trust dedupe·채점 + shuffled 케이스 생성~~ | **완료** — 전수 −.0935, shuffled case-alignment 통제 완료 |

**1번의 숨은 절반**: 아카이브에 이미 non-overlap 3,319 실행이 있다(`docs/
archive/paper_tables_worklog_2026-08-23.md`). 그 실행이 "둘 다 오답" 칸을
11:4, p=0.118로 재현 실패 판정해 탐색적으로 강등했고, README의 "형식 우위
주장 금지"가 거기서 나왔다. 그런데 그 칸은 정의상 `is_correct` 결과이므로
매처 수정이 행을 칸 사이로 옮길 수 있다. **강등을 상속하지 말고 다시
얻어야 한다.**

## 6–8. 판정자가 필요한 것들 (GPU 불필요)

    DRY=1 bash scripts/run_readout_semantic_judge.sh   # 견적
    bash scripts/run_readout_semantic_judge.sh         # 본 실행

**판정자 #3 — Table 1의 빈 채점 주체 칸.** 그 칸이 비어 있던 이유가
확인됐다: `.340/.731/.557`은 쌍 단위 손라벨을 행 가중해 A+B를 센 값이다
(`results_snapshot/*_heldout_pairs_hand_labeled.jsonl`에서 소수점 넷째 자리까지
재현된다). 손채점이라 칸을 비워둔 것이고, 외부 판정자가 그 자리를 채운다.

계획보다 훨씬 작다. `judge_jobs`는 n=1,314(438×3)로 잡았지만 DDXPlus가 고정
문진표에서 소견을 렌더링하므로 **고유 쌍은 238개**(92/72/74)로 5.5배 줄고,
dry-run 견적은 3층 합계 **약 $0.09**다. 결과는 손라벨을 덮어쓰지 않고
나란히 출력한다 — 두 채점자가 같은 238쌍을 봤다는 사실이 어느 한쪽 숫자보다
강한 근거다.

나머지 둘:

    bash scripts/run_reader_trust_judge.sh             # shuffled arm
    python scripts/make_cot_monitor_requests.py ... --no-cot

`--no-cot`은 같은 케이스·같은 답·같은 루브릭에서 추론 블록만 지운다. 모니터
`.7233`에서 이 값을 빼야 체인 자체의 몫이 나온다. 지금 본문이 "CoT만의 순수
증분으로 부르지 않는다"고 적어둔 것이 이 갈래가 없기 때문이다.

## 9. GPU (짧음) — source output-head likelihood

Table 2b의 생성문과 hidden-state probe 사이에 실제 final-logit 기준선이 빠져 있다.
이 값 없이는 probe가 final output distribution에 이미 있는 uncertainty를 다시 읽은
것인지, output head보다 이른 representation에서 추가 정보를 얻은 것인지 구분할
수 없다. [실험 17](17-output-head-likelihood.md)의 canonical-eligible wrong-arm
1,729행만 실행한다.

결과는 `RESULTS_CANONICAL`에 먼저 기록한 뒤 Table 2b의 `▢` 행을 채운다. 과거
all-cue source-error logprob AUROC는 label과 모집단이 다르므로 가져오지 않는다.

## 10. GPU (며칠) — 2번 통과 후에만

MCR wrong-note activation 추출 → Table 3b MCR 칸 → MCR r5.

**단, 사다리 r3/r4는 지금 실행 가능하다** (`run_mcr_ladder.sh`). wrong-note
activation이 필요한 것은 r5뿐이고, r6은 현재 고정-class 설계에서 직접 이전이
안 된다. Table 4d의 절반은 2번 결과와 무관하게 오늘 채울 수 있다.

이 항목이 끝내 들어오지 않아도 논문은 선다 — `docs/paper/README.md`의 주장
경계("82.9% 기전 해부는 DDXPlus만")와 게이트 4번의 대안 조항("본문 주장을
행동 복제까지만 제한한다")이 이미 그 실패 모드를 덮는다. 나머지 항목에는
그런 대안이 없다.

## 11. 스크립트가 못 하는 것

Related Work의 서지·게재 상태와 인용 문장을 원문으로 재확인하는 일.
`docs/paper/README.md` 게이트 7번이다.

## 새로 생긴 계기

| 스크립트 | 하는 일 |
|---|---|
| `preflight_remaining_work.sh` | 단계별 입력 존재 확인 + corpus-300 provenance 판정 |
| `run_remaining_cpu_work.sh` | 1–5단계 드라이버, 입력 없으면 건너뛰고 이름을 찍는다 |
| `run_readout_semantic_judge.sh` | 판정자 #3 (238쌍) |
| `analyze_readout_semantic_judgements.py` | 외부 판정 vs 손라벨, 행 가중 A+B와 kappa |
| `src/paired_stats.py` | Table 3의 페어드 CI·군간 차·추세 검정 (torch 불필요, 테스트됨) |
| `analyze_hint_effect.py --exclude-from` | 개입 표의 non-overlap 부분집합 |
| `make_cot_monitor_requests.py --no-cot` | 모니터 ablation arm |
