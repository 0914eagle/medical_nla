# 남은 작업 실행 런북

> **현재 상태 (2026-08-25, `a21875e`)**: DDXPlus canonical 행동·궤적·탐지·
> 교정 재집계와 reader-trust는 완료됐다. MCR wrong-note activation/readout도
> 생성됐지만, 3,086행이 none/wrong arm을 각각 1,543행씩 포함한다는 사실을
> 무시한 첫 채점은 무효다. 현재 첫 실행은 **arm-aware CPU 재채점**이다.

이 문서는 남은 항목을 의존 순서로 적는 실행 정본이다. 새 수치는 먼저
`RESULTS_CANONICAL_2026-08-24.md`에 모집단·입력 파일·스크립트와 함께 기록한 뒤
표·그림·발표 문서로 옮긴다.

분모는 [모집단 원장](POPULATION_REGISTRY_2026-08-25.md)의 cohort key를 사용한다.
새 분석이 다른 분모를 요구하면 결과를 옮기기 전에 원장에 정의와 사용처부터
추가한다.

## 0. 환경과 입력 확인

```bash
source scripts/env.sh
bash scripts/preflight_remaining_work.sh
python scripts/audit_document_populations.py
```

Preflight가 아직 새 MCR arm-aware 감사와 detector-gated 정책 입력을 모두 검사하지
않으므로, 해당 절의 파일 확인 명령도 함께 사용한다.

## P0-A. 지금 즉시 — MCR arm-aware CPU 감사

정본 절차와 폐기할 수치는 [실험 18](18-mcr-wrong-arm-readout.md)에 있다.

```bash
python scripts/score_readout_against_model.py \
  --readouts "$ART/results/readout_mcr_hint_final_L32.jsonl" \
  --answers "$ART/results/mcr_hint_answers_full_rescored.jsonl" \
  --variant wrong \
  2>&1 | tee "$ART/reports/mcr_wrong_readout_faithfulness.txt"

python scripts/analyze_readout_grounding.py \
  --readouts "$ART/results/readout_mcr_hint_final_L32.jsonl" \
  --by-variant \
  2>&1 | tee "$ART/reports/mcr_readout_grounding_by_variant.txt"
```

첫 3,086행 pooled 점수 `.6361/.0029`는 인용하지 않는다. wrong readout은
`mcr_hint_answers_full_rescored.jsonl`의 **wrong-arm 실제 답**과 비교해야 한다.
이 결과가 MCR r5의 선행 조건이다.

필수 확인:

- wrong 평가행 1,543, `unjoined=0`
- wrong `vs model / vs gold / deranged`
- wrong-arm source-wrong 수와 model-only/gold-only/both/neither 분해
- none/wrong별 cue grounding gap과 반복률

## P0-B. 위 감사와 독립적으로 가능한 GPU 작업

### B1. MCR 사다리 r3/r4

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 RUNGS="3 4" \
  bash scripts/run_mcr_ladder.sh \
  > "$ART/logs/mcr_ladder_r3_r4_launcher.log" 2>&1 &
```

r3/r4는 activation이나 판독 결론을 사용하지 않으므로 P0-A와 병렬 실행한다.

### B2. Source output-head likelihood

[실험 17](17-output-head-likelihood.md)을 canonical-eligible wrong-arm 1,729행에서
실행한다. 이 기준선은 다음 두 질문에 필요하다.

1. Probe가 final output distribution에 이미 있는 신호를 다시 읽는가.
2. Detector-gated correction의 source-confidence selector를 무엇으로 정의할까.

과거 all-cue source-error logprob는 label과 모집단이 달라 대체할 수 없다.

## P0-C. MCR CPU 관문 통과 후

### C1. MCR r5를 conclusion-only와 full로 분리

현재 r5는 `internal conclusion`과 `encoded findings`를 함께 제공한다. Grounding이
약하고 반복 문장이 많으므로 최종 비교는 다음 두 조건이어야 한다.

| 조건 | 제공 내용 | 역할 |
|---|---|---|
| r5-conclusion | internal conclusion만 | 주 비교: 결론 자체의 교정 가치 |
| r5-full | conclusion + encoded findings | 민감도: 현재 판독 전체의 순효과 |

`r5-conclusion` builder flag는 구현 대기다. 이 통제 없이 full r5만 실행하면
grounded explanation의 효과를 주장할 수 없다. Arm-aware 결론이 derangement를
통과하지 못하면 MCR r5를 중단하고 MCR은 행동 복제까지만 주장한다.

### C2. 동일-ID 분석

```bash
python scripts/analyze_correction_ladder.py \
  --rungs \
    "$ART/results/mcr_ladder_r3.jsonl" \
    "$ART/results/mcr_ladder_r4.jsonl" \
    "$ART/results/mcr_ladder_r5.jsonl" \
  --common-ids
```

주 비교는 r5-conclusion 대 r4다. MCR에서는 DDXPlus 49-class r6를 직접 만들지
않는다.

## P0-D. 제출 주장을 결정하는 핵심 실험

### D1. Canonical detector-gated correction

Canonical DDXPlus 1,729건에서 threshold와 policy를 validation에서 고정하고
held-out test에서 평가한다.

| 정책 | 개입 방식 |
|---|---|
| Keep first | 개입 없음 |
| Apply r5 to all | 무선별 재실행 통제 |
| Source-confidence gated | output-head 신호로 r5 선택 |
| LLM-monitor gated | 외부 text monitor로 r5 선택 |
| AV gated | AV 위험 신호로 r5 선택 |
| Probe gated | probe 위험 신호로 r5 선택 |
| Probe argmax replacement | flag 시 probe class로 교체 |
| Oracle moved | true moved에만 개입하는 상한 |

반드시 보고할 값:

- overall accuracy
- moved recovery
- unchanged preservation
- newly broken
- net correction (`wrong→right - right→wrong`)
- intervention rate와 intervention precision
- keep-first 대비 case-paired 95% CI

Validation에서 고정한 정책이 held-out test에서 positive net correction을 내고 CI가
0을 배제하기 전에는 RQ3를 “사후 식별된 moved 사례에서 내부 내용이 유용하다”로
제한한다.

## P1. 해석 교란을 닫는 확인 실험

| 항목 | 선행 조건 | 닫히는 해석 |
|---|---|---|
| 동일 monitor의 no-CoT arm | 빌더 완성 | monitor 성능 중 CoT 자체의 증분 |
| Direct×CoT matched 2×2 | 공통 ID 확인 후 누락 셀 생성 | selection bias 없는 CoT 강건성 |
| realistic matched-neutral | canonical clean 1,204 | 길이·문체 비용과 진단 제안 비용 분리 |
| matched layer/position reader | GPU | Figure A1의 layer와 recipe 교란 분리 |
| MCR cue-position/span swap | P0-A 결과 이후 | 열린 어휘 판독의 위치·사례 특이성 |

Direct×CoT는 두 분석을 분리한다.

1. 정답 여부로 고르지 않은 common cohort: difference-in-differences
2. 두 no-note가 모두 정답인 shared-solvable subset: harmful flip 비교

Realistic matched-neutral 전에는 30.40%p와 짧은 referral 23.75%p의 차이
6.65%p를 현실성의 독립 효과로 해석하지 않는다.

## P2. CPU·판정자·문서 작업

### 남은 작업

- 동일 LLM monitor의 no-CoT 판정
- Table 3 capitulation/newly-broken와 selector-policy paired CI 전사
- Related Work 서지·게재 상태·인용 문장 원문 재확인
- 최종 표/그림 모집단 통일: clean 1,204 / eligible 1,729 / silent 1,628 /
  moved 319 / MCR 1,452
- corpus-300 canonical clean expected 2,137 재집계와 행동 moved 563 / 사다리
  moved 571의 matcher 차이 해소
- RQ3 gated 결과가 나온 뒤 Abstract/Conclusion 성능 향상 문장 동결

### 완료되어 재실행하지 않는 작업

- wording 4종과 CoT의 canonical clean 1,204 재채점
- reader-trust 2,896/2,896 및 shuffled 통제
- 외부 의미 판정 238쌍
- Appendix Figure A2 `.591` 재집계
- corpus-300 non-overlap **ID 확인**과 fixed-cohort appendix audit
- DDXPlus trajectory/detection/correction canonical 1,729 재집계

## 보조 실행기

```bash
bash scripts/run_remaining_cpu_work.sh
DRY=1 bash scripts/run_readout_semantic_judge.sh
python scripts/make_cot_monitor_requests.py ... --no-cot
```

| 스크립트 | 하는 일 |
|---|---|
| `preflight_remaining_work.sh` | 단계별 입력 존재 확인 |
| `run_remaining_cpu_work.sh` | 기존 CPU 감사 드라이버 |
| `src/paired_stats.py` | paired CI·군간 차·추세 검정 |
| `make_cot_monitor_requests.py --no-cot` | monitor ablation arm |
| `score_readout_against_model.py --variant` | arm-aware 결론 충실성 |
| `analyze_readout_grounding.py --by-variant` | arm별 근거 grounding |

## 결과 전파 순서

1. `RESULTS_CANONICAL_2026-08-24.md`에 모집단·분자/분모·입력·스크립트 기록
2. 해당 실험 문서(17 또는 18) 갱신
3. `docs/paper/experiment_summary_2026-08-25.md`
4. `docs/paper/table_camera_ready_2026-08-25.md`와 figure dump
5. `docs/professor/paper_presentation_full_2026-08-25.md`
6. 마지막에 Overleaf 원고

Related Work의 서지·게재 상태와 인용 문장 확인은 스크립트로 대체하지 않는다.
