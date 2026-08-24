# 논문 문서 안내 — 현재 정본

이 폴더의 논문 서사는 **현상 우선**이다. 주인공은 NLA 자체가 아니라,
의뢰 소견서의 의심 진단이 의료 LLM의 출력은 바꾸지만 내부 정답 표상을
대부분 지우지 않는다는 **internal-output dissociation**이다. 프로브는 이를
정밀하게 측정하고, 자연어 판독은 내부 내용을 열린 어휘의 문장으로 서술한다.

## 처음 읽는 순서

1. `experiment_summary_2026-08-25.md` — 실험별 질문, 실측, 상태를 한 번에 본다.
2. `paper_outline_2026-08-24.md` — 논문 테제와 절별 서사를 본다.
3. `table_camera_ready_2026-08-25.md` — 현재 표의 수치와 캡션을 확인한다.
4. `prior_work_2026-08-24.md` — 신규성의 범위와 가장 가까운 선행을 확인한다.
5. `draft_related_work_2026-08-24.md` / `related_work.tex` — 실제 Related Work 원고.

`related_work_2026-08-23.md`는 문헌 조사 원장이고, `reading_*.md`는 최근접
논문의 정독 노트다. `judge_jobs_2026-08-24.md`는 외부 판정자 실험의 실행
대기열이다.

## 현재 논문의 한 문장

의료 LLM이 의뢰 소견서에 앵커링되어 답을 바꿀 때도, 인과 통제된 DDXPlus
실험에서 내부 표상은 대다수 사례에서 정답을 유지한다. 이 내부-출력 결렬은
한 번의 실행에서 탐지할 수 있고 자연어로 판독할 수 있으며, 정확한 내부
내용을 되먹이면 일부 오류를 회복할 수 있다.

영문으로는 다음 범위가 안전하다.

> In a causally controlled diagnostic setting, referral-note anchoring often
> changes what a medical LLM emits without erasing the correct diagnosis from
> its probed internal state. The resulting internal-output rift can be detected
> from a single run, rendered as a natural-language readout, and used as
> corrective evidence when the readout is accurate.

## 확정된 근거와 범위

| 주장 | 현재 근거 | 주장 가능한 범위 |
|---|---|---|
| 오답 소견서가 진단을 움직인다 | DDXPlus −23.1 pp, MCR −27.8 pp; 제안 고유 효과 −17.4/−22.3 pp | 행동 효과는 두 코퍼스 |
| 내부 정답 표상은 대부분 유지된다 | moved 324건 중 268건(82.7%)에서 제안이 어느 랜드마크에서도 probe top-1이 아님 | DDXPlus, 49-class cross-fit probe |
| 소견서는 내부를 밀지만 대개 뒤집지 못한다 | final-token gold-probability cost .007/.055/.187 (kept/lost/adopted) | DDXPlus |
| CoT보다 내부 채널이 강하다 | silent subset AUROC: LLM CoT monitor .695, NL readout .842, probe .984 | DDXPlus, 동일 모집단 |
| 자연어 판독은 벡터에 종속된다 | swap .993, memorization .000, contamination .007; unseen-cue .750 | DDXPlus 검증 배터리 |
| 내부 내용을 되먹이면 회복한다 | moved accuracy .012 → r5 .627 / r6 .830; r5−r4 +22.8 pp | DDXPlus, 선별 개입 필요 |
| 자연어 형식 자체가 교정의 원인이다 | **지지되지 않음**: 내용 정확도 일치 시 r5 vs r6 p=.720 | 주장 금지 |

## 반드시 지킬 주장 경계

- 행동 효과는 DDXPlus와 MCR에서 재현됐지만, **82.7% 기전 해부는 DDXPlus만**이다.
- 닫힌 49-class 탐지는 자연어 판독보다 지도 프로브가 강하다(.984 vs .842,
  silent subset). 자연어 판독의 몫은 최고 정확도가 아니라 서술, 근거, 열린
  진단 어휘다.
- MCR처럼 진단명이 대부분 한 번씩 등장하는 자료에서는 표준적인 고정
  클래스 지도 프로브를 그대로 정의하기 어렵다. 이것을 "어떤 프로브도
  불가능하다"고 쓰지는 않는다.
- "clinician-readable"은 형식에 대한 주장이다. "clinically useful"은 외부
  판정자나 임상의 평가 전에는 주장하지 않는다.
- r5가 r4보다 좋은 것은 판독 **내용**이 유용하다는 증거다. 자연어라는 형식의
  독립 효과는 확인되지 않았다.
- 무선별 재고 요청은 전체 성능을 크게 해친다. 교정은 정밀한 selector와
  결합된 조건부 정책으로만 제안한다.

## 문서 상태

### 현재 정본

- `experiment_summary_2026-08-25.md`: 실험·수치·상태의 짧은 정본.
- `paper_outline_2026-08-24.md`: 서사와 절 배치의 정본.
- `table_camera_ready_2026-08-25.md`: 표 수치와 캡션의 정본.
- `prior_work_2026-08-24.md`: 신규성 판단의 정본.

### 조사·감사 문서

- `related_work_2026-08-23.md`: 문헌 조사 원장. 원고가 아니다.
- `reading_catching_rationalization.md`, `reading_when_truth_is_overridden.md`:
  최근접 선행의 충돌·차이 분석.
- `judge_jobs_2026-08-24.md`: 외부 판정자 실험 계획.
- `table_camera_ready_2026-08-25.md` 안의 MCR 결론 판독 실패 기록은 **감사
  기록**이다. `.052`, `.034`, "6배" 수치는 source-misaligned target으로 얻은
  무효 결과이며 현재 주장의 근거가 아니다.

### 대체된 문서

`docs/archive/paper_tables_worklog_2026-08-23.md`는 표 설계가 여러 번
바뀌는 동안의 작업 로그다. 재현 감사에는 유용하지만 현재 표·수치의 출처로
인용하지 않는다.

## 초안과 투고 준비 상태

초안 작성은 가능하다. 다만 아래 항목 전에는 "submission-ready"가 아니다.

1. r7 자기 CoT 되먹임 대조를 닫아, 내부 되먹임이 단순한 "무언가 재제시"보다
   낫다는 주장을 검증한다.
2. DDXPlus alias/matching 규칙을 하나로 고정하고 Table 2b/2c/4를 재집계한다.
3. DDXPlus correct-note 누락 셀을 동일 모집단에서 다시 실행한다.
4. 외부 판정자 또는 임상의 평가로 자연어 판독의 유용성과 임상적 타당성을
   보조 검증한다.
5. MCR에서는 source-aligned 자연어 판독과 교정 사다리를 완성하거나, 본문
   주장을 행동 복제까지만 명확히 제한한다.
6. Related Work의 최신 논문 서지·게재 상태와 정확한 인용 문장을 원문으로
   재확인한다.

## 갱신 규칙

- 새 수치는 먼저 `table_camera_ready_2026-08-25.md`에 분자/분모·모집단·계기와
  함께 기록한다.
- 서사가 바뀌면 `paper_outline_2026-08-24.md`, 짧은 상태가 바뀌면
  `experiment_summary_2026-08-25.md`를 갱신한다.
- 실패한 실행과 철회된 해석은 삭제하지 않고 `docs/archive/` 또는 명시된
  audit subsection에 남긴다.
- 교수님용 문서는 위 정본에서만 수치를 가져오며, 과거 날짜의 professor 문서는
  당시 스냅샷으로 취급한다.
