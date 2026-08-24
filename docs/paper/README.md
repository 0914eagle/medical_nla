# 논문 문서 안내 — 현재 정본

이 폴더의 논문 서사는 **현상 우선**이다. 주인공은 NLA 자체가 아니라,
의뢰 소견서의 의심 진단이 의료 LLM의 출력은 바꾸지만 내부 정답 신호를
완전히 대체하지 못한다는 **internal-output dissociation**이다. 프로브는 이를
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
실험에서 내부 gold signal은 평균적으로 남고 suggestion은 대다수 사례에서
probe top-1이 되지 않는다. 이 내부-출력 결렬은
한 번의 실행에서 탐지할 수 있고 자연어로 판독할 수 있으며, 정확한 내부
내용을 되먹이면 일부 오류를 회복할 수 있다.

영문으로는 다음 범위가 안전하다.

> In a causally controlled diagnostic setting, referral-note anchoring often
> changes what a medical LLM emits without fully replacing the gold-diagnosis
> signal in its probed internal state. The resulting internal-output rift can be detected
> from a single run, rendered as a natural-language readout, and used as
> corrective evidence when the readout is accurate.

## 확정된 근거와 범위

| 주장 | 현재 근거 | 주장 가능한 범위 |
|---|---|---|
| 오답 소견서가 진단을 움직인다 | DDXPlus main −23.03 pp; 3× larger run −21.30 pp; MCR −26.89 pp | 행동 효과는 두 코퍼스; c300은 독립 표본 아님 |
| suggestion이 내부 top-1을 대개 차지하지 못한다 | moved 321건 중 266건(82.9%)에서 suggestion이 어느 landmark에서도 top-1이 아님 | 그중 gold throughout 151, third-diagnosis path 115 |
| 소견서는 내부 gold signal을 행동별로 다르게 낮춘다 | canonical paired trajectory 확인, exact Table 3 probability 전사 대기 | DDXPlus, 49-class cross-fit probe |
| CoT보다 내부 채널이 강하다 | 정본 silent subset(n=1,641) AUROC: LLM CoT monitor .6829, NL readout .8302; gap +.1473 CI [.0691,.2209] | probe canonical silent 값은 대기 |
| 자연어 판독은 벡터에 종속된다 | swap .993, memorization .000, contamination .007; heldout cue content .751 (n=770, 기계 채점) | DDXPlus 검증 배터리 |
| 내부 내용을 되먹이면 회복한다 | 구 matcher에서 moved accuracy .012 → r5 .627 / r6 .830 | canonical moved=321 재집계와 r7 대기 |
| 자연어 형식 자체가 교정의 원인이다 | **지지되지 않음**: 내용 정확도 일치 시 r5 vs r6 p=.720 | 주장 금지 |

## 반드시 지킬 주장 경계

- 행동 효과는 DDXPlus와 MCR에서 재현됐지만, **82.9% 기전 해부는 DDXPlus만**이다.
  그리고 82.9%는 gold-throughout이 아니라 suggestion-never-top1이다.
- 닫힌 49-class 탐지는 기존 실행에서 자연어 판독보다 지도 프로브가 강했다.
  canonical silent probe 값은 재집계 대기다. 자연어 판독의 몫은 최고 정확도가 아니라 서술, 근거, 열린
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

1. canonical matcher로 wording, CoT, r3–r7 교정 사다리를 모두 재집계한다.
   r7 자기 CoT 되먹임 대조도 이때 함께 닫는다.
2. Table 3의 canonical 1,426/230/91 그룹별 확률과 canonical probe all/silent
   값을 채운다.
3. DDXPlus main neutral/correct 누락 셀을 같은 fixed cohort에서 채운다.
4. 외부 판정자 또는 임상의 평가로 자연어 판독의 유용성과 임상적 타당성을
   보조 검증한다.
5. MCR에서는 source-aligned 자연어 판독과 교정 사다리를 완성하거나, 본문
   주장을 행동 복제까지만 명확히 제한한다.
6. 같은 judge의 no-CoT arm을 추가해 LLM monitor 향상이 CoT 때문인지,
   vignette/note/answer 접근 때문인지 분리한다.
7. Related Work의 최신 논문 서지·게재 상태와 정확한 인용 문장을 원문으로
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
