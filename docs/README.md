# docs/ 안내

## experiments/ — 실험 하나당 문서 하나 (재현용)
- `README.md` — 목록 + **모든 실험이 공유하는 설정**: 모델·하드웨어·활성값
  추출 지점·LoRA 하이퍼파라미터·채점 규칙·공통 용어
- `01`–`15` — 실험별로 질문·설정·표본·절차·실측값·**의미하지 않는 것**·재현 명령
- `docs/paper/`가 조판용 정본이고, 여기는 그 값이 **어떻게 나왔는지**를 적는다

## paper/ — 논문 작업의 살아있는 문서
- `README.md` — **현재 정본 안내**: 테제, 확정 근거, 주장 금지선, 제출 전 관문. 처음에는 이 파일만 보면 됨
- `table_camera_ready_2026-08-25.md` — **표 5종 조판용 원고** (영문 캡션·실측치·설계 규칙·남은 ▢)
- `judge_jobs_2026-08-24.md` — **판정자 작업 8종 전체 목록**: 우선순위·비용(~$6–30)·예상 결과·실행 절차. API 키 오면 여기부터
- `prior_work_2026-08-24.md` — **선행 연구 정독 12편, 신규성 판정의 정본**. 조항 A–E별로 정리: 무엇이 우리 기여가 아닌지, 어디에 한정어가 필요한지, 무엇이 그대로 서는지. §2 인용 목록과 §4.4의 4범주 표 포함
- `experiment_summary_2026-08-25.md` — **한 장 현황**: 실험별 무엇을 보이는지·실측치·상태, 선행 연구 지도와 우리 몫, 논문이 성립하는 이유. 처음 보는 사람은 여기부터
- `paper_outline_2026-08-24.md` — **논문 골격**: RQ 3개, 절별 문단 계획 (4+4 구조)
- `reading_catching_rationalization.md` — 탐지 축 쌍둥이(2603.17199) 정독 노트: 겹침과 결정적 차이
- `reading_when_truth_is_overridden.md` — 최근접 선행(AAAI'26) 정독 노트: 충돌 문장과 그 해소
- `related_work_2026-08-23.md` — 문헌 조사 전체 (필수 인용, 계열별 포지셔닝, 논문 일람, DDXPlus 선행, SHAP/LIME 방어)
- `draft_related_work_2026-08-24.md` — Related Work 산문 초안 (영문)
- `related_work.tex` — 위 초안의 LaTeX판 (\cite 키 + 키→논문 매핑)

## professor/ — 교수님 대면 문서
- `professor_update_2026-08-24.md` — **최신 보고** (주말 결과 + 정정 1건 + 결정 요청)
- `professor_report_2026-08-17_to_22.md` — 8/17–22 진행 보고
- `hypothesis_disposition_2026-08-22.md` — 8/19 명제 13개의 생사
- `project_history_2026-08-22.md` — 첫 독자용 프로젝트 역사
- `professor_presentation_2026-08-17.md` — 초기 발표 자료

## data/ — 데이터셋 문서
- 스펙, 구축 로그, DDXPlus 벤치마크 성질, 소스 답·likelihood 노트

## results/ — 1기 실험 결과 기록 (계기 검증 시절)
- layer/position sweep, counterfactual faithfulness, cue-position 판독,
  수동 라벨 tsv 등 — Table 2의 근거 문서들

## archive/ — 대체된 문서 (참고용)
- 옛 표 설계(8/18·8/19), 옛 가설·스토리라인(8/22), 세션 인수인계,
  상태 스냅샷, `paper_tables_worklog_2026-08-23.md` — 최신 내용은
  `paper/README.md`가 가리키는 정본에 흡수됨
