# Medical-NLA discussion index

## 운영 규칙

1. 한 실험 질문 또는 한 판정 흐름마다 `YYYY-MM-DD-topic.md` 파일 하나를 만든다.
2. 새 결과는 관련 주제 파일에 실제 수치와 artifact 경로를 기록한다.
3. 사람이 승인한 사항만 [`DECISIONS.md`](DECISIONS.md)에 옮긴다.
4. 완료된 주제는 이 index에서 `resolved`로 표시하고 파일은 유지한다.
5. 새 에이전트는 전체 archive를 읽지 않고 이 index, 결정 원장, 활성 문서만 읽는다.
6. Locked-test 접근, threshold 변경, 실패 후 sweep은 기존 동결 결정을 따른다.

## 활성 주제

| 상태 | 주제 | 다음 판정 |
|---|---|---|
| ready to run | [D22 Patchscope same-layer source sweep](2026-08-31-d22-patchscope-same-layer-source-sweep.md) | HS32 고정 target sweep 전부 FAIL; HS16→16/HS24→24/HS32→32 마지막 control gate |
| resolved (FAIL) | [D22 Patchscope feature-interface calibration](2026-08-31-d22-patchscope-feature-interface.md) | entity 2/5 전 layer, relation 1/5·1/5·0/5; clinical 미실행, same-layer source sweep으로 분리 |
| active | [D22 public-AR medical-distribution diagnostic](2026-08-30-d22-public-ar-diagnostic.md) | 공개 AR 불인정; final-marker Patchscope는 token decoding PASS/entity description FAIL, 후속 interface calibration으로 분리 |
| resolved (FAIL) | [D10 budget calibration and program decision](2026-08-29-program-decision-after-d16.md) | budget gate FAIL 확정; 출구는 D20 결과 후 재론 |
| resolved (FAIL) | [Specificity-anchored 2x2 objective (D20)](2026-08-30-specificity-anchored-objective.md) | teacher-forced gate FAIL; detector 차단 성공·changed-gap 신호 부재; D19/D21 ledger 행 사람 승인 대기 |
| active | [Paper table completion](2026-08-29-paper-table-completion.md) | Vanilla protocol freeze, D10 decision, then locked baseline batch |
| resolved (PASS) | [DDXPlus open-text semantic mapper protocol](2026-08-30-ddxplus-semantic-mapper-protocol.md) | V2 G1-G4 전부 통과; aggregate receipt commit 뒤 sealed 10,028행 1회 채점 |
| ready to run | [Locked baseline execution runbook](2026-08-30-locked-baseline-execution.md) | DDXPlus HS32 -> sealed 10,028 generation; D10 뒤 DiReCT 178 batch |
| proposal | [Paper tables under a successful Medical-NLA](2026-08-29-paper-tables-success-scenario.md) | conditional table structure approval |
| active | [Paper table values and reproducibility](2026-08-29-paper-table-values-and-reproducibility.md) | canonical values, denominators, recipes, provenance audit |
| ready to run | [SFT family raw-output audit](2026-08-30-sft-raw-output-audit.md) | DiReCT 50-case census and separate DDXPlus deletion/value-edit 50-case audits |

## 완료·보관

| 범위 | 문서 |
|---|---|
| D14 hard-set OOF teacher (K=5 gate FAIL) | [2026-08-29-d14-oof-teacher.md](2026-08-29-d14-oof-teacher.md) |
| D16 soft auxiliary bottleneck (3-seed gate FAIL, frozen-z 하락) | [2026-08-29-soft-auxiliary-grounding.md](2026-08-29-soft-auxiliary-grounding.md) |
| R1-R20 전체 원문 | [2026-08-29-r01-r20.md](archive/2026-08-29-r01-r20.md) |

## 새 문서 형식

```markdown
# 주제

## 질문
## 현재까지의 근거
## 승인된 규약
## 실행 및 산출물
## 결과
## 판정
## 열린 항목
```

라운드 번호를 전역으로 계속 증가시키지 않는다. 필요한 경우 문서 안에서만
`Discussion 1`, `Decision 1`처럼 지역 번호를 사용한다.
