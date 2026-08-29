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
| implementing | [D16 soft auxiliary bottleneck](2026-08-29-soft-auxiliary-grounding.md) | PCA gate -> control-first smoke -> frozen-z/generation 판정 |

## 완료·보관

| 범위 | 문서 |
|---|---|
| D14 hard-set OOF teacher (K=5 gate FAIL) | [2026-08-29-d14-oof-teacher.md](2026-08-29-d14-oof-teacher.md) |
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
