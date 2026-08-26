# Medical-NLA 문서 안내

이 디렉터리의 활성 문서는 2026-08-26에 확정한 연구 방향을 기준으로 한다.

> 의료 LLM의 CoT와 activation 기반 자연어 판독을 같은 임상 기준에서 비교하고,
> Medical-NLA가 임상적으로 정렬된 설명을 만들 뿐 아니라 해당 사례의 activation에
> 실제로 근거하며, 검증을 통과한 판독이 선택적 개입에 유용한지 평가한다.

과거의 `wrong referral note` 연구는 재현성과 연구 이력을 위해
[`archive/legacy_wrong_note_2026-08-25/`](archive/legacy_wrong_note_2026-08-25/README.md)에
보존한다. 그 문서와 수치는 현재 논문의 주 근거로 사용하지 않는다.

## 처음 읽는 순서

1. [`paper/README.md`](paper/README.md): 현재 주장, 가설, RQ, 주장 금지선
2. [`paper/tables_and_figures.md`](paper/tables_and_figures.md): 최종 표와 그림의 구조
3. [`experiments/README.md`](experiments/README.md): E0-E7 실행 순서와 상태
4. [`data/direct_dataset_and_split.md`](data/direct_dataset_and_split.md): DiReCT 구조와 split
5. [`professor/current_research_brief_2026-08-26.md`](professor/current_research_brief_2026-08-26.md): 교수님 보고용 요약

## 디렉터리 역할

| 디렉터리 | 역할 | 정본 |
|---|---|---|
| `paper/` | 논문 주장, 구성, 표·그림, Overleaf 이전 계획 | `paper/README.md` |
| `experiments/` | E0-E7 질문·입력·출력·평가·중단 조건 | `experiments/README.md` |
| `data/` | 데이터셋 구조, 분할, 구축 기록 | `data/direct_dataset_and_split.md` |
| `professor/` | 현재 연구 방향과 상세 실행 계획 | `professor/README.md` |
| `results/` | 8월 13-21일 DDXPlus pilot 결과 | 참고 근거이며 최종 표 수치가 아님 |
| `archive/` | 폐기·대체된 가설, 표, 소견서 연구 | 현재 주장에 인용 금지 |

## 현재 연구의 세 평가 층

| 층 | 질문 | 데이터 | 통과해야 주장 가능한 것 |
|---|---|---|---|
| Clinical alignment | 설명이 의사 주석의 관찰·관계·진단과 맞는가 | DiReCT | 임상 설명 품질 |
| Activation grounding | 설명이 해당 activation과 사례별로 연결되는가 | DDXPlus | 내부 상태에 근거한 판독 |
| Causal utility | 판독을 편집·복원하면 목표 상태와 행동이 선택적으로 바뀌는가 | DDXPlus | text patching과 성능 개선 가능성 |

DiReCT 점수만 높다고 `faithful`이라고 부르지 않는다. DDXPlus의 짝 깨기,
증거 반사실, round-trip 통제를 통과해야 activation-grounded라는 표현을 사용한다.

## 데이터 보안

DiReCT는 제한 데이터다. 원문, 추출 JSON, patient identifier, private manifest,
모델 입출력의 원문 사례는 Git에 커밋하지 않는다. 문서에는 aggregate 수치와 공개가
허용된 스키마만 기록한다.
