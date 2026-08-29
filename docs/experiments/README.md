# Medical-NLA 실험 원장

이 디렉터리는 현재 논문의 E0-E7만 다룬다. 과거 wrong-note 실험은
[`../archive/legacy_wrong_note_2026-08-25/experiments/`](../archive/legacy_wrong_note_2026-08-25/experiments/)에
있다.

| ID | 실험 | 상태 | 주 산출물 |
|---|---|---|---|
| E0 | DiReCT data/evaluator audit | 완료 | canonical split, evaluator smoke |
| E1 | Source CoT and activation extraction | 완료 | 496 cases, P0/P1/P2 x HS16/24/32 |
| E2 | P0 representation audit | 대부분 완료 | diagnosis/category 및 DDXPlus finding/value locked probe 완료; source-decision 대기 |
| E3 | Medical-NLA training | objective 재설계 | Full/CF SFT는 부분 finding 신호만 복원; cue-level objective가 다음 gate |
| E4 | DiReCT explanation evaluation | validation 완료 | Common SFT 의미 채점 실패; 새 objective 고정 전 locked 72/106 중단 |
| E5 | DDXPlus activation grounding | NLA validation 부분 신호 | probe availability 통과; CF seed17 deletion contrast 증가, value switch 실패; locked test 중단 |
| E6 | Text patching | E5 조건부 | Table 4, Figure 4 |
| E7 | MCR external OOD | 후순위 | frozen-checkpoint OOD table |

## 의존성

```text
E0 -> E1 -> E2 -> E3 -> E4
                  |      |
                  +----> E5 -> [pass] -> E6
                         |
                         +------------> E7 (frozen only)
```

## 공통 규칙

- Train/validation으로 layer, epoch, threshold를 고정하고 test는 마지막에 한 번 평가한다.
- DiReCT 원문과 private artifact는 Git에 올리지 않는다.
- CoT와 NLA의 비교는 같은 source case와 같은 evaluator를 사용한다.
- P0가 primary activation이다. P1은 CoT 내 diagnosis leakage를 분리해 보조 분석한다.
- 공개 AV/AR의 extraction index가 32이므로 primary NLA/round-trip은 HS32다. HS16/24는 sensitivity다.
- Clinical alignment와 activation grounding을 하나의 점수로 합치지 않는다.
- E5를 통과하지 못한 방법으로 E6 성능 개선 주장을 하지 않는다. 현재 value-edit E6는 중단한다.

각 실험의 상세는 `00`-`07` 문서를 따른다.

Medical-NLA 학습 방법의 비교, 현재 실패 원인, 다음 objective와 최종 단일-adapter
파이프라인은
[`medical_nla_tuning_strategy_2026-08-29.md`](medical_nla_tuning_strategy_2026-08-29.md)를
따른다.

이 전략의 검토 결과 — 보완점 6개(HS32 Gate A ceiling, seed 3개 규칙,
value gate 강등, Gate C bar, paired margin 구현, cue support mask)와
1a 감사 결과, smoke 전 실행 순서는
[`medical_nla_tuning_strategy_review_2026-08-29.md`](medical_nla_tuning_strategy_review_2026-08-29.md)를
따른다.

실험 방향에 대한 에이전트 간 토의는
[`AGENT_DISCUSSION.md`](AGENT_DISCUSSION.md)에서 진행한다 — 운영 규칙,
동결된 결정 원장(D1-D8), 라운드 로그.
