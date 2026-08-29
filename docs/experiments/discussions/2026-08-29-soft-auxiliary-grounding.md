# Soft auxiliary grounding after D14

## 질문

D14 hard-set OOF teacher가 calibration gate를 통과하지 못한 뒤에도, probe의
연속적인 activation 신호를 training-only 보조 목적함수로 사용해 하나의
Medical-NLA decoder를 사례 특이적인 자연어 판독기로 만들 수 있는가?

최종 inference 계약은 유지한다.

```text
raw CoT-P0/HS32 activation -> one Medical-NLA decoder -> <observed> findings
```

Probe, DDXPlus ontology, prompt cue, auxiliary head는 inference에서 사용하지 않는다.

## 출발 근거

D14 K=5는 original 상태에서 OOF/full Jaccard `.9437`, recall `.9999`를 보였지만
동결 AND gate는 실패했다.

| failure | observed | gate |
|---|---:|---:|
| original cue precision | .8881 | >= .90 |
| deleted mean-claims relative gap | 18.10% | <= 10% |

따라서 다음 행동은 금지한다.

- K=10 등 추가 fold sweep
- threshold `.5` 재선택
- failed hard set으로 text target 생성
- full-data probe output을 OOF target 대신 사용

동시에 continuous activation에는 강한 판독 가능성이 확인돼 있다.

| evidence | value |
|---|---:|
| locked-test finding micro F1 | .9562 |
| locked-test own-shuffled finding gap | +.1624 |
| locked-test value accuracy | .7659 |
| structured-reader test finding F1 | .9587 |
| structured-reader test deletion phantom | .3593 |

이는 probe를 최종 시스템으로 채택할 근거가 아니라, language decoder가 버리고
있는 activation-conditioned signal이 존재한다는 근거다.

## 후보 방법

| method | 장점 | 핵심 위험 | 현재 판단 |
|---|---|---|---|
| hard-set set-to-text 재시도 | 구현이 단순함 | D15 위반, calibration error를 정답으로 고정 | 금지 |
| soft probability를 문장 target으로 변환 | claim별 불확실성 보존 | token CE가 확률 벡터를 직접 표현하지 못함 | 비권장 |
| training-only soft auxiliary head + language SFT | 연속 신호와 자연어 역할 분리, inference는 단일 decoder | decoder가 auxiliary 신호를 무시할 수 있음 | **우선 검토** |
| structured reader를 최종 방법으로 사용 | finding 판독이 강함 | open NLA가 아니며 ontology/probe가 inference에 필요 | control만 유지 |

## 우선 검토 설계

하나의 activation adapter latent `z=A(h32)`를 language decoder와 training-only
multilabel head가 공유한다.

```text
                         -> language decoder -> <observed>
raw HS32 -> shared z ---|
                         -> auxiliary finding head (training only)
```

학습 역할은 다음처럼 분리한다.

| source | language loss | auxiliary grounding loss |
|---|---|---|
| DiReCT | physician-observation `<observed>` | 선택 사항 |
| DDXPlus original/deleted | 없음 또는 최소 schema loss | OOF soft probability/vector loss |

DDXPlus cue 문장을 곧바로 자연어 정답으로 주지 않으므로 prompt reconstruction을
줄이고, hard threshold를 사용하지 않아 D14의 set calibration 실패를 text target에
고정하지 않는다. Auxiliary head는 배포 전에 제거한다.

## 반드시 검증할 실패 모드

1. **Decoder bypass**: auxiliary head만 activation을 읽고 language decoder는 여전히
   generic text를 생성할 수 있다.
2. **Soft-teacher inheritance**: K=5 teacher의 deletion 과다 선택이 continuous loss에도
   전달될 수 있다.
3. **Dataset shortcut**: DDXPlus auxiliary task와 DiReCT language task가 shared latent에서
   분리되어 전이되지 않을 수 있다.
4. **Fluency-grounding tradeoff**: grounding 개선과 함께 Direct 자연어 품질이 떨어질 수
   있다.

## 최소 판정 실험 제안

새 결정을 승인하기 전에는 구현하지 않는다. 승인된다면 첫 실험은 capacity sweep이
아니라 두 arm의 동일-budget 비교로 제한한다.

| arm | objective |
|---|---|
| control | DiReCT language SFT only |
| proposed | 동일 language SFT + DDXPlus training-only soft auxiliary loss |

필수 판정값:

- Direct activation-target symmetric alignment gap과 category-cluster CI
- Direct semantic observation score
- DDXPlus original/deleted generation hit, phantom, removal, retained preservation
- seed 17/29/43 방향 재현성
- auxiliary head 제거 후 동일 결과인지 여부

우선 실험은 validation/development population만 사용한다. Locked test는 방법과
hyperparameter를 동결한 뒤 한 번만 사용한다.

## 판정

현재 상태: **discussion / 사람 승인 전**.

아직 결정하지 않은 항목:

1. soft target을 probability, logit, rank 중 무엇으로 둘지
2. auxiliary loss weight를 sweep 없이 어떻게 사전 고정할지
3. shared latent의 정확한 위치와 auxiliary head 제거 계약
4. 첫 smoke의 effect floor와 promotion gate
