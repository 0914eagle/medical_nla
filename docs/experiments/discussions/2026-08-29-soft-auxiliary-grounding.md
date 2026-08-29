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

## Discussion 2 — Claude 검토 (2026-08-29)

**[동의] 방법 선택 자체는 사전 등록된 분기 안에 있다.** shared latent `z` +
training-only multilabel head는 방법 J(latent claim bottleneck)의 soft
supervision 형태다. D12의 사전 등록 분기가 "실패 → I/J"였고 I는 control로
고정됐으므로(D13), 이 제안은 새 발명이 아니라 합의된 다음 칸이다. Hard-set
재시도·sweep 금지·단일 decoder inference 계약 준수도 확인했다.

**[반론] 핵심 기제가 하나 빠져 있다 — aux head는 z가 이미 가진 성질을
재생산할 위험이 있다.** D13이 확정한 병목은 "표현에 정보가 없다"가 아니라
"decoder가 그 정보를 쓰지 않는다"였다(raw HS32 선형 decodability `.9607` vs
free generation 실패). Aux loss가 하는 일은 z를 finding-decodable하게 만드는
것인데, **raw activation은 이미 그 성질을 갖고 있다.** 따라서 이 설계가
의미를 가지는 조건은 단 하나다: **z가 실제 bottleneck이어서, language SFT만
받으면 finding 정보를 버리게 되는 구조일 것.** 이 조건을 설계와 판정에
명시해야 한다.

- 설계 조건: z가 activation→decoder의 **유일한** 경로여야 하며(우회 residual
  경로 금지), aux head는 **선형**이어야 한다. Deep head면 head가 과제를
  자체 흡수해 z가 조직화될 필요가 없어지고 bypass가 더 쉬워진다.
- 판정 조건 추가: 학습 후 **frozen z에 대한 선형 probe decodability를 두 arm
  모두에서 보고**한다. (i) aux arm의 z decodability가 control과 같으면 aux
  loss가 아무것도 바꾸지 않은 것이고, (ii) z decodability만 오르고 생성이
  안 바뀌면 decoder bypass가 확정된다. 실패 모드 1을 사후 해석이 아니라
  계측으로 분리하는 장치다.

**[제안] 열린 항목 1 — deleted arm은 절대 soft target이 아니라 paired
delta로 감독한다.** D14 부검의 핵심은 "절대 calibration(precision, mean
claims)은 실패했지만 **paired delta는 처음부터 신뢰할 수 있었다**"는 것이다
(deletion delta 중앙값 `.79`, D9a positive/null 분리). 그러므로:

- original arm: teacher soft probability에 대한 BCE (K=5 original은
  full-data와 Jaccard `.9437`로 근접 — 절대 supervision 정당)
- deleted arm: 절대 set/soft target 금지. **changed label에 대한 paired
  margin** — `head(z_orig)[c] - head(z_del)[c] > 0` — 만 건다. 실패한
  절대 calibration을 계승하지 않고(실패 모드 2 차단), 살아남은 신호만 쓴다.

logit/rank는 비권장: rank는 calibration 정보를 버리고, logit 회귀는 teacher
절대값 오류를 그대로 상속한다.

**[제안] 열린 항목 2 — weight는 gradient-norm parity로 1회 고정.** Sweep
없이 임의성을 제거하는 규칙: 학습 시작 시점에 shared z에 대한 aux gradient
norm이 language gradient norm과 같아지도록 lambda를 한 번 계산해 고정하고
값을 기록한다. 이후 조정은 사람 승인.

**[제안] 열린 항목 3 — 제거 계약.** aux head는 z 위의 단일 선형층. 배포
checkpoint는 decoder+adapter만 포함하고, aux head 제거 전후 decoder/adapter
weight SHA256이 동일함을 기록한다.

**[제안] 열린 항목 4 — effect floor는 control-first 순서로 동결한다.**
DiReCT alignment gap의 역사적 스케일은 `+.0051`(full SFT)로 D5의 `.05`와
자릿수가 다르다. 임의 floor 대신: **control arm(3 seeds)을 먼저 완주하고,
proposed arm을 평가하기 전에** floor = control seed spread 기반 값(예: spread
상한의 2배)을 선언·기록한다. Proposed arm 결과를 본 뒤의 조정은 무효.
DDXPlus generation 지표(hit/phantom/removal/retained)는 D5 구조를 상속한다.

**[동의+한정] 실패 모드 4에 하나 추가.** Control arm(DiReCT SFT only)의
역사적 성능은 Obscomp `.0301`로 매우 낮다. 두 arm 비교가 "상대 개선"을
보여도 둘 다 임상적으로 무의미한 수준일 수 있으므로, smoke 통과와 별개로
Gate C bar(source CoT `.2130` 초과, D7)가 phase 출구임을 이 문서에 명시해
두자.

**[판정 요청]** 위 5개(z-bottleneck 조건 + z-probe 계측, deleted-arm paired
delta 전환, gradient-parity weight, 제거 계약, control-first floor)를 제안
설계에 반영해 사람 승인에 올릴 것인가.
