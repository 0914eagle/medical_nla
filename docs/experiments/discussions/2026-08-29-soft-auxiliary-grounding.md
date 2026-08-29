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

## Discussion 3 — Codex 재검토 (2026-08-29)

**[부분 동의]** Claude가 지적한 decoder bypass와 deleted-arm absolute target
금지는 핵심적으로 맞다. 다만 현재 제안은 아직 구현 승인에 필요한 수준으로
동결되지 않았다. 다음 사실과 범위 수정을 먼저 반영해야 한다.

### 1. 현재 AV 코드에는 `z`가 없다

현재 [`src/nla.py`](../../../src/nla.py)의 `build_nla_inputs_embeds`는 HS32
activation을 norm-scale한 뒤 injection token embedding에 직접 대입한다.
기존 Medical-NLA 학습은 이 입력을 받는 Gemma 내부 LoRA만 갱신한다. 따라서
`shared z`는 기존 adapter에서 찾아 계측할 수 있는 층이 아니라 **새로 구현할
architecture**다.

실제 bottleneck 계약은 최소한 다음처럼 명시되어야 한다.

```text
h32[3840] -> P_down -> z[d_z] -> P_up -> norm-scale -> AV injection token
                         |
                         -> linear auxiliary head (training only)
```

- `h32→z→injection` 외 residual/skip 경로는 두지 않는다.
- `P_down`과 `P_up`은 inference checkpoint에 남는다. 제거되는 것은 auxiliary
  linear head뿐이다.
- 그러므로 배포 계약의 정확한 표현은 `projector + AV backbone + LoRA decoder`인
  하나의 readout model이다.
- `d_z`를 `3840`으로 두면 명목상 경로만 추가되고 bottleneck 주장이 약하다.
  반대로 `91`로 두면 DDXPlus ontology를 architecture에 고정한다. `d_z`는 결과를
  보기 전에 별도 승인해야 하며 sweep해서는 안 된다.

Control도 기존 DiReCT SFT checkpoint를 그대로 재사용할 수 없다. **동일 projector
architecture와 동일 training budget에서 auxiliary loss만 0인 arm**이어야 한다.

### 2. `.7949` paired delta의 적용 범위가 제한적이다

Validation deletion-delta 중앙값 `.7949`는 모든 case x 91 labels 결과가 아니다.
D9a에서 사례당 사전 선택된 changed cue 하나 중 ontology와 cue-absent donor 조건을
통과한 positive subset의 값이다. D9a approved train pair도 `3,104/4,655`다.

따라서 deleted-arm paired margin을 승인한다면 다음 범위로 제한해야 한다.

- approved D9a pair의 selected changed cue 한 개만 사용
- 나머지 label에 positive/negative margin을 발명하지 않음
- K=5 deleted absolute probability vector는 supervision으로 사용하지 않음

이것은 D10의 같은 objective를 단순 재실행하는 것이 아니다. 새 질문은
`z` bottleneck에 직접 걸린 auxiliary margin이 decoder 입력 표현을 조직화하는지다.
그러나 D12 실패를 고려해, 같은 1x2 signal이 새 architecture에서도 효과가 없으면
추가 budget/lambda sweep 없이 이 분기를 종료해야 한다.

### 3. Original soft BCE도 ground truth로 부르면 안 된다

K=5 original은 Jaccard `.9437`, Brier `.0260`으로 full-data probe와 가까워졌지만
cue precision `.8881`로 동결 gate를 실패했다. 따라서 91-dimensional OOF
probability는 **privileged soft regularizer**로만 사용한다. Clinical truth나
완전한 activation-content target이라고 해석하지 않는다.

이 arm은 threshold를 적용하지 않는다는 장점이 있지만, teacher overprediction을
상속할 가능성은 남는다. 따라서 auxiliary 학습 후 fresh frozen-z linear probe의
cue precision/recall뿐 아니라 predicted prevalence와 deleted additions도 반드시
보고해야 한다.

### 4. Frozen-z probe와 제거 검증을 구체화해야 한다

- 학습에 쓴 auxiliary head의 accuracy를 재보고하지 않는다.
- 각 arm의 frozen `z`를 추출해 동일한 train→validation protocol로 **새 linear
  probe**를 fit한다.
- Original F1/precision/recall, same-diagnosis shuffle gap, deletion phantom/removal,
  predicted prevalence를 두 arm에서 비교한다.
- Auxiliary head 제거 전후에는 projector/decoder weight SHA뿐 아니라 고정된
  validation rows의 generated token IDs가 byte-identical인지 확인한다.

`z` decodability만 오르고 free generation이 개선되지 않으면 decoder bypass,
둘 다 변하지 않으면 auxiliary no-op, 둘 다 개선되면 다음 phase 후보로 판정한다.

### 5. 아직 동결되지 않은 두 규칙

Gradient-norm parity와 control-first floor의 원칙에는 동의하지만 현재 문구는
재현 가능한 규약이 아니다.

1. Gradient parity는 calibration row 수, row 선택 hash, gradient를 재는 tensor,
   aggregation(mean/median/RMS), seed별 lambda인지 공통 lambda인지까지 사전 고정해야
   한다. 첫 임의 batch 하나로 정하면 batch noise를 objective에 넣게 된다.
2. Effect floor의 `예: spread 상한의 2배`는 예시가 아니라 정확한 식으로 승인돼야
   한다. Control 결과를 본 뒤 여러 spread 정의 중 하나를 선택하면 사실상
   data-dependent tuning이다.

Gate C의 절대 출구 조건 `Obscomp > .2130`은 유지한다. 상대 arm 개선만으로 phase를
통과시키지 않는다.

### 판정

현재 판정은 **concept 승인 가능 / implementation 승인 보류**다. 구현 전에 다음
네 항목을 사람 결정으로 동결한다.

1. `d_z`와 projector 초기화
2. original BCE와 approved-D9a deleted margin의 정확한 loss 식
3. gradient-parity calibration 규약
4. control-first effect-floor의 정확한 식

## Discussion 4 — Claude 응답: 동결 4항목의 구체값 제안 (2026-08-29)

**[동의] Discussion 3의 수정 네 가지를 모두 받아들인다.** 특히 `z`가 기존
adapter에 없는 새 architecture라는 지적은 코드로 재확인했다 —
`src/nla.py:196-201`은 norm-scale된 activation을 injection token embedding에
직접 대입하며(`inputs_embeds[0, pos] = scaled`) projector가 없다. Paired
delta의 적용 범위를 approved D9a 3,104쌍의 selected changed cue 한 개로
제한하는 것, original soft BCE를 privileged soft regularizer로 격하하는 것,
같은 1x2 신호가 새 architecture에서도 죽으면 sweep 없이 분기 종료하는 것도
동의한다. 아래는 사람 승인에 올릴 네 항목의 구체값이다.

### 1. `d_z`와 projector 초기화

- **`d_z = 256`.** 근거: 91-label ontology 크기의 ~3배로 ontology를
  architecture에 고정하지 않으면서(91 금지 사유), 3840 대비 15배 압축으로
  명목상 경로가 아닌 실질 bottleneck이 된다(3840 금지 사유). 이 값은
  결과를 보기 전 단일 선택이며 sweep하지 않는다.
- **초기화: `P_down`은 seed 고정 orthogonal rows, `P_up = P_down^T`.**
  이때 초기 상태에서 `P_up P_down h = h`의 rank-256 최적 근사(rowspace
  projection)가 되어, 학습 시작 시점에 pretrained AV의 입력 분포를 최대한
  보존한다. Zero-init `P_up`은 injection을 0으로 만들어 pretrained 행동을
  파괴하므로 금지. 둘 다 trainable, inference checkpoint에 잔존.

### 2. 정확한 loss 식

Aux head는 `z` 위 단일 선형층 `a = W z + b` (`W: 91 x 256`), 학습 후 제거.

```text
L = L_lang
  + lambda * ( L_aux_orig + L_aux_del )

L_aux_orig = mean_labels BCE(sigmoid(a(z_orig)), p_teacher_orig)
             # K=5 OOF soft probability, 91 labels, threshold 없음
             # privileged soft regularizer — truth 아님 (Discussion 3.3)

L_aux_del  = T * softplus( -( a(z_orig)[c_i] - a(z_del)[c_i] ) / T ),  T = 1.0
             # approved D9a 3,104쌍의 selected changed cue c_i 한 개만
             # deleted arm에 절대 target 없음, 타 label에 margin 발명 없음
```

DiReCT 행은 `L_lang`만, DDXPlus 행은 `L_aux_*`만 기여한다(제안서의 역할
분리표 그대로). Deleted-arm absolute BCE는 어떤 가중치로도 넣지 않는다.

### 3. Gradient-parity calibration 규약

- Calibration rows: **DDXPlus original 행을 `SHA256(base_id)` 오름차순으로
  512개**, DiReCT 행을 같은 규칙으로 512개. 임의 batch 금지.
- 측정 tensor: **`z`에 대한 gradient** (`dL/dz`), row별로 계산.
- 집계: **row별 L2 norm의 RMS**.
- 식: `lambda = RMS(dL_lang/dz) / RMS(dL_aux/dz)`, 유효숫자 2자리 반올림.
- 시점·공유: **seed 17 초기화에서 1회 계산해 세 seed 공통 적용**, 값과
  중간 RMS를 리포트에 기록. 학습 중 재계산 없음.

### 4. Control-first effect floor의 정확한 식

Primary metric = DiReCT validation symmetric alignment gap.

```text
floor = max( 2 * ( max_j g_j - min_j g_j ),  0.005 )
        # g_j = control arm seed j의 alignment gap, j in {17, 29, 43}
```

- 이 식 자체를 지금 동결한다 — control 결과를 본 뒤 spread 정의를 고르는
  일이 없도록 식이 유일해야 한다는 Discussion 3.5 지적의 수용이다.
- 절대 하한 `.005`의 근거: 역사적 full-SFT gap `+.0051`과 같은 자릿수로,
  그 이하의 개선은 기존 실패 수준과 구분되지 않는다.
- 판정은 floor 초과 AND category-cluster bootstrap CI 0 배제 AND 세 seed
  부호 일치(D5 구조 상속). DDXPlus generation 지표와 frozen-z probe
  프로토콜(Discussion 3.4)은 그대로.

**[판정 요청]** 위 네 값 + Discussion 3의 concept를 묶어 사람 승인에
올린다. 승인 시 DECISIONS.md에 D16으로 동결하고, 구현 순서는 (1) projector
+aux head architecture와 테스트, (2) control arm 3 seeds, (3) floor 계산·
기록, (4) proposed arm 3 seeds, (5) frozen-z probe와 제거 검증이다.
