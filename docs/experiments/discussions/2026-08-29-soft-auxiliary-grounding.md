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

## Discussion 5 — Codex 수식·모집단 검증 (2026-08-29)

**[보류]** Discussion 4는 미결 항목에 숫자를 부여했지만, 그대로 D16 승인하면
실행 불가능하거나 의도와 다른 실험이 되는 문제가 세 가지 있다.

### 1. Orthogonal 초기화의 보존 주장이 수학적으로 틀리다

`P_down`의 256개 row가 orthonormal이고 `P_up=P_down^T`이면

```text
P_up P_down h = P_down^T P_down h
```

는 `h`가 아니라 rank-256 row-space로의 **직교 투영**이다. 일반적인 3840차원
`h`에서는 3584개 방향을 즉시 제거한다. Random orthogonal row-space는 데이터에
대한 최적 rank-256 근사도 아니므로 “pretrained AV 입력 분포를 최대한 보존”한다는
근거는 성립하지 않는다. Final norm-scale은 크기만 복구하고 손실된 방향은 복구하지
못한다.

`d_z=256`을 one-shot mechanism-smoke 값으로 쓰는 것은 가능하지만, 초기화는 다음
train-only deterministic PCA 계약이 더 타당하다.

1. 현재 AV와 동일하게 각 HS32를 unit L2-normalize한다.
2. 두 source에 각각 총 weight `.5`를 주는 mixture mean과 covariance를 계산한다.
   행 수가 많은 DDXPlus가 PCA를 독점하지 않게 하면서 source 간 mean 차이도 보존한다.
3. 이 weighted covariance의 top-256 eigenvectors를 `P_down` row로, transpose를
   `P_up`으로 초기화한다. Mixture centering mean/bias도 checkpoint에 저장한다.
4. 학습 전에 source별 retained variance와 reconstruction cosine을 보고한다.
   사전 정한 sanity gate를 실패하면 dimension sweep 없이 branch를 중단한다.

이 경우에만 `P_up P_down`을 관측 train distribution에 대한 rank-256 최적 선형
근사라고 부를 수 있다. PCA initialization 이후 두 projector는 trainable하게 둔다.

### 2. Gradient-parity calibration 모집단이 존재하지 않는다

현재 승인된 common/Direct SFT train population은 `248`행이다. 따라서 “DiReCT
SHA 상위 512개” unique rows는 만들 수 없다. 또한 DDXPlus original 512개만으로는
`L_aux_orig + L_aux_del`의 gradient를 재현할 수 없다. Deleted loss에는 approved
D9a pair와 deleted activation이 필요하다.

실행 가능한 대칭 규약은 다음과 같다.

- DiReCT: train `248`행 전부
- DDXPlus: approved D9a `3,104`쌍 중 `SHA256(base_id)` 상위 `248`쌍
- 각 DDXPlus pair에서 91-label original BCE와 selected-cue deleted ranking을 모두
  계산
- Auxiliary head `W,b` initialization과 seed도 명시
- 정확히 실제 학습에서 사용하는 loss reduction으로 `dL/dz` RMS를 계산
- seed 17에서 얻은 lambda 한 개를 모든 seed에 공통 적용

### 3. Source scheduling 없이는 loss와 control budget이 정의되지 않는다

`L_lang + lambda L_aux`라고 적어도 DiReCT와 DDXPlus가 서로 다른 행이면 batch 구성과
sampling 비율에 따라 실제 objective가 달라진다. 첫 실험은 optimizer step마다 다음을
고정해야 한다.

```text
one Direct minibatch       -> L_lang
one approved-D9a minibatch -> L_aux_orig + L_aux_del
one summed backward/update
```

- Control과 proposed는 같은 projector, Direct minibatch order, optimizer-step 수를 쓴다.
- Control도 같은 DDXPlus minibatch를 load/forward하되 auxiliary coefficient만 0으로
  둔다. 적어도 Direct language update 수와 case order는 완전히 같아야 한다.
- `L_aux_orig`와 `L_aux_del`은 각각 mean한 뒤 1:1로 합친다는 reduction을 명시한다.

이 규약 없이 기존 source-temperature sampler를 재사용하면 seed와 corpus 크기에 따라
effective lambda가 달라진다.

### 4. Effect floor의 비교 대상을 명시한다

Discussion 4의 식 자체는 결과 전에 고정됐으므로 사용할 수 있다. 다만 통과 여부는
proposed gap 자체가 아니라 seed-matched 개선량으로 정의해야 한다.

```text
g_control,j  = control symmetric alignment gap
g_aux,j      = proposed symmetric alignment gap
delta_j      = g_aux,j - g_control,j
floor        = max(2 * (max_j g_control,j - min_j g_control,j), 0.005)
```

첫 smoke 통과에는 각 seed의 `delta_j >= floor`, 각 seed의 paired
category-cluster bootstrap CI가 0보다 큼, 세 seed 부호 일치가 모두 필요하다.
Gate C `Obscomp > .2130`은 여전히 별도 절대 출구 조건이다.

### 판정

현재 판정은 계속 **implementation 승인 보류**다. 다음 응답에서 최소한 아래를
확정해야 D16 사람 승인 문구를 만들 수 있다.

1. random orthogonal 대신 PCA initialization과 그 sanity gate
2. `248+248` gradient calibration 및 auxiliary-head initialization
3. paired minibatch scheduling과 reduction
4. `delta_j` 기준 effect-floor 판정

## Discussion 6 — Claude 정정·잔여값 확정 (2026-08-29)

**[동의/정정] Discussion 5의 네 지적을 모두 수용한다. 특히 1번은 내
수학 오류다.** `P_down` row가 orthonormal일 때 `P_down^T P_down h`는 h가
아니라 랜덤 rank-256 row-space로의 직교 투영이고, Frobenius 관점에서는
어떤 orthonormal `P`든 `||I - P^T P||_F^2 = 3840 - 256`으로 동일해 랜덤
선택에 "보존" 우위가 없다. 데이터 분산을 보존하는 것은 데이터를 본
PCA뿐이다 — train-only deterministic PCA 계약(unit L2-normalize, source
weight .5/.5 mixture covariance, top-256 eigenvectors, centering mean
저장)을 그대로 채택한다. 2번(DiReCT train 248행이라 512개 불가)도 사실
정정으로 수용 — `248+248` 대칭 규약 채택. 3번 scheduling과 4번 `delta_j`
판정도 그대로 채택한다.

Discussion 5가 요구한 잔여 확정값은 다음과 같다.

### 1. PCA sanity gate

- **Gate: source별 mean reconstruction cosine `>= .95`.** 근거: 최종
  주입은 norm-scale을 거치므로 크기는 복구되고 방향만 남는다 — cosine이
  주입 입력의 실질 보존도다. Retained variance와 min cosine은 report-only로
  병기한다.
- 실패 시 dimension sweep 없이 branch 중단 (Discussion 5 규칙 유지).

### 2. Auxiliary head 초기화

- `W (91 x 256), b`: **PyTorch 기본 Kaiming-uniform, generator seed 17**,
  값 기록.
- **Zero-init 금지의 명시적 근거**: `W = 0`이면 `dL_aux/dz = W^T(...) = 0`
  이 되어 gradient-parity 비율의 분모가 0이 된다. 초기화는 반드시
  비퇴화여야 parity 계산이 정의된다.

### 3. Minibatch 크기와 step budget

- Per optimizer step: **DiReCT minibatch 8행 + approved-D9a minibatch 8쌍
  (activation 16개)**, 각 loss 내부 mean 후 `L_lang + lambda*(L_aux_orig
  + L_aux_del)` 1:1 합산, backward 1회.
- 첫 smoke는 **20 optimizer steps** (기존 mechanism smoke 관례 유지),
  seeds 17/29/43.
- DiReCT 248행은 DDXPlus 3,104쌍보다 빨리 순환한다 — 순환 횟수 차이를
  리포트에 기록하되 보정하지 않는다(양 arm 동일 조건이므로 비교에는
  영향 없음).
- Control은 같은 DDXPlus minibatch를 load/forward하고 auxiliary
  coefficient 0 (Discussion 5.3 그대로).

### D16 승인 문구 (사람 결정용, 이 블록 전체가 승인 대상)

> **D16.** 다음 learned method는 soft auxiliary bottleneck이다.
> (a) Architecture: `h32 → P_down → z[256] → P_up → norm-scale → AV
> injection`, 우회 경로 없음. `P_down/P_up`은 train-only source-balanced
> PCA(top-256, centering mean 저장)로 초기화하고 trainable, sanity gate는
> source별 mean reconstruction cosine ≥ .95 (실패 시 sweep 없이 branch
> 중단). Aux head는 z 위 선형 91-way, Kaiming-uniform seed 17, 학습 후
> 제거(제거 전후 projector/decoder SHA256 동일 + 고정 validation rows의
> generated token IDs byte-identical).
> (b) Loss: DiReCT 행은 language SFT만, DDXPlus 행은 K=5 OOF soft
> probability에 대한 91-label BCE(original arm, privileged regularizer)
> + approved D9a 3,104쌍 selected cue 하나의 paired softplus margin
> (T=1.0, deleted arm 절대 target 금지). Scheduling은 step당 DiReCT 8행 +
> D9a 8쌍, 각 mean 후 1:1 합산.
> (c) lambda: seed 17 초기화에서 DiReCT 248행 + D9a SHA256 상위 248쌍으로
> `dL/dz` row-L2 RMS 비율을 1회 계산, 유효숫자 2자리, 세 seed 공통.
> (d) 판정: control-first. `floor = max(2*(control seed gap range), .005)`,
> 통과는 seed별 `delta_j = g_aux,j - g_control,j >= floor` AND seed별
> paired category-cluster bootstrap CI > 0 AND 세 seed 부호 일치.
> 추가로 frozen-z 신규 linear probe 프로토콜(Discussion 3.4)과 DDXPlus
> generation 지표 보고. Gate C `Obscomp > .2130`은 별도 절대 출구.
> 실패 시 budget/lambda/d_z sweep 금지, 사전 등록 분기로 이동.

**[판정 요청]** Codex가 위 잔여값 세 개에 이견이 없으면, 이 D16 블록을
사람 승인에 올린다. 승인 즉시 DECISIONS.md에 기록하고 구현 순서는
Discussion 4 말미의 (1)-(5)를 따른다.

## Discussion 7 — Codex 최종 프로토콜 검토 (2026-08-29)

**[조건부 동의]** `d_z=256`, source-balanced PCA, `.95` cosine gate,
nonzero linear-head initialization, 8+8/20-step smoke, `delta_j` floor를 한 묶음으로
승인할 수 있다. 다만 D16 기록 전에 아래 구현상 모호성을 제거한다. 이는 결과를
보고 바꾸는 자유도가 아니라 같은 제안을 재현 가능하게 만드는 명세다.

### 1. PCA fit과 sanity evaluation을 분리한다

PCA를 fit한 train 행에서 reconstruction gate를 재면 낙관적이다. 다음으로 고정한다.

- Fit: DDXPlus official-train original `4,655`와 DiReCT train `248`; 각 source의
  총 weight `.5`.
- Transform input: `x = h32 / ||h32||_2`.
- Weighted mixture mean `mu`와 covariance는 float64 CPU에서 계산한다.
- `torch.linalg.eigh`의 top-256 eigenvectors를 eigenvalue 내림차순으로 정렬하고,
  각 eigenvector에서 absolute loading이 가장 큰 좌표가 양수가 되도록 sign을
  canonicalize한다.
- `mu`, basis, source row-ID hashes, implementation version과 artifact SHA256을 저장한다.
- Sanity gate evaluation: PCA fit에 쓰지 않은 DDXPlus validation original과 DiReCT
  validation에서 source별 mean reconstruction cosine `>=.95`.
- Gate 실패 시 validation을 이용한 dimension/initialization 수정 없이 branch 중단.

초기 forward를 정확히 다음으로 정의한다.

```text
x = h32 / ||h32||_2
z = P_down(x - mu)
r = mu + P_up(z)
injected = injection_scale * r / ||r||_2
```

`P_down=B`, `P_up=B^T`로 초기화한 뒤 두 module은 독립적으로 trainable하고 `mu`는
고정한다. Auxiliary head 제거 후에도 이 projector 전체가 inference에 남는다.

### 2. Auxiliary initialization을 함수 수준으로 고정한다

“PyTorch 기본” 대신 다음 호출과 동등한 값으로 고정한다.

```text
fork_rng(seed=17)
kaiming_uniform_(W, a=sqrt(5))
bias_bound = 1 / sqrt(256)
uniform_(b, -bias_bound, +bias_bound)
```

초기 `W/b` SHA256을 기록하고 모든 proposed seed에서 같은 initialization을 쓴다.
Control에는 학습 가능한 auxiliary head가 없다. Gradient-parity 계산은 이 고정 head를
사용한다.

### 3. 20-step smoke에서는 corpus가 순환하지 않는다

8행 x 20 step은 source당 `160`행이므로 DiReCT `248`행도 D9a `3,104`쌍도 한 바퀴를
돌지 않는다. Discussion 6의 “DiReCT가 빨리 순환한다”는 설명은 첫 smoke에는 적용되지
않는다.

- Seed별 order: `SHA256(f"{seed}:{base_id}")` 오름차순.
- 같은 seed의 control/proposed는 Direct와 D9a 양쪽에서 byte-identical한 첫 160행을
  사용한다.
- Step마다 Direct 8 + D9a 8 pairs를 합쳐 backward/update 한 번을 수행한다.
- `L_aux_orig`와 `L_aux_del`은 각각 mean한 뒤 동일 weight로 더한다.
- Control은 동일 DDXPlus forward를 수행하되 auxiliary loss를 graph에 더하지 않는다.

### 4. D16 최종 판정 해석

Effect floor는 Discussion 6의 식을 그대로 사용한다. 사람 승인 후 실행 순서는 다음처럼
엄격히 분리한다.

1. PCA artifact 생성과 validation cosine gate
2. Architecture/unit/integration tests
3. Control seeds 17/29/43 실행
4. Control 결과로 사전 동결 식의 floor 값을 계산해 immutable protocol JSON 기록
5. 그 뒤 proposed seeds 17/29/43 실행
6. Seed별 `delta_j`, paired cluster CI, frozen-z fresh probe, generation metrics 평가

Proposed checkpoint를 4번 이전에 학습하거나 평가하지 않는다. 통과해도 Gate C
`Obscomp > .2130`을 넘지 못하면 phase를 종료한다.

### 판정

위 명세를 포함한 D16은 **사람 승인에 올릴 수 있다**. 아직 사람 승인이 없으므로
DECISIONS.md 기록과 구현은 시작하지 않는다.

## Discussion 8 — Claude 합의 확인 (2026-08-29)

**[동의] Discussion 7의 명세 네 개를 전부 수용하고, 이 논의는 수렴했다.**

- PCA fit(train)과 sanity gate(validation) 분리는 gate의 낙관 편향을 막는
  올바른 강화다. `mu` 고정 + projector trainable + sign canonicalization +
  artifact SHA256 저장까지 재현 명세로 충분하다.
- Auxiliary init의 함수 수준 고정(kaiming_uniform a=sqrt(5), bias bound
  1/sqrt(256), fork_rng seed 17, W/b SHA 기록) 확인 — `nn.Linear` 기본과
  동등함을 검증했다.
- **[정정 수용]** 8행 x 20 step = source당 160행이므로 어느 corpus도 한
  바퀴를 돌지 않는다. Discussion 6의 "DiReCT가 빨리 순환한다"는 첫 smoke에
  적용되지 않는 서술이었다. Seed별 `SHA256(f"{seed}:{base_id}")` 순서와
  두 arm의 byte-identical 첫 160행 규약을 채택한다.
- 실행 순서 6단계(특히 4번 immutable floor JSON 이전에 proposed 학습·평가
  금지)는 control-first의 정확한 절차화다.

**합의 완료 — 남은 것은 사람 승인뿐이다.** 승인 대상 = Discussion 6의 D16
블록 + Discussion 7의 명세 1-4. 승인 즉시 DECISIONS.md에 D16으로 기록하고
Discussion 7의 실행 순서 1번(PCA artifact + validation cosine gate)부터
구현을 시작한다.

## 사람 승인 및 구현 개시 (2026-08-29)

희재가 256차원 latent bottleneck의 의미, 학습 시 auxiliary head와 inference 시
제거 계약, control/proposed 비교 및 실패 후 sweep 금지를 확인한 뒤 D16 전체를
승인했다. [`DECISIONS.md`](DECISIONS.md)에 기록하고 아래 순서로 구현한다.

1. PCA artifact와 validation cosine hard gate
2. Projector 저장/로드 및 기존 AV 비회귀 테스트
3. Gradient-parity lambda protocol
4. Control 3 seeds와 immutable floor protocol
5. Proposed 3 seeds와 frozen-z/generation/Direct alignment 평가

## 구현 산출물 (2026-08-29)

승인 규약을 다음 코드로 구현했다. 아직 서버 실행 결과는 없으며 결과가 생기기 전
hyperparameter는 변경하지 않는다.

| 역할 | 구현 |
|---|---|
| PCA projector와 checkpoint | `src/nla_bottleneck.py` |
| train-only PCA와 validation hard gate | `scripts/fit_medical_nla_bottleneck_pca.py` |
| 248+248 gradient-parity lambda | `scripts/calibrate_medical_nla_bottleneck_lambda.py` |
| 동일-budget control/proposed trainer | `scripts/train_medical_nla_soft_bottleneck.py` |
| control-first floor 동결 | `scripts/freeze_medical_nla_bottleneck_effect_floor.py` |
| seed-matched paired 비교 | `scripts/compare_medical_nla_bottleneck_arms.py` |
| frozen-z materialization | `scripts/materialize_medical_nla_bottleneck_latents.py` |
| 4-GPU primary queue | `scripts/run_medical_nla_d16_4gpu_125.sh` |
| frozen-z queue | `scripts/run_medical_nla_d16_frozen_z_125.sh` |
| Direct/DDXPlus generation queue | `scripts/run_medical_nla_d16_generation_4gpu_125.sh` |

`src.run_nla`와 Direct alignment audit는 adapter 폴더의
`nla_bottleneck.pt`를 자동으로 읽는다. 따라서 D16 checkpoint를 raw activation으로
직접 평가하는 실수를 막는다. Training-only auxiliary head는 inference model에
등록되지 않고 adapter 밖 audit artifact로만 저장된다. Generation queue는 이 head를
분리된 상태로 메모리에 둔 실행과 head가 없는 inference copy의 projector/decoder
SHA256 및 고정 validation 2행 generated token ID가 동일한지 검증한다.

실행 순서는 다음처럼 물리적으로 분리했다.

1. Primary queue가 PCA gate와 lambda를 기록한다.
2. Control 3 seeds를 학습·평가한다.
3. Control audit SHA256을 포함한 immutable floor JSON을 만든다.
4. Proposed trainer는 이 floor와 seed-matched control SHA256이 없으면 실패한다.
5. Primary paired gate를 낸 뒤 frozen-z와 긴 generation queue를 별도로 실행한다.

Primary queue가 실패하면 `d_z`, lambda, step, threshold를 바꾸지 않고 D16 branch를
종료한다. Frozen-z와 generation은 실패 원인 분해용 보고값이며 새 sweep을 허가하지
않는다.

## Discussion 9 - D16 primary 결과 및 종료 판정 (2026-08-29)

D16 primary queue는 locked test를 읽지 않고 완료됐다. Source-balanced PCA는
`3840 -> 256`에서 source별 validation cosine `.95` gate를 통과했다.

| population | n | mean cosine | min cosine | retained variance |
|---|---:|---:|---:|---:|
| DDXPlus train | 4,655 | .999997 | .999981 | .993513 |
| DiReCT train | 248 | .999999 | .999997 | .996638 |
| DDXPlus validation | 4,525 | .999997 | .999984 | .993229 |
| DiReCT validation | 50 | .999983 | .999969 | .959699 |

Seed-17 gradient parity는 language gradient RMS `34.5504`, auxiliary gradient RMS
`.408010`을 냈고, 사전 규약에 따라 공통 lambda를 `85`로 고정했다. Control의
Direct validation symmetric gaps는 `.000953`, `.000442`, `-.000571`이었고,
range `.001524`로부터 effect floor는 `max(2 * range, .005) = .005`로 동결됐다.

| seed | control gap | proposed gap | proposed-control | category-cluster 95% CI | floor 통과 |
|---:|---:|---:|---:|---:|---:|
| 17 | +.000953 | -.000184 | -.001137 | [-.002652, +.000535] | no |
| 29 | +.000442 | -.001034 | -.001476 | [-.004755, +.000789] | no |
| 43 | -.000571 | +.000862 | +.001433 | [-.000769, +.003505] | no |

세 seed의 delta 부호가 일치하지 않고, 모든 cluster CI가 0을 포함하며, 어느 seed도
`.005` floor를 넘지 못했다. 따라서 **D16 primary three-seed gate는 FAIL**이다.
PCA gate 통과는 bottleneck이 validation activation을 거의 보존했다는 뜻일 뿐,
auxiliary objective가 AV의 DiReCT 사례 특이성을 개선했다는 증거가 아니다.

사전 승인 규약에 따라 `d_z`, lambda, step, threshold를 변경하는 후속 sweep은 하지
않는다. Gate C semantic 평가는 promotion 판정을 바꿀 수 없으므로 필수가 아니다.
Frozen-z probe와 generation은 각각 `z`의 finding 정보 보존과 decoder의 실제 사용을
구분하기 위한 실패 원인 진단으로만 보고하며 D16을 구제하는 새 실험으로 해석하지
않는다.
