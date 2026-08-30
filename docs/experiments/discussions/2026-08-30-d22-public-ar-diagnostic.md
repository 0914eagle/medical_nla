# D22: 공개 AR 의료 분포 진단

## 질문

D10/D20은 surrogate cue objective의 실패를 확정했지만, 원 NLA의 핵심인
text-to-activation AR reconstruction을 사용하지 않았다. D22의 첫 단계는 공개 HS32 AR가
의료 설명을 자기 activation과 같은 진단의 다른 사례 activation 사이에서 구별할 수 있는지
validation에서 확인한다.

## 사전 고정

- 공개 AR: `kitft/nla-gemma3-12b-L32-ar`
- 위치: CoT-P0, HS32
- locked test: 읽지 않음
- control: 같은 diagnosis stratum의 다른 `base_id`, SHA256 결정론적 순환 배정
- 같은 reconstructed vector를 own/control activation에 각각 비교하므로 text length는 두
  cosine에 동일하게 작용한다. arm별 mean word count도 함께 보고한다.
- restricted DiReCT 원문과 reconstructed vector/row score는
  `/data1/heejae/restricted/direct/e4` 아래에만 둔다.

## 양성 대조

1. DDXPlus structured reader validation text: frozen probe가 렌더링했고 finding F1 `.9607`인
   사례 특이적 텍스트
2. DiReCT Source CoT validation text

두 arm 모두 matched-over-shuffled mean cosine gap의 row-bootstrap 95% CI 하한이 0보다
커야 공개 AR를 의료 분포의 측정기로 인정한다. 실패는 텍스트나 activation에 임상 정보가
없다는 뜻이 아니라 공개 AR의 distribution mismatch를 뜻하며 Medical-AR adaptation을 먼저
요구한다.

Vanilla와 기존 SFT 5종은 report-only다. Reconstruction cosine은 학습 reward 후보일 뿐
Medical-NLA promotion metric이 아니며, 이후에도 semantic alignment와 counterfactual
specificity gate를 대체하지 않는다.

## 진단 결과 (2026-08-30 실행, 사람 전달 수치)

| arm | n | own cosine | shuffled cosine | 판정 |
|---|---:|---:|---:|---|
| Structured reader (양성 대조 1) | 20 | .9765 | .9765 | gap ≈ 0 |
| Source CoT (양성 대조 2) | 20 | .9835 | .9834 | CI가 0 포함 |
| Vanilla (report-only) | 20 | .9962 | .9961 | gap +.0001; 전 arm 중 최고 |

**사전 등록 기준에 따른 판정: 공개 AR 불인정.** 양성 대조 두 arm 모두
matched-over-shuffled CI 하한이 0을 넘지 못했다. 사례 특이성이 보장된 reader
텍스트(finding F1 .96)조차 구별하지 못하므로, 실패는 텍스트가 아니라 공개 AR의
의료 CoT-P0/HS32 distribution mismatch다.

세 가지 구분 (사전 등록 문구 그대로 적용):

1. SFT 출력의 판독 실패를 이 실험이 **추가 확정한 것은 아니다.**
2. 확정된 것은 **공개 AR가 이 분포의 환자별 차이를 측정하지 못한다**는 것이다.
3. AV-AR 접근의 실패가 아니라 **Medical-AR 선행 학습이라는 전제**가 생긴 것이다.

가장 특이성이 약한 Vanilla가 .9962로 최고라는 사실은 mean-direction
설명과 **일치하는 현상**이지만, 그 자체로 확정 증거는 아니다. 아래 A1–A5가
우리 activation 분포에서 평균 방향 영향을 실측한다. 다만 이 cosine을 검증 없이
GRPO reward로 쓰면 평균 방향 맞추기가 고보상 해가 될 수 있음을 예고한다.

## Claude 검토 (2026-08-30)

### 검증 두 건

1. **`model.norm.weight MISSING`은 정상 — 확인 완료.** 공식
   [`nla_inference.py`](https://github.com/kitft/natural_language_autoencoders/blob/main/nla_inference.py)를
   직접 확인했다: AR는 최종 LayerNorm을 의도적으로 `Identity`로 교체하고 lm_head를
   제거한 뒤 `Linear(d,d)` value head를 쓴다. 우리 로더는 공식 규약을 따르며 이번
   결과는 로딩 결함이 아니다.
2. **Anisotropy 인용 출처 교체.** 결과 해석에 인용된
   [`sidaraslanoglu.com` 분석](https://sidaraslanoglu.com/papers/nla-autoencoders.pdf)은
   현재 접근 가능하지만 self-hosted 비심사 자료다. `Gemma HS32 평균 cosine ≈ .975`는
   참고 수치로만 두고 동결 논문의 핵심 근거로 사용하지 않는다. 일반 현상은
   transformer hidden state의 anisotropy가 무관한 state 간 cosine을 부풀린다는
   [EACL 2024 최종판](https://arxiv.org/abs/2401.12143)과 Ethayarajh 2019로
   근거를 대고, 구체 수치는 A1의 자체 실측값으로 대체한다.

### Geometry audit 사전 등록 (소규모 AR 재실행 + CPU 집계)

비싼 학습 전에 다음 다섯 항목을 계산한다. locked test는 읽지 않는다. 현재
`private_scores.jsonl`은 cosine scalar만 저장했으므로 A2/A3/A5에 필요한 reconstructed
vector가 없다. 기존 160개 text에 대한 AR forward를 한 번 재실행해 row별 vector를
restricted 경로에 저장한 뒤 집계는 CPU에서 한다. A1/A4는 기존 activation과 scalar
결과로 바로 계산 가능하다.

| id | 항목 | 무엇을 분리하나 |
|---|---|---|
| A1 | activation 쌍별 cosine baseline (same-diagnosis / different-diagnosis) | 우리 분포의 anisotropy 바닥 실측 — 외부 인용 대체 |
| A2 | train activation 공통 방향 투영 제거 후 matched-over-shuffled gap | anisotropy 공통 방향 제거 후 사례 신호 |
| A3 | train empirical-mean 예측 대비 direction-normalized FVE | "평균만 맞추기"와 실제 복원의 분리 |
| A4 | same-diagnosis donor vs different-diagnosis donor gap | 구별 난이도의 계층 |
| A5 | 후보 activation 중 own-case retrieval rank | threshold 없는 최강 판별 시험 |

보고 규칙: generator별(reader/CoT/vanilla) 분리, 진단 cluster bootstrap CI, 기존
관례의 aggregate-only(DiReCT 원문은 restricted 경로 밖 반출 금지).

- DDXPlus 공통 방향/평균은 official-train original CoT-P0/HS32에서만 계산한다.
- DiReCT 공통 방향/평균은 train 248 CoT-P0/HS32에서만 계산한다.
- A2는 train mean direction μ에 대해 gold/reconstruction 모두에서
  `proj_μ(v) = (v·μ)μ`를 뺀 잔차 방향 간 cosine으로 고정한다.
- A3는 unit-normalized gold/reconstruction의 squared error를 쓰고,
  `1 - sum(model_error) / sum(train_mean_predictor_error)`로 계산한다.
- A5는 각 reconstruction을 같은 diagnosis stratum의 모든 validation original
  activation과 비교해 top-1, MRR, median rank, candidate count를 보고한다. 후보 2개
  미만 stratum은 제외하고 cosine tie는 average rank로 처리한다.

**판정 기준 (실행 전 동결):**

- **제한적 진단 도구 인정**: 양성 대조 arm에서 A2 CI 하한 > 0
  **또는** A5 own-case retrieval이 chance 대비 CI로 우월 → 공개 AR를 초기화·비교
  대상으로만 재론.
- **AV reward 인정**: 양성 대조 두 arm에서 A2와 A5가 모두 통과하고 A3 FVE > 0.
  이 조건 전에는 raw/centered cosine을 AV 학습 reward로 쓰지 않는다.
- **폐기**: A2/A5가 둘 다 실패 → 공개 AR를 의료 경로에서 제외하고 Medical-AR
  (text → CoT-P0/HS32) 학습을 전제로 확정.
- 실행 후 기준 조정은 무효.

### Medical-AR 학습 위치 제약 (사전 결정 필요)

"DDXPlus 4,655 + DiReCT 248" 학습안은 그대로 실행하면 **DiReCT RunPod 반출 금지
규칙과 충돌**한다. 선택지를 사람 결정으로 고정해야 한다:

1. Pod에서 DDXPlus-only 1차 학습 → 125에서 DiReCT 248 2차 adaptation
2. 전체를 125에서 학습 — AR는 LoRA + `Linear(d,d)` value head 회귀라 AV 생성
   학습보다 가볍고 2×4090 가능성이 있다 (실측 전 보장 아님)

어느 쪽이든 D22 본 학습 사전 등록에 명시한다.

### 선행 연구 지형 (관련 문서 통합, 2026-08-30 검색 검증)

현재 검색에서 **peer-reviewed medical-specific AV–AR 학습 연구**는 확인되지
않았다. 코드 도메인에 데이터를 교체해 AR/AV를 재학습하는 비심사 구현은
존재하지만, 성공이 확립된 발표 결과로 보지는 않는다. 인접 계열과 우리 논쟁에
주는 함의:

| 계열 | 대표 | 함의 |
|---|---|---|
| Unsupervised AV-AR RL (원 NLA) | [Anthropic NLA 2026](https://transformer-circuits.pub/2026/nla/), [kitft 구현](https://github.com/kitft/natural_language_autoencoders) | D22가 따르는 canonical 경로 |
| Supervised activation decoder | [LatentQA/LIT, ICLR 2026](https://arxiv.org/abs/2412.08686) | (activation, QA) SFT — 우리 supervised 시도의 방법론적 친척, 단 counterfactual gate 없음 |
| 대규모·고다양성 supervised 확장 | [Activation Oracles](https://arxiv.org/abs/2512.15674) | "양과 다양성 스케일링만으로 개선" — 우리 diversity limitation의 **외부 근거** |
| 도메인 특화 decoder 선례 | STATEWITNESS ([arXiv 2606.17478](https://arxiv.org/abs/2606.17478)) | vertical 특화 학습이 가능하다는 선례 (비의료) |
| 무학습 inference-time | [Patchscopes](https://arxiv.org/abs/2401.06102), SelfIE | related work 절용, 튜닝 선행 아님 |
| 의료 적용 | [BlakeMasters preprint](https://github.com/BlakeMasters/medical_language_autoencoders) | 유일 사례, vanilla 평가만. cosine .828 vs heuristic alignment 5.5% — **cosine 단독 지표 위험의 실증** → 우리 "cosine은 reward만, gate는 semantic" 규칙의 근거 |

### NLA-KTH에서 참고할 도메인 재학습 레시피

[`mohamedibrahim26/nla-kth`](https://github.com/mohamedibrahim26/nla-kth)는 원 NLA를
소규모로 재구현한 비심사 프로젝트다. 일반 실험은 `Qwen2.5-0.5B-Instruct`의
layer 16 마지막 token activation 8,000개를 추출하고, 그중 4,000개에
`Qwen2.5-3B-Instruct` teacher summary를 붙였다.

1. **AR SFT**: summary를 읽은 마지막 hidden state에 `Linear(d,d)` head를 붙여
   dimension-wise z-score activation을 예측했다. attention/MLP LoRA rank 32, batch 16,
   learning rate `2e-4`, 12 epochs이며 validation FVE 최고는 .050(epoch 2)이었다.
2. **AV SFT**: unit-normalized activation을 learned linear projection으로 token embedding 크기에
   맞춰 soft activation token으로 넣고 teacher summary CE를 학습했다. batch 4,
   learning rate `2e-4`, 3 epochs이었다.
3. **GRPO**: activation당 설명 후보 `G=8`을 생성하고 frozen AR의 `-MSE`를 reward로
   사용했다. KL 계수는 .05였다. 그러나 보고된 FVE는 .0070 → .0071로
   개선이 +.0001에 그쳐, 약한 AR/warm start 위에서 RL을 시작하는 것의 위험을 보였다.

코드 확장은 `Qwen2.5-Coder-0.5B` + CodeSearchNet Python에 AST로 off-by-one,
wrong operator/variable/comparison 변이를 만들고, `Qwen2.5-Coder-3B-Instruct`가 teacher
summary를 작성한 뒤 같은 AR/AV trainer를 재사용한다. 공개 pipeline의 코드 분기는
주로 AR/AV SFT와 correct/buggy FVE·keyword shift 평가며, domain GRPO 성공을
실증한 결과로 보지 않는다.

우리가 가져올 규칙은 (a) Medical-AR에 activation dimension 표준화를 검토하고,
(b) oracle/reader text validation FVE로 AR을 먼저 선택하며, (c) 양성 대조 AR가 약하면
AV RL을 열지 않는 것이다. 이 구현의 4,000 teacher summary에서도 AR validation
FVE가 .050이었다는 점은 우리의 DDXPlus 4,655를 최종 학습량이 아닌 smoke로
보고, official train에서 47k–100k로 확장해야 할 근거다.

### Patchscopes: 학습 없는 native-layer decoder baseline

[`Patchscopes`](https://arxiv.org/abs/2401.06102)는 NLA가 아니다. AV처럼 HS32
activation을 layer-0 token embedding으로 변환해 읽게 하지 않고, source prompt에서 추출한
hidden state를 별도 target prompt의 placeholder 위치, 같은 native layer에 직접 덩어쓴다.

```text
source clinical prompt --Gemma layers 0..32--> h_source

target inspection prompt --Gemma layers 0..32--> h_placeholder
                                                |
                                                +-- replace with h_source
                                                    |
                                                    +-- layers 33..47 --> answer
```

같은 Gemma-3-12B의 HS32를 같은 Gemma-3-12B HS32에 patch하는 1차 baseline은 가중치
학습이나 공간 변환이 필요 없다. 해석 prompt는 `Which clinical findings are represented?`,
`Is pulmonary embolism represented?`, `What value is represented for chest-pain location?`처럼
질문을 교체할 수 있다. 다른 model family/width로 cross-model patching하는 경우에만
affine mapping 학습을 검토한다.

장점은 우리 AV SFT가 학습해야 했던 `HS32 -> layer-0 embedding`의 domain/position
변환을 피하고, activation을 원래 존재하던 layer에 직접 넣는다는 점이다. 단,
target prompt와 model prior만으로 질환 전형을 만들 수 있으므로 다음 대조를 모두
동결한다.

| patch 조건 | 질문 |
|---|---|
| real activation | 실제 환자 정보 |
| same-diagnosis shuffled | 환자별 특이성 |
| train-mean activation | 평균 방향 baseline |
| no patch | prompt/model prior |
| cue-deleted activation | changed finding 선택적 제거 |
| value-edited activation | old/new value 전환 |

Patchscope는 AR/reconstruction 병목 없이 activation의 정보가 상위 layer에서 자연어로
추출 가능한지 보는 **비학습 baseline**이다. 성공하면 activation은 읽힐 수 있지만
기존 AV injection/training이 병목이었다는 원인 분리가 가능하다. probe는 성공하고
Patchscope도 실패하면, 자연어 decoder가 환자별 신호를 발화하는 부분이 핵심
병목으로 남는다.

**채점기 동결 (실행 전 고정)**: Patchscope 출력도 open text이므로 채점 규칙을
사전에 고정한다. 이미 G1–G4를 통과한 frozen semantic mapper 프로토콜
(SHA `12e4500f...`)을 validation 용도로 사용한다 — cache key가 protocol-bound라
새 prompt/조건 조합에도 stale 재사용이 없다. 빠른 screen이 필요하면 lexical
매칭을 보조로 병기하되, 판정은 mapper 결과로만 한다. 실행 후 채점기 교체나
기준 조정은 무효.

### D22 다음 실행 순서

1. 기존 160 text의 reconstruction vector를 저장하는 소규모 AR 재실행.
2. A1–A5 geometry audit 실행.
3. 독립 비학습 baseline으로 DDXPlus validation 50-case Patchscope smoke:
   original/deletion/value-edit x real/shuffled/mean/no-patch.
4. 공개 AR가 reward gate를 통과하지 못하면 DDXPlus 4,655 Medical-AR pipeline smoke.
5. smoke의 oracle/reader FVE가 양수이면 official train 47k–100k로 확대 —
   **단, 이 단계는 자동 진행이 아니다.** 47k–100k 규모는 source CoT 생성과
   activation 추출부터 새로 하는 대형 GPU 작업이므로, smoke 통과 후 별도
   사전 등록(비용 추정, 실행 위치 — DDXPlus-only면 pod 가능 — 고정, gate 수치
   동결)과 사람 비용 승인을 거쳐야 연다.
6. Medical-AR positive control이 통과한 후에만 Medical-AV SFT → AR-reward optimization을 연다.

순서 1–3은 validation-only 소규모 실행이라 사람 승인 후 즉시 가능하다. 4 이후는
각 단계의 gate 통과 + 해당 단계 사전 등록이 선행 조건이다.

검증된 경로는 둘이다: ① AR-reward RL (원 NLA), ② 대규모 diverse supervised
(LatentQA/AO). D22가 ①을 택한 이유는 공개 checkpoint 호환과 원 방법 재현성이며,
②의 존재와 미실행 사유를 사전 기재해 "왜 AO 방식은 안 했나" 심사 질문을 미리
닫는다. 우리 8건 실패는 어느 경로의 검증된 작동점에서도 실행된 것이 아니었다는
것이 D19–D22 서사의 정확한 위치다.

### 결정 구조 (사람 승인 대기 정리)

1. **D19 승인**: D10 budget calibration FAIL, unanchored 계열 종결.
2. **D21 축소 승인**: D20 FAIL + surrogate 계열(SFT/ranking/anchor/bottleneck)
   종결만. 생성형 전체 종료·주표 행 영구 제외는 승인하지 않음 — 기존 조건부
   규칙(gate 통과 시에만 행 추가) 유지.
3. **D22 개방**: 이 문서의 진단은 실행 완료(공개 AR 불인정). 다음 단계는
   geometry audit(위 동결 기준) → 결과에 따라 Medical-AR 학습 사전 등록.
4. Baseline 논문 트랙(DiReCT locked batch)은 D22와 독립적으로 진행 가능 —
   일정 결정(선제출 vs D22 대기)은 별도 사람 결정.
