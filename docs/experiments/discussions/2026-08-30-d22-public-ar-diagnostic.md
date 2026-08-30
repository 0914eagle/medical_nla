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

| arm | own cosine | shuffled cosine | 판정 |
|---|---:|---:|---|
| Structured reader (양성 대조 1) | .9765 | .9765 | gap ≈ 0 |
| Source CoT (양성 대조 2) | .9835 | .9834 | CI가 0 포함 |
| Vanilla (report-only) | .9962 | — | 전 arm 중 최고 |

**사전 등록 기준에 따른 판정: 공개 AR 불인정.** 양성 대조 두 arm 모두
matched-over-shuffled CI 하한이 0을 넘지 못했다. 사례 특이성이 보장된 reader
텍스트(finding F1 .96)조차 구별하지 못하므로, 실패는 텍스트가 아니라 공개 AR의
의료 CoT-P0/HS32 distribution mismatch다.

세 가지 구분 (사전 등록 문구 그대로 적용):

1. SFT 출력의 판독 실패를 이 실험이 **추가 확정한 것은 아니다.**
2. 확정된 것은 **공개 AR가 이 분포의 환자별 차이를 측정하지 못한다**는 것이다.
3. AV-AR 접근의 실패가 아니라 **Medical-AR 선행 학습이라는 전제**가 생긴 것이다.

가장 특이성이 약한 Vanilla가 .9962로 최고라는 사실 자체가 mean-direction 설명의
증거이자, 이 cosine을 그대로 GRPO reward로 썼을 때 생길 hacking(평균 방향 맞추기)의
실증적 예고다. 이 진단은 그 비싼 실패를 사전에 차단했다.

## Claude 검토 (2026-08-30)

### 검증 두 건

1. **`model.norm.weight MISSING`은 정상 — 확인 완료.** 공식
   [`nla_inference.py`](https://github.com/kitft/natural_language_autoencoders/blob/main/nla_inference.py)를
   직접 확인했다: AR는 최종 LayerNorm을 의도적으로 `Identity`로 교체하고 lm_head를
   제거한 뒤 `Linear(d,d)` value head를 쓴다. 우리 로더는 공식 규약을 따르며 이번
   결과는 로딩 결함이 아니다.
2. **Anisotropy 인용 출처 교체 필요.** 결과 해석에 인용된
   `sidaraslanoglu.com/papers/nla-autoencoders.pdf`는 접근 불가이고 웹 검색에서도
   저자·문서의 독립 흔적이 없다. **"Gemma HS32 평균 cosine ≈ .975"라는 구체 수치는
   검증 불가이므로 동결 문서와 논문에 이 링크를 인용하지 않는다.** 일반 현상은 정식
   문헌으로 충분하다 — transformer hidden state의 anisotropy가 무관한 state 간
   cosine을 부풀리고 mean-centering이 표준 처방이라는 것
   ([arXiv 2306.07656](https://arxiv.org/abs/2306.07656),
   [arXiv 2401.12143](https://arxiv.org/abs/2401.12143), Ethayarajh 2019).
   아래 geometry audit A1이 그 수치를 **우리 데이터의 실측값으로 대체**한다.

### Geometry audit 사전 등록 (CPU-only, 기존 160개 결과 재사용)

비싼 학습 전에 다음 다섯 항목을 계산한다. 신규 GPU 없음, locked test 없음.

| id | 항목 | 무엇을 분리하나 |
|---|---|---|
| A1 | activation 쌍별 cosine baseline (same-diagnosis / different-diagnosis) | 우리 분포의 anisotropy 바닥 실측 — 외부 인용 대체 |
| A2 | 평균 제거 centered cosine의 matched-over-shuffled gap | anisotropy 공통 방향 제거 후 사례 신호 |
| A3 | empirical-mean 예측 대비 FVE | "평균만 맞추기"와 실제 복원의 분리 |
| A4 | same-diagnosis donor vs different-diagnosis donor gap | 구별 난이도의 계층 |
| A5 | 후보 activation 중 own-case retrieval rank | threshold 없는 최강 판별 시험 |

보고 규칙: generator별(reader/CoT/vanilla) 분리, 진단 cluster bootstrap CI, 기존
관례의 aggregate-only(DiReCT 원문은 restricted 경로 밖 반출 금지).

**판정 기준 (실행 전 동결):**

- **부분 인정**: 양성 대조 arm에서 A2 centered matched-over-shuffled CI 하한 > 0
  **또는** A5 own-case retrieval이 chance 대비 CI로 우월 → 공개 AR의 제한적 역할
  (초기화·비교 대상) 재론.
- **폐기**: 둘 다 실패 → 공개 AR를 의료 경로에서 제외하고 Medical-AR
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

NLA 자체를 도메인 특화 fine-tuning한 발표 연구는 없다 — 그 공백이 이 논문의
자리다. 인접 계열과 우리 논쟁에 주는 함의:

| 계열 | 대표 | 함의 |
|---|---|---|
| Unsupervised AV-AR RL (원 NLA) | [Anthropic NLA 2026](https://transformer-circuits.pub/2026/nla/), [kitft 구현](https://github.com/kitft/natural_language_autoencoders) | D22가 따르는 canonical 경로 |
| Supervised activation decoder | [LatentQA/LIT, ICLR 2026](https://arxiv.org/abs/2412.08686) | (activation, QA) SFT — 우리 supervised 시도의 방법론적 친척, 단 counterfactual gate 없음 |
| 대규모·고다양성 supervised 확장 | [Activation Oracles](https://arxiv.org/abs/2512.15674) | "양과 다양성 스케일링만으로 개선" — 우리 diversity limitation의 **외부 근거** |
| 도메인 특화 decoder 선례 | STATEWITNESS ([arXiv 2606.17478](https://arxiv.org/abs/2606.17478)) | vertical 특화 학습이 가능하다는 선례 (비의료) |
| 무학습 inference-time | [Patchscopes](https://arxiv.org/abs/2401.06102), SelfIE | related work 절용, 튜닝 선행 아님 |
| 의료 적용 | [BlakeMasters preprint](https://github.com/BlakeMasters/medical_language_autoencoders) | 유일 사례, vanilla 평가만. cosine .828 vs heuristic alignment 5.5% — **cosine 단독 지표 위험의 실증** → 우리 "cosine은 reward만, gate는 semantic" 규칙의 근거 |

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
