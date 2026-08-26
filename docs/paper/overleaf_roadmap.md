# Overleaf 이전 순서

## 지금 옮길 수 있는 부분

1. Introduction: CoT faithfulness 문제, closed probe와 open readout의 경계, 두 관문 설계
2. Related Work: CoT faithfulness, probing/SAE, NLA/AV-AR, medical explanation benchmark
3. Data: DiReCT 511 raw / 496 eligible, patient-disjoint split, DDXPlus 역할
4. Method: P0/P1/P2, Medical-NLA objectives, clinical alignment와 grounding 분리
5. Experimental protocol: E0-E7, model/decoding, split, evaluator, security

## 결과가 들어온 뒤 옮길 부분

- E2 완료 후 Table 1 확정. 공개 AV/AR 호환 때문에 HS32가 primary이며 HS16/24는 sensitivity
- E4 완료 후 Table 2와 Figure 2
- E5 완료 후 Table 3과 Figure 3, `activation-grounded` 문구 확정
- E6 완료 후 Table 4와 Figure 4, 성능 개선 문구 확정

## Appendix

- DiReCT official metric 구현과 unsmoothed sensitivity
- split 상세와 held-out PDD 목록
- layer/position sweep와 P1 leakage sensitivity
- 모든 prompt, decoding, LoRA hyperparameter
- 추가 seed, by-category/PDD 결과, human audit rubric
- 실패한 SFT-only/classifier-collapse 및 vanilla NLA 사례

## 제출 전 일관성 검사

- 모든 표에서 동일한 split 이름과 n을 사용한다.
- `faithful`은 Table 3 통과 뒤에만 사용한다.
- DiReCT 원문 또는 patient identifier가 포함되지 않았는지 확인한다.
- test set으로 layer, epoch, threshold를 선택하지 않는다.
- MCR은 checkpoint 선택에 쓰지 않고 frozen external test로만 사용한다.
