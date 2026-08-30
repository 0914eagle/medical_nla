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
