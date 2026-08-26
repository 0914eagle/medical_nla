# E2. Capability baselines

## 질문

생성 전 P0 activation에서 닫힌 진단 label과 열린 임상 내용을 각 방법이 얼마나 읽는가?

## 비교 방법

1. Source output head likelihood
2. Linear probe
3. Source CoT
4. Vanilla NLA/AV
5. P2 positive leakage control

## 평가

- PDD/category top-1, top-k, MRR
- Seen vs PDD-heldout
- Source answer와 gold를 분리한 decision fidelity
- Open observation/rationale는 DiReCT official evaluator의 호환 가능한 열
- P0/P1/P2 및 L16/L24/L32 sweep

Probe는 closed-label upper bound다. Open evidence text 열은 `N/A`이며 실패 0점으로
처리하지 않는다. Vanilla NLA의 자연어 점수가 낮아도 P0 activation에 정보가 없다는
결론을 바로 내리지 않고 probe와 output head를 같이 본다.

## 실행 상태

Test P0/L32 vanilla AV는 171/171행 생성 및 `<explanation>` parsing에 성공했고 빈 출력은
없었다. 출력 길이는 637--741자(중앙값 697, 평균 696.9)였다. 길이 안정성은 내용의
사례 특이성과 별개다. 길이 범위가 매우 좁으므로 exact/normalized 반복률과
own-case-versus-shuffled specificity를 우선 확인한 뒤 clinical alignment를 평가한다.

예비 same-category lexical derangement에서 164행의 own-prompt trigram containment와
shuffled-prompt containment가 모두 0.0013이었고 gap은 -0.0001이었다. 따라서 P0 vanilla
AV가 prompt의 사례 고유 표현을 그대로 복원한다는 증거는 없다. 이 검사는 paraphrase를
인정하지 않으므로 최종 실패 판정으로 쓰지 않고, 동일 claim extractor와 semantic
matcher를 이용한 own-versus-shuffled 평가의 필요성을 확인한 sanity check로만 둔다.

## Model selection

Primary layer와 probe regularization은 train/val_seen으로 정한다. Test_seen과
PDD-heldout은 선택 이후 한 번만 평가한다.

## 산출물

- Table 1
- primary layer 결정
- E3에서 사용할 vanilla checkpoint와 prompt 고정
