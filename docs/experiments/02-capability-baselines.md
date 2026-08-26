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

## Model selection

Primary layer와 probe regularization은 train/val_seen으로 정한다. Test_seen과
PDD-heldout은 선택 이후 한 번만 평가한다.

## 산출물

- Table 1
- primary layer 결정
- E3에서 사용할 vanilla checkpoint와 prompt 고정
