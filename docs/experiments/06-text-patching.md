# E6. Text patching

## 선행 조건

E5에서 full Medical-NLA가 사례별 pair와 evidence counterfactual을 읽는다는 증거가 있어야
한다. 그렇지 않으면 text edit는 activation patching이 아니라 외부 힌트 주입이 된다.

## 단계

1. Identity: 판독 text를 수정하지 않고 AR로 복원
2. No-op preservation: 원 answer, target logits, non-target distribution 보존
3. Dataset-native edit: DDXPlus evidence value 하나만 변경
4. Targeted state test: edited attribute probe/readout 변화
5. Behavior test: diagnosis likelihood와 final answer 변화

## 비교군

- No intervention
- Original activation
- CoT/plain prompt에 같은 text를 넣는 baseline
- Oracle activation from matched counterfactual
- Random or shuffled text edit

## 지표

No-op preservation, edited attribute success, target logit delta, off-target KL, diagnosis change,
wrong-to-right와 right-to-wrong을 모두 보고한다. 성능 개선은 net correction과 paired CI가
0을 배제할 때만 주장한다.
