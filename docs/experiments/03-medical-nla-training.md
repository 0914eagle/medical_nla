# E3. Medical-NLA training

## 질문

의료 설명 supervision이 vanilla NLA를 개선하면서도 분류기 붕괴와 activation 무시를
피할 수 있는가?

## 학습군

| Method | Clinical text | Reconstruction | Pair specificity |
|---|---:|---:|---:|
| Vanilla NLA | No | pretrained | No |
| SFT only | Yes | No | No |
| Reconstruction only | No | Yes | optional |
| Full Medical-NLA | Yes | Yes | Yes |

Clinical text는 DiReCT의 physician deduction structure에서 만든다. Activation은 P0를
주 입력으로 한다. Source-wrong 행에서 gold physician text를 activation의 현재 결론처럼
무조건 매핑하면 misalignment가 생기므로 다음을 분리한다.

- source-correct: clinical alignment supervision 가능
- source-wrong: decision fidelity 평가 및 activation-grounding 학습에 사용
- gold diagnosis를 강제로 말하게 하는 loss와 source-state를 읽는 loss를 혼합하지 않음

## 필수 통제

- Patient-disjoint split
- PDD-heldout 5개는 train에서 완전 제외
- 3 random seeds
- 동일 LoRA rank/target modules/token budget
- Early stopping은 val_seen
- 진단명 제거 또는 masking sensitivity

## 중단 기준

Seen 점수만 높고 PDD-heldout, hard shuffle gap, cue counterfactual이 낮으면 분류기 또는
문구 암기로 판정한다. 이 경우 모델 크기나 epoch를 늘리기 전에 objective를 수정한다.
