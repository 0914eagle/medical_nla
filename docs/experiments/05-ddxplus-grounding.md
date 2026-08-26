# E5. DDXPlus activation grounding

## 질문

Medical-NLA의 자연어가 의료 지식 prior로 만든 그럴듯한 문장이 아니라 해당 사례
activation에 의존하는가?

## 통제

1. Matched vs hard-shuffled pair: 같은 diagnosis와 유사 길이 안에서 짝을 바꿈
2. Mean/zero activation: 언어 모델 prior 바닥
3. Activation swap: 사례 A metadata에 사례 B activation
4. Cue deletion: source prompt에서 evidence 하나 제거 후 재추출
5. Cue/value edit: DDXPlus가 정의한 attribute value만 변경
6. AV->text->AR identity round-trip

## 지표

- Own-pair score와 shuffled score의 paired gap
- 제거한 cue readout 감소율
- 건드리지 않은 cue retention
- Attribute/value exact 또는 semantic match
- Round-trip cosine, MSE, FVE
- Diagnosis만 같고 evidence가 다른 hard negative 구분

## 통과 조건

절대 threshold 하나로 정하지 않고 validation에서 effect-size와 bootstrap CI를 고정한다.
Test에서 pair gap과 cue-specific change가 0을 배제하고 untouched retention이 유지되어야
grounding 통과로 판정한다. AR absolute cosine만으로 faithfulness를 판정하지 않는다.

## 산출물

Table 3과 Figure 3. 실패하면 E4 결과를 좋은 explanation generation으로만 해석하고
E6 patching을 주 실험으로 진행하지 않는다.
