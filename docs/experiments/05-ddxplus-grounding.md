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

## Population split

DDXPlus를 평가에만 쓸지 grounding 학습에도 쓸지를 E3 전에 고정한다. Primary transfer
설정은 DiReCT-only adaptation 뒤 DDXPlus를 cross-corpus test로 사용하는 것이다. DDXPlus
counterfactual을 학습에 쓰는 보조 설정에서는 base case, cue/value 조합과 donor pool을
train/validation/test 사이에 분리한다. Test pair나 test cue/value로 prompt, reward weight,
shuffle 난이도를 선택하지 않는다. 학습에 쓴 DDXPlus 행을 같은 Table 3 분모에 넣지 않는다.

공개 AV/AR가 hidden-state extraction index 32용이므로 primary grounding과 round-trip은
**CoT-P0/HS32**로 고정한다. 여기서 CoT-P0는 CoT instruction을 포함한 chat prompt의
경계에서, 모델이 reasoning을 생성하기 전 마지막 hidden state다. DiReCT Medical-NLA도
동일한 CoT-P0로 학습되므로 DDXPlus primary에서 instruction distribution을 맞춘다.
Direct-P0는 같은 validation base case의 paired instruction-sensitivity 통제로만 사용하며
primary 분모와 locked test에는 넣지 않는다. HS16/HS24를 같은 decoder에 넣는 값은
representation 차이와 decoder distribution shift를 분리하지 못해 appendix
sensitivity로만 다룬다.

정본 데이터는 공식 `validate.csv`와 `test.csv`에서 각각 독립적으로 진단당 100건을
reservoir sampling한다(seed 17; 최대 49 diagnosis). 적격 조건은 clean rendered cue가
3개 이상이고 prompt에 gold diagnosis/alias가 직접 나오지 않는 것이다. 따라서 계획 분모는
validation 4,900과 locked test 4,900이며, 실제 값은 빌더가 49개 diagnosis의 quota를 모두
채웠을 때만 확정한다. 세부 생성 규칙과 실행 명령은
[`docs/data/ddxplus_e5_canonical.md`](../data/ddxplus_e5_canonical.md)에 고정한다.

Cue deletion은 모든 적격 case에 만들고, value edit은 같은 `evidence_id`에 release가
명시한 다른 value가 있으며 정상 문장으로 렌더링되는 case에만 만든다. 전역 cue 어휘에서
무관한 문장을 뽑는 과거 swap은 사용하지 않는다. Validation mean activation만 mean-control로
사용하고 test mean은 계산하지 않는다.

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
