# 논문 구성

## 1. Introduction

의료 LLM 설명은 임상적으로 그럴듯한 것과 실제 내부 계산을 반영하는 것이 다르다는
문제에서 시작한다. CoT 비충실성은 선행 연구가 이미 보였지만, 내부 도구의 대안도
완전하지 않다. Probe는 닫힌 label에서는 강하지만 하나의 판독기로 사례별 관찰과
관계를 열린 자연어로 설명하지 않는다. NLA는 열린 판독을 제공할 가능성이 있지만
의료 문구를 지어내거나 분류기로 붕괴할 수 있다.

기여는 세 가지다. 첫째, DiReCT에서 CoT와 activation readout을 같은 의사 주석 기준으로
평가한다. 둘째, clinical alignment와 activation grounding을 분리하고 둘 다 통과해야
faithful이라는 검증 원칙을 제시한다. 셋째, 통과한 판독만 text patching에 사용해
설명과 행동 개선 사이의 연결을 시험한다.

## 2. Related Work

1. CoT faithfulness와 post-hoc rationalization
2. 의료 reasoning 설명 평가와 DiReCT
3. Linear probe, tuned lens, SAE 등 내부 분석
4. Natural-language activation readout과 AV-AR
5. Activation patching과 자연어 기반 개입

신규성은 `내부를 본다` 자체가 아니다. 의료 expert-reference 설명 평가와 사례별
activation grounding을 한 방법에 동시에 요구하고, 자연어 병목의 인과적 효용까지
단계적으로 검증하는 데 둔다.

## 3. Data and Setup

DiReCT restricted release의 구조, 511 raw/496 eligible, patient-disjoint split과 PDD-heldout을
설명한다. DDXPlus는 evidence가 명시된 통제 데이터로 grounding과 patching에 사용한다.
MCR은 frozen external OOD로만 쓴다. Backbone은 Gemma-3-12B-IT이다.

## 4. Method

P0/P1/P2와 L16/L24/L32 추출을 정의한다. P0가 주 입력이고 P2는 leakage control이다.
Vanilla NLA에서 시작해 SFT-only, reconstruction-only, full Medical-NLA를 학습한다.
Full objective는 임상 text supervision, activation reconstruction, pair specificity를
결합하되 각 항을 ablation으로 분리한다.

## 5. Experiments

### 5.1 RQ1: Capability boundary

Output head, probe, CoT, vanilla NLA가 진단과 열린 관찰 정보를 얼마나 복원하는지 본다.

### 5.2 RQ2a: Clinical alignment

DiReCT official evaluator와 제한된 human audit로 Table 2를 채운다.

### 5.3 RQ2b: Activation grounding

DDXPlus hard shuffle, cue deletion, retention, round-trip으로 Table 3을 채운다.

### 5.4 RQ3: Causal utility

Grounding을 통과한 모델만 identity patch와 dataset-native edit를 수행한다.

## 6. Results

결과 절은 Table 1 -> Table 2 -> Table 3 -> Table 4 순서다. 각 절은 앞 절의 성공이
다음 주장의 필요조건임을 명시한다. Table 3이 실패하면 Table 2를 faithfulness로
해석하지 않는다. Table 4가 실패해도 RQ1-RQ2 결과는 유지한다.

## 7. Discussion and Limitations

DiReCT 표본 크기와 PDD 불균형, 12B backbone 하나, restricted data 재현성, 공식 semantic
judge 의존성, NLA의 언어 prior, P0 한 토큰이 전체 상태를 대표한다는 제한을 논의한다.
Text patching이 성공해도 임상 배포나 안전성을 뜻하지 않는다.
