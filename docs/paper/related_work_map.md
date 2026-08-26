# Related Work 지도

이 문서는 현재 논리와 직접 연결되는 문헌 범주를 정리한다. 최종 BibTeX와 정확한 서지
정보는 Overleaf 이전 전에 원문으로 다시 확인한다.

## 1. CoT는 항상 faithful하지 않다

- *Language Models Don't Always Say What They Think: Unfaithful Explanations in
  Chain-of-Thought Prompting* (arXiv:2305.04388)
- *Reasoning Models Don't Always Say What They Think* (arXiv:2505.05410)
- 의료 CoT의 plausibility와 faithfulness를 분리해 평가하는 최근 연구

이 문헌들은 H1의 문제 제기를 뒷받침한다. 본 논문의 신규성은 CoT 비충실성을 처음
발견했다는 것이 아니라, 같은 의료 사례에서 내부 자연어 판독과 임상 설명 평가를
연결한다는 데 둔다.

## 2. 내부 상태의 닫힌 판독

Linear probe, tuned lens, logit lens는 사전 정의한 concept/class가 representation에
존재하는지 정량화한다. SAE는 feature 단위 분해를 제공하지만 feature 의미 부여와
조합적 문장 복원은 별도 문제다. 이들은 Table 1의 closed-label baseline이다.

`Probe는 못한다`고 일반화하지 않는다. 오히려 closed-label diagnosis에서는 강한
upper bound임을 인정하고, NLA는 open evidence/relation text라는 다른 출력 공간을 맡는다.

## 3. Natural-language activation readout

NLA/AV-AR 계열은 activation을 자연어 bottleneck으로 변환하고 text에서 activation을
복원한다. 본 연구는 이 구조를 의료에 적응시키되 reconstruction만으로는 임상 정보가
보존되지 않을 수 있고, SFT만으로는 분류기·template generator로 붕괴할 수 있다는
문제에서 출발한다.

## 4. 의료 설명 benchmark

DiReCT는 physician annotation으로 observation, rationale, diagnosis chain을 평가할 수
있다. 이는 clinical alignment 기준이지 activation faithfulness의 정답이 아니다.
따라서 DDXPlus의 pair/counterfactual 통제와 함께 사용한다.

## 5. 본 논문의 위치

기존 요소 각각은 선행 연구에 있다: CoT 비충실성, probe, NLA, 의료 설명 평가,
activation patching. 본 연구가 검증할 결합은 다음이다.

1. 같은 의료 source run에서 CoT와 pre-generation natural-language readout 비교
2. Expert-reference alignment와 activation grounding을 별도 관문으로 운영
3. 두 관문을 통과한 자연어 판독만 text-mediated intervention에 사용

이 세 항목이 모두 실험으로 닫히기 전에는 `최초`나 `유일`이라는 표현을 쓰지 않는다.
