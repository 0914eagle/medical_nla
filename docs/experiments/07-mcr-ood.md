# E7. MedCaseReasoning external OOD

## 역할

MCR은 자연 임상 case report와 긴 꼬리 진단에서 frozen Medical-NLA의 외적 일반화를 보는
후순위 데이터다. Gold evidence span이 DiReCT/DDXPlus처럼 구조화되어 있다고 가정하지 않는다.

## 규칙

- MCR로 checkpoint, layer, epoch, threshold를 선택하지 않는다.
- E3-E5에서 고정한 모델을 그대로 평가한다.
- Diagnosis accuracy, decision fidelity, natural-text truncation, readout diversity를 보고한다.
- 근거 평가는 자동 lexical score만으로 결론내지 않고 제한된 blinded semantic/human audit를 쓴다.

## 주의

과거 MCR wrong-note readout 결과는 현재 E7이 아니다. 해당 실험은 archive에 남기며 새로운
OOD 표에 섞지 않는다.
