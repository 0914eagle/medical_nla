# E4. DiReCT clinical explanation evaluation

## 질문

CoT, vanilla NLA, Medical-NLA가 의사가 주석한 observation-rationale-diagnosis 구조를
얼마나 보존하는가?

## 주 평가

Official evaluator의 Accdiag, Obspre, Obsrec, Obscomp, Expcom, Expall을 사용한다.
같은 case, 같은 output schema, 같은 Llama-3-8B semantic judge를 사용한다. 공식 `+1`
denominator를 유지한다.

CoT와 각 NLA의 free text는 공통 claim extractor로 observation-rationale-diagnosis schema에
변환한다. Extractor는 method 이름, gold annotation, 원 note를 받지 않는다. Parse 또는
extraction 실패는 분모에서 삭제하지 않고 0점 처리하며 extraction coverage를 함께 보고한다.

## 보조 평가

- Unsmooothed observation precision/recall
- Seen vs PDD-heldout
- Source-correct vs source-wrong
- By-category 결과와 macro average
- 100개 이하의 blinded human/clinician audit: factuality, missing key evidence, unsupported claim

LLM-as-a-judge는 official semantic matching에만 사용한다. 독립적인 faithfulness 판정자로
간주하지 않는다. Judge agreement는 일부 표본에서 수동 검사한다.

## 해석

Table 2가 개선되면 `clinically aligned`라고 말할 수 있다. 하지만 해당 문장이 activation을
읽었다는 결론은 E5가 통과해야 한다.
