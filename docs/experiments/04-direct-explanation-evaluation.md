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

## 현재 validation 결과

동일한 50-case validation에서 CoT, vanilla NLA, Medical-AV SFT seeds 17/29/43을 공통
quote-constrained extractor와 official-compatible evaluator로 평가했다. 이 값은 방법 진단용이며
locked `test_seen=72`와 `test_pdd_heldout=106`의 Table 2 결과가 아니다.

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | 0 | .3009 | .3903 | .2349 | .0573 | .0144 |
| Vanilla NLA | 0/50 | 0 | 0 | 0 | 0 | 0 | 0 |
| Medical-AV SFT, seed 17 | 50/50 | 0 | .0771 | .0435 | .0343 | 0 | 0 |
| Medical-AV SFT, seed 29 | 50/50 | 0 | .0133 | .0047 | .0047 | 0 | 0 |
| Medical-AV SFT, seed 43 | 50/50 | 0 | .0200 | .0029 | .0032 | 0 | 0 |

SFT-only는 structured observation을 생성하게 했지만 CoT보다 임상 정렬이 낮고 seed 편차가
컸다. 현재 target에 rationale가 없으므로 `Expcom/Expall=0`은 구조상 예상되지만 observation
계열도 충분하지 않다. 따라서 이 결과를 Medical-NLA 성공으로 쓰지 않고, reconstruction과
pair-specificity objective가 필요한 근거로 사용한다.

## Common-schema mixed pilot 후속 의미 채점

Common validation readout은 DiReCT 50행과 DDXPlus 50행을 함께 담는다. 임의의 앞 50행을
사용하지 않고 `source_dataset=direct`를 먼저 적용한 뒤, 기존 DiReCT validation cohort와
base ID 집합이 정확히 같을 때만 request를 만든다. `READOUTS_DIR`와
`READOUT_SOURCE_DATASET=direct`를 지정해 같은 E4 extractor/evaluator를 재사용한다.

이 검사는 약칭과 의역 때문에 lexical cue score가 0이 된 경우를 찾기 위한 validation
diagnostic이다. 여기서도 observation 계열이 개선되지 않으면 mixed SFT v1은 내용 판독에
실패한 것으로 판정한다. 성공 여부와 관계없이 locked test와 text patching은 다음 objective를
고정하기 전까지 실행하지 않는다.
