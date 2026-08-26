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

## Validation readout 실행

세 seed를 동일한 SFT validation 50행에서 생성한다. 이 50행은 frozen `val_seen` 52행 중
gold label이 note에 정확히 적힌 2행을 E3와 같은 규칙으로 제외한 집합이다. Seed별 best
epoch는 이미 validation content loss로 선택됐으며, 여기서 seed 하나를 골라 test를 보고하지
않는다. 세 seed의 평균과 표준편차를 보고한다.

서버 62는 seed 17/43과 공통 vanilla를 생성한다.

```bash
DATA_ROOT=/data/heejae GPUS=2,3 SEEDS="17 43" RUN_VANILLA=1 \
  nohup bash scripts/run_direct_e4_validation_readouts.sh \
  > /data/heejae/medical_nla/logs/direct_e4_validation_17_43.log 2>&1 &
```

서버 125는 seed 29만 생성한다.

```bash
DATA_ROOT=/data1/heejae GPUS=0,1 SEEDS="29" RUN_VANILLA=0 \
  nohup bash scripts/run_direct_e4_validation_readouts.sh \
  > /data1/heejae/medical_nla/logs/direct_e4_validation_29.log 2>&1 &
```

각 Medical-NLA 파일은 정확히 50행이어야 한다. Vanilla도 같은 50행을 사용하므로 과거
52행 E2 prompt audit 값과 직접 섞지 않는다. 이 단계는 생성까지만 담당하며 official schema
claim extraction과 Llama-3 semantic matching은 별도 E4 evaluator 단계에서 모든 방법에
동일하게 적용한다.

## 해석

Table 2가 개선되면 `clinically aligned`라고 말할 수 있다. 하지만 해당 문장이 activation을
읽었다는 결론은 E5가 통과해야 한다.
