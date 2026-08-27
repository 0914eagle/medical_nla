# E2. Capability baselines

## 질문

생성 전 P0 activation에서 Medical-NLA가 설명하려는 진단, source decision, finding presence,
finding value가 먼저 decode 가능한가?

## 비교 방법

1. Source early forced-answer candidate sequence likelihood: backbone 행동 기준선
2. Gold diagnosis/category linear probe: 현재 완료된 closed-label audit
3. Source-decision linear probe: 모델이 실제로 낼 답의 사전 상태 audit
4. Finding-presence multi-label probe: evidence ID별 표현 audit
5. Finding-value conditional probe: native value/polarity 표현 audit
6. Vanilla NLA/AV: open-text baseline이며 probe와 같은 accuracy로 합치지 않음

## 평가

- PDD/category/source-decision: top-1, top-k, MRR
- Finding presence: micro/macro AUROC, F1, precision, recall
- Finding value: conditional accuracy와 macro F1
- Seen vs PDD-heldout 또는 value-combination OOD
- Label shuffle, answer shuffle, same-diagnosis hard shuffle
- HS16/HS24/HS32 validation sensitivity

Probe는 Medical-NLA 학습 전 feasibility audit이자 closed-label 기준선이다. 진단마다 별도
probe를 만들지 않는다. Diagnosis/category는 하나의 multiclass head, finding presence는 하나의
multi-label head, finding value는 충분한 표본이 있는 native value를 대상으로 한 conditional
head를 사용한다. Open evidence text는 probe에 정의되지 않으므로 `N/A`이며 실패 0점으로
처리하지 않는다. Vanilla NLA의 자연어 결과는 Table 2와 Table 3에서 별도로 평가한다.

## Table 1 보고 규칙

Table 1A에는 동일 locked IDs의 Direct/CoT 행동을 보고한다. Table 1B에는 validation에서 고정한
decoder의 locked-test decodability를 보고한다. `Layer`는 주표의 반복 열이 아니다.
HS16/HS24/HS32 세 값은 아래 validation sensitivity 표와 Figure 2에 모두 제시하고, 주표에는
선택된 layer 하나만 사용한다. 현재 diagnosis/category에서는 HS24가 선택됐지만 Medical-NLA와
AR의 primary index는 공개 checkpoint 호환 때문에 HS32로 유지한다. 이 둘을 같은 결과라고
부르지 않는다.

| Target | HS16 | HS24 | HS32 | Majority | 상태 |
|---|---:|---:|---:|---:|---|
| Disease category, 25-way | .5000 | **.5962** | .5192 | .0577 | validation 완료 |
| Canonical PDD, 49-way | .3846 | **.4423** | .3846 | .0962 | validation 완료 |
| Source decision | TBD | TBD | TBD | TBD | 실행 필요 |
| Finding presence, micro F1 | .9636 | **.9607** | .9607 | N/A | validation 및 locked test 완료 |
| Finding value, conditional accuracy | .7641 | **.7700** | .6990 | N/A | 6 evidence tasks에서 완료 |

`Source decision`은 source model의 자유 생성 문자열을 그대로 class ID로 쓰지 않는다. 먼저
train/validation에서 normalized answer 재사용률과 frozen PDD/category ontology의 unique mapping
coverage를 감사한다. Validation label이 train에 충분히 나타나지 않거나 unmatched/ambiguous 비율이
높으면 이 target은 closed probe에서 제외하고 Table 3의 open-text source fidelity로만 평가한다.

```bash
python scripts/audit_direct_source_decision_labels.py \
  --split-dir "$DATA_ROOT/restricted/direct/splits/direct_patient_pdd_confirmatory_v1" \
  --answers \
    "$DATA_ROOT/restricted/direct/e1/direct_e1_trainval_v1/source_cot_answers.jsonl" \
    "$DATA_ROOT/restricted/direct/e1/direct_e1_test_v1/source_cot_answers.jsonl" \
  --output-json "$DATA_ROOT/restricted/direct/e2/source_decision_label_audit_v1.json" \
  --summary-md "$DATA_ROOT/restricted/direct/e2/source_decision_label_audit_v1_summary.md"
```

Finding presence/value는 DiReCT physician deduction을 임의의 fixed label로 바꾸지 않고,
evidence/value ID가 원자료에 정의된 DDXPlus CoT-P0에서 평가한다. 따라서 Table 1B는 DiReCT
diagnosis/source-decision panel과 DDXPlus finding/value panel을 분리한다.

Validation에서 HS24를 고정한 locked test 결과는 finding F1 `.9562`, 같은 진단 hard-shuffle
`.7938`, gap `+.1624 [.1576,.1672]`다. Conditional native-value accuracy는 `.7659`, shuffle
`.5791`, gap `+.1868 [.1650,.2091]`이며 train-supported 6 evidence task/32 value class와 test
single-value target의 `.7161`만 포괄한다. Cue deletion은 target probability drop `+.6103`,
removal success `.6407`이었지만 native value edit replacement hit는 `.1466`, clean switch는
`.0804`였다. 따라서 finding availability는 통과, value counterfactual faithfulness는 실패로
분리 판정한다.

Candidate-likelihood baseline은 단일 다음-token logit이나 저장된 P0 벡터의 unembedding이
아니다. P0 prompt가 먼저 추론하라고 요구하므로 `The answer is`를 강제로 붙인 뒤 각
사전등록 candidate label을 teacher-force하고 label token들의 평균 log probability로
순위를 매긴다. 따라서 명칭은 **CoT-P0 early forced-answer candidate likelihood**로 고정한다.
이는 reasoning 없이 바로 답하게 했을 때 backbone이 가진 닫힌 진단 선호를 보는 행동
기준선이며, 실제 CoT의 next-token distribution이나 P0 한 벡터만의 정보라고 부르지 않는다.
별도 분류 head는 없지만 평가 label ontology를 제공받는 closed candidate-ranking
baseline이다. Held-out PDD를 candidate list에 넣은 결과도 zero-shot open generation이
아니라 ontology-given ranking으로 표기한다.

Validation 실행은 `scripts/run_direct_e2_forced_answer_baseline.sh`로 고정한다. 이 wrapper는
오직 HS32/P0 `manifest_val_seen.jsonl`만 읽으며 locked-test path를 인터페이스에 두지 않는다.
전체 canonical manifest에서 결과와 무관하게 사전등록된 61 PDD 또는 25 category label만
추출해 candidate set을 고정한다. Primary ranking은 multi-token 길이를 정규화한
`logprob_mean`이다.

Wrapper의 기본값은 위 full-ontology/raw 실행을 그대로 보존한다. 공정 비교나 prior
감사를 위해서는 다음 환경 변수를 명시할 수 있다.

- `ONTOLOGY_MANIFEST`: candidate ontology를 만들 manifest. PDD probe와 맞춘 49-way
  비교에는 frozen confirmatory `train.jsonl`을 사용한다.
- `OUTPUT_NAME`: 기존 raw 결과를 덮어쓰지 않는 출력 디렉터리 이름.
- `RANK_FIELD`: `logprob_mean` 또는 `calibrated_logprob_mean`.
- `CALIBRATION_PROMPT`: calibrated ranking에서 차감할 content-free candidate prior를
  만드는 고정 prompt.

두 label space는 서로 독립이므로 두 서버에서 병렬 실행한다.

```bash
# Server 125: 61-way canonical PDD
DATA_ROOT=/data1/heejae GPUS=0,1 LABEL_FIELD=canonical_pdd \
  nohup bash scripts/run_direct_e2_forced_answer_baseline.sh \
  > /data1/heejae/medical_nla/logs/direct_e2_forced_answer_pdd_val_v1.log 2>&1 &

# Server 62: 25-way disease category
DATA_ROOT=/data/heejae GPUS=2,3 LABEL_FIELD=disease_category \
  nohup bash scripts/run_direct_e2_forced_answer_baseline.sh \
  > /data/heejae/medical_nla/logs/direct_e2_forced_answer_category_val_v1.log 2>&1 &
```

두 실행 모두 validation 52행만 점수화한다. PDD와 category 결과는 후보 수와 label
granularity가 다르므로 서로 정확도를 직접 비교하지 않고 각각 같은 label space의 probe와
비교한다.

### Early forced-answer validation 결과와 판정

두 raw 실행은 52/52행을 완주했다.

| Target | Candidates | Top-1 | Top-5 | MRR | Mean gold rank |
|---|---:|---:|---:|---:|---:|
| Disease category, raw likelihood | 25 | 0.4808 | 0.6731 | 0.5814 | 5.02 |
| Disease category, calibrated sensitivity | 25 | 0.2308 | 0.3077 | 0.3091 | 9.58 |
| Canonical PDD, raw likelihood | 61 | 0.1538 | 0.4423 | 0.3168 | 8.77 |
| Canonical PDD, train-ontology raw | 49 | 0.1538 | 0.5192 | 0.3250 | 7.92 |
| Canonical PDD, train-ontology calibrated sensitivity | 49 | 0.0577 | 0.1346 | 0.1486 | 15.83 |

Category 25-way는 동일 label space의 HS24 probe보다 top-1 0.1154, top-5 0.2307,
MRR 0.1470 낮았다. 이는 supervised probe와 무학습 candidate ranking의 비교이므로 동일
학습량 우월성으로 해석하지 않고, closed-label activation readout의 capability boundary로
해석한다.

Raw PDD ranking은 52행 중 35행에서 corpus 빈도 1인
`Arrhythmogenic Right Ventricular Cardiomyopathy`를 top-1으로 골랐다. 61개 후보 중 실제
top-1으로 나온 label도 8개뿐이었다. 따라서 raw `logprob_mean`은 사례 정보뿐 아니라 후보
문자열·tokenization prior에 크게 오염되어 있으며 primary method comparison에 그대로 쓰지
않는다. 두 validation 통제도 52/52행을 완주했다.

1. 49-way train ontology에서도 같은 후보가 35/52 top-1이어서 후보 수 불일치가 붕괴의
   원인이 아니었다. 이 raw 49-way만 후보 공간이 같은 probe와 직접 비교할 수 있지만,
   후보 prior 오염을 함께 보고한다.
2. content-free prior 차감은 category top-1을 0.4808에서 0.2308로, PDD top-1을
   0.1538에서 0.0577로 낮추고 각각 소수의 다른 후보에 다시 집중됐다. 따라서 이 보정은
   primary ranking으로 채택하지 않고 sensitivity audit으로 보존한다.

고정 calibration prompt는 임상 정보가 없는 동일 과제 형식인
`Clinical case:\nN/A\n\nWhat is the most likely diagnosis?`로 한다. 이 보정은 모델의
확률 calibration을 주장하는 절차가 아니라 후보명 prior 차감이다. 이번 결과는 단일
content-free prompt 차감 자체가 안정적인 진단 상태 추정기가 아님을 보여준다. Table 1의
early forced-answer 행은 raw matched 결과를 행동 기준선으로 쓰되, probe보다 낮고 후보명
prior에 취약하다는 제한을 캡션에 명시한다.

## 실행 상태

### P0 linear probe: frozen validation

동일한 train 266행으로 학습하고 `val_seen` 52행에서 validation NLL로 hyperparameter와
stopping epoch를 선택했다. Locked test manifest는 읽지 않았다.

| Target | HS | Classes | Majority | Top-1 | Top-5 | MRR | Macro recall | Val NLL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Canonical PDD | 16 | 49 | 0.0962 | 0.3846 | 0.6923 | 0.5294 | 0.3597 | 2.5533 |
| Canonical PDD | **24** | 49 | 0.0962 | **0.4423** | **0.7692** | **0.5762** | **0.3868** | **2.0489** |
| Canonical PDD | 32 | 49 | 0.0962 | 0.3846 | 0.6923 | 0.5335 | 0.2771 | 2.3784 |
| Disease category | 16 | 25 | 0.0577 | 0.5000 | 0.7885 | 0.6374 | 0.4833 | 1.9679 |
| Disease category | **24** | 25 | 0.0577 | **0.5962** | **0.9038** | **0.7284** | **0.5000** | **1.3961** |
| Disease category | 32 | 25 | 0.0577 | 0.5192 | 0.8654 | 0.6609 | 0.4426 | 1.6869 |

두 label space에서 HS24가 모든 핵심 validation 지표와 NLL에서 가장 좋았다. 이는 생성 전
P0 activation에 닫힌 ontology의 진단 정보가 선형적으로 읽힌다는 증거다. 예를 들어
HS24 category top-1은 majority 0.0577보다 10배 이상 높고 top-5는 0.9038이다. 반면
같은 P0에서 vanilla AV의 source answer, gold PDD, category literal mention은 모두 0이다.
따라서 현재의 정확한 결론은 **P0에 정보가 없다는 것이 아니라, supervised closed-label
probe가 읽는 정보를 vanilla AV가 자연어로 꺼내지 못한다**는 것이다.

Probe는 train에서 정의된 49 PDD 또는 25 category 중 하나를 고르는 분류기다. 새로운
PDD, 관찰, 관계, 근거 문장을 생성하지 못하므로 설명 품질 기준선이 아니며 open-evidence
열은 `N/A`다. 또한 위 수치는 validation 결과이므로 최종 성능 추정치가 아니다. HS24의
우세는 layer sensitivity 결과로 보존하되, 공개 AV/AR checkpoint가 HS32용이므로
Medical-NLA와 round-trip의 primary index는 계속 HS32로 둔다.

### Vanilla AV prompt comparison: frozen validation

Frozen validation 52행의 HS32/P0 prompt comparison은 다음과 같다.

| Prompt | Parse | Source-answer mention | Gold-PDD mention | Category mention | Own-donor source gap | Prompt trigram gap |
|---|---:|---:|---:|---:|---:|---:|
| Default | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Task-aligned suffix | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0007 |

여기서 `mention`은 의미 채점이 아니라 `src.answer_matching.is_correct`의 엄격한 문자열
포함 진단이다. 대소문자, 구두점, 단복수, 일부 영미 철자와 manifest에 등록된 gold alias는
처리하지만, source answer와 category에는 별도 alias를 제공하지 않았고 `GERD`, `PE` 같은
약칭이나 등록되지 않은 임상 동의어를 추론하지 않는다. 따라서 위의 0은 **lexical lower
bound**이며 `semantic mention=0` 또는 `activation information=0`으로 인용하지 않는다.

Frozen validation에서 생성한 vanilla AV의 정확한 수는 다음과 같다.

| 범위 | 계산 | readout rows | 용도 |
|---|---:|---:|---|
| P0 primary+sensitivity | 52 cases x 2 prompts x 3 layers | **312** | 현재 P0 lexical 결과의 전체 모집단 |
| P1/P2 positive controls | 52 x 2 prompts x 2 positions | **208** | answer/reasoning 노출 통제 |
| validation total | 312 + 208 | **520** | 방법 선택용, locked test 아님 |
| old exploratory test | 171 x 3 positions | **513** | 이미 본 pilot; 최종 test로 재사용 금지 |

따라서 지금까지 materialize된 readout은 총 1,033행이지만 서로 다른 목적의 행을 합친
운영상 총계다. 주결과의 primary arm은 frozen validation `default/HS32/P0` 52행이고,
312행은 prompt/layer sensitivity 전체다.

약칭·동의어 누락을 닫기 위해 P0 312행 전부에 blinded semantic audit을 수행한다.
Judge에는 환자 note를 주지 않고 readout과 순서를 무작위화한 source answer, gold PDD,
disease category만 준다. `match=true`일 때 readout 안의 exact evidence quote를 의무화하고,
인용이 실제 readout에서 확인되지 않으면 불일치로 처리한다. 이 LLM-as-a-judge 평가는
의미상 약칭/동의어 누락을 보완하지만 activation faithfulness 자체를 증명하지는 않는다.
Faithfulness는 이후 own-vs-shuffled, counterfactual, patching 통제로 별도로 검증한다.

자동 semantic audit은 312/312행 parse에 성공했다. Exact evidence quote가 실제 readout에
포함된 경우에만 semantic match를 인정했다.

| Prompt | HS | Source answer | Gold PDD | Category |
|---|---:|---:|---:|---:|
| Default | 16 | 0/52 | 0/52 | 1/52 |
| Default | 24 | 0/52 | 0/52 | 0/52 |
| **Default** | **32** | **0/52** | **0/52** | **0/52** |
| Task-aligned | 16 | 0/52 | 0/52 | 1/52 |
| Task-aligned | 24 | 0/52 | 0/52 | 0/52 |
| Task-aligned | 32 | 0/52 | 0/52 | 0/52 |

따라서 primary `default_HS32/P0`의 lexical 0은 단순 약칭·임상 동의어 누락으로 설명되지
않는다. Prompt suffix도 개선하지 않았다. 다만 이 audit은 readout이 세 진단 target을
명시적으로 이름 붙였는지만 측정하며 physician observation/rationale의 품질이나 activation
grounding을 측정하지 않는다. 따라서 Table 1의 semantic diagnostic 열은 위 AI 판정으로
확정하되, 이를 human-validated score라고 부르지 않는다. 현재 판정자는 DiReCT 공식 평가에
사용되는 local Llama-3-8B이고, single-judge validation이라는 한계를 함께 보고한다.

`manual_default_hs32.jsonl`은 자동 판정을 필요할 때 사람이 재감사할 수 있도록 만든 선택적
보조 산출물이다. 이 파일의 `human_*` 필드는 현재 0/52로 비어 있지만, 더 이상 E2 완료나
Table 1 확정의 필수 조건은 아니다. 향후 수동 감사 또는 독립된 두 번째 AI judge를 추가하면
민감도 분석으로 보고하며 primary AI 판정을 사후 교체하지 않는다. 선택적으로 수동 감사할
때만 다음을 실행한다.

```bash
AUDIT=/data/heejae/restricted/direct/e2/direct_e2_val_v1/semantic_audit_v1
python scripts/review_direct_e2_semantic_audit.py \
  --input-jsonl "$AUDIT/manual_default_hs32.jsonl" \
  --output-jsonl "$AUDIT/manual_default_hs32_reviewed.jsonl" \
  --summary-md "$AUDIT/manual_default_hs32_reviewed_summary.md" \
  --reviewer heejae
```

각 행에서 source answer, gold PDD, disease category가 readout에 의미상 나타나는지를 차례로
`y n n`처럼 입력한다. `q`는 현재까지의 판정을 저장하고 종료한다. 두 파일 모두 restricted
data이므로 커밋하지 않는다.

Server 62에 여섯 P0 파일을 모은 뒤 다음처럼 실행한다. 먼저 `LIMIT=8`로 schema와 인용
검증을 smoke-test하고, 같은 출력에 full run을 resume한다. Restricted readout과 judge
response는 모두 `${DATA_ROOT}/restricted` 아래에 남기며 커밋하지 않는다.

```bash
# task-aligned HS16/24가 server 125에만 있으면 server 62에서 먼저 복사
scp \
  eagle0914@165.132.76.125:/data1/heejae/restricted/direct/e2/direct_e2_val_v1/vanilla_av_task_aligned_p0_hs16_val.jsonl \
  eagle0914@165.132.76.125:/data1/heejae/restricted/direct/e2/direct_e2_val_v1/vanilla_av_task_aligned_p0_hs24_val.jsonl \
  /data/heejae/restricted/direct/e2/direct_e2_val_v1/

DATA_ROOT=/data/heejae GPU=2 LIMIT=8 \
  bash scripts/run_direct_e2_semantic_audit.sh

DATA_ROOT=/data/heejae GPU=2 \
  nohup bash scripts/run_direct_e2_semantic_audit.sh \
  > /data/heejae/medical_nla/logs/direct_e2_semantic_audit_v1.log 2>&1 &
```

Task-aligned suffix가 lexical 또는 semantic diagnostic recovery를 개선하지 않아 default를
vanilla primary로 유지한다. P1/P2 validation에서는 source-answer mention이 각각
default 0.5192/0.5962, task-aligned 0.5577/0.5000이었지만, P1 leakage-free subset은
5행이고 두 prompt 모두 0/5였다. P1은 CoT 문자열 누출 분석, P2는 answer-exposed positive
control로만 사용한다.

같은 52행에서 HS16/24/32 P0 activation을 HS32용 vanilla AV decoder에 넣은 layer
sensitivity도 완료했다.

| Prompt | HS | Parse | Source answer | Gold PDD | Category | Own-donor source gap | Prompt trigram gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Default | 16 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Default | 24 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0007 |
| Default | 32 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Task-aligned | 16 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Task-aligned | 24 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Task-aligned | 32 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0007 |

Prompt와 입력 layer를 바꿔도 생성 전 diagnosis/category의 literal recovery와
same-category donor discrimination은 개선되지 않았다. 특히 HS24는 같은 validation에서
probe 성능이 가장 높았으므로 `HS24 activation에 진단 정보가 없다`는 해석은 맞지 않는다.
정확한 해석은 **HS32용 vanilla AV decoder가 어느 입력 layer에서도 그 정보를 현재
출력 과제로 표현하지 못했다**는 것이다. HS16/24 행에는 input layer와 decoder training
layer 불일치가 있으므로 주 성능표가 아닌 sensitivity로만 보고한다.

Test P0/L32 vanilla AV는 171/171행 생성 및 `<explanation>` parsing에 성공했고 빈 출력은
없었다. 출력 길이는 637--741자(중앙값 697, 평균 696.9)였다. 길이 안정성은 내용의
사례 특이성과 별개다. 길이 범위가 매우 좁으므로 exact/normalized 반복률과
own-case-versus-shuffled specificity를 우선 확인한 뒤 clinical alignment를 평가한다.

예비 same-category lexical derangement에서 164행의 own-prompt trigram containment와
shuffled-prompt containment가 모두 0.0013이었고 gap은 -0.0001이었다. 따라서 P0 vanilla
AV가 prompt의 사례 고유 표현을 그대로 복원한다는 증거는 없다. 이 검사는 paraphrase를
인정하지 않으므로 최종 실패 판정으로 쓰지 않고, 동일 claim extractor와 semantic
matcher를 이용한 own-versus-shuffled 평가의 필요성을 확인한 sanity check로만 둔다.

P1/P2 L32는 각각 171/171행 parse됐고 빈 출력과 normalized exact duplicate가 없었다.
Lexical own/shuffled는 P1 0.0067/0.0064(gap +0.0003), P2
0.0017/0.0018(gap -0.0001)이었다. 따라서 reasoning/answer 이후 위치에서도 현재
trigram 검사는 사례 특이성을 찾지 못했다. P1의 절대 overlap 증가는 동일 category의
다른 사례에서도 유지되어 공유 임상 어휘로 설명된다. P2는 답을 이미 본 위치지만 진단
label이 1--2단어인 경우 trigram에 기여하지 않으므로, 이 결과만으로 positive control
실패를 확정하지 않고 phrase-level source-answer recovery를 다음 검사로 둔다.

Phrase-level 결과는 다음과 같다.

| Position | Source-answer mention | Gold-PDD mention | Category mention | Own-vs-donor source gap |
|---|---:|---:|---:|---:|
| P0 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| P1 | 0.4912 | 0.1404 | 0.5848 | +0.4146 |
| P2 | 0.3918 | 0.0819 | 0.4854 | +0.3598 |

Own-vs-donor는 같은 disease category이되 다른 source answer를 가진 164행에서 계산했다.
P1은 source answer alias가 reasoning에 없던 15행에서는 1/15=0.0667만 source answer를
언급했다. 따라서 P1 전체의 높은 specificity는 pre-answer 내부 판독보다 CoT 문자열
누출 상한으로 해석한다. P2의 양의 gap은 answer-exposed positive control을 통과한
것으로, vanilla AV가 모든 DiReCT activation에서 무조건 실패하는 것은 아님을 보인다.
반면 생성 전 P0의 diagnosis/category phrase recovery는 0/171이다. 이것은 Medical-NLA가
개선해야 할 baseline failure지만, P0 evidence/rationale의 semantic recovery까지 0이라는
뜻은 아니므로 Table 2 claim extraction과 E5 grounding을 계속 분리한다.

## Model selection

현재 test_seen과 PDD-heldout 171행은 이미 위치 및 vanilla AV 설계 점검에 사용했으므로
exploratory pilot다. 공개 AV/AR와 호환되는 HS32를 primary로 고정한다. HS16/HS24는 같은
L32 decoder의 distribution shift가 섞인 sensitivity다. Probe 자체는 HS24가 validation에서
최고였지만, 이것이 HS24 Medical-NLA decoder를 선택했다는 뜻은 아니다. Task-aligned
vanilla prompt와 probe regularization은 train/validation에서만 정한다. 새 locked test는
설정과 분석 코드를 동결한 뒤 한 번만 평가한다.

## 산출물

- Table 1
- HS32 primary baseline과 HS16/HS24 sensitivity 완료
- E3에서 사용할 vanilla checkpoint와 prompt 고정
