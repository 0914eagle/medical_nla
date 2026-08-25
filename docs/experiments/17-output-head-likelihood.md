# 17 — Source output-head likelihood baseline

## 질문

Hidden-state probe가 `moved`를 잘 판별하는 이유가 정말 **최종 출력분포에는 아직
드러나지 않은 내부 정보**를 읽기 때문인가? 아니면 source model의 마지막 logits에
이미 나타난 불확실성이나 진단 경쟁을 probe가 다시 읽은 것인가?

현재 Table 2b에는 생성된 answer text와 hidden-state 채널 사이의 이 기준선이 없다.

```text
L32 activation
  ├─ linear probe / AV readout
  ↓
final model logits over diagnosis continuations   <- 이 실험
  ↓
greedy generated answer
```

따라서 이 실험은 선택적 부록이 아니라, **왜 activation 접근이 필요한가**를 묻는
필수 gray-box baseline이다.

## 정확히 무엇을 점수화하는가

Canonical direct arm과 같은 user prompt 및 assistant prefill
`The answer is`를 사용한다. 그 뒤에 DDXPlus 진단 후보를 하나씩 teacher forcing해
source model의 log probability를 계산한다. Free generation을 새로 채점하는 것이
아니며, 실제 answer가 나온 것과 동일한 prefix에서 후보분포를 읽는다.

진단명은 여러 token일 수 있으므로 “출력 직전 likelihood”는 단일 숫자가 아니다.

| score | 의미 | 주의 |
|---|---|---|
| `first_token_logprob` | 실제 greedy decoder의 첫 선택에 가장 가까움 | 첫 token을 공유하는 진단을 구분하지 못함 |
| `logprob_sum` | 진단명 전체 continuation likelihood | 짧은 진단명을 선호 |
| `logprob_mean` | token당 평균 likelihood | 평범한 token으로 된 이름을 선호할 수 있음 |
| `calibrated_*` | content-free prompt의 후보명 prior를 차감 | 실제 raw decoder 분포가 아니라 보정된 분석치 |

Primary는 `first_token_logprob`로 고정한다. 전체 진단명 결과는
`logprob_sum`/`logprob_mean`, 후보명 빈도 효과는 calibrated score로 sensitivity
analysis한다. 가장 잘 나온 score 하나를 사후 선택해 본문에 싣지 않는다.

이 실험은 L32를 vocabulary로 바로 투영하는 **logit lens가 아니다**. 모든
transformer block을 지난 source model의 실제 final output head를 읽는다. L32 logit
lens를 추가한다면 별도 행으로 표시해야 한다.

## 평가 label과 모집단

- 모집단: canonical DDXPlus no-note source-correct cases, `n=1,747`
- 입력: 각 사례의 **wrong-note run 하나만**
- 정답 label: 같은 사례의 no-note/wrong-note pair로 사후 정의한 `moved`
- silent: wrong answer가 suggestion 이름을 말하지 않는 `n=1,641`
- 주 지표: diagnosis-stratified AUROC, all/silent

No-note arm은 label 생성에만 사용하며 likelihood detector에는 주지 않는다.

### 배포 가능한 특징

- candidate entropy
- 낮은 top-1 probability
- 작은 top-1/top-2 probability margin
- `p(suggestion)` — suggestion은 wrong note에 쓰여 있으므로 관측 가능
- output-head top-1이 suggestion인지
- output-head top-1과 실제 generated answer가 다른지

`p(gold)` 및 `p(suggestion)-p(gold)`는 기전 분석에는 유용하지만 실제 배포에서는
gold를 모르므로 detector 성능 주장에 사용하지 않는다.

## 실행

```bash
cd /home/eagle0914/medical_nla
source /data1/heejae/uv/medical_nla/bin/activate
export PYTHONPATH=/home/eagle0914/medical_nla
source scripts/env.sh

CUDA_VISIBLE_DEVICES=0 python scripts/score_source_diagnosis_logprobs.py \
  --config configs/default.yaml \
  --input "$DATA/ddxplus_hint_cases_v2.jsonl" \
  --hint-variant wrong \
  --rank-field first_token_logprob \
  --top-k-output 49 \
  --candidate-batch-size 8 \
  --output-jsonl "$ART/results/ddxplus_wrong_output_head_logprobs.jsonl" \
  --summary-md "$ART/results/ddxplus_wrong_output_head_logprobs_summary.md"

python scripts/evaluate_output_head_attribution.py \
  --answers "$ART/results/ddxplus_hint_answers_v2_rescored.jsonl" \
  --cases "$DATA/ddxplus_hint_cases_v2.jsonl" \
  --logprobs "$ART/results/ddxplus_wrong_output_head_logprobs.jsonl" \
  --rank-field first_token_logprob \
  --output-jsonl "$ART/results/ddxplus_output_head_attribution.jsonl" \
  --summary-md "$ART/results/ddxplus_output_head_attribution_summary.md"
```

`--hint-variant wrong`은 계산할 row만 줄인다. Candidate set은 필터 전 전체 input에서
수집하므로 wrong arm만 실행해도 진단 선택지는 줄어들지 않는다.

## 결과 해석을 사전에 고정한다

| 결과 | 허용되는 해석 |
|---|---|
| output-head AUROC가 probe와 비슷함 | probe 우위의 상당 부분이 final logits에도 이미 있음. hidden-only 정보라는 주장은 약화 |
| output-head가 text monitor보다 높지만 probe보다 낮음 | output distribution에 일부 신호가 있으나 L32 representation이 추가 정보 제공 |
| output-head가 text monitor 수준이고 probe가 높음 | 생성문과 최종 logits 모두 놓치는 소견서 영향 신호가 hidden representation에 존재 |
| `p(gold)>p(suggestion)`인데 suggestion 출력 | Figure 3의 internal-output mismatch가 실제 output head와 decoding 사이에서도 나타나는 사례 |

결과가 어느 방향이든 보고한다. Output-head가 probe와 같으면 “내부를 봐야만
가능하다”는 주장을 철회하고, probe는 compact supervised detector로 재해석한다.

## 상태

- ✅ source candidate scorer 존재
- ✅ wrong-arm 전용 필터 및 canonical evaluator 구현
- ▢ GPU likelihood 실행
- ▢ Table 2b all/silent 값 반영
- ▢ raw full-sequence 및 calibrated sensitivity

