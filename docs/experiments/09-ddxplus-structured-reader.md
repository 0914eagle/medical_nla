# DDXPlus Probe-Guided Structured Reader

## 목적

D10 1x2 ranking은 validation `3,032` pair/seed에서 full D5 gate를 실패했다.
Ranking-minus-control changed-gap delta는 seed 17/29/43에서
`+.0005/+.0028/+.0030`이었고, 동결 최소 효과 `.05`에 모두 미달했다. Seed 17은
specificity delta가 `-.0005`이고 cluster 95% CI `[-.0020,+.0010]`이 0을
포함했다.

다음 질문은 single AV decoder가 병목인지 분리하는 것이다.

> Frozen probe가 읽은 structured clinical state를 자유 생성 decoder 없이
> 자연어 claim으로 정확히 옮길 수 있는가?

이 방법은 **open-ended NLA가 아니라 structured monitor baseline**이다. 성공해도
probe보다 더 많은 정보를 읽었다고 주장하지 않는다.

## 동결 구조

```text
CoT-P0/HS24 activation
  -> validation-selected frozen finding/value probe
  -> finding threshold를 넘는 evidence/value set
  -> official train에서 만든 deterministic phrase lexicon
  -> <observed> bullet list
```

- Layer: validation-selected `HS24`
- Finding threshold/head/value head: 기존 frozen probe artifact 그대로 사용
- Finding selection: probability가 frozen threshold 이상인 label 전부
- Ordering: finding probability 내림차순, 이후 evidence ID 오름차순
- Verbalizer: official DDXPlus train에서 evidence/value별 가장 빈도가 높은 exact
  rendered cue phrase
- Phrase 동률: 사전순으로 결정
- 입력 prompt text: 예측 및 출력 구성에 사용하지 않음
- Claim 개수: 고정하지 않음
- 새 학습, threshold 선택, layer 선택: 없음

HS24를 쓰는 이유는 이 방법이 공개 HS32 AV checkpoint를 사용하지 않기 때문이다.
Probe validation에서 사전 고정 규칙이 HS24를 선택했고 finding/value validation은
각각 `.9607/.7700`이었다. Generative AV와 비교할 때는 `structured monitor`라는
method class를 별도로 표시한다.

## 평가

Validation에서 먼저 다음을 산출한다.

- finding micro F1
- same-diagnosis hard-shuffled F1과 own-shuffled gap
- native-value end-to-end accuracy와 emission coverage
- deletion original hit, deleted phantom, conditional removal success
- unchanged finding conditional preservation
- value-edit replacement hit, old-value persistence, conditional clean switch

Finding set 점수는 frozen probe와 수학적으로 동일하다. Structured Reader의 추가
가치는 그 set을 prompt 복사나 free generation 없이 명시적인 임상 문구로 변환하는
경로를 제공하는 것이다. 따라서 이 행은 generative Medical-NLA의 **grounding
upper baseline / architecture diagnostic**로 보고한다.

## 실행 순서

### 1. Server 62 validation

Server 62 (`165.132.76.62`, `/data/heejae`)에서 실행한다. HS24 linear heads만
forward하므로 GPU 하나면 충분하다.

```bash
cd /home/eagle0914/medical_nla
git pull origin main

DATA_ROOT=/data/heejae GPU=2 MODE=validation \
  nohup bash scripts/run_ddxplus_structured_reader.sh \
  > /data/heejae/medical_nla/logs/ddxplus_structured_reader_validation_v1.log 2>&1 &

tail -f /data/heejae/medical_nla/logs/ddxplus_structured_reader_validation_v1.log
```

완료 확인:

```bash
OUT=/data/heejae/medical_nla/results/ddxplus_structured_reader_validation_v1
wc -l "$OUT/readouts.jsonl"
cat "$OUT/summary.md"
```

Validation manifest는 original/deletion/value-edit 합계 `10,006`행이다.

### 2. Frozen protocol 확인 후 locked test

Validation 실행은
`/data/heejae/medical_nla/results/ddxplus_finding_value_probe_val_v1/structured_reader_hs24_protocol_v1.json`
을 한 번 생성한다. 이 파일은 train-only lexicon, artifact hash, threshold, ordering을
고정한다. Validation 결과를 보고 이 계약을 수정하지 않는다.

Locked test activation이 server 62에 이미 병합돼 있을 때만 다음을 실행한다.
Wrapper는 validation `results.json`의 protocol SHA256이 현재 protocol과 같은지도
검증한다. Validation 영수증이 없거나 protocol이 달라졌으면 locked test를 거부한다.

```bash
DATA_ROOT=/data/heejae GPU=2 MODE=locked_test \
  nohup bash scripts/run_ddxplus_structured_reader.sh \
  > /data/heejae/medical_nla/logs/ddxplus_structured_reader_locked_test_v1.log 2>&1 &

tail -f /data/heejae/medical_nla/logs/ddxplus_structured_reader_locked_test_v1.log
```

완료 확인:

```bash
OUT=/data/heejae/medical_nla/results/ddxplus_structured_reader_locked_test_v1
wc -l "$OUT/readouts.jsonl"
cat "$OUT/summary.md"
```

## 판정

- Structured Reader가 strong counterfactual response를 보이면: activation selection은
  가능하고 free-generating AV decoder가 병목이라는 증거다. 다음은 frozen structured
  state를 입력으로 받는 constrained set-to-text verbalizer다.
- Structured Reader도 deletion/value response가 약하면: AV decoder 이전의
  support/ontology/value-state 정의가 병목이다. Set decoder 학습 전에 structured
  target을 다시 검토한다.
- 어떤 경우에도 D10의 lambda, temperature, step을 결과 후 sweep하지 않는다.
