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

## 결과

동일한 frozen protocol로 validation `10,006`행과 locked test `10,028`행을
실행했다. Original case는 각각 `4,525/4,543`개였다.

| metric | validation | locked test |
|---|---:|---:|
| mean emitted claims | 4.9485 | 4.9353 |
| finding micro F1 | .9607 | .9587 |
| same-diagnosis shuffled F1 | .7954 | .7938 |
| own-shuffled finding gap | +.1630 | +.1624 |
| native value end-to-end accuracy | .7700 | .7654 |
| native value emission coverage | 1.0000 | .9995 |
| deletion original hit | 1.0000 | 1.0000 |
| deletion phantom | .3626 | .3593 |
| removal success given original hit | .6374 | .6407 |
| retained finding preservation | .9985 | .9987 |
| value-edit replacement hit | .1407 | .1466 |
| value-edit old persistence | .5722 | .5955 |
| clean value switch | .1038 (`n=395`) | .0804 (`n=398`) |

Hard-shuffle pair 수는 validation/test `4,106/4,121`, native-value target은
`2,183/2,136`, deletion pair는 `4,523/4,540`, value-edit pair는 `533/539`였다.

### 해석

1. **정적 finding selection은 강하고 재현된다.** Finding F1 `.9587`과
   own-shuffled gap `+.1624`는 validation `.9607/+.1630`과 거의 같다. Prompt를
   출력 생성에 사용하지 않고 평균 약 4.94개 train-derived clinical phrase를
   안정적으로 렌더링할 수 있다.
2. **기존 free-generating AV는 중요한 병목이다.** Frozen probe state를 직접
   verbalize하면 기존 generative readout보다 훨씬 많은 finding을 노출한다. 단,
   이 차이는 structured selector를 사용한 결과이므로 open NLA의 우월성으로
   주장하지 않는다.
3. **Counterfactual state는 아직 완전하지 않다.** 삭제 cue는 original에서 전부
   검출됐지만 삭제 후에도 `.3593`이 남았다. 반면 untouched finding preservation은
   `.9987`이므로 단순 전체 예측 붕괴는 아니다.
4. **Value state가 가장 약하다.** Original native value accuracy는 `.7654`지만
   value edit 후 replacement hit `.1466`, old persistence `.5955`, clean switch
   `.0804`다. 정적 value 분류 가능성과 개입에 따른 value update는 다른 능력이다.

따라서 결과는 **decoder-only bottleneck**도 **representation-only bottleneck**도
아니다. 정적 finding은 probe selector와 deterministic verbalizer로 잘 노출되지만,
삭제·특히 value edit에 대한 내부 state update는 불완전하다.

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

- Structured Reader는 최종 **structured-monitor upper baseline**으로 유지한다.
- Finding set-to-text 경로는 다음 learned verbalizer의 입력/출력 계약으로 사용할 수
  있다. 이때 selector는 frozen하고 verbalizer만 학습해 두 병목을 다시 섞지 않는다.
- Value edit는 현재 learned verbalizer target으로 승격하지 않는다. 먼저 value-state
  intervention 반응을 개선하거나 불확실성/abstention을 명시하는 별도 설계가 필요하다.
- 삭제 finding도 `.3593` phantom을 upper baseline의 한계로 함께 보고한다.
- 어떤 경우에도 D10의 lambda, temperature, step을 결과 후 sweep하지 않는다.
