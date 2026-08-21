# 판독 첫 결과 — L24, 시드 17, 1에폭 (2026-08-21)

8/19 감사로 파일럿 수치를 전부 버린 뒤 **재구축 파이프라인에서 처음 나온 실제
숫자**다. 시드 하나, 에폭 하나, 레이어 하나이므로 논문 수치가 아니라 방향
확인이다.

---

## 1. 숫자

어댑터: `ddxplus_L24_s17`, 1에폭, 학습 행 10,195. 평가 pool 각 770행
(seen은 2,940에서 시드 17로 무작위 추출).

| | seen | heldout | 격차 |
|---|---:|---:|---:|
| 형식 준수 (`<observed>` 방출) | 1.0000 | 1.0000 | 0 |
| 비라틴 문자 오염 | 0.0000 | 0.0000 | 0 |
| 평균 출력 길이 | 47자 | 52자 | |
| **문자열 축자 재현** | **0.9104** | **0.1532** | **-0.7571** |
| 완전 내용 일치 (f1 = 1.00) | 0.9156 | 0.2896 | -0.6260 |
| **내용 일치 f1 ≥ 0.5** | 0.9857 | **0.7130** | -0.2727 |
| 평균 f1 | 0.9744 | 0.6490 | -0.3254 |

---

## 2. 해석

**seen에서 일어나는 일은 암송이다.** 축자 재현 91%는 판독이 아니라 학습에서 본
문자열을 그대로 뱉는 것이다. **seen 수치는 논문에 쓸 수 없다.**

**heldout에서는 암송이 불가능하다.** 축자 재현이 15%로 무너진다. 그런데 내용은
71%가 맞는다. heldout cue는 어떤 학습 행에서도 감독된 적이 없으므로, 외워서
맞힐 경로가 없다. **남은 0.713은 벡터를 실제로 읽어낸 몫이다.**

격차 -0.27 자체는 결함이 아니다. 격차가 **없었다면** 오히려 이상했을 것이다.
중요한 것은 세 가지다.

1. heldout 0.713은 0에서 한참 위다 → 판독이 작동한다
2. seen 0.986에는 암기가 섞여 있다 → 대표 숫자가 될 수 없다
3. **논문의 대표 숫자는 heldout이다**

heldout pool이 없었다면 0.99를 보고하고 틀렸을 것이다. cue-string heldout 설계가
자기 역할을 했다.

파일럿이 L24 heldout에서 주장한 73.1%와 자릿수가 맞는다. 파일럿 수치는 다른
이유로 무효지만, 재구축이 전혀 다른 곳에 착지하지 않았다는 신호는 된다.

---

## 3. 채점 규칙이 이 결론을 좌우했다

같은 파일, 같은 판독물을 옛 v2 규칙(축자 포함)으로 채점하면 **heldout 0.1532**다.
그 숫자를 그대로 읽었다면 "판독 실패"로 결론 내리고 설계를 갈아엎었을 것이다.

두 단계로 고쳤고, 둘 다 판단이 아니라 기계적 처리다.

**(a) 내용어 겹침으로 전환** — cue는 최대 21단어의 설문 문장이고 판독기는
바꿔 말한다. 축자 포함은 이 경우 규칙이 아니라 사고다.

**(b) 어형과 설문 괄호** — 전 구간 **+15%p**를 회수했다.

| 사례 | 이전 | 이후 |
|---|---:|---:|
| `a fever (either felt or measured with a thermometer)` → `had a fever (defined as 100F or higher)` | 0.22 | **1.00** |
| `coughing up blood` → `recently had a cough that produced blood` | 0.29 | 0.57 |
| `had chills or shivers` → `they feel like they are shivering` | 0.00 | 0.40 |

DDXPlus는 cue를 설문 문항의 해설 괄호까지 붙여 렌더링한다. 소견은 "fever"인데
gold는 측정 방법까지 포함하므로, 정확한 판독이 구조적으로 감점된다.

### 그래도 임계값으로는 못 가른다

`undergo dialysis` → `have to dialyze`는 **명백한 정답인데 0.00**이다. 어간이
`dialy-sis`/`dialy-ze`로 갈려 접미사 규칙으로 붙지 않는다. 가장 나쁜 오답
(0.18)보다 낮은 정답이 존재한다. 테스트에 이 사실을 고정해 두었다.

**따라서 overlap은 지표가 아니라 정렬 도구다.** LLM 판정기가 "있으면 좋은 것"이
아니라 필수인 근거가 이것이다.

최하위 25행을 수기 판정한 결과: **A 15 / B 6 / C 4.** 하위 구간의 60%가 정답이므로
0.713도 하한이다.

---

## 3-2. 수기 판정 60건 (두 pool 무작위 표본)

| | seen (n=32) | heldout (n=28) |
|---|---:|---:|
| **A 같은 소견** | 31 (96.9%) | **19 (67.9%)** |
| **B 부분** | 0 | 8 (28.6%) |
| **C 오답** | 1 (3.1%) | 1 (3.6%) |

**heldout A+B = 96.4%, C는 1건.** 이것이 가장 중요한 관측이다 — 판독기가
**엉뚱한 소견을 지어내는 일이 거의 없다.** 실패는 "다른 것을 말한다"가 아니라
"덜 정확하게 말한다"이며, 충실성 주장에서 이 차이는 결정적이다.

### B 8건은 전부 속성 오류다

| 유형 | 건수 | 예 |
|---|---:|---|
| 속성 값 오독 | 3 | `itching rated 10` → `rated 3` / `high pitched sound` → `wheezing` (×2) |
| 없는 수치 생성 | 2 | `involuntary weight loss` → `lost 2 pounds` / `more than 5 lbs` |
| 복합 cue의 한 조각 | 3 | `상복부에서 인후로 타는 느낌 + 입 쓴맛` → `쓴맛`만 |

**소견 자체는 맞고 속성이 틀린다.** 파일럿이 "속성 해상도(C 26.0%)"로 뭉뚱그린
것의 유형 분해이며, 이 표는 논문에 그대로 쓸 수 있다.

seen에도 오답이 1건 있다(`intense coughing fits` → `intense muscle spasms`).
암송조차 완전하지 않다.

### 임계값과 수기의 일치

heldout 28건에서 `f1 ≥ 0.5` 규칙은 21건(75%)을 통과시키고 수기 A는 19건(67.9%)이다.
양방향으로 틀린다 — f1 ≥ 0.5인데 B가 4건, f1 < 0.5인데 A가 2건 — 오차가 상쇄되어
**총계는 7%p 안에서 맞는다.** 즉 overlap은 **집계 근사치로는 쓸 만하고 개별 행
판정으로는 못 쓴다.**

### 한계

heldout 28건이므로 67.9%의 95% 신뢰구간이 **±17%p**(51~85%)다. 논문 수치로는
너무 넓다. 수기를 200건으로 늘리는 것보다, **판정기로 770건 전량을 판정하고 이
60건 + 최하위 25건을 검증 표본으로 κ를 보고**하는 편이 낫다. E5를 P0으로 올린
결정이 여기서 다시 확인된다.

---

## 4. 품질 결함 두 가지

**반복 붕괴.** `numbness or tingling affected their upper gum, labia majora
majora majora majora(R), bottom lip(R), ...` greedy 디코딩 병리이며 셀 수 있다.

**위치 어휘 살포.** numbness 케이스 3건이 모두 위치를 틀렸고
(`arms and legs and mouth` → `upper gum, bottom lip`), 틀린 위치가 DDXPlus 신체부위
어휘에서 나왔다. **어댑터가 위치를 읽는 대신 외운 어휘에서 고를 가능성**이 있다.
반사실 실험(위치 cue 교체 시 판독이 따라가는가)에서 직접 검증된다.

---

## 5. 이 결과가 정하는 것

**에폭.** seen이 1에폭에 이미 0.986으로 포화다. 에폭을 더 주면 seen만 오를
가능성이 크지만 **검증하지 않았다.** 9시간(1에폭×18런)과 27시간(3에폭×18런)이
걸린 문제라 대조군 하나를 돌린다: `ddxplus_L24_s17_ep3`의 heldout이 0.713보다
유의하게 높은지 본다.

**source-correct 필터.** heldout이 무너진 시나리오가 아니므로 급하지 않다.
부차 실험으로 내린다.

**아직 답하지 못한 것.** "LoRA가 판독 능력 자체를 향상시켰는가"는 **바닐라 L24**가
있어야 답한다. 현재 도는 바닐라는 L16이라 레이어가 달라 비교 불가.

바닐라 표본 3건을 눈으로 본 결과는 예상과 반대였다 — 바닐라도 heldout cue를
읽어낸다. 못 하는 것은 읽기가 아니라 **멈추기**다: 태그를 방출하지 않고 1,600자를
쓰며, 내용이 다음 토큰 예측 서술에 묻힌다. 행당 20초로 어댑터의 4.3배가 걸리는
이유도 이것이다.

사실이라면 표1·표2(B)의 서사가 바뀐다. "바닐라는 못 읽는다"가 아니라
**"바닐라는 읽지만 쓸 수 없다"** 이고, 후자가 더 정확할 뿐 아니라 더 강하다 —
활성 벡터에 소견이 들어 있음을 어댑터 없이도 보이기 때문이다. 어댑터의 기여는
그것을 **감사 가능한 형태로 만드는 것**이 된다.

---

## 6. 재현 명령

```bash
source scripts/env.sh
SPLIT=$ART/train/ddxplus_cuepos_L24
AD=$ART/train/adapters/ddxplus_L24_s17

python -m src.run_nla --config configs/default.yaml \
  --manifest $SPLIT/manifest_test_heldout_cue.jsonl \
  --output $ART/results/readout_L24_s17_heldout_ep1.jsonl --adapter-id $AD

python -m src.run_nla --config configs/default.yaml \
  --manifest $SPLIT/manifest_test_seen_cue.jsonl \
  --output $ART/results/readout_L24_s17_seen_ep1.jsonl \
  --adapter-id $AD --limit 770

python scripts/score_cue_position_readouts.py \
  --heldout $ART/results/readout_L24_s17_heldout_ep1.jsonl \
  --seen $ART/results/readout_L24_s17_seen_ep1.jsonl \
  --dump-sample 60 --dump $ART/reports/readout_L24_s17_sample.tsv
```

생성은 행당 4.2초, pool당 약 55분.
