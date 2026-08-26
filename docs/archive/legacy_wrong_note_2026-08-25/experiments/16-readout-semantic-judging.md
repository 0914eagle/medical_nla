# 16 — 판독 의미 채점: 손채점과 외부 판정자 (Appendix Table A1)

**질문**: Appendix Table A1의 `.340 / .731 / .557`은 **누가** 매긴 값인가. 그리고 그
숫자는 채점자를 바꾸면 얼마나 움직이는가.

**상태**: ✅ 완료 2026-08-24. 두 채점자 전수, 파싱 실패 0.

---

## 왜 이 실험이 필요했는가

heldout cue 판독의 기계 채점(어휘 재현율 `.751`)은 표현이 다르면 놓친다.
"undergo dialysis"와 "have to dialyze"는 같은 소견인데 단어가 겹치지 않는다.
그래서 별도로 **의미 채점**을 했고 그 값이 Appendix Table A1에 들어갔다.

그런데 **채점 주체 칸이 비어 있었다.** 이번에 복원한 결과 이유가 드러났다 —
**저자 본인이 손으로 매긴 값**이다. 심사에서 이건 바로 찔린다: 자기가 만든
판독을 자기가 채점해서 잘 나왔다고 하는 것 아니냐.

그래서 **백본이 아닌 외부 모델에게 같은 쌍을 같은 루브릭으로 다시 채점**
시켰다. 목적은 손채점을 대체하는 것이 아니라 **두 채점자가 얼마나 갈리는지를
재는 것**이다.

---

## 설정

| | |
|---|---|
| 판독 대상 | heldout cue 판독 (학습에서 제외한 소견 문자열) |
| 층 | L16_v5 / L24_v5 / v4(=L32) |
| 행 | 438 / 층 |
| **고유 쌍** | **72 / 74 / 92 = 238** |
| 손채점자 | **저자 본인** (2026-08-17) |
| 외부 판정자 | **`gpt-5.6-sol`** via codex CLI (2026-08-24) |
| 판정 실패 | 0 |
| 비용 | dry-run 견적 3층 합계 **~$0.10** |

**어댑터 주의**: 이 스윕은 L16/24가 2에폭, L32가 3에폭이다. 레이어 효과에
에폭 효과가 섞여 있으므로 층 간 비교를 인과로 읽지 않는다.

### 438행이 238쌍으로 접히는 이유

DDXPlus는 고정 문진표에서 소견을 렌더링한다. 그래서 같은 (gold, readout)
쌍이 반복된다 — 한 쌍이 최대 **53행**을 차지한다. 판정자에게는 **쌍마다 한
번만** 묻는다. 그래야 같은 쌍이 어떤 행에서는 A, 다른 행에서는 B로 나오는
일이 없다. 비용도 5.5배 준다.

`judge_jobs_2026-08-24.md`가 이 작업을 n=1,314(438×3)로 잡았던 것은 행 기준
추정이었다. 실제 규모는 238이다.

---

## 판정자에게 보낸 프롬프트 (전문)

`scripts/make_readout_judge_requests.py`의 `SINGLE_RUBRIC`. 쌍마다 이 텍스트
전체가 GOLD/READ만 바뀌어 전송된다.

```
You are scoring whether a readout describes the same clinical finding as the
gold text. The readout was produced by a model reading an internal activation
vector; the gold is the finding that vector was taken from. Judge meaning, not
wording.

  A  the same finding, however differently worded
     ("undergo dialysis" / "have to dialyze" is A)
     ("a fever (either felt or measured with a thermometer)" / "had a fever
      (defined as 100F or higher)" is A -- the gloss is the questionnaire's,
      not part of the finding)
  B  the finding is right but an attribute is wrong, missing or invented, or
     only one clause of a compound gold is read
     ("the itching rated 10 for severity" / "rated 3 for severity" is B)
     ("involuntary weight loss over 3 months" / "lost 2 pounds in the last 3
      months" is B -- the quantity is not in the gold)
  C  a different finding
  D  empty, refused, or no clinical content

When A and B are both arguable, answer B. When B and C are both arguable,
answer C. The rate being measured is a claim about faithfulness, so err
against it.

Reply with exactly one character and nothing else: A, B, C or D.

GOLD: {gold}
READ: {read}
```

실제 전송 예시 (L24_v5, id=1):

```
GOLD: that muscle spasms or soreness in their neck are keeping the patient
      from turning their head to one side
READ: a painful contraction in one side of the neck
```

### 프롬프트 설계에서 의도한 것

- **한 쌍에 한 요청.** 초기 설계는 120쌍을 한 프롬프트에 넣고 `1=A 2=C ...`로
  받았다. 판정자가 한 줄을 건너뛰거나 번호를 다시 매기면 **그 뒤 판정이 전부
  한 칸씩 밀린다.** 조용히 틀린다. 쌍마다 요청하면 어긋날 수가 없고, 죽어도
  쌍 단위로 재개된다. 대가는 루브릭 반복 비용뿐이다.
- **불리한 쪽으로 기울이라는 지시.** 측정 대상이 "판독이 충실한가"라는
  우리 쪽 주장이므로, 애매하면 낮은 등급으로 가라고 명시했다.
- **백본 거부.** `run_judge.py`가 판정자 이름에 `gemma`가 있으면 거부한다.
  Gemma 판독을 Gemma로 채점하는 것이 가장 뻔한 반론이다.

---

## 결과

### 두 채점자 × 두 가중

| 층 | | A | B | C | D | **쌍 단위 A+B** | **행 가중 A+B** |
|---|---|---:|---:|---:|---:|---:|---:|
| L16_v5 | 손채점 | 8 | 17 | 39 | 8 | .3472 | **.3402** |
| | 판정자 | 13 | 17 | 42 | 0 | — | **.5525** |
| L24_v5 | 손채점 | 9 | 37 | 24 | 4 | .6216 | **.7306** |
| | 판정자 | 35 | 17 | 22 | 0 | — | **.7740** |
| v4 (L32) | 손채점 | 11 | 38 | 37 | 6 | .5326 | **.5571** |
| | 판정자 | 28 | 27 | 37 | 0 | — | **.6393** |

### 두 채점자의 일치

| 층 | exact 4-way | collapsed A+B / C+D | Cohen's κ |
|---|---:|---:|---:|
| L16_v5 | .6944 | .8750 | **.4976** |
| L24_v5 | .5270 | .9189 | **.3531** |
| v4 (L32) | .6413 | .8696 | **.4730** |

---

## 읽는 법

### 1. 손채점은 낙관이 아니라 보수 편향이었다

세 층 모두 외부 판정자가 **더 후하다**(+.21 / +.04 / +.08). 반대였다면
"자기 채점이라 부풀렸다"가 되지만, 실제로는 저자가 더 엄격했다. 지금까지
인용한 `.340/.731/.557`은 상한이 아니라 **하한**이다.

### 2. 가중 방식이 채점자보다 숫자를 더 움직인다

| 층 | 상위 5쌍이 덮는 행 | 최대 단일 쌍 |
|---|---:|---:|
| L16_v5 | 146 (33.3%) | 53행 (12.1%) |
| L24_v5 | 128 (29.2%) | 35행 (8.0%) |
| v4 | 129 (29.5%) | 38행 (8.7%) |

**쌍 하나가 C↔B로 뒤집히면 행 가중 값이 최대 12%p 움직인다.** L16의 +.2123은
채점 성향의 전면적 차이가 아니라 **큰 쌍 두셋이 넘어간 결과**로 보아야 한다.

그리고 L24에서 **가중 차이(.6216 vs .7306 = 10.9%p)가 채점자 차이(+4.3%p)보다
크다.** 누가 채점했느냐보다 어떻게 세느냐가 더 중요하다. 행 가중만 쓰면
"문진표가 무엇을 자주 렌더링했는가"가 판독 성능으로 읽힌다.

→ **표에는 쌍 단위와 행 가중을 함께 싣는다.**

### 3. 판정자의 여유분은 좌우·부위에 몰려 있다

손채점 C → 판정자 B로 올라간 쌍들:

| GOLD | READ |
|---|---|
| swelling located **thigh(L)** | swelling located **thigh(R)** |
| swelling located **cheek(R)** | swelling located **cheek(L)** |
| affected region **bottom lip(R)** | **upper lip(R)** |
| **dorsal aspect** of the foot(R) | **lateral side** of the foot(R) |
| stools that were **black (like coal)** | **light red blood or blood clots** in stool |

루브릭은 "B와 C가 둘 다 그럴듯하면 C"라고 했는데 판정자는 반대로 갔다.
마지막 행은 흑색변 vs 혈변으로 출혈 부위가 다른 별개 소견이다.

**이건 판정자 잘못이라기보다 루브릭의 빈칸이다.** B의 예시 두 개가 전부
**수량** 속성("강도 10 vs 3", "3개월 체중감소 vs 2파운드")이고, 좌우/부위를
어떻게 다룰지는 적혀 있지 않다. 두 채점자가 각자 다르게 추측했다.

`analyze_readout_semantic_judgements.py`의 `differs_only_by_site()`가 이
분해를 자동으로 센다. 위치를 가리키는 단어(`left/right`, `(L)/(R)`,
`upper/lower`, `dorsal/lateral`, `aspect/side/region` 등)에서만 다른 쌍을
따로 집계하고, 여유분의 1/3 이상이 거기서 나오면 경고한다. 흑색변 vs 혈변,
"밤에 숨 막혀 깬다" vs "밤에 증상이 심하다"처럼 단어는 겹치지만 다른 소견인
경우는 걸러지도록 테스트 7개(`tests/test_semantic_judgement_audit.py`)로
고정했다.

### 4. 철회 — D는 문제가 아니었다

판정자가 D를 한 번도 주지 않은 것을 처음에 "척도의 바닥을 안 써서 그 위가
밀려 올라갔다"고 적었다. **틀렸다.**

손채점이 D로 준 쌍(6/8/4)의 판독을 열어보면 전부 멀쩡한 임상 문장이다:

```
"a cough that produces colored or more abundant sputum"
"been coughing up blood or something resembling coffee grounds"
"a loss of appetite or do the patient get full more quickly than usual"
```

루브릭의 D는 "비었거나 거부했거나 **임상 내용이 없음**"이므로 이것들은 **C**가
맞다. **D를 안 쓴 판정자가 루브릭대로 한 것이고, 손채점이 D를 "완전 오답"으로
오용했다.** 그리고 C와 D는 둘 다 A+B 밖이라 **보고 수치에 영향이 없고**,
층 간 격차를 설명하지도 않는다.

---

## 무엇을 의미하지 않는다

- **외부 판정자가 임상의는 아니다.** 이 실험은 "같은 루브릭 아래 다른
  채점자가 얼마나 갈리는가"를 잰다. 판독의 **임상적 유용성**은 여기서
  답하지 않는다 — 그건 [14](14-reader-trust.md)이고, 거기서는 판독이
  독자에게 **해롭다**(−.0921 [−.135,−.046]).
- **κ .35–.50은 "중간"이다.** 이 과제가 채점자에 따라 흔들린다는 뜻이고,
  그 사실 자체를 숨기지 않고 보고한다.
- **층 간 비교는 에폭이 섞여 있다.** L24가 가장 높은 것을 "L24가 최적 층"의
  증거로 쓰지 않는다.
- **`gpt-5.6-sol`은 codex 실행 당시의 라우팅 이름이다.** 고정된 공개 API
  snapshot을 보장하지 않는다 → [15](15-judge-infrastructure.md).

---

## 재실행할 것인가

**권고: 하지 않는다.** 숫자가 마음에 안 들어서 루브릭을 조여 다시 돌리는
것은 원하는 답이 나올 때까지 채점자를 고르는 것이고, 외부 검증의 의미가
사라진다.

정당한 재실행은 하나뿐이다 — **좌우/부위 규정을 루브릭에 추가하는 것**.
이건 결과가 싫어서가 아니라 계기에 빈칸이 있어서다. 다만 그 개정판을 돌린다면
**첫 실행을 지우지 않고 셋 다 싣고**, 루브릭이 결과를 본 뒤에 고쳐졌음을
명시해야 한다. 좌우 문제는 이미 `differs_only_by_site`로 수치화되므로
재실행 없이도 표에 쓸 수 있다.

---

## 표에 어떻게 싣는가

```
heldout cue 의미 채점 (L24, n=438 / 74쌍)
  손채점(저자)        쌍 .6216  행 .7306
  gpt-5.6-sol         쌍 ——     행 .7740        κ = .3531
  판정자 여유분 중 좌우·부위만 다른 쌍: (differs_only_by_site 출력)
```

한 숫자를 방어하는 것보다 **두 채점자 × 두 가중을 다 보여주는 쪽**이 강하다.
심사자가 어느 조합으로 읽어도 우리가 이미 그 값을 알고 있었던 것이 된다.

---

## 재현

```bash
source scripts/env.sh

# 견적
DRY=1 bash scripts/run_readout_semantic_judge.sh

# 본 실행 (3층, ~$0.10)
bash scripts/run_readout_semantic_judge.sh

# 손채점만 다시 집계 (판정 없이, 리포지토리만으로 가능)
python scripts/make_readout_judge_requests.py \
  --readouts results_snapshot/L24_v5_test_heldout_cue_scored_compact.jsonl \
  --out-dir  /tmp/j3/L24_v5
python scripts/analyze_readout_semantic_judgements.py \
  --index /tmp/j3/L24_v5/judge_index.jsonl \
  --hand  results_snapshot/L24_v5_heldout_pairs_hand_labeled.jsonl \
  --label L24_v5

# 판정자 id 확인
python -c "import json;print(json.loads(open('\$ART/results/judge_readout_semantic_L24_v5.jsonl').readline()).get('judge_model'))"
```

입력은 전부 `results_snapshot/`에 있다 — **GPU도 `\$ART`도 필요 없다.**

| 파일 | 내용 |
|---|---|
| `{층}_test_heldout_cue_scored_compact.jsonl` | 438행 판독 (튜닝판) |
| `{층}_heldout_vanilla_compact.jsonl` | 438행 무학습 대조 |
| `{층}_heldout_pairs_hand_labeled.jsonl` | 쌍 단위 손라벨 A/B/C/D |
| `$ART/results/judge_readout_semantic_{층}.jsonl` | 판정자 응답 + `judge_model` |

관련: [01](01-readout-instrument-validation.md) 계기 검증 ·
[02](02-layer-sweep.md) 레이어 스윕 ·
[15](15-judge-infrastructure.md) 판정자 기반 ·
[14](14-reader-trust.md) 독자 신뢰
