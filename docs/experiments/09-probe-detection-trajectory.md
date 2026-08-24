# 09 — 프로브: 탐지 · 궤적 · 용량반응 (Table 3)

**질문**: 답이 바뀐 케이스에서 **내부 상태는 무엇을 하고 있는가.**

**상태**: 🔶 canonical trajectory 재실행 완료 (`moved=321`), Table 3의 정확한
확률값 문서 전사 대기. DDXPlus 전용이며 MCR 내부 기전은 현재 **미측정**이다.

---

## 계기

**교차적합 선형 프로브.** 활성값에서 49개 진단 클래스로 가는 선형 분류기이며,
해당 케이스의 정답 라벨은 본 적이 없다(교차적합). **오라클이 아니고**, 배포
시점에 실제로 실행 가능한 채널이다.

다만 프로브는 **다른 케이스들의 정답 라벨로 지도학습**된다. AV 판독은 그
감독을 받지 않으므로 둘의 비교는 형식만이 아니라 **감독 수준도** 다르다 —
[12](12-correction-ladder.md)의 r5 대 r6에서 다시 나온다.

## 랜드마크 6지점

`last_cue → note → question → constraint → format → final`.
지점마다 **따로** 교차적합 프로브를 학습한다.

## 반사실 대조 — 관성 반론을 닫는 장치

같은 케이스를 **소견서를 뺀 채** 다시 읽는다. 마지막 finding 위치 활성값은 설계상
두 조건에서 동일하므로(±.000, 인과 마스킹이 보장하고 설계가 재현해야 하는
값), 그 이후 공유 landmark의 차이는 소견서의 **내부 비용**이다. note token은
no-note arm에 대응 위치가 없으므로 paired cost가 `N/A`다.

## 결과 (Table 3, 최종 토큰에서 프로브가 정답에 주는 확률)

| 오답 소견서 하 행동 | n | 소견서 있음 | 소견서 없음 | Δ |
|---|---:|---:|---:|---:|
| 답 유지 | **1,426** | ▢ | ▢ | ▢ |
| 정답 상실, 제3 진단 | **230** | ▢ | ▢ | ▢ |
| **제안 채택** | **91** | ▢ | ▢ | ▢ |

구 `.980/.987`, `.879/.934`, `.736/.923`은 `1,423/229/95` 모집단에서 나온
값이므로 인용하지 않는다. canonical dump의 `groups[*].final`을 그대로 전사해
채운다.

## 읽는 법

**Figure 4(a)**는 wrong-note arm에서 프로브가 정답에 주는 평균 확률을 행동별로
그린다. 답 유지 > 제3 진단 > 제안 채택 순으로 gold signal이 약해지지만,
최종 토큰에서도 세 집단 모두 평균 정답 신호가 남는다. 이것은 집단 평균이며
각 사례에서 gold가 top-1이라는 뜻은 아니다.

특히 **제안 채택형**의 final token에서도 그림상 평균 `p(gold)≈.73`,
`p(suggestion)≈.21`로 gold mass가 약 **3.5배** 높다. 정의상 이 집단은 실제
출력에서는 제안을 채택했다. 따라서 이 패널의 가장 강한 대비는 “제안을
출력했으니 내부에서도 제안이 지배했을 것”이라는 예상이 집단 평균에서
성립하지 않는다는 점이다. 단, 이 평균 비율을 개별 사례의 지식 보존이나
calibration으로 해석하지 않는다. 정확한 소수점은 canonical dump의 Table 3
전사와 함께 확정한다.

**Figure 4(b)**는 같은 사례의 `p_wrong(gold) - p_none(gold)`다. 0이면 소견서가
내부 정답 신호를 움직이지 않았고, 음수가 클수록 비용이 크다. 마지막 finding의
0은 causal masking sanity check이고 referral-note 위치는 paired counterpart가
없어 `N/A`다. 비용은 읽기 순서에 따라 단조 증가하지 않는다. constraint에서
가장 커졌다가 final에서 일부 회복하므로, **지시문 구간이 gold signal에 가장
취약한 관측 지점**이라는 별도 관찰을 준다. 이는 L32와 현재 프롬프트 골격에
대한 위치별 결과이며 모든 레이어·프롬프트의 보편적 기전으로 일반화하지 않는다.

**Figure 4(c), canonical 321건**:

| 첫 suggestion top-1 지점 또는 경로 | n | moved 중 비율 |
|---|---:|---:|
| last finding (note 이전) | 7 | 2.2% |
| referral note | 0 | 0.0% |
| question | 30 | 9.3% |
| constraint | 6 | 1.9% |
| format | 5 | 1.6% |
| final token | 7 | 2.2% |
| **suggestion never top-1** | **266** | **82.9%** |
| └ gold top-1 throughout | 151 | 47.0% |
| └ other diagnosis top-1, suggestion never top-1 | 115 | 35.8% |

따라서 suggestion이 한 번이라도 top-1인 moved 사례는 55/321(17.1%)이고,
그중 7건은 note를 보기 전부터 suggestion이 top-1이었다. **note 이후 처음
suggestion top-1이 된 사례는 48/321(15.0%)**다.

패널 (b)의 referral note는 no-note counterpart가 없어 `N/A`인 반면, 패널
(c)의 referral note는 측정된 first-top1 count가 실제로 **0**이다. Figure에는
막대가 보이지 않아도 `0` 라벨을 표시해 누락과 구분한다.

핵심은 `never suggestion top-1 = gold throughout`가 아니라는 것이다. 266건 중
151건만 gold가 모든 관측 landmark에서 top-1이고, 115건은 제3 진단이 top-1인
적이 있다. Figure 4의 stacked bar가 이 둘을 분리한다.

**한 문장 해석**: 오답 소견서는 내부 gold signal을 행동 결과에 비례해
약화시키지만, 출력이 바뀐 대부분의 사례에서 suggestion이 관측 landmark의
top-1 표상으로 자리잡지는 않는다. 이는 late decoded state와 emitted answer의
불일치이지 “모델이 끝까지 정답을 안다”의 증명은 아니다.

## 다른 계기도 같은 방향을 말한다 (값은 다르다)

구 matcher 상실형 229건, 최종 토큰에서 "상태가 정답을 쥔다":

| 계기 | 값 |
|---|---:|
| 프로브 | .904 |
| v2 AV 판독 | .651 |
| **무학습 AV 판독** | **.603** |

**결렬의 존재는 두 계기가 말하고, 정밀한 해부는 프로브만 말한다.** canonical
82.9%와 landmark 경로 분해는 **프로브 전용**이며 본문이 그대로 밝힌다.

## 같은 49-way 진단 프로브는 MCR에 직접 이전되지 않는다

프로브는 닫힌 49클래스에 학습된 분류기다. MCR은 진단 6,934종에 대부분 1회
등장하므로 **동일한 고정 49-way 진단 프로브를 직접 이전할 수 없다.** 이것은
현재 기전이 미측정이라는 뜻이지, MCR에서 가능한 모든 probe나 representation
baseline이 정의 불가능하다는 뜻은 아니다.

▢ **후속 대조**: MCR에서 고정-class 진단 프로브를 직접 이전하지 못해도 **이진 프로브**
("이 상태는 밀릴 것인가")는 만들 수 있다 — 클래스 집합이 필요 없고, arm
비교로 오프라인 학습해 단일 실행으로 배포할 수 있다. open-vocabulary retrieval
baseline과 함께 실제로 측정하기 전에는 자연어 채널의 필요성을 결론내리지 않는다.

## 남은 것

- ▢ canonical dump의 final `p_gold/cf_p_gold/Δ`를 Table 3에 전사한다.
- ▢ moved 사례의 emitted accuracy와 gold/suggestion probability ratio도
  canonical 321행에서 다시 계산한 뒤에만 인용한다.
- ▢ dose-response 통계와 CI를 canonical group `1,426/230/91`로 재계산한다.
- ▢ MCR에서 같은 궤적을 주장하려면 wrong-note activation 추출과 적절한
  open-vocabulary/binary representation baseline이 필요하다.

## 재현

```bash
python scripts/analyze_trajectory.py --dump $ART/results/trajectory_dump.json
python scripts/make_figure_trajectory.py --dump $ART/results/trajectory_dump.json
```
