# 09 — 프로브: 탐지 · 궤적 · 용량반응 (Table 2a · Figure 3)

**질문**: 답이 바뀐 케이스에서 **내부 상태는 무엇을 하고 있는가.**

**상태**: ✅ canonical-eligible trajectory와 Table 2a 전사 완료 (`n=1,729`, `moved=319`).
DDXPlus 전용이며 MCR 내부 기전은 현재 **미측정**이다.

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

## 결과 (Table 2a, 최종 토큰에서 프로브가 정답에 주는 확률)

| 오답 소견서 하 행동 | n | 소견서 있음 | 소견서 없음 | Δ |
|---|---:|---:|---:|---:|
| 답 유지 | **1,410** | **.981** | **.987** | **−.006** |
| 정답 상실, 제3 진단 | **230** | **.878** | **.932** | **−.054** |
| **제안 채택** | **89** | **.730** | **.929** | **−.199** |

canonical-eligible 그룹에서도 내부 비용은 답 유지 < 제3 진단 < 제안 채택
순으로 커진다. 행동 심각도 순서와 final cost의 Spearman ρ는 **−.282**, 95% CI
**[−.328, −.233]**으로 0을 배제한다.

## 읽는 법

**Figure 3(a)**는 wrong-note arm에서 프로브가 정답에 주는 평균 확률을 행동별로
그린다. 답 유지 > 제3 진단 > 제안 채택 순으로 gold signal이 약해지지만,
최종 토큰에서도 세 집단 모두 평균 정답 신호가 남는다. 이것은 집단 평균이며
각 사례에서 gold가 top-1이라는 뜻은 아니다.

특히 **제안 채택형**의 final token에서도 평균 `p(gold)=.730`,
`p(suggestion)=.212`로 gold mass가 약 **3.4배** 높다. 정의상 이 집단은 실제
출력에서는 제안을 채택했다. 따라서 이 패널의 가장 강한 대비는 “제안을
출력했으니 내부에서도 제안이 지배했을 것”이라는 예상이 집단 평균에서
성립하지 않는다는 점이다. 단, 이 평균 비율을 개별 사례의 지식 보존이나
calibration이나 모델의 실제 next-token 확률로 해석하지 않는다. 둘 다
49-way probe가 디코드한 집단 평균 확률이다.

**Figure 3(b)**는 같은 사례의 `p_wrong(gold) - p_none(gold)`다. 0이면 소견서가
내부 정답 신호를 움직이지 않았고, 음수가 클수록 비용이 크다. 마지막 finding의
0은 causal masking sanity check이고 referral-note 위치는 paired counterpart가
없어 `N/A`다. 비용은 읽기 순서에 따라 단조 증가하지 않는다. constraint에서
가장 커졌다가 final에서 일부 회복하므로, **지시문 구간이 gold signal에 가장
취약한 관측 지점**이라는 별도 관찰을 준다. 이는 L32와 현재 프롬프트 골격에
대한 위치별 결과이며, 랜드마크마다 별도 probe를 학습했으므로 모든
레이어·프롬프트의 보편적 기전이나 단일 probe의 시간 변화로 일반화하지 않는다.

canonical paired cost의 핵심 지점은 다음과 같다.

| 지점 | 제안 채택 Δ | 정답 상실 Δ |
|---|---:|---:|
| question | −.171 | −.060 |
| **constraint** | **−.467** | **−.299** |
| format | −.188 | −.189 |
| final | −.199 | −.054 |

두 moved 집단 모두 constraint에서 비용이 가장 크고 final prompt token에서
일부 회복한다. 그런데 이 사례들은 이후 잘못된 답을 생성한다. 따라서 관측된
gold signal의 회복은 올바른 출력에 **충분하지 않았다**. 이를 “회복 신호가
출력 경로에 전달되지 않았다”는 인과 주장으로 확대하지는 않는다.

**Figure 3(c), canonical-eligible 319건**:

| 첫 suggestion top-1 지점 또는 경로 | n | moved 중 비율 |
|---|---:|---:|
| last finding (note 이전) | 7 | 2.2% |
| referral note | 0 | 0.0% |
| question | 29 | 9.1% |
| constraint | 10 | 3.1% |
| format | 5 | 1.6% |
| final token | 6 | 1.9% |
| **suggestion never top-1** | **262** | **82.1%** |
| └ gold top-1 throughout | 147 | 46.1% |
| └ other diagnosis top-1, suggestion never top-1 | 115 | 36.1% |

따라서 suggestion이 한 번이라도 top-1인 moved 사례는 57/319(17.9%)이고,
그중 7건은 note를 보기 전부터 suggestion이 top-1이었다. **note 이후 처음
suggestion top-1이 된 사례는 50/319(15.7%)**다.

패널 (b)의 referral note는 no-note counterpart가 없어 `N/A`인 반면, 패널
(c)의 referral note는 측정된 first-top1 count가 실제로 **0**이다. Figure에는
막대가 보이지 않아도 `0` 라벨을 표시해 누락과 구분한다.

note landmark에서 gold-label probe가 준 `p(suggestion)`은 세 집단 모두
표시 정밀도에서 `.000`이다. 이는 **이 probe가 그 지점에서 suggestion을
진단 top-1 신호로 디코드하지 못했다**는 뜻이다. suggestion 정보가 activation에
전혀 없다는 뜻은 아니다. 그 부재를 검증하려면 suggestion identity를 직접
라벨로 둔 probe나 matched-vs-mismatched retrieval 검사가 필요하다.

핵심은 `never suggestion top-1 = gold throughout`가 아니라는 것이다. 262건 중
147건만 gold가 모든 관측 landmark에서 top-1이고, 115건은 제3 진단이 top-1인
적이 있다. Figure 3의 stacked bar가 이 둘을 분리한다.

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
82.1%와 landmark 경로 분해는 **프로브 전용**이며 본문이 그대로 밝힌다.

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

- ✅ canonical final `p_gold/cf_p_gold/Δ`와 채택형 `p(suggestion)=.212` 전사.
- ✅ Table 2a의 paired bootstrap CI와 행동 심각도 순서의 Spearman 추세 검정
  전사. final ρ=−.282 [−.328,−.233].
- ▢ constraint 최대값이 landmark별 probe calibration 차이인지 확인하려면
  각 위치 probe의 heldout 성능·calibration을 함께 보고한다.
- ▢ suggestion 표상 부재를 주장하려면 hint-label probe/retrieval 대조를 추가한다.
- ▢ MCR에서 같은 궤적을 주장하려면 wrong-note activation 추출과 적절한
  open-vocabulary/binary representation baseline이 필요하다.

## 재현

```bash
python scripts/analyze_trajectory.py --dump $ART/results/trajectory_dump_canonical_eligible.json
python scripts/make_figure_trajectory.py \
  --dump $ART/results/trajectory_dump_canonical_eligible.json \
  --output $ART/results/paper_figures/figure3_trajectory_canonical_eligible.png
```
