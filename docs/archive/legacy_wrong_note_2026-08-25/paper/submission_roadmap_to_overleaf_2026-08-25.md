# 제출까지 남은 실험·문서·Overleaf 이전 로드맵 (2026-08-25)

이 문서는 현재 정본 결과에서 **무엇이 이미 성립하고**, **무엇은 추가 실험 전에는
말할 수 없으며**, **어떤 순서로 결과를 동결해 Overleaf 원고로 옮길지**를 정한다.
새 수치는 먼저 `docs/experiments/RESULTS_CANONICAL_2026-08-24.md`에 기록하고,
그 뒤 표·그림·본문·발표 문서로 전파한다.

---

## 1. 지금 논문이 이미 말할 수 있는 것

### RQ1 — 행동 효과와 내부 궤적

- Primary behavior cohort는 explicit gold-name leakage를 제외하고 canonical
  no-note 정답인 **DDXPlus clean 1,204건**이다. Wrong note 정확도는 `.7625`로,
  no-note 기준 `1.0000`에서 **23.75%p** 하락한다.
- MedCaseReasoning(MCR) canonical cohort **1,452건**에서도 wrong-note 비용은
  **29.34%p**다. 행동 현상은 합성 문진과 실제 case report 양쪽에서 재현된다.
- 내부 분석은 activation이 이미 추출된 DDXPlus canonical-eligible **1,729건**
  (moved 319)을 쓴다. 이 모집단은 Figure 2의 clean 1,204와 다르며 섞어 쓰지
  않는다.
- Moved 319건 중 suggestion이 관측한 여섯 landmark에서 한 번도 probe top-1이
  아닌 사례가 **262건(82.1%)**이다. 그중 gold top-1 throughout는 147건,
  suggestion 이외의 진단이 top-1인 경로는 115건이다.

따라서 확립된 주장은 “모델이 속으로 항상 정답을 안다”가 아니라, **행동적
답변 이동과 내부에서 decode되는 suggestion dominance가 같은 사건이 아니다**이다.

### RQ2 — wrong-note 단일 실행에서의 영향 판별

실험자는 같은 case의 no-note/wrong-note pair로 `moved` 정답 라벨을 만든다.
그러나 detector에는 gold도 no-note 실행도 주지 않고 **현재 wrong-note 실행
하나의 관측치만** 준다.

- Linear probe: all/silent AUROC **`.9330/.9881`**
- AV readout: **`.7511/.8319`**
- LLM monitor: **`.7305/.6904`**
- Answer heuristic: all **`.6632`**
- Rule-based CoT: all **`.5434`**

이는 DDXPlus의 49-class 지도 probe가 가장 강한 detector임을 뜻한다. AV는 probe를
이겼다는 결과가 아니며, 자연어 후보와 열린 어휘 확장 가능성을 시험하는 보조
계기다.

### RQ3 — 현재 확립된 것은 oracle/conditional information value

현재 교정 사다리는 **사후 평가에서 moved로 확인된 사례**에 내부 content가
유용한지를 보여준다.

- First wrong answer moved accuracy: `.0031`
- Reconsideration r3: `.4545`
- Findings re-shown r4: `.4044`
- AV content r5: `.6301`
- Probe label r6: `.8339`
- r5 − r4: **+22.6%p**

따라서 다음 문장은 성립한다.

> 이미 harmful movement가 발생한 사례에서, 내부 판독 내용은 단순 재고나
> 소견 재제시보다 교정에 유용한 정보를 제공한다.

그러나 r5를 전 사례에 적용하면 전체 정확도가 `.8161 → .4083`으로 떨어진다.
따라서 다음 문장은 아직 강하게 쓸 수 없다.

> 내부 판독을 사용하면 전체 의료 QA 성능이 향상된다.

과거 fixed-cohort에서 `probe disagreement → r5` 정책은 전체 `.9141`,
`probe disagreement → probe argmax replacement`는 `.9651`을 기록했다. 이는
예비 end-to-end 증거지만, 최신 canonical 1,729 코호트 재집계, 정책 선택의
validation 고정, paired uncertainty가 아직 없다. 그러므로 현재 본문에서는
**proof of concept**로만 다루고 제출 전 P0 실험으로 다시 닫는다.

---

## 2. RQ3의 정본 질문과 배포 시나리오

### 정본 RQ3

> **When harmful note-induced movement is identified from a single wrong-note
> run, can internal-state readouts support correction better than behavioral
> reconsideration without breaking unaffected answers?**

한국어로는 다음과 같다.

> **오답 소견서 때문에 답이 움직였을 가능성이 높은 사례를 단일 실행에서
> 식별했을 때, 내부 상태 판독을 이용한 선택적 개입이 단순 재고보다 moved
> 사례를 더 복구하면서 unaffected 답을 보존하는가?**

### 실험 평가와 실제 사용의 차이

```text
실험 평가:
  no-note/wrong-note pair + gold
  -> moved 정답 라벨과 최종 정확도 계산에만 사용

실제 selector 입력:
  현재 wrong-note run의 output / CoT / logits / activation / AV
  -> moved 위험 점수 계산

배포 정책:
  위험 점수 >= validation에서 고정한 threshold
      -> r5 또는 비교 교정 적용
  그 외
      -> 첫 답 유지
```

`moved`를 알아야 selector를 평가할 수 있지만, selector가 배포 시 `moved`나 gold를
입력으로 받는 것은 아니다. 이 구분을 Methods와 RQ3 결과 첫 문단에 반복한다.

---

## 3. P0 — 제출 주장을 결정하는 필수 실험

### P0-1. Canonical detector-gated correction

#### 목적

RQ2 detector와 RQ3 correction을 실제 정책으로 결합해, 조건부 정보 가치가
**전체 성능의 순이득**으로 이어지는지 검증한다.

#### 데이터 분할과 누출 방지

1. Canonical-eligible DDXPlus 1,729건을 diagnosis-stratified train/validation/test
   또는 기존 cross-fitting fold에 맞춰 나눈다.
2. Probe prediction은 반드시 해당 case를 학습하지 않은 fold에서 생성한다.
3. Threshold와 intervention policy는 validation에서 한 번 고정한다.
4. Test의 moved label, gold, no-note answer를 threshold 선택에 사용하지 않는다.
5. 여러 threshold 중 test 최고값을 보고하지 않는다. 전체 curve는 보조 그림으로,
   사전 고정한 operating point 한 개를 주 결과로 보고한다.

#### 비교 정책

| 정책 | selector 입력 | flag 시 행동 | 역할 |
|---|---|---|---|
| Keep first | 없음 | 개입 없음 | 배포 기준선 |
| Apply r5 to all | 없음 | 전부 r5 | 무선별 개입 실패 통제 |
| Source-confidence gated | output-head likelihood | r5 | 표면 불확실성 기준선 |
| CoT/LLM-monitor gated | wrong-run text channels | r5 | 외부 텍스트 기준선 |
| AV gated | AV 위험 점수 | r5 | 자연어 내부 채널 정책 |
| Probe gated | probe 위험 점수 | r5 | 내부 selector + 자연어 교정 |
| Probe argmax replacement | probe 위험 점수 | argmax로 직접 교체 | 닫힌 label-space 정책 |
| Oracle moved | true moved | r5/r6 | 배포 불가능한 교정 상한 |

기존 binary `probe disagreement`도 그대로 보고하되, continuous score의 threshold를
validation에서 고정한 정책을 함께 둔다. 그래야 AUROC와 실제 개입 정책을 연결할
수 있다.

#### 주 지표

| 지표 | 계산과 의미 |
|---|---|
| Overall accuracy | 정책 적용 후 전체 test 정확도 |
| Moved recovery | 실제 moved 중 wrong→right가 된 비율 |
| Unchanged preservation | 실제 non-moved/첫 답 정답 중 그대로 맞게 남은 비율 |
| Newly broken | first right→policy wrong 사례 수와 비율 |
| Net correction | `wrong→right - right→wrong` |
| Intervention rate | 전체 중 second pass/교체가 적용된 비율 |
| Precision among interventions | 개입된 사례 중 실제 moved 비율 |
| Risk-coverage curve | 개입 예산이 바뀔 때 정확도와 coverage의 trade-off |

Overall accuracy 차이는 case-paired bootstrap CI와 paired McNemar test를 보고한다.
Moved recovery 하나만 높고 newly broken이 더 크면 성공으로 판정하지 않는다.

#### 성공/실패 판정

- **강한 RQ3 성공**: validation에서 고정한 AV/probe-gated r5 정책이 test에서
  Keep-first보다 positive net correction을 보이고, paired 95% CI가 0을 배제한다.
- **제한적 성공**: 점추정은 양수지만 CI가 0을 포함한다. “feasibility signal”로만
  쓰고 성능 향상 claim은 하지 않는다.
- **실패**: overall이 같거나 낮다. RQ3는 moved oracle subset의 conditional
  information-value 분석으로 제한한다.
- **Probe replacement만 성공**: hidden-state detection의 utility는 지지되지만,
  AV feedback utility는 지지되지 않는다. 논문의 해결 축을 “internal-state
  selective correction”으로 쓰고 “natural-language correction”은 낮춘다.

### P0-2. Source output-head likelihood baseline

- 같은 canonical wrong-note 1,729건과 49 diagnosis candidate set을 사용한다.
- Source output head의 gold/suggestion/first-answer likelihood, margin, entropy를
  계산한다.
- Diagnosis-stratified all/silent AUROC를 probe/AV/monitor와 같은 표에 넣는다.
- Detector-gated correction의 source-confidence selector로도 사용한다.
- Output head가 probe와 비슷하면 “hidden-only information” 주장을 약화한다.
  Probe가 silent paired comparison에서 더 높으면 hidden-state 추가 가치를 지지한다.

### P0-3. Canonical 수치와 통계 전사

- Table 2b/Figure 4(a)의 channel-gap paired bootstrap CI
- Table 3 canonical capitulation/newly-broken counts
- Selector policy별 paired CI와 net correction
- Appendix Figure A2의 canonical `.591`
- corpus-300 non-overlap canonical clean refresh(expected 2,137)와 행동/사다리
  moved 563/571 불일치 해소; 전에는 fixed-cohort appendix audit로만 표기
- 모든 표의 모집단을 `clean 1,204`, `all eligible 1,729`, `silent 1,628`,
  `moved 319`, `MCR 1,452` 중 하나로 명시

---

## 4. P1 — 논문 범위를 넓히는 실험

### P1-1. 동일 LLM monitor의 no-CoT arm

현재 monitor는 vignette, note, CoT, answer를 함께 본다. CoT를 제거한 동일 judge
arm을 실행해 `.7305/.6904`가 CoT에서 온 것인지 vignette/note/answer 접근에서 온
것인지 분리한다. 이 실험 전에는 monitor 성능을 CoT만의 순수 기여라고 부르지 않는다.

### P1-2. MCR wrong-note activation과 detection

1. Canonical MCR none/wrong prompt의 L32 activation과 readout 3,086행 생성은
   완료됐다(arm당 1,543행).
2. 첫 채점은 arm을 pooling하고 wrong readout을 no-note 답에 조인해 무효다.
   `a21875e`의 `(base_id, hint_variant)` 조인과 `--variant wrong`으로 재채점한다.
3. Fixed 49-class probe는 직접 적용하지 않는다.
4. Source output likelihood, LLM monitor, source-aligned AV answer field를 비교한다.
5. Wrong-arm derangement를 통과한 answer field만 사용한다. Supporting-ground field는
   none/wrong별 grounding 통제에서 다시 실패하면 표에서 제외한다.
6. 성공하면 DDXPlus 전용 기전이라는 한계를 줄이고, 실패하면 MCR은 행동 효과의
   외적 재현까지만 담당한다.

첫 pooled 값 `.6361/.0029`는 인용하지 않는다. 실행·판정 관문은
`docs/experiments/18-mcr-wrong-arm-readout.md`를 따른다.

### P1-3. MCR correction ladder

- r3/r4는 activation 없이 실행 가능하다.
- r5에 필요한 MCR wrong-note activation + source-aligned readout은 생성됐다.
  Arm-aware 결론 감사가 통과한 뒤 **conclusion-only**와
  **conclusion+supporting-ground**를 분리해 실행한다.
- r7은 MCR source CoT가 필요하다.
- r6의 49-class probe label은 열린 진단 어휘에 직접 정의되지 않으므로 `n.a.`다.

MCR에서 r5가 r4를 이기면 open-vocabulary correction의 의미가 커진다. 이기지
못하면 DDXPlus r5를 닫힌 진단 공간의 조건부 결과로 제한한다.

### P1-4. Robustness controls

#### Direct × CoT matched 2×2

현재 canonical clean 1,204건은 Direct-none 정답으로 선정되어 Direct baseline은
1.0이지만 CoT baseline은 .7068이다. 따라서 23.75%p 대 4.40%p만으로 CoT가
anchoring을 줄인다고 결론 내리지 않는다.

같은 gold-absent base ID에서 Direct/CoT × no-note/wrong-note 네 셀을 맞춘다.
정답 여부로 선정하지 않은 common cohort에서는 네 accuracy와
`[CoT wrong-CoT none] - [Direct wrong-Direct none]` interaction을 보고한다.
Direct-none과 CoT-none이 모두 정답인 shared-solvable subset에서는 harmful flip,
suggestion adoption, third-diagnosis movement, newly corrected를 비교한다.
Case-level paired bootstrap CI와 paired permutation test를 사용하며,
`answer_forced` 포함/제외 민감도 분석을 붙인다. 기존 출력의 공통 coverage를 먼저
확인하고 누락 셀만 생성한다.

#### Realistic matched-neutral control

현재 realistic wrong arm은 canonical clean 1,204건에서 no-note `.9917`, wrong
`.6877`, paired cost **30.40%p**, moved 376, suggestion adoption 219로 가장 큰
효과를 보인다. 그러나 짧은 referral arm과 비교하면 진단 제안뿐 아니라 문장 길이,
clinical register, 정중한 공식 의뢰 형식이 함께 바뀐다. 따라서 현재는
`현실형 문구에서도 효과가 유지된다`고만 말하고, 추가 **6.65%p**를 현실성 또는
문체의 독립 효과로 부르지 않는다.

같은 1,204개 ID와 같은 삽입 위치에서 다음 고정 문구를 paired 실행한다.

```text
realistic wrong:
Referral note: Thank you for seeing this patient. Given the presentation,
we are concerned about possible {d} and would appreciate your assessment.

realistic neutral:
Referral note: Thank you for seeing this patient. Given the presentation,
we would appreciate your assessment and recommendations regarding further
evaluation and management.
```

Gemma tokenizer 기준 길이 차이를 생성 전에 감사하고, 결과를 보기 전에 neutral
template을 한 번만 고정한다. 주 추정량은 paired
`accuracy(realistic neutral) - accuracy(realistic wrong)`이며, no-note 대비 neutral
비용과 no-note 대비 wrong 총비용도 함께 보고한다. Accuracy, moved, adoption,
third-diagnosis 이동, paired bootstrap 95% CI, McNemar test를 기록한다.

이 통제가 유의한 추가 비용을 보일 때만 `현실적 임상 문서 안에서도 진단 제안
내용 자체가 해롭다`고 쓴다. 통과하지 못하면 realistic arm은 생태적 강건성
검사로만 남긴다.

#### Other robustness controls

- Appendix Figure A1의 layer/position 비교에서 reader recipe와 학습량 일치
- 필요 시 cue-position/span counterfactual swap
- 임상 전문가 소표본 재현은 secondary validation으로만 사용

---

## 5. P2 — 부록과 재현성 완성

- Related Work의 게재 상태, DOI/arXiv 버전, 직접 인용 문장을 원문으로 재확인
- Prompt 전문: no-note, neutral, wrong, correct, CoT, r3-r7
- Wrong suggestion 생성법과 MCR/DDXPlus provenance 차이
- Matcher alias 목록, 과거 버그와 canonical rescore 규칙
- Cohort flow: 원자료 → source-correct → gold leakage 제거 → activation subset
- Hyperparameter 표: backbone, layer, landmarks, probe folds, LoRA, decoding
- Compute 표와 hardware/runtime
- Failure examples: AV false alarm, MCR grounding failure, suggestion-never-top1 세 경로
- 재현 명령과 artifact manifest

P2는 P0 결과를 기다리는 동안 병행할 수 있다.

---

## 6. 문서 갱신 순서

1. `docs/experiments/RESULTS_CANONICAL_2026-08-24.md`: 분자/분모, cohort,
   seed/fold, 입력 파일, 실행 명령, artifact를 먼저 기록한다.
2. 실험별 문서: detector-gated correction은 `12-correction-ladder.md`, output
   head와 no-CoT monitor는 해당 실험 문서에 추가한다.
3. `docs/paper/table_camera_ready_2026-08-25.md`
4. `docs/paper/experiment_summary_2026-08-25.md`
5. `docs/paper/paper_outline_2026-08-24.md`
6. `docs/professor/paper_presentation_full_2026-08-25.md`

과거 fixed-cohort 수치는 삭제하지 않고 audit/appendix로 남기되 canonical 값과
같은 표에 섞지 않는다.

---

## 7. 본문 집필과 claim 동결

### P0 전에도 쓸 수 있는 절

- Introduction, Related Work, Methodology
- RQ1 Results: Figure 2와 Figure 3
- RQ2 detector comparison: Figure 4(a)
- Limitations의 backbone/corpus/readout 범위

### P0 뒤에 확정할 절

- RQ3 Results의 detector-gated policy 표와 Figure 4(b) 설명
- Abstract의 “supports correction” 강도
- Contribution 마지막 항목
- Conclusion의 utility 문장

**Gated policy 성공 시**

> A single-run internal-state detector can selectively trigger correction,
> yielding positive net correction while preserving unaffected answers.

**Gated policy 실패 시**

> Internal readouts contain conditionally useful corrective information on
> oracle-identified moved cases, but our detector-gated policy does not improve
> overall accuracy; end-to-end correction remains unresolved.

실패해도 RQ1/RQ2와 무선별 재고의 위험은 독립적으로 남는다.

---

## 8. 표와 그림 동결 조건

### Main paper

1. Figure 1: experimental pipeline
2. Figure 2: primary clean behavior, DDX 1,204 / MCR 1,452
3. Figure 3: DDX all-eligible trajectory, n=1,729 / moved=319
4. Figure 4(a): single-run detection, all 1,729 / silent 1,628
5. Figure 4(b): correction ladder; P0 뒤 detector-gated policy panel 또는 표 추가
6. Table 1: measurement validation, 필요하면 본문 축약·부록 전체
7. Table 2: paired internal trajectory statistics
8. Table 3: correction ladder + selective policy net effect

Figure 2의 DDX behavior는 clean 1,204를 유지한다. Figure 3/4는 activation이 있는
canonical all 1,729를 쓰며 캡션에서 분모 전환을 명시한다.

### Appendix

- Wording/CoT robustness
- Reader-trust + shuffled control
- MCR readout/derangement/grounding
- Content-correct r5/r6 comparison
- Corpus-300 replication
- Layer/position map
- Full prompts and matcher audit

표·그림은 PDF/SVG 벡터 출력을 정본으로 하고 PNG는 발표·검수용으로만 사용한다.

---

## 9. Overleaf 이전 계획

### 이전 전 게이트

- P0 detector-gated correction 판정 완료
- Source output-head baseline 완료 또는 명시적 pending 처리
- Canonical 원장과 table document의 숫자 일치
- Figure 2 clean / Figure 3·4 all-eligible 분모 확정
- 본문에서 fixed-cohort legacy 수치 제거

### 권장 LaTeX 구조

```text
paper/
  main.tex
  macros.tex
  numbers.tex
  references.bib
  sections/
    01_introduction.tex
    02_related_work.tex
    03_methodology.tex
    04_results_rq1.tex
    05_results_rq2.tex
    06_results_rq3.tex
    07_discussion_limitations.tex
    08_conclusion.tex
  tables/
    table_measurement.tex
    table_trajectory.tex
    table_correction.tex
  figures/
    figure1_pipeline.pdf
    figure2_behavior.pdf
    figure3_trajectory.pdf
    figure4_detection_correction.pdf
  appendix/
    appendix_methods.tex
    appendix_results.tex
    appendix_prompts.tex
```

### 숫자 불일치 방지

`numbers.tex`에 자주 쓰는 값을 한 번만 정의한다.

```tex
\newcommand{\DDXCleanN}{1,204}
\newcommand{\DDXEligibleN}{1,729}
\newcommand{\DDXMovedN}{319}
\newcommand{\DDXSilentN}{1,628}
\newcommand{\MCRN}{1,452}
\newcommand{\WrongNoteCostDDX}{23.75}
\newcommand{\WrongNoteCostMCR}{29.34}
```

본문·캡션·표에서 숫자를 직접 반복 입력하지 않는다.

### 실제 이전 순서

1. 로컬 repo에 `paper/` LaTeX source를 만든다.
2. Markdown outline을 절별 `.tex`로 옮기고 citation key를 먼저 맞춘다.
3. `table_camera_ready`의 표를 별도 `.tex` 파일로 변환한다.
4. Figure를 PDF로 재생성하고 캡션에 population/metric/CI를 넣는다.
5. `latexmk -pdf main.tex`로 로컬 compile한다.
6. Undefined reference/citation, overfull box, 표·그림 순서, 익명화, 페이지 제한을
   확인한다.
7. 로컬 `paper/`를 zip 또는 Git bridge로 Overleaf에 올린다.
8. 이후 정본은 Git repo로 유지하며 Overleaf와 로컬을 동시에 수동 수정하지 않는다.

### Overleaf 최종 점검

- RQ마다 Methods와 Results가 1:1로 대응하는가
- 모든 `n`이 clean/all/silent/moved 중 무엇인지 캡션에 있는가
- AUROC가 diagnosis-stratified인지 pooled인지 적혀 있는가
- Probe probability와 source next-token probability를 혼동하지 않는가
- `moved`가 배포 입력이 아니라 평가 label임을 명시했는가
- RQ3가 oracle conditional analysis와 detector-gated policy를 분리했는가
- AV가 probe보다 강하다고 쓰지 않았는가
- MCR 행동 복제와 DDX 내부 기전을 분리했는가
- Reader-trust와 MCR grounds의 negative result를 숨기지 않았는가
- Code/data/model license, compute, ethics, clinical non-deployment statement가 있는가

---

## 10. 권장 실행 순서

```text
1. Canonical selector+r5/r6/argmax policy 재집계
2. Validation-frozen threshold + test paired CI
3. Source output-head likelihood baseline을 같은 policy 표에 결합
4. RQ3 claim을 성공/제한적 성공/실패 중 하나로 동결
5. no-CoT monitor와 남은 CPU 통계·전사
6. MCR wrong-note activation/detection/correction (범위 확장)
7. 표·그림 번호와 canonical n 동결
8. 본문 초안 완성
9. LaTeX/Overleaf 이전
10. Related Work 원문 검증, 부록, 재현성·익명화 최종 감사
```

가장 먼저 할 일은 새로운 대규모 학습이 아니라 **최신 canonical 1,729건에서
detector-gated correction을 다시 계산하고, validation에서 고정한 정책이 test
전체 정확도에 순이득을 주는지 확인하는 것**이다. 이 결과가 RQ3를 “조건부 정보
가치”로 둘지 “실제 선택적 교정 시스템”으로 올릴지를 결정한다.
