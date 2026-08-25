# 논문 문서 안내 — 현재 정본

> **08-25 행동 재집계 완료, 파생 분석 진행 중.** Canonical no-note correctness를
> 다시 요구한 DDXPlus 전체 1,729/clean 1,204와 MCR 1,452에서 Figure 2 행동값을
> 갱신했다. Trajectory/detection/correction과 reader-trust도 같은 eligibility로
> 재집계됐다. Table 2b paired CI와 Table 3 capitulation만 로그 전사 대기다.
> 과거 1,747/321 수치는 generation-time fixed-cohort 감사값으로만 인용한다.

모든 본문 분모와 표·그림의 모집단 대응은
[`POPULATION_REGISTRY_2026-08-25.md`](../experiments/POPULATION_REGISTRY_2026-08-25.md)를
따른다. 현재 정본은 DDXPlus clean 1,204, all eligible 1,729, silent 1,628,
MCR behavior 1,452다.

이 폴더의 논문 서사는 **현상 우선**이다. 주인공은 NLA 자체가 아니라,
의뢰 소견서의 의심 진단이 의료 LLM의 출력은 바꾸지만 내부 정답 신호를
완전히 대체하지 못한다는 **internal-output dissociation**이다. 프로브는 이를
정밀하게 측정하고, 자연어 판독은 내부 상태를 탐색하는 계측 채널이다. 현재
판독은 열린 어휘 결론에서 예비 신호가 있지만 근거 접지와 독자 인터페이스에서는
검증을 통과하지 못했다.

## 처음 읽는 순서

1. `experiment_summary_2026-08-25.md` — 실험별 질문, 실측, 상태를 한 번에 본다.
2. `paper_outline_2026-08-24.md` — 논문 테제와 절별 서사를 본다.
3. `draft_introduction_2026-08-25.md` — 영어 Introduction 초안, H/RQ, 원문 출처 지도를 본다.
4. `table_camera_ready_2026-08-25.md` — 현재 표의 수치와 캡션을 확인한다.
5. `figure_order_and_generation_2026-08-25.md` — 그림 순서·생성 명령·해석 경계를 확인한다.
6. `prior_work_2026-08-24.md` — 신규성의 범위와 가장 가까운 선행을 확인한다.
7. `prior_work_verification.md` — 선행의 각 주장을 우리 실험대에서 직접
   시험한 결과. 재현 실패 둘(Turpin의 은폐, Yuan의 비인과성)이 여기 있다.
8. `draft_related_work_2026-08-24.md` / `related_work.tex` — 실제 Related Work 원고.

`related_work_2026-08-23.md`는 문헌 조사 원장이고, `reading_*.md`는 최근접
논문의 정독 노트다. `judge_jobs_2026-08-24.md`는 외부 판정자 실험의 실행
대기열이다.

## Camera-ready asset order

본문은 **Figure 1 설계 → Table 1/Figure 2 행동 → Table 2a/Figure 3 궤적 →
Table 2b/Figure 4(a) 탐지 → Table 3/Figure 4(b) 교정** 순서다. AV 계기 검증과
layer map은 핵심 인과 주장의 전제가 아니라 AV 채널에만 필요한 측정 관문이므로
**Appendix Table A1/Figure A1**로 이동한다. Myocarditis case study는 Appendix
Figure A2다. 과거 실험 문서에 남은 Table 1/Figure 2 표기는 legacy numbering이며,
현재 번호의 정본은 `table_camera_ready_2026-08-25.md`다.

## 현재 논문의 한 문장

의료 LLM이 의뢰 소견서에 앵커링되어 답을 바꿀 때도, 인과 통제된 DDXPlus
실험에서 내부 gold signal은 평균적으로 남고 suggestion은 대다수 사례에서
probe top-1이 되지 않는다. 이 내부-출력 결렬은
한 번의 실행에서 탐지할 수 있고 자연어로 탐색할 수 있으며, 정확한 내부
내용을 조건부로 되먹이면 일부 오류를 회복할 수 있다. 현재 자연어 판독을
임상의 대면 설명으로 사용하는 것은 지지되지 않는다.

영문으로는 다음 범위가 안전하다.

> In a causally controlled diagnostic setting, referral-note anchoring often
> changes what a medical LLM emits without fully replacing the gold-diagnosis
> signal in its probed internal state. The resulting internal-output rift can be
> detected from a single run. Natural-language activation readouts provide an
> exploratory measurement channel, but their current precision is insufficient
> for a clinician-facing interface; accurate internal content can nevertheless
> support conditional correction.

## 확정된 근거와 범위

| 주장 | 현재 근거 | 주장 가능한 범위 |
|---|---|---|
| 오답 소견서가 진단을 움직인다 | canonical-eligible DDXPlus main **−23.75 pp**; MCR **−29.34 pp** | 행동 효과는 두 코퍼스; non-overlap primary refresh 대기 |
| moved의 다수는 제안 복사가 아니다 | primary behavior에서 DDXPlus clean **201/287(70.0%)**, MCR **300/427(70.3%)**가 제3 진단으로 이동 | DDX 전체 eligible 민감도는 **230/319(72.1%)** |
| corpus-300의 ID 독립성은 확인됐다 | 주 실행 id를 뺀 **미관측 3,319**(fixed-cohort clean 2,192)에서 오답 조건 **.7682** vs 초집합 .7670 — 0.12 pp 차 | 행동 효과의 appendix audit; canonical clean 2,137 refresh 전에는 primary row로 쓰지 않음 |
| suggestion이 내부 top-1을 대개 차지하지 못한다 | canonical moved 319건 중 **262건(82.1%)**에서 suggestion이 어느 landmark에서도 top-1이 아님 | 그중 gold throughout 147, third-diagnosis path 115 |
| 소견서는 내부 gold signal을 행동별로 다르게 낮춘다 | final Δ: 유지 **−.006**, 제3 진단 **−.054**, 제안 채택 **−.199** | paired CI 모두 0 배제; final trend ρ=−.282 [−.328,−.233] |
| CoT보다 내부 채널이 강하다 | 정본 silent subset(n=1,628) AUROC: LLM CoT monitor .6904, NL readout .8319, probe .9881; readout−monitor 점추정 +.1415, 새 CI 전사 대기 | DDXPlus, 진단 내 층화 |
| 자연어 판독은 벡터에 종속된다 | swap .993, memorization .000, contamination .007; heldout cue content .751 (n=770, 기계 채점) | DDXPlus 검증 배터리 |
| 내부 내용을 되먹이면 회복한다 | canonical moved=319: 첫 답 .0031 → r5 .6301 / r6 .8339 | 내용 정확도가 지배; 무선별 재실행은 해로움 |
| 자기 CoT 되먹임은 교정하지 못한다 | 공통 1,151 id에서 r7 moved 회복 .1236, r5 .5281, r6 .7416 | 쉬운 공통 집합; 기전은 고착과 일관되나 미확정 |
| 현재 판독은 독자 인터페이스로 부적합하다 | canonical controlled reader-trust: no-account 대비 ΔAUROC **−.0998 [−.135,−.065]**; probe **+.0692 [.042,.098]**, CoT −.0217 (0 포함) | n=716/channel; shuffled readout n=715 |
| 판독의 **내용**은 독자에게 실재한다 | shuffled 통제: readout .7342 vs shuffled_readout .4488 (**+.2854**) | 내용은 실재하나 순효과는 여전히 음수 |
| MCR 판독의 답 필드는 사례를 읽는다 | 821행 vs model **.2643** vs derangement **.0049** (gap +.2594); source-wrong 710행 .2127 vs .0042 | 근거 필드는 통과 못 함(gap +.025, 반복 70%) |
| 자연어 형식 자체가 교정의 원인이다 | **확립되지 않음**: canonical 전체 correct/correct에서 p=1.000; moved에서는 맨 라벨 점추정 우위 p=.016이나 Bonferroni .0125 미달 | 형식 우위 주장 금지 |

## 반드시 지킬 주장 경계

- 행동 효과는 DDXPlus와 MCR에서 재현됐지만, **82.1% 기전 해부는 DDXPlus만**이다.
  그리고 82.1%는 gold-throughout이 아니라 suggestion-never-top1이다.
- 닫힌 49-class 탐지에서는 지도 프로브가 자연어 판독보다 강하다(.9330/.9881
  all/silent 대 .7511/.8319). 자연어 판독의 현재 몫은 최고 정확도나 임상의용
  설명이 아니라 **계측·오류 유형 탐색과 열린 어휘 가설 생성**이다.
- MCR처럼 진단명이 대부분 한 번씩 등장하는 자료에서는 표준적인 고정
  클래스 지도 프로브를 그대로 정의하기 어렵다. 이것을 "어떤 프로브도
  불가능하다"고 쓰지는 않는다.
- reader-trust에서 판독은 아무것도 안 보여주는 것보다 **나쁘다**
  (canonical controlled, −.0998 [−.135,−.065]). "clinician-readable"을 효용 주장처럼
  쓰지 않는다. 형식상 읽을 수 있다는 것과 독자에게 도움이 된다는 것은 다르다.
  기전도 적어둔다: 판독을 본 독자는 moved를 .928로 의심하지만 kept도
  **.579**로 의심한다(무판독 .078). 판별이 아니라 **무차별 의심**을 준다.
- **그러나 "판독은 내용이 없다"로 쓰면 틀린다.** shuffled 통제에서 다른
  케이스의 판독으로 바꾸면 .7342 → **.4488**로 무너진다(내용 기여 **+.2854**).
  산문은 벡터가 derangement에서 통과한 것과 같은 시험을 통과했다. 정확한
  문장은 **"내용은 실재하지만 설명을 제시하는 비용을 갚지 못한다"**이다 —
  probe는 갚고(+.4814 기여로 −.4123을 넘김) 판독은 못 갚는다.
- **뒤섞인 설명은 무정보가 아니라 적극적 오도다.** shuffled 세 팔이 모두
  −.30~−.41이고 `shuffled_probe`는 **.4218로 동전 던지기 아래**다. 자신 있게
  제시된 틀린 진단명이 가장 해롭다는 뜻이므로, 배포 논의에서 **정밀도 없는
  자동 설명**을 권고하지 않는 근거로 쓴다.
- MCR 판독은 **답 필드만** 열렸다. derangement 통제에서 답 필드는
  .2643 vs .0049로 사례 특이적이지만, 근거 필드는 gap +.025에 cue 문장의
  70%가 반복 정형문구다. "MCR에서 판독이 통한다"고 뭉뜽그려 쓰지 않는다.
  그리고 gap이 크다는 것은 **사례를 읽는다**는 뜻이지 **정확하다**는 뜻이
  아니다 — 절대값은 .2127이다.
- r5가 r4보다 좋은 것은 판독 **내용**이 유용하다는 증거다. 자연어라는 형식의
  독립 효과는 확인되지 않았다.
- 무선별 재고 요청은 전체 성능을 크게 해친다. 교정은 정밀한 selector와
  결합된 조건부 정책으로만 제안한다.

## 문서 상태

### 현재 정본

- `experiment_summary_2026-08-25.md`: 실험·수치·상태의 짧은 정본.
- `paper_outline_2026-08-24.md`: 서사와 절 배치의 정본.
- `table_camera_ready_2026-08-25.md`: 표 수치와 캡션의 정본.
- `prior_work_2026-08-24.md`: 신규성 판단의 정본.

### 조사·감사 문서

- `related_work_2026-08-23.md`: 문헌 조사 원장. 원고가 아니다.
- `reading_catching_rationalization.md`, `reading_when_truth_is_overridden.md`:
  최근접 선행의 충돌·차이 분석.
- `judge_jobs_2026-08-24.md`: 외부 판정자 실험 계획.
- `table_camera_ready_2026-08-25.md` 안의 MCR 결론 판독 실패 기록은 **감사
  기록**이다. `.052`, `.034`, "6배" 수치는 source-misaligned target으로 얻은
  무효 결과이며 현재 주장의 근거가 아니다.

### 대체된 문서

`docs/archive/paper_tables_worklog_2026-08-23.md`는 표 설계가 여러 번
바뀌는 동안의 작업 로그다. 재현 감사에는 유용하지만 현재 표·수치의 출처로
인용하지 않는다.

## 초안과 투고 준비 상태

초안 작성은 가능하다. 다만 아래 항목 전에는 "submission-ready"가 아니다.

1. ~~wording과 CoT 파생 실행을 동일 canonical clean cohort로 재집계한다.~~
   **완료 (08-25, n=1,204)** — wrong accuracy `.7625/.7757/.8480/.6877`,
   paired 비용 `23.75/21.93/14.45/30.40%p`; Direct-selected CoT gap은
   **4.40%p**이나 matched 2×2 전에는 완화 효과로 해석하지 않는다.
2. ~~Table 2a의 행동군 Δ 차이에 paired CI 또는 추세 검정을 추가한다.~~
   **계기 완료** (`src/paired_stats.py`) — 궤적 재실행에서 값이 나온다.
3. ~~reader-trust 2,896 완주와 shuffled 통제.~~ **canonical 재집계 완료 (08-25)**:
   readout **−.0998 [−.135,−.065]**, probe **+.0692**, CoT 0 포함.
   shuffled로 내용/제시를 분해했다 — 판독 내용 기여 **+.2854**, 그러나
   뒤섞임 바닥이 −.3848이라 순효과는 음수로 남는다.
4. ~~MCR 결론 과제 판독에 derangement baseline을 붙인다.~~ **완료 — 통과
   (08-24)**: 별도 conclusion-task 821행의 답 필드 .2643 vs .0049. Wrong-note
   L32 추출과 readout 3,086행 생성도 완료됐다. 다만 이 파일은 none/wrong을
   1,543행씩 포함하며 첫 채점은 두 arm을 pooling하고 wrong readout을 no-note
   답에 붙여 **무효**다. `a21875e`의 arm-aware 조인으로 wrong 1,543행을 다시
   채점한 뒤에만 MCR r5를 진행한다. Grounding이 약하므로 r5는 conclusion-only와
   conclusion+grounds를 분리한다. 세부 절차는
   [`18-mcr-wrong-arm-readout.md`](../experiments/18-mcr-wrong-arm-readout.md)다.
5. ~~외부 판정자로 판독의 임상적 타당성을 보조 검증한다.~~ **완료 (08-24)**:
   238쌍 전수. 손채점 `.3402/.7306/.5571` vs 판정자 `.5525/.7740/.6393`,
   kappa .35–.50. **손채점은 하한이었다** — 세 층 모두 외부 판정자가 더
   후하다. 단 판정자는 D를 한 번도 주지 않았고 여유분이 좌우·부위 오류에
   몰려 있어, 표에는 값과 그 분해를 함께 싣는다. **임상의 평가는 별개로
   남는다** — 판정자는 모델이지 임상의가 아니다.
6. 같은 judge의 no-CoT arm을 추가해 LLM monitor 향상이 CoT 때문인지,
   vignette/note/answer 접근 때문인지 분리한다. (`--no-cot` 빌더 완료)
7. Related Work의 최신 논문 서지·게재 상태와 정확한 인용 문장을 원문으로
   재확인한다.
8. Appendix Figure A2의 옛 `64.1%`를 **`.591`**로 교체한다 (canonical 재집계 완료).
9. corpus-300은 **base-ID non-overlap이 확인된 fixed-cohort 독립 감사**로
   서술한다. Canonical clean 2,137 refresh가 끝난 뒤에만 primary replication
   row로 승격한다.
10. **Source output-head likelihood 기준선**을 canonical wrong-note 1,729건에서
    실행한다. 생성문과 hidden-state probe 사이의 필수 비교이며, 과거 source-error
    logprob 결과로 대체하지 않는다.
11. **Detector-gated correction을 canonical 1,729건에서 제출 수준으로 다시
    검증한다.** 과거 fixed-cohort의 probe-selector+r5 `.9141`과 argmax replacement
    `.9651`은 proof of concept다. Validation에서 threshold/policy를 고정하고,
    held-out test에서 overall accuracy, moved recovery, unchanged preservation,
    newly broken, net correction, intervention rate와 paired CI를 보고하기 전에는
    RQ3를 전체 성능 향상으로 쓰지 않는다.
12. **Realistic matched-neutral control을 canonical clean 1,204건에서 실행한다.**
    현재 realistic arm의 30.40%p 비용은 길이·임상 문체·정중한 의뢰 형식과 진단
    제안이 함께 바뀐 총효과다. 같은 길이와 레지스터의 neutral referral note를
    paired 비교하기 전에는 짧은 referral 대비 추가 6.65%p를 현실성 또는 제안
    내용의 독립 효과로 쓰지 않는다.
13. **Direct×CoT matched 2×2 비교를 실행한다.** 현재 1,204건은 Direct-none 정답으로
    선정되어 `23.75%p vs 4.40%p`가 selection bias와 floor effect를 포함한다.
    정답 여부로 고르지 않은 공통 cohort에서 difference-in-differences를, 두
    no-note가 모두 정답인 shared-solvable subset에서 harmful flip을 paired
    비교하기 전에는 CoT가 anchoring을 완화한다고 확정하지 않는다.

전체 의존관계와 Overleaf 이전 순서는
[`submission_roadmap_to_overleaf_2026-08-25.md`](submission_roadmap_to_overleaf_2026-08-25.md)를
정본으로 따른다.

## 갱신 규칙

- 새 수치는 먼저 `../experiments/RESULTS_CANONICAL_2026-08-24.md`에
  분자/분모·모집단·계기·입력 파일과 함께 기록한 뒤 표 문서로 옮긴다.
- 서사가 바뀌면 `paper_outline_2026-08-24.md`, 짧은 상태가 바뀌면
  `experiment_summary_2026-08-25.md`를 갱신한다.
- 실패한 실행과 철회된 해석은 삭제하지 않고 `docs/archive/` 또는 명시된
  audit subsection에 남긴다.
- 교수님용 문서는 위 정본에서만 수치를 가져오며, 과거 날짜의 professor 문서는
  당시 스냅샷으로 취급한다.
