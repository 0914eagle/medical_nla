# Paper figure order and generation (2026-08-25)

이 문서는 **Figure 1 실험 파이프라인을 제외하고**, 현재 결과만으로 그릴 수 있는
모든 본문/부록 그림의 정본 순서, 입력, 실행법, 해석 경계를 고정한다.

## 1. Camera-ready 순서

| 순서 | 자산 | 바로 답하는 질문 | 본문/부록 |
|---:|---|---|---|
| 1 | Figure 1 | 실험 개입과 측정 채널은 무엇인가 | 본문, 개념도; 이 코드 묶음에서 제외 |
| 2 | **Figure 2** | wrong referral note가 실제 답을 움직이는가 | 본문 |
| 3 | **Figure 3** | 움직인 답의 내부 진단 신호는 어디서 어떻게 변하는가 | 본문 |
| 4 | **Figure 4** | 한 번의 실행에서 탐지하고, 선택적으로 고칠 수 있는가 | 본문 |
| A1 | **Appendix Figure A1** | AV 판독 계기를 어느 층/위치에서 읽을 수 있는가 | 부록 |
| A2 | **Appendix Figure A2** | 한 사례에서 chart/note/CoT/output/internal channels가 어떻게 어긋나는가 | 부록, 수작업 case panel |

본문 표는 Table 1 행동, Table 2a 궤적, Table 2b 탐지, Table 3 교정 순서다.
Figure 2(a)가 네 arm의 원시 정확도를 담고, Table 1은 그를 pp 비용으로
분해하여 효과크기와 non-overlap 재현을 담당한다. 두 자산은 같은
열을 반복하지 않는다.
AV validation은 Appendix Table A1이다. Discussion의 도구 선택표는 번호를 주지
않고 산문으로 축약한다.

## 2. Figure 2 — Behavioral intervention

생성 코드: `scripts/make_figure_intervention.py`.

- **(a) three displayed intervention arms**: neutral/wrong/correct referral note.
  No-note는 canonical eligibility 선정 조건이라 1.0 기준선으로만 표시한다. Gold가 chart에
  노출되지 않은 `clean` cohort를 쓴다. Wrong만 낮고 neutral은 덜 낮아야 단순
  문장 삽입 비용이 아니라 suggestion-specific effect로 읽을 수 있다.
- **(b) moved destination**: 전체 canonical no-note-correct cohort에서 causally
  moved된 답을 suggestion 채택과 제3 진단 이동으로 나눈다. DDXPlus **89/230**,
  MCR **127/300**이다.

DDXPlus canonical moved는 **319/1,729**이고 그중 suggestion 채택 89, 제3 진단
이동 230이다. Clean/explicit-gold 분해는 `287/1,204`(suggestion 86, other
diagnosis 201) 대 `32/525`(suggestion 3, other diagnosis 29)다. 즉 moved의
287/319(90.0%)이 clean에서 발생하고 clean에서도 other-diagnosis 이동이
201/287(70.0%)다.

**중요:** (a)와 (b)는 분모가 다르다. 코드 기본값이 각각 `clean`과 `all`이며,
각 패널 x축에 n을 따로 인쇄한다. 두 패널 n을 같은 모집단처럼 설명하지 않는다.
여기서 `clean`은 train-test leakage가 아니라 **presentation에 gold name/alias가
직접 나오지 않는 행**이라는 뜻이다.

최신 `analyze_hint_effect.py`는 dump에 `moved.n`, `to_suggestion`,
`to_third_diagnosis`를 기록한다. 기존 dump에는 이 block이 없으므로 먼저 다시 만든다.

```bash
python scripts/analyze_hint_effect.py \
  --answers $ART/results/ddxplus_hint_answers_v2_rescored.jsonl \
            $ART/results/ddxplus_hint_answers_neutral_rescored.jsonl \
            $ART/results/ddxplus_hint_answers_correct_rescored.jsonl \
  --require-canonical-no-note-correct \
  --dump $ART/results/figure2_ddx_dump.json

python scripts/analyze_hint_effect.py \
  --answers $ART/results/mcr_hint_answers_full_rescored.jsonl \
  --require-canonical-no-note-correct \
  --dump $ART/results/figure2_mcr_dump.json

python scripts/make_figure_intervention.py \
  --dumps $ART/results/figure2_ddx_dump.json $ART/results/figure2_mcr_dump.json \
  --labels DDXPlus MedCaseReasoning \
  --accuracy-population clean \
  --destination-population all \
  --omit-no-note \
  --output $ART/results/figure2_behavior.png
```

`--require-canonical-no-note-correct`를 쓰면 none accuracy는 1.0이므로 Figure 2(a)는
none 막대를 반복하지 않는다. `--omit-no-note`는 1.0 기준선을 대신 그리고, dump의
none accuracy가 실제로 1.0이 아니면 오류로 중단한다. 기존 generation-time fixed
cohort를 감사 목적으로 재현할 때만 두 옵션을 생략한다.

**현재 primary 패널 값**: DDXPlus clean `n=1,204`에서 neutral/wrong/correct
`.9460/.7625/.9302`; MCR `n=1,452`에서 `.9339/.7066/.8388`. Figure 2(b)는
DDXPlus `319=89+230`, MCR `427=127+300`을 표시해야 한다.
Moved는 `lost_gold OR causally_adopted_suggestion`이므로 각 코퍼스의 1건처럼
gold를 유지한 채 suggestion을 추가한 사례도 포함한다. 따라서 panel (b)는
**Composition of causally affected cases**로 제목을 붙이고, 범례는
`adopted suggestion` 대 `lost gold; other diagnosis`로 쓴다.

## 3. Figure 3 — Internal trajectory

생성 코드: `scripts/make_figure_trajectory.py`; 입력은
`scripts/analyze_trajectory.py --dump`의 정본 JSON이다.

- **(a) absolute signal**: wrong-note arm의 행동군별 `p(gold)`, 그리고 suggestion
  채택군의 `p(suggestion)`을 그린다. 채택군 final에서도 `.725` 대 `.211`로
  gold가 3.4배 높다는 것이 핵심이다. 단, 이는 source next-token probability가
  아니라 landmark별 cross-fit diagnosis probe probability다.
- **(b) paired note cost**: 같은 case의 `p_wrong(gold)-p_none(gold)`. Constraint에서
  비용이 최대이고 final에서 일부 회복한다. 단조 누적이 아니며, 현재 prompt
  skeleton의 위치 효과이지 보편적인 `constraint token` 기전이 아니다.
- **(c) first suggestion top-1**: canonical-eligible 모집단에서 다시 계산한 moved
  사례를 first top-1 landmark별로 나눈다. 옛 321/266/151/115는 matcher-era
  고정 코호트 값이므로 본문에서 폐기한다. Note의 0 label은 측정된 0이며
  누락이 아니다.

```bash
python scripts/make_figure_trajectory.py \
  --dump $ART/results/trajectory_dump_canonical_eligible.json \
  --output $ART/results/figure3_trajectory_canonical_eligible.png
```

## 4. Figure 4 — Detection to correction

생성 코드: `scripts/make_figure_detection_correction.py`; 정본 값은
`run_canonical_eligible_downstream.sh`가 per-case 산출물에서 생성하는
`$ART/reports/figure4_detection_correction_canonical_eligible.json`에 있다.

- **(a) detection**: wrong-note 실행 하나만 보고 moved를 탐지하는 within-diagnosis
  AUROC다. `silent`는 답이 suggestion을 직접 명명한 경우를 제외한다. 수치는
  canonical 1,729 코호트에서 probe를 재학습한 뒤 채운다. 옛 1,747 코호트의
  `.9280/.9840`, `.7506/.8302`, `.7233/.6829`는 감사 기록으로만 남긴다.
- **(b) correction**: 전체 정확도와 moved recovery를 분리한다. R3-R6도 같은
  1,729 ID로 재집계한다. 방향이 유지되더라도 옛 `.8117/.4098/.4568` 등의
  값을 새 모집단의 값으로 간주하지 않는다.

```bash
python scripts/make_figure_detection_correction.py \
  --values $ART/reports/figure4_detection_correction_canonical_eligible.json \
  --output $ART/results/figure4_detection_correction_canonical_eligible.png
```

## 5. Appendix Figure A1 — AV readability map

생성 코드: `scripts/make_figure_readout_map.py`.

- (a)는 per-cue reader의 held-out cue-string sweep이다.
- (b)는 cue-first final-prompt-token reader의 diagnosis-heldout sweep이다.
- 두 패널은 reader recipe, target, held-out axis가 다르므로 **패널 사이의 절대값을
  position ablation처럼 비교하지 않는다**. 안전한 비교는 각 패널 내부 layer와
  (b)의 seen-vs-heldout뿐이다.

```bash
python scripts/make_figure_readout_map.py \
  --output $ART/results/appendix_figure_a1_readout_map.png
```

## 6. 한 번에 그리기

Figure 1과 수작업 case study A2를 제외한 네 장을 한 번에 만든다.

가장 안전한 명령은 Figure 2 분석 dump도 canonical answer에서 다시 만드는 wrapper다.

```bash
cd /home/eagle0914/medical_nla
bash scripts/run_paper_figures_without_figure1.sh /data1/heejae
```

PDF가 필요하면:

```bash
FORMAT=pdf bash scripts/run_paper_figures_without_figure1.sh /data1/heejae
```

이미 dump가 준비되어 있어 plot만 다시 그릴 때는 아래 Python 드라이버를 쓴다.

```bash
python scripts/make_paper_figures.py \
  --ddx-dump $ART/results/figure2_ddx_canonical_eligible_dump.json \
  --mcr-dump $ART/results/figure2_mcr_canonical_eligible_dump.json \
  --trajectory-dump $ART/results/trajectory_dump_canonical_eligible.json \
  --detection-values $ART/reports/figure4_detection_correction_canonical_eligible.json \
  --out-dir $ART/results/paper_figures \
  --format png
```

생성물:

```text
figure2_behavior.png
figure3_trajectory.png
figure4_detection_correction.png
appendix_figure_a1_readout_map.png
```

현재 Figure 4의 detection panel은 확정된 channel만 그린다. 실험 17의 source
output-head likelihood가 끝나면 canonical JSON과 표를 먼저 갱신한 뒤 Figure 4를
다시 그린다. `▢` 값을 임의로 plot하지 않는다.

## 7. 아직 그림에 넣지 않는 것

- **Reader-trust**는 2,896/2,896과 shuffled control까지 완료됐다. 다만
  Figure 4의 single-run detector와 다른 외부 독자 효용 과제이므로 그 패널에
  섞지 않고 Appendix table 또는 별도 자산으로 보고한다.
- **MCR internal trajectory/correction**은 현재 DDXPlus 기전 그림에 섞지 않는다.
- **AV MCR supporting cues**는 grounding gate를 통과하지 못했으므로 성공 그림으로
  만들지 않는다.
- **Figure 1 pipeline**은 데이터 도식이므로 별도 벡터 편집 자산으로 만든다.
- **Appendix Figure A2 case study**는 자유 텍스트와 여러 채널을 조판해야 하므로
  자동 plot이 아니라 원고 레이아웃에서 만든다.

## 8. 남은 검증

1,747 fixed-cohort 파생값을 폐기하고 canonical 1,729에서 필요한 재학습·재집계를
한 번에 실행하는 명령은 다음과 같다.

```bash
nohup bash scripts/run_canonical_eligible_downstream.sh /data1/heejae \
  > /data1/heejae/medical_nla/logs/canonical_eligible_downstream.log 2>&1 &
```

이 실행은 final probe와 landmark probe를 새 코호트에서 다시 학습하고,
`channel_scores`, correction ladder, reader-trust를 같은 eligibility ID로 필터한 뒤
Figure 3/4를 다시 그린다. 완료 후 우선 확인할 파일은 다음 두 개다.

```text
$ART/reports/figure4_detection_correction_canonical_eligible_summary.md
$ART/reports/trajectory_canonical_eligible.log
```

1. Figure 2용 DDX/MCR dump를 최신 스키마로 재생성하고 표의 n/accuracy와 대조한다.
2. Figure 3과 Figure 4는 `run_canonical_eligible_downstream.sh`가 만든
   `_canonical_eligible` 산출물만 사용한다. 옛 `trajectory_dump.json`,
   `channel_scores.jsonl`, `probe_verdicts_canonical.jsonl`의 1,747건 결과는
   matcher-era 감사 기록일 뿐 본문 결과가 아니다.
3. Figure 4 값은 새 실험이 확정될 때 per-case JSONL → canonical values JSON
   → 원장 → 표 문서 → 그림 순으로 갱신한다. 플롯 코드에 수치를 직접 옮기지 않는다.
4. 모든 PNG/PDF를 흑백 인쇄와 1-column/2-column 폭에서 확인한다.
5. 본문 캡션에는 모집단, 계기, silent 정의, probe supervision을 반드시 적는다.
