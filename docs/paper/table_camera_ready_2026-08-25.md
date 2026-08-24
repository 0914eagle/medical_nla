# Camera-ready tables, v2 (2026-08-25)

설계 규칙 (v1의 실패에서):
- **표 하나 = 지표 하나.** 단위가 다른 값은 같은 열에 두지 않는다.
- **열 = 조건/방법, 행 = 측정 대상.** 파생 통계(차이, 배수)는 별도 열이
  아니라 본문 또는 명시된 Δ열.
- 셀 하나에 값 하나. 슬래시로 두 값을 넣지 않는다.
- 숫자 열에 텍스트 금지. 정의되지 않는 칸은 – 와 표 각주.
- 소수 자리 통일 (비율 .xxx, pp는 정수 또는 x.x).
- **캡션이 계기를 명시한다.** T1은 자연어 판독, T3은 프로브, T3b는 채널별.
  같은 절 안에서도 어느 계기가 잰 값인지 독자가 표만 보고 알아야 한다.

---

## Table 1 — Instrument validation (§4.1)

지표: 비율 하나로 통일. 각 행에 기준값(우연 또는 무학습)을 **자기 열**로.
길이(1,557→52자)와 형식(0.05→1.00)은 단위가 달라 **본문 문장으로** 이동.

**Table 1.** Validation of the readout as a measuring instrument. Each row is
one test; Reference gives the value the test must beat (chance) or the
untuned checkpoint's score on the same items.

| Test | n | Readout | Reference | Reference type |
|---|---:|---:|---:|---|
| Swap tracking | 438 | .993 | — | (higher is better) |
| Context memorization | 438 | .000 | — | (lower is better) |
| Cross-patient contamination | 438 | .007 | .015 | chance |
| Description precision | 438 | .671 | .075 | untuned |
| Held-out description rate | 2,122 | .751 | .088 | chance |
| Unseen-cue description | 438 | .750 | .720 | untuned |
| Conclusion at the answer position | 217 | .682 | .636 | untuned |

*Format compliance and output length are reported in the text (0.05 → 1.00;
1,557 → 52 characters): they establish that the readout is machine-scorable,
not that it is faithful. Swap tracking / memorization are the core: editing
one finding moves the description 99.3% of the time and never leaves the
original wording behind.*

**채점자 표기 (08-25 결정)**: unseen-cue 서술률 0.75는 1기의 438행 수동
채점(A/B/C/D 4등급, A+B를 성공으로) 결과다. **채점자 신원·인원 표기는
비워 두고, 외부 API 판정자를 확보하면 그 판정으로 대체한다** — 사람
2차 채점이나 자기 일치율로 메우지 않는다. 그때까지 본문에는 수치만 싣고
채점 주체는 서술하지 않으며, 최종 원고에서 판정자 절차로 채운다.

**답-위치 행이 이 표에서 가장 이상하게 읽히는 행이고, 그것이 요점이다
(08-24).** 다른 행과 달리 Reference가 Readout에 가깝다 — 무학습
체크포인트도 답이 바뀐 케이스의 63.6%에서 정답을 짚는다. **결렬은 우리
어댑터의 산물이 아니라 활성값의 성질이다**, 라는 뜻이고 이쪽이 더 강한
결과다. 본문은 여기에 두 번째 수치를 붙인다: vanilla는 판독 하나당
진단명을 1.21개 부르고 v2는 1.01개를 부르므로, 포함 검사로는 vanilla가
유리하고 이름 하나당으로는 .526 vs .675로 뒤집힌다. **어댑터가 사는 값은
적중률이 아니라 정밀도다** — 이 표의 나머지 행이 전부 그 이야기다.

▢ 남은 것: shuffle-control 값, swap/memorization의 정확한 n,
**MCR 산문 서술률 행** — 계기가 실제 임상 문장도 읽는지는 1기 mcr_sweep
산출물 재집계(CPU)로 채운다. 이 행이 있어야 T1이 DDXPlus 전용 검증표가
아니게 된다.

---

## Table 2 — Intervention accuracy (§4.2)

행 = 코퍼스, 열 = 조건. 지표는 정확도 하나. 케이스 수·낙폭은 본문.

**Table 2.** Accuracy by arm, on cases answered correctly with no note.
Finding-position activations are bit-identical across arms by construction.

| Corpus | n | No note | Neutral | Wrong | Correct |
|---|---:|---:|---:|---:|---:|
| DDXPlus | 1,220 | .991 | .934 | **.760** | .932 |
| MedCaseReasoning | 1,543 | .981 | .926 | **.703** | .839 |

*The wrong note costs 23.1 pp on DDXPlus and 27.8 pp on MedCaseReasoning;
the neutral note costs 5.7 and 5.5 pp, so the suggestion-specific effect is
17.4 and 22.3 pp — 4.1× and 5.1× the cost of insertion alone.*

**Table 2b.** Where the moved answers go.

| Corpus | Moved | To the suggestion | To a third diagnosis |
|---|---:|---:|---:|
| DDXPlus | 324 | 95 | 229 |
| MedCaseReasoning | 441 | 138 | 303 |

**Table 2c** (지면 되면; 아니면 부록). Speaker/wording variants, DDXPlus,
n = 1,747 each.

| Wording | Accuracy | Moved | Adopted |
|---|---:|---:|---:|
| Referral note (one line) | .814 | 324 | 95 |
| Colleague | .821 | 305 | 107 |
| Patient | .867 | 224 | 17 |
| Realistic multi-sentence note | **.745** | **445** | **236** |

**Table 2d** (2c와 한 쌍 — 같은 해리의 두 번째 조작). Suggestion source on
MedCaseReasoning: the model's own confusions vs. a cue-similar neighbour's
diagnosis (no differential field exists).

| Suggestion source | n | Wrong-arm acc. | Moved | Adopted |
|---|---:|---:|---:|---:|
| Model's own confusion | 849 | .682 | 257 | 106 |
| Nearest-neighbour diagnosis | 694 | .728 | 182 | 32 |

*Destabilisation barely differs (30.3% vs 26.2% moved, z = 1.75, n.s.);
persuasion differs 2.3× (41.2% vs 17.6% adopted, z = 5.26). The same
dissociation the patient-voice wording shows in 2c, from an unrelated
manipulation: persuasion tracks the note's properties, destabilisation
mostly does not.*

---

## Table 3 — What the note does inside (§4.3)

**08-24 신설.** 이 논문의 중심 주장이 지금까지 표가 없이 Figure 4에만
있었다. 그림은 정확히 인용되지 않고, 관성 반론을 닫는 것이 이 세 줄이므로
표가 있어야 한다. 지표 하나(최종 토큰에서 프로브가 정답에 주는 확률),
셀당 값 하나, Δ는 명시된 파생 열.

**Table 3.** Probability the cross-fit linear probe places on the gold
diagnosis at the final token, by what the model then did. "No note" reads the
same cases with the note removed; finding-position activations are identical
across the two by construction, so Δ is the note's internal cost.

| Behaviour under the wrong note | n | With the note | No note | Δ |
|---|---:|---:|---:|---:|
| Answer unchanged | 1,423 | .980 | .987 | −.007 |
| Lost the gold, answered elsewhere | 229 | .879 | .934 | −.055 |
| Adopted the suggestion | 95 | .736 | .923 | **−.187** |

*The cost grows with the behavioural outcome, so the state is not merely
carrying an earlier answer forward: it reads the note and moves as much as it
reads. It does not move enough to be overturned. In the bottom row the probe
still puts 3.5× more mass on the gold than on the suggestion, while by
definition every one of those cases emitted the suggestion. Across all six
landmarks, 268 of the 324 moved cases (82.7%) never once read the suggestion
as top-1, against an emitted accuracy of .012 on the same cases. At the
finding positions the cost is ±.000 to three decimals, which is what causal
masking guarantees and therefore what the design must reproduce.*

**계기 표기**: 이 표는 **프로브**다. 자연어 판독은 같은 방향을 독립적으로
말하지만 값이 다르다 — 상실형 최종 토큰에서 "상태가 정답을 쥠"이 프로브
.904, v2 판독 .682, 무학습 판독 .636. **결렬의 존재는 두 계기가, 정밀한
해부는 프로브만 말한다**는 것을 본문이 그대로 밝힌다. 프로브는 닫힌
49클래스에 학습된 분류기이므로 이 표는 **MCR에서 미측정이 아니라 정의
불가**다 — 그 사실이 Table 3b 마지막 열과 Table 5의 근거가 된다.

---

## Table 3b — Single-run attribution (§4.3)

셀당 값 하나: All / Silent를 **열 두 개**로. MCR은 숫자 열이 아니라
**적용 가능 여부 열**로 — 값이 아니라 정의의 문제라서.

**Table 3b.** Within-diagnosis AUROC for identifying moved cases from the
wrong-note run alone. Silent: cases whose answer differs from the suggestion
(70% of moved), where output-only signals are blind by construction. The
last column states whether the channel is definable when the diagnosis space
is open (6,934 labels, most occurring once).

| Channel | Internals | AUROC, all | AUROC, silent | Open vocab. |
|---|:-:|---:|---:|:-:|
| Chain-of-thought features | – | .53 | .53 | yes |
| Answer equals suggestion | – | .664 | –ᵃ | yes |
| Linear probe, final token | ✓ | **.924** | **.984** | noᵇ |
| Verified NL readout (ours) | ✓ | .755 | .842 | yes |

ᵃ Undefined on the silent subset: the feature is the subset's defining
condition. ᵇ No class set exists to train on.

*CoT 값은 특징 3종(0.50–0.53) 중 최댓값 하나로 통일해 셀당 값 하나 규칙을
지킨다; 특징별 값은 부록. **MCR의 출력 채널 AUROC는 완료된 개입 답에서 지금
CPU로 계산 가능(▢ 최우선); CoT 채널은 MCR CoT 실행이 필요(GPU ~1–2h,
prompt_cot는 케이스 파일에 이미 있음)** — 나오면 "AUROC, MCR (behavioural)" 열을
추가한다: 두 행동 채널 + probe "n.a." + readout "ᵈ"로 열이 완성되고, 열린
어휘에서 행동 채널도 우연 수준이면 "MCR에서는 아직 아무도 탐지 못 한다"가
어댑터 동기의 마지막 조각이 된다.*

---

## Table 4 — Correction ladder (§4.4)

**Table 4.** Second-pass accuracy with the wrong note still in place. Moved:
the 324 causally moved cases. Capitulation: share of newly broken answers
landing on the suggested diagnosis (first-pass counterpart .293).

| Rung | Appended | Overall | Moved | Capitulation |
|---|---|---:|---:|---:|
| r3 | reconsider request only | .424 | .460 | .450 |
| r4 | + findings re-shown (control) | .417 | .398 | .644 |
| r5 | + readout conclusion & grounds | .418 | .627 | .498 |
| r6 | + probe class label | .467 | .830 | .527 |
| r7 | + the model's own chain (▢ 실행 대기) | – | – | – |

*First-pass baseline: overall .814, moved .012. r5 − r4 = +22.8 pp on moved
(+17.7 pp on the 3× replication); r5 has the lowest capitulation
(z = 6.1/10.6).*

**Table 4d (예정) — 같은 사다리, MedCaseReasoning.** 어느 단이 존재하는지를
코퍼스가 정한다. 이 표의 빈칸은 미실시가 아니라 **결과**다.

| Rung | DDXPlus | MedCaseReasoning | 왜 |
|---|:-:|:-:|---|
| r3 reconsider only | .460 | ▢ 실행 가능 | 어댑터 불필요 |
| r4 findings re-shown | .398 | ▢ 실행 가능 | 어댑터 불필요 |
| r7 own chain | ▢ | ▢ (CoT 실행 필요) | GPU ~1–2h |
| r5 readout conclusion | .627 | ▢ 어댑터 학습 중 | 결론 어댑터가 여는 칸 |
| r6 probe class label | .830 | **✕ 존재 불가** | 진단 6,934종·대부분 1회 — 되먹일 클래스 집합이 없다 |

*r6의 불가능이 §4.4의 마지막 문장이다. DDXPlus만 보면 probe가 교정 비교에서
이기고, 독자는 자연어 채널이 잉여라고 결론지어도 좋다. **그 채널만이 존재하는
코퍼스**가 그 결론의 답이며, 주장이 아니라 측정으로 보여야 한다.
실행: `scripts/run_mcr_ladder.sh` (기본 rungs 3 4 7; 어댑터가 나오면
`RUNGS="3 4 5 7" READOUTS=…`).*

**이 표가 확립하는 명제는 "내부를 되먹여라"이지 "자연어로 되먹여라"가
아니다 (08-25).** r5와 r6이 둘 다 r4(입력 재제시)를 이기고, 내용 정확도를
맞추면 둘 사이 차이는 0이다(4b 1행). 교정 축의 결론은 **채널 중립**이며,
자연어가 필요해지는 곳은 회복률이 아니라 **클래스 채널이 존재하지 않는
코퍼스**와 **근거 제시가 요구되는 상황**이다(Table 5).

**r7이 이 명제를 완성한다 (▢ 실행 대기).** 지금까지 내부 되먹임의 비교
대상은 "입력을 다시 보여주기"였고, 가장 명백한 경쟁자 — **모델 자신의 CoT를
되먹이기** — 는 측정된 적이 없다. 그것 없이는 "내부를 되먹여라"가 서지
않는다. 어댑터 불필요, DDXPlus CoT 산출물 재사용
(`make_correction_ladder_cases.py --rungs 7 --cot-answers`). CoT 실행의 답이
direct 첫 답과 다른 케이스는 제외되므로 r7의 모집단이 작다 — **같은 id로
제한한 r3–r6과만 비교하고, 표에는 그 제한된 열을 따로 싣는다.**

**r6은 제안하는 방법이 아니라 통제다 (08-25 명시).** 표를 처음 보는 독자는
r6의 moved .830을 "probe가 이긴다"로 읽고, 곧바로 **"클래스명을 되먹이는
건 정답을 쥐여 주는 것 아닌가"**라고 되묻는다. 그 되물음은 옳고, 수치가
그대로 인정한다: probe argmax의 정답률은 moved에서 **.8642**(전체 .9599),
AV 판독 결론은 **.5185**(전체 .6754)다. r6의 .830은 .8642를 거의 그대로
따라간다. r6이 존재하는 이유가 바로 이것이다 — r5가 r4를 이긴 것이
**문장이라서**인지 **내용이 맞아서**인지 가르려면 내용만 있고 문장이 없는
단이 필요했고, Table 4b가 그 교란을 제거한다. 답은 내용이다.

세 가지를 본문에 함께 적는다. ① probe는 오라클이 아니라 활성값만 읽는
**교차적합** 분류기이며 해당 케이스의 정답 라벨을 본 적이 없다 — 배포
시점에 실제로 실행 가능한 채널이므로 정답지 누출이 아니다. ② 그러나 probe는
다른 케이스들의 **정답 라벨로 지도학습**되고 AV 판독은 그 감독을 받지
않으므로, r5 vs r6은 형식만이 아니라 **감독 수준도** 다르다. ③ probe가
정의되는 코퍼스라면 r6은 애초에 쓸 정책이 아니다 — Table 4c에서 재실행
없는 argmax 교체(.966)가 r6 재실행(.954)보다 낫다. r6은 사다리의 통제로서
자기 일을 했고, 배포 권고에는 들어가지 않는다.

**Table 4b.** r5 vs. r6 with fed-back content accuracy held fixed. Unseen
replication set (n = 3,319), exact McNemar on discordant pairs.

| Content (readout / probe) | n | r5 | r6 | r5-only : r6-only | p |
|---|---:|---:|---:|:-:|---:|
| correct / correct | 2,189 | .514 | .511 | 144 : 137 | .72 |
| wrong / correct | 1,017 | .223 | .437 | 33 : 250 | <.001 |
| wrong / wrong | 78 | .282 | .192 | 11 : 4 | .12 |
| correct / wrong | 35 | .600 | .086 | 19 : 1 | <.001 |

*Form contributes nothing once content accuracy is matched (row 1); the two
one-sided rows reflect content accuracy, not form. Where form does show is
row 3: when both channels hand over a wrong diagnosis, prose is the safer
carrier (main run .400 vs .240, 8 : 0, p = .008) — a bare name has nothing to
check against the chart, a conclusion with its grounds does.*

**Table 4c.** Deployment policies.

| Policy | DDXPlus | 3× replication |
|---|---:|---:|
| Keep first answer | .814 | .824 |
| Probe selects, argmax replaces | **.966** | **.973** |
| Probe selects, re-ask r6 | .954 | .966 |
| Probe selects, re-ask r5 | .915 | .927 |

---

## Table 5 — Discussion guidance (§5)

**Table 5.** When to use which instrument, from the measurements in §4.

| Setting | Instrument | Basis (§) |
|---|---|---|
| Closed label set, training labels available | Supervised probe | 4.3, 4.4 |
| Open diagnosis space | NL readout | 4.2–4.3 |
| Clinician-facing grounds required | NL readout | 4.1, 4.3 |
| Self-correction by re-asking | Neither — avoid | 4.4 |

---

## 그림이 나르는 것 (표에 안 넣는 수치)

- 위치별 비용 곡선(랜드마크 6지점, 그룹 3종) → **Figure 4**. 최종 토큰의
  세 값과 never-flip 268/324는 **Table 3으로 옮겼다** — 그림은 모양을,
  표는 인용 가능한 값을 나른다.
- 사례 서술(심근염 케이스) → **Figure 5**
- layer×position → **Figure 2**

## 남은 ▢ (표 전반)

- **별칭 매칭 규칙 통일** — 채택 건수가 규칙에 따라 95 / 107 / 139로 갈린다.
  T2b·T2c·T4의 항복률이 모두 이 숫자에 매달려 있으므로, 규칙을 하나로
  정하고 세 표를 같은 규칙으로 다시 집계한다. MCR의 "답 바뀜" 정의도 같은
  통일에 딸려 있다.
- T1: shuffle-control 값, swap/memorization/specificity의 정확한 n,
  답 위치 vanilla 행, MCR 산문 서술률 행
- T3b: MCR 출력 채널 AUROC(CPU 가능), MCR CoT 채널(GPU), logit lens 칸
- T4: **MCR 사다리(Table 4d)** — r3/r4는 지금 실행 가능, r7은 MCR CoT
  실행 필요, r5는 결론 어댑터 대기, r6은 존재 불가(결과). `run_mcr_ladder.sh`
- T4: **r7(자기 설명 되먹임)** — DDXPlus·MCR 양쪽. 이 단이 없으면 4.4는
  "내부를 되먹여라"가 아니라 "뭐라도 되먹여라"까지만 주장한다

## v2 → v3에서 바뀐 것 (08-24)

- **T3 신설**: 기전(대조 곡선·never-flip)이 표 없이 그림에만 있었다.
  관성 반론을 닫는 세 줄이므로 인용 가능한 표가 필요하다.
- 구 T3(채널별 귀속) → **T3b**. 둘 다 §4.3이고, 2b/2c/2d와 같은 관례다.
- T1에 **답-위치 vanilla 행** 추가 (실행 완료).
- 캡션이 계기를 명시하도록 설계 규칙에 한 줄 추가.

## v1 대비 바뀐 것

- T1: 길이·형식 행 제거(단위 불일치 → 본문), Reference 열 신설
- T2: 파생 통계 행 제거(본문), 행=코퍼스·열=조건으로 전치, 2b/2c 분리
- T3b(구 T3): 슬래시 셀 제거(All/Silent 열 분리), MCR을 boolean 열로, CoT 셀당 한 값
- T4: 첫 패스 행을 캡션으로, 4b 열 정리
- T5: 근거 문장 대신 절 참조
