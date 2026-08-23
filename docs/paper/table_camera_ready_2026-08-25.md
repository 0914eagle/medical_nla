# Camera-ready tables, v2 (2026-08-25)

설계 규칙 (v1의 실패에서):
- **표 하나 = 지표 하나.** 단위가 다른 값은 같은 열에 두지 않는다.
- **열 = 조건/방법, 행 = 측정 대상.** 파생 통계(차이, 배수)는 별도 열이
  아니라 본문 또는 명시된 Δ열.
- 셀 하나에 값 하나. 슬래시로 두 값을 넣지 않는다.
- 숫자 열에 텍스트 금지. 정의되지 않는 칸은 – 와 표 각주.
- 소수 자리 통일 (비율 .xxx, pp는 정수 또는 x.x).

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

▢ 남은 것: shuffle-control 값, swap/memorization의 정확한 n, 답-위치
vanilla 행(실행 중), **MCR 산문 서술률 행** — 계기가 실제 임상 문장도
읽는지는 1기 mcr_sweep 산출물 재집계(CPU)로 채운다. 이 행이 있어야 T1이
DDXPlus 전용 검증표가 아니게 된다.

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

## Table 3 — Single-run attribution (§4.3)

셀당 값 하나: All / Silent를 **열 두 개**로. MCR은 숫자 열이 아니라
**적용 가능 여부 열**로 — 값이 아니라 정의의 문제라서.

**Table 3.** Within-diagnosis AUROC for identifying moved cases from the
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

*First-pass baseline: overall .814, moved .012. r5 − r4 = +22.8 pp on moved
(+17.7 pp on the 3× replication); r5 has the lowest capitulation
(z = 6.1/10.6). The ladder is DDXPlus-only so far: on MedCaseReasoning, r3
and r4 need no adapter (▢ planned), r5 requires the conclusion adapter (in
training), and r6 cannot exist — no class set to feed back. That asymmetry
is the point §4.4 ends on.*

**Table 4b.** r5 vs. r6 with fed-back content accuracy held fixed. Unseen
replication set (n = 3,319), exact McNemar on discordant pairs.

| Content (readout / probe) | n | r5 | r6 | r5-only : r6-only | p |
|---|---:|---:|---:|:-:|---:|
| correct / correct | 2,189 | .514 | .511 | 144 : 137 | .72 |
| wrong / correct | 1,017 | .223 | .437 | 33 : 250 | <.001 |
| wrong / wrong | 78 | .282 | .192 | 11 : 4 | .12 |
| correct / wrong | 35 | .600 | .086 | 19 : 1 | <.001 |

*Form contributes nothing once content accuracy is matched (row 1); the two
one-sided rows reflect content accuracy, not form.*

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

- 84.8% never-flip, 대조 곡선 0.007/.055/.187, 위치별 비용 → **Figure 5**
- 사례 서술(심근염 케이스) → **Figure 4**
- layer×position → **Figure 2**

## v1 대비 바뀐 것

- T1: 길이·형식 행 제거(단위 불일치 → 본문), Reference 열 신설
- T2: 파생 통계 행 제거(본문), 행=코퍼스·열=조건으로 전치, 2b/2c 분리
- T3: 슬래시 셀 제거(All/Silent 열 분리), MCR을 boolean 열로, CoT 셀당 한 값
- T4: 첫 패스 행을 캡션으로, 4b 열 정리
- T5: 근거 문장 대신 절 참조
