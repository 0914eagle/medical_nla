# 논문 표 완결 계획

## 질문

현재까지 확정된 양성·음성 결과로 어떤 표를 본문에 남기고, 각 셀을 어떤
artifact로 채우며, 남은 계산을 최소화할 것인가?

이 문서는 성공할 생성형 Medical-NLA를 가정해 빈 행을 만드는 계획이 아니다. 검증
관문을 통과한 결과만 본문에 넣고, 탈락한 방법은 validation 개발 결과로 보고한다.

## 표 구성 원칙

1. 본문 표는 **representation/behavior**, **DiReCT clinical alignment**,
   **DDXPlus grounding**의 세 개로 닫는다.
2. 생성형 NLA가 validation gate를 통과하지 못하면 locked-test 행을 만들지 않는다.
   `TBD`나 0으로 채우지 않고 그 행 자체를 제외한다.
3. Probe와 structured reader를 open-ended NLA처럼 부르지 않는다. 각각
   `closed decoder`와 `structured monitor`로 명시한다.
4. D10/D14/D16 같은 방법 개발 결과는 동일 metric이 아니므로 clinical alignment나
   grounding 주표에 억지로 합치지 않고 별도 validation ablation 표로 둔다.
5. Table 3 grounding gate를 통과한 free-generating NLA가 없으므로 기존 text-patching
   Table 4는 실행·게재하지 않는다. 이는 누락이 아니라 사전 등록된 gate의 결과다.
6. Validation과 locked test를 같은 열에서 혼합하지 않는다. Method selection 수치는
   appendix/development table, 동결 후 한 번 계산한 값은 main result table에 둔다.

## Main Table 1 — Internal-state availability and readout boundary

Table 1은 backbone 행동, closed probe, vanilla open readout의 역할을 분리한다. 한 행의
accuracy가 서로 다른 의미를 갖지 않도록 세 panel로 구성한다.

### Panel A. Backbone diagnostic behavior

동결 split의 동일 case ID에서 기존 source output을 재집계한다. 새 backbone generation은
필요 없다.

| Generation | Pool | n | Parse | Strict PDD | Disease category |
|---|---|---:|---:|---:|---:|
| Direct, answer-prefilled | test seen | 72 | pending reindex | pending | pending |
| Source CoT | test seen | 72 | pending reindex | pending | pending |
| Direct, answer-prefilled | PDD-heldout | 106 | pending reindex | pending | pending |
| Source CoT | PDD-heldout | 106 | pending reindex | pending | pending |

이미 계산된 171-case exploratory 참고값은 Direct/CoT strict PDD
`.2105/.1930`, category `.5029/.5088`이다. 이 값은 새 72/106 두 pool의 셀에
복사하지 않는다. 재집계 결과와 대조하는 audit 값으로만 사용한다.

`Official semantic diagnosis` 열은 필수 열에서 뺀다. 공식 DiReCT evaluator의 주 목적은
explanation alignment이고, strict/category와 별도로 source diagnosis semantic judge를
다시 돌리는 것이 핵심 주장에 필요하지 않기 때문이다. 실행 시간이 남으면 appendix
sensitivity로만 추가한다.

### Panel B. Closed P0 decodability

| Dataset / target | Decoder | Validation | Locked test | Control |
|---|---|---:|---:|---:|
| DiReCT disease category | HS24 linear, 25-way | `.5962` | test-seen pending | label-shuffle pending |
| DiReCT canonical PDD | HS24 linear, 49 train labels | `.4423` | test-seen pending | label-shuffle pending |
| DDXPlus finding presence | HS24 multi-label, 91 IDs | `.9607` | `.9562` | shuffled `.7938`; gap `+.1624 [.1576,.1672]` |
| DDXPlus native value | HS24 conditional, 6 tasks/32 values | `.7700` | `.7659` | shuffled `.5791`; gap `+.1868 [.1650,.2091]` |

Canonical PDD는 train ontology에 없는 PDD-heldout node에 대해 top-1이 정의되지 않으므로
PDD-heldout 셀을 만들지 않는다. Disease category는 실제 train-supported category 여부를
감사한 뒤 정의되는 행만 보고한다. Source-decision probe는 논문의 필수 결론에 필요하지
않고 ontology 추가 결정이 남아 있으므로 본문 표에서 제외한다.

Layer sensitivity는 Table 1의 열을 늘리지 않고 Figure 2 또는 appendix에 다음 확정값을
그대로 사용한다.

| Target | HS16 | HS24 | HS32 |
|---|---:|---:|---:|
| DiReCT category top-1 | `.5000` | `.5962` | `.5192` |
| DiReCT PDD top-1 | `.3846` | `.4423` | `.3846` |
| DDXPlus finding micro F1 | `.9636` | `.9607` | `.9607` |
| DDXPlus value accuracy | `.7641` | `.7700` | `.6990` |

### Panel C. Open vanilla AV boundary

Validation 52 cases × 2 prompts × 3 layers의 semantic audit 312행을 짧은 boundary
panel 또는 본문 문장으로 보고한다.

| Readout | source answer | gold PDD | category |
|---|---:|---:|---:|
| Vanilla AV, default HS32 | `0/52` | `0/52` | `0/52` |
| Vanilla AV, task-aligned HS32 | `0/52` | `0/52` | `0/52` |

이는 observation grounding 0이 아니라 **명시적 diagnosis readout 0**이다. Exact readout
quote를 요구한 single Llama-3-8B judge 결과임을 caption에 쓴다. HS16 category의
`1/52`는 appendix layer sensitivity로 둔다.

## Main Table 2 — DiReCT clinical explanation alignment

### Locked-test 본문 표

72 test-seen과 106 PDD-heldout을 두 panel로 보고한다. 지금 확정적으로 실행할 행은
Source CoT와 Vanilla NLA 두 개다.

| Method | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|---:|
| Source CoT | pending | pending | pending | pending | pending | pending | pending |
| Vanilla NLA | pending | pending | pending | pending | pending | pending | pending |
| Selected generative Medical-NLA | gate 통과 시에만 생성 |  |  |  |  |  |  |

마지막 행은 D10 1,552-step 결과가 teacher-forced D5 gate를 통과하고, 이어지는 generation
validation에서 `Obscomp > .2130`을 통과할 때만 materialize한다. 실패하면 빈칸을 남기지
않고 행을 삭제한다. D10이 DDXPlus one-claim objective라는 이유만으로 곧바로 DiReCT
locked test를 읽지 않는다.

### Appendix development table

생성형 방법 실패는 숨기지 않고 validation 표로 보고한다. 서로 다른 extractor run의
Source CoT 값이 조금 다르므로 아래 두 block을 합치지 않는다.

**Common 248+248 SFT, same 50 Direct cases**

| Method | Obs. rows | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | `.3110` | `.4069` | `.2399` | `.0657` | `.0168` |
| Vanilla NLA | 10/50 | `0` | `0` | `0` | `0` | `0` |
| Common SFT seed 17 | 50/50 | `.0100` | `.0037` | `.0034` | `0` | `0` |
| Common SFT seed 29 | 50/50 | `0` | `0` | `0` | `0` | `0` |
| Common SFT seed 43 | 50/50 | `.0070` | `.0054` | `.0043` | `0` | `0` |

**Full-data canonical-target SFT, same 50 Direct cases**

| Method | Obs. rows | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---:|---:|---:|---:|---:|---:|
| Source CoT | 50/50 | `.2835` | `.3726` | `.2130` | `.0650` | `.0153` |
| Full-data SFT seed 17 | 50/50 | `.0544` | `.0502` | `.0301` | `0` | `0` |
| Full-data SFT seed 29 | 50/50 | `.0553` | `.0388` | `.0296` | `0` | `0` |

이 표의 결론은 “Medical-NLA가 0점”이 아니라, 형식 생성은 성공했지만 physician
observation과 사례별로 정렬되지 않았다는 것이다. 따라서 seed를 골라 locked test로
보내지 않았다.

## Main Table 3 — DDXPlus activation grounding

Probe와 structured monitor의 locked core rows는 완결됐고, branch-independent Vanilla baseline이
남았다. Probe와 structured monitor를 나란히 두되 동일한 종류의 모델이라고 해석하지 않는다.

### Panel A. Static availability and case specificity

| Method | Finding F1 | Same-diagnosis shuffled | Pair gap | Native-value accuracy |
|---|---:|---:|---:|---:|
| Frozen closed probe | `.9562` | `.7938` | `+.1624 [.1576,.1672]` | `.7659` |
| Probe-guided structured monitor | `.9587` | `.7938` | `+.1624` | `.7654` |
| Vanilla NLA | pending full locked baseline | pending | pending | pending |

Structured monitor의 mean emitted claims는 `4.9353`, native-value emission coverage는
`.9995`다. Prompt text는 prediction construction에 사용하지 않았다.

### Panel B. Counterfactual response

| Method | Deletion phantom | Removal success | Untouched retention | Replacement hit | Old persistence | Clean switch |
|---|---:|---:|---:|---:|---:|---:|
| Probe-guided structured monitor | `.3593` | `.6407` | `.9987` | `.1466` | `.5955` | `.0804` |
| Vanilla NLA | pending | pending | pending | pending | pending | method-specific denominator |

분모는 deletion `4,540`, value edit `539`, clean-switch eligible `398`이다. 이 결과는
정적 finding은 강하게 읽히지만 삭제 state가 완전히 사라지지 않고 native value update는
약하다는 양면 결론을 만든다.

공개 checkpoint의 **Vanilla NLA는 branch-independent baseline**이므로 actor prompt와 semantic
mapper를 먼저 동결한 뒤 full locked row를 계산한다. Adapted SFT/D10 행만 validation generation
grounding을 통과한 경우에 추가한다. Promotion을 통과하지 못한 adapted method의 locked-test
`TBD` 행은 만들지 않는다.

## Appendix Table — Generative method development gates

서로 다른 실패를 하나의 accuracy로 합치지 않고 각 방법의 사전 지정 gate를 숫자로
보고한다.

| Method | Primary validation statistic | Frozen requirement | Result |
|---|---|---|---|
| Full-data SFT | DiReCT Obscomp | `>.2130` | `.0301/.0296`, fail |
| D10 1×2, 20 steps | ranking-control changed-gap delta, seeds 17/29/43 | each `>=.05`, CI > 0, specificity | `+.0005/+.0028/+.0030`, fail |
| D14 K=5 OOF teacher | original cue precision | `>=.90` plus six calibration gates | `.8881`, fail |
| D16 soft bottleneck | proposed-control Direct alignment delta | each `>=.005`, CI > 0 | `-.001137/-.001476/+.001433`, fail |
| D16 frozen-z | auxiliary-control finding F1 | positive across seeds | `-.0009/-.0007/-.0016`, fail |
| D10 budget calibration | final step 1,552 D5 gate | same D5 gate, no extension | running/pending |

D16 frozen-z의 own-shuffled gap delta는 `-.0050/-.0046/-.0058`, value accuracy delta는
`-.0137/-.0096/-.0160`, deletion-drop delta는 `-.0167/-.0141/-.0151`이었다.

이 표는 여섯 objective를 같은 metric으로 순위화하지 않는다. 각 objective가 자기
promotion gate를 통과했는지만 보인다. D10 budget run은 기존 결과를 본 뒤 승인된
post-hoc exploratory calibration이라고 명시한다.

## 삭제할 기존 표와 그림

### Text patching Table 4

삭제한다. Free-generating NLA가 grounding gate를 통과하지 않았으므로 identity patch,
edit patch, behavioral utility를 수행하면 사전 등록 순서를 위반한다. `not run because the
grounding prerequisite failed`를 Results와 Limitations에 한 문장으로 기록한다.

### Figure 4 text bottleneck intervention

같은 이유로 삭제한다. 성공하지 않은 미래 방법을 전제로 빈 pipeline 그림을 남기지 않는다.

### 유지할 그림

1. Figure 1: 전체 evaluation pipeline과 두 gate.
2. Figure 2: HS16/24/32 probe layer sensitivity.
3. Figure 3: DDXPlus original/deletion/value-edit paired response 분포. 평균만 반복하지
   않고 deletion probability drop, phantom, untouched retention 분포를 시각화한다.

## 실행 원장: Medical-NLA 성공과 무관하게 채울 수 있는 셀

여기서 “성공과 무관하다”는 두 의미를 구분한다.

1. **결과 분기와 무관**: Medical-NLA가 성공하든 실패하든 최종 논문에 필요한 baseline이다.
2. **지금 즉시 열어도 됨**: locked-label 접근 순서까지 만족해 현재 실행해도 되는 작업이다.

DiReCT Table 1A/1B와 Table 2 baseline은 첫 번째에는 해당하지만 두 번째에는 해당하지 않는다.
이들은 Medical-NLA 성공 여부와 무관하게 반드시 필요하지만, 사전 기록한 접근 규약에 따라 D10
최종 분기와 final recipe hash를 고정한 뒤 한 batch로만 연다. 반대로 DDXPlus locked test는 frozen
probe/structured-reader protocol로 이미 한 번 평가됐으므로, 완전히 고정한 Vanilla baseline을
추가하는 것은 새 method selection을 하지 않는다는 조건에서 지금 실행할 수 있다.

### 전체 작업 원장

| 우선순위 | 작업 | 논문 셀/그림 | 모집단 | 계산 자원 | 현재 상태 | 실행 시점 |
|---:|---|---|---|---|---|---|
| 0 | 승인된 고정 셀 동기화 | Table 1B DDXPlus, Table 3 structured reader, Appendix gates | 기존 summary | CPU | 완료/재검증 가능 | 지금 |
| 0 | Vanilla actor prompt와 환경 provenance 동결 | Table 1C/2/3 Methods | checkpoint sidecar | CPU + 2-GPU load smoke | 미완료 | 지금 |
| 1 | DDXPlus Vanilla full readout | Table 3A/3B open-generator baseline | locked test 10,028 activation rows | 2x4090 두 job, 총 4 GPUs 권장 | 미계산 | prompt/evaluator hash 동결 직후 |
| 1 | Probe layer-sensitivity plot | Figure 2 | 기존 HS16/24/32 validation summaries | CPU | 수치는 완료, 전용 plot 미구현 | 지금 |
| 1 | DDXPlus paired-response plot | Figure 3 | 기존 validation/locked probe-reader scores | CPU | 수치는 완료, 전용 plot 미구현 | 지금 |
| 1 | D10 budget trajectory/final gate | Appendix + Medical-NLA branch 결정 | train 3,104 pairs, validation 3,032 | RunPod A100 80GB | 실행 중/결과 대기 | 중복 실행 금지 |
| 2 | Source Direct/CoT 재index와 진단 집계 | Table 1A | DiReCT frozen 72/106, 기존 496 outputs | CPU | 미계산 | D10 종료 후 single batch |
| 2 | DiReCT HS24 probe locked 적용 | Table 1B DiReCT locked 열 | test_seen 72 | CPU 또는 1 GPU | 미계산 | 같은 single batch |
| 2 | Source CoT clinical alignment | Table 2 baseline | frozen 72/106 = 178 outputs | CPU/Codex extraction + 1x4090 official judge | 미계산 | 같은 single batch |
| 2 | Vanilla DiReCT clinical alignment | Table 2 baseline | frozen 72/106 = 178 activations | 2x4090 generation + extraction + 1x4090 judge | 미계산 | 같은 single batch |
| 3 | Medical-NLA final generation | Table 2/3 conditional row | validation 후 locked populations | GPU | method 성공에 조건부 | promotion 통과 시만 |
| 4 | AR identity/patching | Table 3C/4 conditional | 아직 미동결 | GPU | 미실행 | AV/AR gate가 열릴 때만 |

이 표에서 우선순위 0-2가 Medical-NLA 최종 성공과 무관한 논문 완결 작업이다. 우선순위 3-4는
성공한 생성형 readout이 있어야만 의미가 있으므로 실패 시 빈 셀로 두지 않고 행/표를 제거한다.

### 계산 없이 지금 확정할 셀

다음은 새 결과를 계산하거나 locked test를 읽지 않고 승인된 14개 row를 canonical paper table에
idempotent하게 materialize/검증한다.

```bash
python scripts/sync_paper_table_fixed_cells.py --write
python scripts/sync_paper_table_fixed_cells.py --check
```

포함되는 핵심 값은 DDXPlus locked finding/value probe `.9562/.7659`, structured-reader
finding/value `.9587/.7654`, deletion phantom `.3593`, removal `.6407`, retention `.9987`,
replacement `.1466`, old persistence `.5955`, clean switch `.0804`, 그리고 D9a/D10/D14/D16
development 결과다. `--write`는 DiReCT locked cells와 conditional generative rows를 건드리지
않는다.

### 지금 4090으로 실행 가능한 독립 baseline

#### A. DDXPlus Vanilla full readout

목적은 Medical-NLA 성공 여부와 관계없이 공개 Vanilla AV가 같은 CoT-P0 activations에서 어느
정도 clinical state와 counterfactual change를 읽는지 Table 3A/3B에 넣는 것이다.

- input: locked-test HS32 manifest 전체 10,028 rows
- generated variants: original 4,543 + cue deletion 4,543 + value edit 942 = 10,028
- paper metric eligibility: deletion 4,540, value edit 539, clean switch는 method별 original-old hit 수
- model: `kitft/nla-gemma3-12b-L32-av`
- decoding 후보가 아니라 동결값: greedy, `do_sample=false`, max new tokens 512, batch 4
- hardware: 12B bf16 model 하나당 4090 두 장; 4 GPUs면 두 manifest shards를 병렬 실행
- 과거 pilot 관측 처리량: 약 14.8 s/row per 2-GPU job
- 단순 wall-time 추정: `10,028 x 14.8 / 2 = 74,207 s`, 약 20.6시간. I/O와 shard 불균형에 따라
  달라지므로 보장 시간이 아니다.

단, 바로 10,028행을 생성하기 전에 아래 두 항목을 먼저 동결한다.

1. `--dump-actor-prompt-template`로 실제 sidecar prompt를 저장하고 byte SHA-256을 기록한다.
2. Open text를 evidence ID/value로 채점할 evaluator를 동결한다. 현재 lexical pilot scorer만으로는
   약칭·의역을 놓칠 수 있으므로 paper용 수치로 바로 쓰지 않는다. Method-blind semantic mapper,
   exact-quote validator, candidate ontology, model/version, prompt hash를 먼저 고정해야 한다.

즉 **generation 자체는 독립 baseline이라 지금 가능하지만, evaluator 동결 전에 long job을
시작하는 것은 권하지 않는다.** Prompt가 바뀌면 10,028 outputs를 재사용할 수 없기 때문이다.

Prompt dump는 server-local valid manifest path를 넣어 다음처럼 만든다. Dump mode에서도 CLI가
`--manifest/--output`을 요구하지만 generation output은 만들지 않는다.

```bash
OUT=/data1/heejae/medical_nla/results/ddxplus_vanilla_locked_v1
MANIFEST=/path/to/server125-remapped/locked_test_hs32_manifest.jsonl
mkdir -p "$OUT/provenance"

python -m src.run_nla \
  --config configs/default.yaml \
  --manifest "$MANIFEST" \
  --output "$OUT/unused_dump_mode.jsonl" \
  --dump-actor-prompt-template \
  > "$OUT/provenance/vanilla_actor_prompt.txt"

sha256sum "$OUT/provenance/vanilla_actor_prompt.txt" \
  > "$OUT/provenance/vanilla_actor_prompt.sha256"
```

현재 generic generation entry point는 `src.run_nla`가 있지만, 두 shard 생성·merge·population
검증·semantic scoring까지 한 번에 수행하는 paper-safe wrapper는 아직 없다. 구현할 wrapper는
`scripts/run_ddxplus_vanilla_locked_baseline_4gpu.sh` 하나로 고정하고 다음을 hard fail해야 한다.

- 두 shard의 `base_id/variant` union이 canonical 10,028 rows와 정확히 일치
- duplicate/missing activation 0
- actor prompt hash와 model revision 일치
- generation 설정 일치
- semantic evaluator protocol hash 일치

#### B. Figure 2와 Figure 3

두 그림은 이미 계산된 숫자로 만들 수 있어 GPU가 필요 없다.

- Figure 2: DiReCT category/PDD와 DDXPlus finding/value의 HS16/24/32 validation sensitivity
- Figure 3: DDXPlus original/deletion/value-edit에서 probability drop, phantom, retained
  preservation, replacement/old persistence 분포

기존 `scripts/run_paper_figures_without_figure1.sh`는 이전 hint-intervention 논문의 Figure 2-4를
그리는 legacy pipeline이므로 이 새 표 구조에 사용하지 않는다. 새 canonical figures에는 별도의
`make_medical_nla_probe_layer_figure.py`와 `make_ddxplus_counterfactual_figure.py`가 필요하다.

### D10 종료 뒤 한 번만 실행할 DiReCT baseline batch

D10 결과가 성공이든 실패든 아래 baseline은 모두 논문에 들어간다. 단, 분석자가 72/106 label을
먼저 보고 final recipe를 바꾸는 경로를 막기 위해 D10 final checkpoint 판정과 recipe hash 기록이
선행돼야 한다.

1. 기존 496 source Direct/CoT outputs을 confirmatory split으로 reindex한다. Backbone generation은
   다시 하지 않는다.
2. Table 1A의 parse, strict PDD, disease category, official semantic diagnosis, paired McNemar를
   72/106별로 계산한다.
3. 이미 validation에서 HS24로 동결한 DiReCT category/PDD probes를 test_seen 72에 한 번 적용하고,
   label-shuffle control과 patient-group cluster CI를 계산한다. Heldout PDD ontology 밖 cell은 `N/A`다.
4. Source CoT 178 outputs를 method-blind extractor와 official evaluator에 보낸다.
5. Vanilla의 저장 output이 prompt/decoding/model hash까지 일치하면 겹치는 rows만 재사용하고,
   하나라도 다르면 178건 전부 다시 생성한다.
6. Source CoT와 Vanilla에 **동일한 extractor requests, candidate ontology, judge version, official
   evaluator**를 사용하고 72/106 panel을 함께 materialize한다.

4090 네 장을 쓸 때의 병렬화는 다음이 안전하다.

- CPU: source reindex, Table 1A lexical 집계, probe input join, extraction request 생성
- GPUs 0-1: Vanilla 178 readouts 생성
- GPU 2: 준비된 Source CoT claims의 official Llama evaluator
- GPU 3: probe 적용 또는 evaluator 후속 queue

과거 DiReCT E2 Vanilla 관측치는 2x4090에서 171행, max 256 tokens가 약 4.73 s/row였다. 같은
recipe라면 178행 generation 자체는 약 14분 규모지만, quote extraction과 official semantic
evaluation 시간이 별도로 든다. 이 값은 runtime 참고치이지 논문 protocol은 아니다.

현재 `scripts/run_direct_e4_validation_evaluator.sh`는 50-case `val_seen` 전용이며 cohort와 expected
count가 hard-coded돼 있다. 이 파일에 test path를 임의로 넣어 재사용하지 않는다. Final batch에는
다음 세 wrapper를 먼저 구현하고 preflight를 통과시킨다.

```text
scripts/reindex_and_score_direct_locked_source_outputs.py
scripts/evaluate_direct_locked_probes.py
scripts/run_direct_locked_baseline_batch.sh
```

마지막 wrapper는 D10 decision record, recipe hash, 72/106 split hashes, prompt hash가 없으면 실행을
거부해야 한다.

### D10 budget run과 조건부 branch

D10 budget calibration은 현재 RunPod A100-SXM4-80GB 한 장에서 동일 objective/data를 유지하고
20 -> 1,552 steps만 변화시킨 dose-response다. 같은 실험을 4090에서 중복 실행하지 않는다.

1. Step 1,552 final checkpoint만 frozen D5 gate로 판정한다.
2. 실패하면 Appendix trajectory/final gate를 확정하고 generative final 행을 제거한다.
3. 통과하면 validation generation을 한 번 수행한다.
4. DiReCT `Obscomp>.2130`과 `Expcom>.0650`, DDXPlus generation grounding gate를 모두 통과한
   단일 recipe만 locked test로 보낸다.
5. 1,552 step 이후 epoch/lambda/temperature 자동 탐색은 하지 않는다.

### 지금 돌리지 않을 작업

- promotion을 통과하지 않은 SFT/D10/D16 checkpoint의 locked-test generation
- test 결과를 보고 threshold/layer/prompt를 다시 고르는 DDXPlus sweep
- 이미 locked 수치가 확정된 probe와 structured reader의 불필요한 재학습
- identity와 matched-over-shuffled protocol이 동결되지 않은 AR patching
- old pilot 71/100을 새 72/106 결과로 바꾸어 쓰는 재집계

### 가장 빠른 실제 순서

1. 지금 CPU에서 fixed-cell sync/check와 provenance 파일 목록을 완결한다.
2. Vanilla sidecar prompt와 DDXPlus semantic mapper를 동결한다.
3. 4090 네 장이 비면 DDXPlus Vanilla 10,028행을 두 2-GPU shards로 실행한다.
4. 동시에 CPU에서 Figure 2/3 전용 plotter와 DiReCT locked-batch wrappers를 구현·fixture test한다.
5. RunPod D10이 끝나면 final branch/recipe hash를 기록한다.
6. 같은 날 DiReCT Table 1A -> 1B -> Table 2 Source CoT/Vanilla를 single batch로 실행한다.
7. Medical-NLA promotion 성공 시에만 conditional row를 추가하고, 실패 시 structured reader를
   main positive result로 유지한다.

## 완료 정의

다음 조건을 모두 만족하면 branch-independent 숫자 표가 완료된다.

- Table 1A frozen 72/106 Direct/CoT 재집계 완료
- Table 1B DiReCT test-seen closed probe와 label-shuffle 완료
- Table 2 Source CoT/Vanilla frozen baseline 완료
- DDXPlus Vanilla full baseline의 prompt/evaluator protocol과 결과 완료
- D10 step 1,552의 최종 분기 판정 완료
- Figure 2/3 canonical plot과 source artifact hash 기록 완료

Medical-NLA가 promotion에 실패하면 conditional 행과 AR 표를 삭제하는 것이 완료 상태다. 성공한
경우에만 validation generation gate와 locked-test conditional row가 추가된다.

## 판정

실행 순서의 현재 canonical 규칙은 다음과 같다.

1. Fixed cells와 CPU figures/provenance는 지금 완결한다.
2. DDXPlus Vanilla는 prompt와 semantic mapper를 동결한 뒤 실행한다.
3. DiReCT 72/106 locked-label batch는 D10 final branch와 recipe hash 기록 뒤 한 번만 연다.
4. Promotion을 통과하지 않은 adapted generator와 AR은 locked test에서 실행하지 않는다.

## 실행 기록

### 2026-08-29

사람이 "지금은 기존 확정값과 표 구조만 반영"을 승인했다.
`docs/paper/tables_and_figures.md`에 다음을 반영했다.

- Table 3을 method-class panel 구조로 교체하고 probe/structured monitor locked 행 기입
- Table 2의 가상 행 제거, Source CoT/Vanilla와 조건부 Medical-NLA 규칙만 유지
- Appendix development gates에 Full SFT/D10/D14/D16 수치 기입
- D9a protocol 기록 (`P=.90/D=0/M=0`, coverage .9993, false support .0378, pairs 3,104)
- Vanilla artifact 재사용은 prompt/model/decoding hash가 일치할 때만 허용

### 2026-08-30

실행 계획을 수치 원장에서 이 문서로 분리했다. 수치·분모·hyperparameter의 canonical 기록은
`2026-08-29-paper-table-values-and-reproducibility.md`, 서버·GPU·실행 순서의 canonical 기록은
이 문서가 담당한다.

#### 논문 수치 계산기 구현

다음 전용 스크립트를 구현했다. 기존 validation runner나 legacy figure script에 test path를
임의로 바꾸어 넣지 않는다.

| script | 채우는 항목 | 핵심 fail-closed 조건 |
|---|---|---|
| `reindex_and_score_direct_locked_source_outputs.py` | Table 1A 72/106 Direct·CoT | 정확한 72/106 ID, 중복 0, 명시적 locked confirmation |
| `evaluate_direct_locked_probes.py` | Table 1B DiReCT HS24 test-seen | HS24, 72행, 두 target artifact, 승인된 control protocol |
| `make_medical_nla_probe_layer_figure.py` | Figure 2 | DiReCT/DDXPlus validation JSON에 HS16/24/32가 모두 존재 |
| `make_ddxplus_counterfactual_figure.py` | Figure 3 | frozen HS24 artifact, paired original/deletion/value-edit activation |
| `validate_nla_readout_population.py` | 두 Vanilla baseline population receipt | manifest/output ID exact union, activation 존재, decoding 일치 |
| `run_ddxplus_vanilla_locked_baseline_4gpu.sh` | Table 3 Vanilla locked | prompt·semantic protocol·semantic scorer hash 일치 전 GPU 시작 금지 |
| `run_direct_locked_baseline_batch.sh` | Table 1A/1B/2 single batch | D10 decision·recipe·split·prompt·control hash 일치 |

Figure 2는 두 DiReCT probe job이 별도 directory에 있으므로 `--direct-results`를 두 번 줄 수 있다.
서버 62에서 필요한 JSON을 한곳에 모은 뒤 다음처럼 실행한다.

```bash
python scripts/make_medical_nla_probe_layer_figure.py \
  --direct-results /path/to/direct_e2_probe_pdd_val_v1/validation_results.json \
  --direct-results /path/to/direct_e2_probe_category_val_v1/validation_results.json \
  --ddxplus-results /data/heejae/medical_nla/results/ddxplus_finding_value_probe_val_v1/results.json \
  --output /data/heejae/medical_nla/results/paper_figures/figure2_probe_layers.pdf \
  --values-json /data/heejae/medical_nla/results/paper_figures/figure2_probe_layers_values.json
```

Figure 3은 locked manifest의 activation path가 실행 서버에서 실제로 존재해야 한다. `/data`에서
만든 manifest를 `/data1`에서 읽을 때는 `--path-map /data/heejae=/data1/heejae`를 명시한다.

```bash
python scripts/make_ddxplus_counterfactual_figure.py \
  --artifact /data/heejae/medical_nla/results/ddxplus_finding_value_probe_val_v1/finding_value_hs24.pt \
  --manifest /data/heejae/medical_nla/data/ddxplus_e5_canonical_v1/activations/ddxplus_e5_test_cot_p0_hs24_merged_v1/layer24/last_token/manifest.jsonl \
  --output /data/heejae/medical_nla/results/paper_figures/figure3_counterfactual.pdf \
  --values-json /data/heejae/medical_nla/results/paper_figures/figure3_counterfactual_values.json
```

Figure 2/3은 서버 62에 있는 DDXPlus artifact가 기준이므로 canonical queue도 서버 62에서 CPU로
실행한다. 서버 125에서 필요한 것은 category probe `validation_results.json` 하나뿐이다. Password가
필요한 `scp`를 `nohup` 안에 넣지 말고 먼저 한 번 복사한다.

```bash
mkdir -p /data/heejae/medical_nla/imports/server125/direct_e2_probe_category_val_v1

scp \
  eagle0914@165.132.76.125:/data1/heejae/restricted/direct/e2/direct_e2_probe_category_val_v1/validation_results.json \
  /data/heejae/medical_nla/imports/server125/direct_e2_probe_category_val_v1/
```

그다음 서버 62에서 다음 한 queue만 실행한다.

```bash
cd /home/eagle0914/medical_nla
git pull origin main
mkdir -p /data/heejae/medical_nla/logs

nohup env DATA_ROOT=/data/heejae \
  bash scripts/run_paper_cpu_metrics_62.sh \
  > /data/heejae/medical_nla/logs/paper_cpu_metrics_62_v1.log 2>&1 &
```

이 queue는 GPU를 사용하지 않으며 Figure 2/3 PDF, source-hashed values JSON, 모든 핵심 수치와
실제 분모가 적힌 `summary.md`를 함께 만든다. 기본 output은
`/data/heejae/medical_nla/results/paper_cpu_metrics_62_v1/`이다.

DDXPlus Vanilla wrapper는 현재 **실행 준비 완료, protocol 입력 대기** 상태다. 아래 세 파일/hash가
없으면 10,028행 generation을 시작하지 않는다.

1. `src.run_nla --dump-actor-prompt-template`로 얻은 actor prompt와 승인 SHA-256
2. method-blind semantic mapper protocol JSON과 승인 SHA-256
3. 그 protocol을 구현한 scorer script와 승인 SHA-256

이는 lexical scorer를 paper metric으로 잘못 쓰는 것을 막기 위한 의도적 정지점이다. Semantic
protocol이 승인되면 `run_ddxplus_vanilla_locked_baseline_4gpu.sh`가 two-shard generation,
10,028행 exact-union 검증, provenance 기록, semantic scoring을 한 queue로 끝낸다.

DiReCT locked batch 전에는 patient-group label-shuffle와 cluster bootstrap을 담은 JSON을 먼저
승인·hash 고정한다. 최소 schema는 다음과 같다.

```json
{
  "shuffle_unit": "patient_group",
  "shuffle_seed": 17,
  "control_init_seed": 17,
  "bootstrap_seed": 17,
  "bootstrap_replicates": 10000
}
```

여기서 control은 test label을 섞어 재채점하는 것이 아니다. Train patient-group label을 결정론적으로
derange한 뒤 validation에서 이미 선택된 learning rate, weight decay, class balancing, epoch 수를
그대로 사용해 새 linear head를 학습하고 원래 test label에 평가한다. 이 값은 구현 default가 아니라
locked access 전에 승인해야 하는 protocol이다. D10 final decision,
final recipe, split protocol, Vanilla prompt와 이 control protocol의 SHA-256을 모두 전달해야
`run_direct_locked_baseline_batch.sh`가 Table 1A -> 1B -> Vanilla 178 -> 두 pool의 동일 extractor와
official evaluator 순서로 진행한다.

## 검토 (Claude, 2026-08-30)

**[동의] 원장/실행 분리와 "성공과 무관"의 두 의미 구분이 이 문서의 핵심
기여다.** 특히 후자 — (1) 결과 분기와 무관 vs (2) 지금 열어도 됨 — 는 내가
앞서 제기한 "test 일괄 개봉" 반론을 정확히 해소한다: DiReCT 1A/1B/2는 (1)
이지만 (2)가 아니므로 batch 유지, DDXPlus Vanilla는 locked test가 이미
"locked downstream method evaluation"으로 열려 있고 baseline이 완전
동결·무선택이므로 (2)에도 해당한다. 이 구분을 받아들인다.

**[동의] Vanilla 10,028행 전에 prompt dump/hash와 semantic mapper 동결을
강제한 것.** "Prompt가 바뀌면 10,028 outputs를 재사용할 수 없다"가 결정적
이유다. Lexical pilot scorer를 paper 수치로 바로 쓰지 않는다는 것도 맞다.

**[확인] Fixed-cell sync 검증 완료.** `sync_paper_table_fixed_cells.py
--check`가 canonical `docs/paper/tables_and_figures.md`에서 `[ok] 14 fixed
rows`로 통과함을 재확인했다(2026-08-30).

**[제안] Vanilla 10,028행의 실행 위치로 유휴 RunPod pod를 검토하라.**
D10이 끝나면 A100 80GB pod가 놀게 된다. 계산:

- 4090 추정치는 2-GPU sharded 14.8 s/row 기준 ~20.6시간. A100 단일
  device는 sharding/PCIe/pipeline bubble이 없어 큰 폭 단축이 예상된다
  (정확한 배수는 20행 실측 후 확정).
- Locked-test activation은 현재 pod bundle에 없으므로 두 번째 DDXPlus-only
  bundle(manifest + 10,028 tensors, ~150MB)이 필요하다. DiReCT 금지 규칙
  동일 적용.
- 이 baseline은 arm 간 비교가 아니므로 D10의 "6 runs 동일 하드웨어" 조건과
  무관하다. 단 greedy decoding도 hardware에 따라 미세하게 달라질 수 있으니
  GPU model을 provenance에 기록한다.
- 비용은 pod 요금 vs lab 4090 무료·20시간 점유의 교환 — 사람 결정 사항.

**[제안] 지금 CPU로 구현 가능한 세 가지는 내가 잡을 수 있다.**
(a) `make_medical_nla_probe_layer_figure.py`, (b)
`make_ddxplus_counterfactual_figure.py` — 둘 다 기존 summary JSON을 입력으로
받는 plotter + fixture test, (c) DiReCT locked-batch wrapper 3종의 뼈대
(D10 decision record/recipe hash 없으면 실행 거부하는 preflight 포함).
D10 대기 중 병렬 작업으로 적합하다.

나머지 — Table 1A semantic diagnosis 열의 appendix 강등, legacy figure
pipeline 불사용 선언, "지금 돌리지 않을 작업" 목록 — 전부 동의한다.
