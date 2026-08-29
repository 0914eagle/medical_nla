# 논문 표 수치 원장과 실험 재현 규약

## 문서 목적

이 문서는 현재 논문 표에 들어갈 수치를 한곳에 모으고, 각 숫자가 어떤 모집단, 모델,
prompt, activation, 학습 절차, metric으로 계산되었는지를 재현 가능한 수준으로 기록한다.
Medical-NLA 성공을 전제한 표 구조는
[`2026-08-29-paper-tables-success-scenario.md`](2026-08-29-paper-tables-success-scenario.md)를
따르되, 이 문서는 성공을 선언하지 않는다.

숫자의 상태는 다음 네 종류로 구분한다.

- **Locked**: protocol을 동결한 뒤 locked-test에서 한 번 계산한 최종 수치
- **Validation**: 모델·layer·threshold 선택에 사용한 개발 수치
- **Exploratory**: 과거 split 또는 pilot에서 얻어 방향성만 해석할 수 있는 수치
- **Not computed**: 표 구조는 확정했지만 해당 모집단에서 아직 계산하지 않은 셀

Validation 숫자를 locked-test 숫자처럼 옮기거나, 과거 71/100 split을 현재 72/106
split의 결과처럼 쓰지 않는다.

---

## 1. 공통 데이터와 backbone protocol

### 1.1 DiReCT 모집단

Canonical private manifest는 raw 511 notes를 읽었다. 충돌, ID 실패, 중복 등 15행을
eligible population에서 제외해 496 notes를 사용한다.

- logical population SHA-256:
  `7d0a89a880fa868959099b7146c369cccaac5e7701d7ce5d8f01356ecfb68894`
- split seed: `17`
- split 단위: patient group
- PDD-heldout 구성은 label component 단위로 분리
- 이미 pilot에서 사용한 PDD 5개는 confirmatory heldout 후보에서 금지

| Split | notes | patient groups | PDDs | categories | exact gold label in note | 역할 |
|---|---:|---:|---:|---:|---:|---|
| train | 266 | 244 | 49 | 25 | 18 | probe 학습, Medical-NLA adaptation |
| val_seen | 52 | 47 | 24 | 18 | layer, epoch, promotion gate |
| test_seen | 72 | 64 | 25 | 21 | seen-PDD locked evaluation |
| test_pdd_heldout | 106 | 103 | 12 | 10 | unseen-PDD locked evaluation |

Generative SFT에서는 note에 exact canonical gold-label phrase가 있는 행을 제외해
train/validation `248/50`을 사용한다. Closed probe의 `266/52`와 분모가 다르다. Test의
exact-label-exposed `3/72`, `5/106`은 primary에서 사후 제외하지 않고 sensitivity로만
추가한다.

동일 환자의 여러 notes가 있으므로 confidence interval은 note-level iid bootstrap이 아니라
`patient_group` cluster bootstrap을 사용한다. 또한 496행 모두에서 source output이 이미
materialize되었으므로 이 평가는 “dataset-level untouched test”가 아니라 **locked downstream
method evaluation**이다.

관련 코드와 protocol:

- `scripts/make_direct_canonical_manifest.py`
- `scripts/make_direct_patient_pdd_splits.py`
- `scripts/reindex_direct_activations.py`
- `docs/data/direct_dataset_and_split.md`

### 1.2 DDXPlus 모집단

DDXPlus는 official train/validation/test를 섞지 않는다. Train에서 ontology와 probe를 만들고,
validation에서 layer와 threshold를 선택한 뒤, locked test에서는 어떤 선택도 다시 하지 않는다.

| Population | original cases | activation/intervention rows | 역할 |
|---|---:|---:|---|
| official train development | 4,655 | 4,655 CoT-P0 originals | ontology, probe, SFT |
| D9a supported pairs | 3,104 | original/deleted pairs | ranking objective |
| validation | 4,525 | 10,006 | layer, threshold, promotion gate |
| locked test | 4,543 | 10,028 | final grounding evaluation |

Metric마다 eligibility가 달라 분모도 다르다.

| Metric family | validation | locked test |
|---|---:|---:|
| same-diagnosis hard-shuffle pairs | 4,106 | 4,121 |
| native-value targets | 2,183 | 2,136 |
| cue-deletion pairs | 4,523 | 4,540 |
| native-value-edit pairs | 533 | 539 |
| clean-switch eligible | 395 | 398 |

이 숫자는 임의의 표본 크기가 아니라 metric별 eligibility를 적용한 뒤 남은 분모다.

- **Same-diagnosis hard-shuffle pairs**: own case와 진단은 같지만 `base_id`가 다른 donor를
  만들 수 있는 original cases다. Own activation의 prediction을 donor case의 target과 비교해
  진단 template만 말한 것인지 환자별 state를 읽은 것인지 검사한다. Diagnosis bucket 안에
  valid donor가 없는 case는 제외되어 validation/test가 4,106/4,121이다.
- **Native-value targets**: train ontology가 지원하는 multi-value evidence ID가 original case에
  있고, 그 case의 native value도 train-supported class인 occurrence다. Finding이 “있다”를 넘어서
  severity/location/character 같은 값을 구분하는 conditional value accuracy의 분모이며
  2,183/2,136이다. 전체 환자 수가 아니라 eligible evidence-value occurrence 수다.
- **Cue-deletion pairs**: canonical builder가 original에서 target cue 하나만 물리적으로 삭제하고
  나머지 presentation을 유지한 derived arm을 정상 생성한 original/derived pair다. Deletion
  probability drop, phantom, removal success, untouched retention에 사용하며 4,523/4,540이다.
  Validation 2건과 test 3건은 canonical single-cue deletion family를 만들 수 없어 제외됐다.
- **Native-value-edit pairs**: value-bearing cue 하나를 같은 evidence ID의 다른 valid native value로
  바꾼 original/edited pair다. Replacement hit와 old-value persistence의 분모이며 533/539다.
  Binary presence cue나 대체 native value가 없는 cue는 이 과제가 정의되지 않는다.
- **Clean-switch eligible**: value-edit pair 중 original arm에서 reader가 old value를 정확히 읽은
  cases다. 이미 원래 값도 못 읽은 case를 “전환 실패”로 세지 않기 위해 조건부 분모를 쓰며
  395/398이다. Edited arm에서 new value는 맞고 old value는 사라져야 clean switch다.

따라서 이 다섯 줄은 서로 다른 subset이고 더하거나 서로 직접 비교하면 안 된다. Builder가
variant family를 만들고 frozen train ontology가 support를 결정한 뒤, evaluator가 이 조건을
적용해 수를 계산했다. Test score를 본 뒤 분모를 조정하지 않았다.

DDXPlus primary activation은 DiReCT와 instruction 조건을 맞춘 **CoT-P0**다. 과거 local cue
position activation 결과는 별도 선행 실험이며 이 표에 섞지 않는다.

관련 코드:

- `scripts/prepare_ddxplus_e5.py`
- `scripts/prepare_ddxplus_probe_train.py`
- `scripts/train_ddxplus_finding_value_probes.py`
- `scripts/evaluate_ddxplus_finding_value_probes.py`

### 1.3 Source prompt와 generation

모든 source prompt의 공통 header는 다음과 같다.

```text
You are an expert physician. A patient presents as follows:

{note_text}
```

DiReCT CoT suffix:

```text
Work through this case as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Which diagnoses fit the presentation and which do not
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is <diagnosis>."
```

Direct suffix:

```text
What is the single most likely diagnosis?

Give the diagnosis only. Do not explain your reasoning.

You MUST end your response with exactly "The answer is <diagnosis>."
```

Direct arm은 assistant prefill `The answer is`를 넣었다. CoT arm은 prefill하지 않았다.
DDXPlus는 patient demographic과 evidence cues를 bullet presentation으로 rendering한 뒤 같은 CoT
suffix를 사용한다.

| Setting | CoT | Direct |
|---|---|---|
| model | `google/gemma-3-12b-it` | 동일 |
| dtype | bfloat16 | 동일 |
| placement | `device_map=auto`, GPU당 `22GiB` | 동일 |
| decoding | greedy, `do_sample=false` | 동일 |
| max new tokens | 2,048 | 64 |
| batch size | 1 | 4 |
| forced answer | false | false |
| assistant prefill | none | `The answer is` |
| sample/split seed | 17 | 17 |

Temperature와 top-p는 sampling을 사용하지 않아 별도 지정하지 않았다. Hugging Face chat template에
user message 하나를 넣고 `add_generation_prompt=true`로 tokenization했다. Padding은 left,
pad token이 없으면 EOS를 pad로 사용했다. 최종 answer는 response에서 마지막
`The answer is ...` pattern을 parsing했다.

관련 코드:

- `src/case_prompts.py`
- `scripts/run_source_answers.py`
- `configs/default.yaml`

### 1.4 Activation 정의

Gemma hidden dimension은 3,840이다. HS16, HS24, HS32를 float32 tensor로 저장했다.
각 layer/selection directory는 row ID의 CRC32로 정한 256개 shard directory를 사용한다.
Extraction batch size는 DiReCT full run에서 1이었고 resume가 아닌 `--no-resume` run으로
완전 grid를 만든 뒤 case x position x layer completeness를 검증했다.

| Position | 정의 | selection |
|---|---|---|
| CoT-P0 | clinical presentation과 CoT instruction을 모두 읽고 첫 response token을 생성하기 직전 | prompt의 `last_token` |
| P1 | 생성된 reasoning과 마지막 `The answer is` marker를 읽고 diagnosis를 쓰기 직전 | marker의 `last_subtoken` |
| P2 | 생성 diagnosis의 마지막 subtoken | diagnosis `last_subtoken` |

P1/P2는 source response를 teacher forcing한 동일 transcript에서 추출했다. 이 문서의 primary
Medical-NLA와 DDXPlus grounding 표는 **CoT-P0**를 사용한다. P1/P2는 문자열 노출이 있는
positive/sensitivity control이지 primary가 아니다.

관련 코드: `src/extract_activations.py`, `scripts/make_direct_transcript_activation_rows.py`.

---

## 2. Table 1A: Backbone diagnostic behavior

### 2.1 최종 표에 들어갈 셀

현재 frozen 72/106 split의 셀은 아직 재집계하지 않았다.

| Pool | Generation | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis | 상태 |
|---|---|---:|---:|---:|---:|---|
| test_seen, n=72 | Direct, answer-prefilled | Not computed | Not computed | Not computed | Not computed | frozen output 재집계 필요 |
| test_seen, n=72 | Source CoT | Not computed | Not computed | Not computed | Not computed | frozen output 재집계 필요 |
| PDD-heldout, n=106 | Direct, answer-prefilled | Not computed | Not computed | Not computed | Not computed | frozen output 재집계 필요 |
| PDD-heldout, n=106 | Source CoT | Not computed | Not computed | Not computed | Not computed | frozen output 재집계 필요 |

### 2.2 기존 exploratory 결과

과거 pilot split 171행 결과는 방향성 확인용으로만 보관한다.

| Pilot pool | Method | Strict PDD | Disease category |
|---|---|---:|---:|
| overall, n=171 | Direct | .2105 | .5029 |
| overall, n=171 | CoT | .1930 | .5088 |
| old seen, n=71 | Direct | .2535 | .3944 |
| old seen, n=71 | CoT | .2254 | .3803 |
| old heldout, n=100 | Direct | .1800 | .5800 |
| old heldout, n=100 | CoT | .1700 | .6000 |

Strict-PDD paired table은 both correct 26, Direct-only 10, CoT-only 7, neither 128이었다.
CoT-minus-Direct는 `-.0175`, exact McNemar `p=.6291`이었다. Category 차이는 `+.0058`,
`p=1.0000`이었다. 따라서 CoT가 strict PDD를 낮춘다고 확정할 수 없고, 이 수치는 현재
72/106 결과를 대신할 수 없다.

이 subsection의 171-case pilot 표는 **논문 main table에는 들어가지 않는다**. Main Table 1A는
frozen 72/106 결과만 사용한다. 171-case 값은 appendix의 development history 또는 Results의
“split correction 전 exploratory audit” 한 문장으로만 보고할 수 있다. 표 공간이 부족하면
논문에서는 제외하고 발표/내부 문서에만 남긴다.

### 2.3 계산 정의

- **Parse coverage**: formatted final answer가 parsing된 rows / 전체 rows
- **Strict PDD**: parsed answer가 canonical PDD 또는 사전 정의 alias와 일치한 rows / 전체 rows
- **Disease category**: parsed answer가 상위 disease category와 일치한 rows / 전체 rows
- **Official semantic diagnosis**: strict alias가 아닌 official semantic diagnosis matcher가
  gold diagnosis와 의미상 같다고 판정한 rows / 전체 rows
- Pairwise Direct-CoT 비교: 같은 `base_id`끼리 결합하고 discordant pair에 exact McNemar test
- Test CI: `patient_group` cluster bootstrap

Frozen 셀을 채울 때는 기존 496행 source answer artifact를 confirmatory split ID로 재index한다.
Backbone을 다시 생성하거나 prompt를 바꾸지 않는다.

이 locked label batch는 D10 최종 분기와 final recipe hash를 먼저 동결한 뒤 Table 1B와 Table 2
baseline과 함께 한 번만 연다. CPU 작업이라는 이유로 지금 미리 집계하지 않는다.

---

## 3. Table 1B: Closed P0 decodability

### 3.1 현재 수치

| Dataset / target | Decoder | Validation | Locked evaluation | Control | 상태 |
|---|---|---:|---:|---:|---|
| DiReCT category | HS24, 25-way linear | .5962 | test_seen n=72: Not computed | label shuffle pending | Validation |
| DiReCT canonical PDD | HS24, 49-way linear | .4423 | test_seen n=72: Not computed | label shuffle pending | Validation |
| DDXPlus finding | HS24, 91-label multi-label | .9607 | **.9562** | shuffled .7938; gap **+.1624** | Locked |
| DDXPlus native value | HS24, 6 tasks / 32 values | .7700 | **.7659** | shuffled .5791; gap **+.1868** | Locked |

DDXPlus locked finding-gap 95% CI는 `[.1576,.1672]`, value-gap CI는
`[.1650,.2091]`이다. Finding coverage는 `.9996`, value coverage는 `.7161`이다.
PDD-heldout canonical-PDD cell은 train output ontology에 없는 PDD가 있으므로 `N/A`이며 0으로
기록하지 않는다.

이 표의 읽는 법은 다음과 같다.

- DiReCT `.5962/.4423`은 52-case validation에서 HS24를 선택할 때 얻은 top-1 accuracy이며
  main table의 validation 열에는 들어간다. 아직 72-case test-seen 결과가 아니므로 locked 열은
  비워 둔다.
- DDXPlus `.9607/.7700`은 validation에서 layer/threshold를 고정한 수치다.
- DDXPlus `.9562/.7659`는 동결한 HS24 probe를 4,543 original locked-test cases에 한 번 적용한
  주 결과다.
- `.7938/.5791`은 같은 diagnosis donor control이고, `+.1624/+.1868`은 matched case가 donor보다
  나은 paired gap이다. 단순 majority 또는 diagnosis template으로 높은 점수가 나온 부분을
  통제한다.
- Coverage `.9996/.7161`은 전체 test 중 해당 metric이 정의되는 비율이다. Value accuracy는
  모든 4,543명에 대한 accuracy가 아니다.

### 3.2 DiReCT probe 학습

Input은 각 case의 CoT-P0 activation 한 개다. 진단마다 별도 probe를 만드는 것이 아니라 target
ontology 전체를 예측하는 **하나의 multiclass linear softmax head**를 학습한다.

| Hyperparameter | 값 |
|---|---|
| train / validation | 266 / 52 |
| layers | HS16, HS24, HS32 |
| feature normalization | train mean/std로 standardization |
| optimizer | AdamW, full batch |
| learning rates | `3e-4`, `1e-3` |
| weight decay | `0`, `1e-4`, `1e-3`, `1e-2` |
| class-balanced loss | off/on |
| maximum epochs / patience | 300 / 30 |
| seed | 17 |
| selection | lowest validation NLL; tie: higher acc1, then lower WD |

Layer sensitivity:

| Target | HS16 | HS24 | HS32 | Majority |
|---|---:|---:|---:|---:|
| disease category, 25-way | .5000 | **.5962** | .5192 | .0577 |
| canonical PDD, 49-way | .3846 | **.4423** | .3846 | .0962 |

Accuracy는 보고 metric이고 hyperparameter 선택은 NLL로 했다. 따라서 우연히 validation accuracy가
가장 높은 epoch를 직접 고른 것이 아니다.

Probe는 `h -> Wh+b` 하나뿐인 linear readout이다. Category probe 하나가 25 logits, PDD probe
하나가 49 logits를 동시에 낸다. “PDD마다 probe 하나”를 49개 학습한 것이 아니다. 이 절은
P0에 진단 관련 정보가 선형적으로 decode 가능한지 확인하는 representation audit이며,
Medical-NLA 자연어 생성 성능이나 causal faithfulness를 직접 측정하지 않는다.

### 3.3 DDXPlus finding/value probe 학습

Finding head는 91개 evidence ID에 대한 multi-label linear classifier다. Value head는 value가
여러 개인 evidence ID 6개에 대해 evidence-conditioned class partition을 가진 linear classifier다.

| Hyperparameter | Finding | Value |
|---|---|---|
| train / validation originals | 4,655 / 4,525 | 동일 |
| train support criterion | evidence count >=20 | value count >=10 |
| layers | HS16/24/32 | HS16/24/32 |
| normalization | train mean/std | 동일 |
| loss | BCEWithLogits | evidence-conditioned CE |
| optimizer | AdamW | AdamW |
| learning rates | `1e-3`, `3e-3` | 동일 |
| weight decay | `0`, `1e-3` | 동일 |
| positive weighting | off/on; neg/pos clipped 1..20 | N/A |
| threshold grid | `.1,.2,.3,.4,.5` | argmax |
| max epochs / patience | 80 / 8 | 80 / 8 |
| batch size | 512 | full batch |
| seed | 17 | 17 |

Finding hyperparameter는 validation BCE, threshold는 micro F1, macro F1, `.5`와의 거리 순으로
선택한다. Value는 validation NLL로 선택한다. Layer는 finding own-minus-shuffled gap, value gap,
낮은 layer 순으로 고정했고 HS24가 선택되었다.

Validation layer 결과:

| Layer | Finding micro F1 | Finding macro F1 | Value accuracy |
|---:|---:|---:|---:|
| HS16 | .9636 | .9097 | .7641 |
| HS24 | .9607 | .9049 | **.7700** |
| HS32 | .9607 | .9134 | .6990 |

### 3.4 Metric 의미

- Finding micro F1: 모든 case-label의 TP/FP/FN을 합쳐
  `2TP / (2TP + FP + FN)`으로 계산
- Native-value accuracy: train-supported multi-value evidence가 실제 존재하는 eligible target에서
  predicted native value가 gold와 같은 비율
- Hard-shuffled score: 같은 diagnosis의 다른 case에 own prediction을 대입해 계산
- Own-shuffled gap: matched case score minus hard-shuffled score
- Bootstrap: paired `base_id` 단위, 2,000 repetitions, seed 17

높은 probe score는 activation에 해당 정보를 linearly decodable하게 표현했다는 뜻이다. Open
generator가 그것을 충실한 자연어로 읽는다는 뜻은 아니다.

---

## 4. Table 1C: Open vanilla NLA boundary

### 4.1 모델과 prompt

- model: `kitft/nla-gemma3-12b-L32-av`
- input activation: primarily CoT-P0/HS32, dimension 3,840
- dtype/placement: bfloat16, `device_map=auto`, two GPUs, GPU당 `22GiB`
- decoding: greedy, `do_sample=false`
- E2 validation generation: max new tokens 256, batch size 4
- seed: 17

Task-aligned suffix는 다음과 같다.

```text
For this run, focus specifically on the medical diagnosis, clinical finding,
mechanism, anatomy, physiology, pathology, medication, or treatment-related
content represented by the activation. Prefer concrete biomedical content over
descriptions of the question format, answer style, or task structure. If the
activation appears to encode only format or task structure rather than medical
content, state that directly.
```

### 4.2 Validation 결과

DiReCT `val_seen` 52행에서 default와 task-aligned prompt, HS16/24/32의 6 arms, 총 312
readouts를 만들었다. Primary HS32 결과는 다음과 같다.

| Prompt | Source answer semantic mention | Gold PDD | Disease category |
|---|---:|---:|---:|
| default HS32 | 0/52 | 0/52 | 0/52 |
| task-aligned HS32 | 0/52 | 0/52 | 0/52 |

이는 diagnosis semantic readout의 boundary다. Physician observation alignment까지 0이라는
뜻은 아니다. HS16의 category 1/52는 appendix sensitivity로만 남긴다.

Semantic audit은 readout에서 target을 지지하는 **exact quote**를 요구했다. Judge가 yes라고
해도 quote가 원문 readout에 없으면 불일치 처리한다.

여기서 312는 `52 cases x 2 prompts x 3 layers`다. Default prompt는 공개 AV checkpoint의
sidecar prompt이고 task-aligned arm은 같은 prompt에 medical-content suffix만 추가했다. 각
readout에 대해 source answer, gold PDD, disease category가 의미상 명시돼 있는지를 검사했다.
HS32 두 prompt에서 세 target 모두 0/52였다는 뜻은 vanilla AV가 **진단명을 명시적으로
복원하지 못했다**는 뜻이다. Clinical finding이 전혀 없었다거나 activation에 진단 정보가 없다는
뜻은 아니다. 전자는 Table 2/3, 후자는 closed probe로 따로 측정한다. 이 결과는 main table의
짧은 open-readout boundary 또는 Results 본문에 들어가며, locked-test 성능으로 부르지 않는다.

### 4.3 현재 재현성 결손

Vanilla의 exact default actor prompt는 model sidecar `nla_meta.yaml`에서 runtime에 읽으며 현재
git에 고정되어 있지 않다. 논문 제출 전 실제 checkpoint에서
`python -m src.run_nla ... --dump-actor-prompt-template`로 prompt를 덤프하고 SHA-256과 함께
artifact에 저장해야 한다. 위 task-aligned 문자열은 git의
`prompts/medical_actor_prompt_suffix.txt`에 고정돼 있다. Default prompt를 추정해 appendix에
쓰면 안 된다.

---

## 5. Table 2: DiReCT clinical explanation alignment

### 5.1 현재 validation 수치

현재 frozen test 72/106은 아직 계산하지 않았다. 아래는 generative method selection에 사용한
gold-label-absent 50행 validation 결과다.

대상은 DiReCT `val_seen` 52 notes 중 note에 canonical gold-label phrase가 exact하게 노출된 2행을
제외한 **동일한 50 notes**다. DDXPlus validation이나 과거 171-case test가 아니다. Source CoT는
Gemma backbone의 해당 50개 response, Vanilla/SFT는 같은 50개 CoT-P0/HS32 activation의
readout이다. Mixed pilot 모델은 DiReCT 248 + DDXPlus 248로, full-data 모델은 DiReCT 248 +
DDXPlus 4,655로 학습했다.

두 Source CoT 행의 수치가 다른 것은 모집단이 달라서가 아니라 common-pilot 평가와 full-data
평가에서 quote extractor를 별도로 다시 실행했기 때문이다. Official semantic evaluator는
고정돼 있지만 앞단 Codex extraction은 model/version이 명시되지 않은 별도 materialization이라
결과가 조금 달라졌다. 따라서 두 block을 평균내지 않고, final baseline batch에서는 extractor
model/version, prompt hash, request hash를 고정해 한 번만 다시 계산한다.

| Method | parsed observed | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall | 상태 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Source CoT, mixed-pilot population | 50/50 | 0 | .3110 | .4069 | .2399 | .0657 | .0168 | Validation |
| Source CoT, full-data run population | 50/50 | 0 | .2835 | .3726 | .2130 | .0650 | .0153 | Validation floor |
| Vanilla NLA | 10/50 | 0 | 0 | 0 | 0 | 0 | 0 | Validation |
| Common SFT seed17, mixed pilot | 50/50 | 0 | .0100 | .0037 | .0034 | 0 | 0 | Validation |
| Common SFT seed29, mixed pilot | 50/50 | 0 | 0 | 0 | 0 | 0 | 0 | Validation |
| Common SFT seed43, mixed pilot | 4/50 Direct | 0 | .0070 | .0054 | .0043 | 0 | 0 | Validation |
| Full-data common SFT seed17 | 50/50 | 0 | .0544 | .0502 | .0301 | 0 | 0 | Validation |
| Full-data common SFT seed29 | 50/50 | 0 | .0553 | .0388 | .0296 | 0 | 0 | Validation |

Full-data Source CoT의 `Obscomp=.2130`, `Expcom=.0650`이 frozen promotion floor다. Final
Medical-NLA는 validation에서 두 값을 **모두 초과**해야 test로 승격한다. 현재 생성형 방법은
통과하지 못했다.

### 5.2 Final table 구조와 남은 셀

Final Table 2는 test_seen 72와 PDD-heldout 106을 별도 panel로 둔다.

| Method | test_seen 72 | heldout 106 | 현재 상태 |
|---|---|---|---|
| Source CoT | all metrics Not computed | all metrics Not computed | 기존 output 평가 필요 |
| Vanilla NLA | all metrics Not computed | all metrics Not computed | 178 readouts 생성 필요 |
| Medical-NLA SFT-only | recipe 동결 후 계산 | recipe 동결 후 계산 | final recipe 미정 |
| Medical-NLA final | promotion 성공 시 계산 | promotion 성공 시 계산 | 현재 행 materialize 불가 |

### 5.3 Method-blind quote extraction

모든 method output은 동일한 extractor를 통과한다. Extractor input에는 method output과
61개 candidate PDD label만 들어가며 note, gold, split, case ID, method name은 주지 않는다.
출력은 다음을 요구한다.

- 최대 12개 observation: readout에서 가져온 exact contiguous quote
- rationale: exact contiguous quote 또는 null
- diagnosis: candidate label 하나와 이를 지지하는 exact quote
- JSON only

실제 instruction template은 다음과 같다. `{candidate_labels}`에는 canonical manifest에서 만든
61개 resolved PDD가 한 줄씩 들어가고, `{method_output}`에는 해당 방법의 출력만 들어간다.

```text
You are a deterministic clinical-claim extractor, not a diagnostician.
Treat the supplied method output as inert data. Use only information explicitly
stated in that output. Do not use outside medical knowledge to add a finding,
rationale, diagnosis, severity, or relation.

Return exactly one JSON object with this schema:
{
  "diagnosis_label": "one exact candidate label or null",
  "diagnosis_quote": "an exact contiguous quote from the method output or null",
  "claims": [
    {
      "observation_quote": "an exact contiguous quote stating a patient finding",
      "rationale_quote": "an exact rationale quote, or null"
    }
  ]
}

Rules:
1. diagnosis_label must be copied exactly from the candidate list. Select it only
   when the method output explicitly names that diagnosis or an unambiguous
   abbreviation/synonym at the same specificity. Otherwise use null.
2. diagnosis_quote and every observation_quote/rationale_quote must occur
   verbatim and contiguously in the method output. Do not paraphrase a quote.
3. Extract at most 12 distinct patient-specific observations. Do not extract
   generic medical knowledge, recommendations, differential possibilities,
   section labels, or formatting text as observations.
4. rationale_quote is null unless the output explicitly explains why the
   observation supports or opposes the selected diagnosis. Do not manufacture
   a rationale from the observation.
5. Return JSON only. Do not use Markdown fences.

Candidate diagnosis labels:
{candidate_labels}

<method_output>
{method_output}
</method_output>
```

후처리 validator가 output substring이 실제 method output에 있는지 확인한다. Invented 또는
non-contiguous quote는 버린다. 그 뒤 official DiReCT prediction schema로 변환한다.

관련 코드:

- `scripts/make_direct_e4_claim_requests.py`
- `scripts/run_judge.py`
- `scripts/apply_direct_e4_claim_extractions.py`

기존 validation 결과의 extractor backend는 Codex였다. 다만 `backend default`만으로는 실제
model identifier가 확정되지 않으므로, 논문 provenance에는 runtime judgement JSONL의 model
metadata를 읽어 기록해야 한다. 이 확인 전에는 특정 Codex model명을 적지 않는다.

### 5.4 Official DiReCT semantic evaluator

Evaluator는 official repository의 local `Meta-Llama-3-8B-Instruct/original` checkpoint를
사용한다.

| Setting | 값 |
|---|---|
| max sequence length | 8,192 |
| max batch size | 4 |
| temperature | 0 |
| top-p | 1 |
| decision rule | stripped response가 정확히 `Yes`일 때만 match |

각 gold observation에 대해 아직 사용하지 않은 predicted observation을 순서대로 judge하고,
첫 semantic match를 greedy pairing한다. Rationale는 paired observation의 diagnosis가 exact
match하고 rationale judge가 `Yes`인 경우만 correct다. Missing/invalid output은 분모에서
제거하지 않고 0점이다.

Observation/rationale prompt는 official repository의
`discriminate_similarity_observation()`과 `discriminate_similarity_reason()`을 수정 없이
사용했다. 이 두 함수의 exact text는 현재 public workspace에 복제돼 있지 않으므로 제출용
artifact에는 official source file과 SHA-256을 보존해야 한다. Wrapper 설명으로 prompt를
재작성해 대체하지 않는다.

### 5.5 Metric 공식과 의미

한 case에서 다음을 정의한다.

- `G`: gold observations 수
- `P`: predicted observations 수
- `m`: semantic paired observations 수
- `U = G + P - m`: union 크기
- `R`: paired observation 중 diagnosis exact match와 rationale semantic match를 모두 만족한 수

공식 metric은 다음과 같다.

```text
Obspre  = m / (P + 1)
Obsrec  = m / (G + 1)
Obscomp = m / U
Expcom  = R / m
Expall  = R / U
```

`Acccat`은 predicted chain의 category node와 gold category, `Accdiag`는 predicted chain의
마지막 diagnosis와 gold final diagnosis의 exact normalized equality다. 현재 converter의
category node는 extractor가 고른 canonical PDD에서 deterministic ontology mapping된 값이므로,
독립적으로 생성한 category reasoning score로 해석하지 않는다.

`Obspre`와 `Obsrec`에는 official `+1` smoothing이 있다. 따라서 완벽한 oracle도 두 값은
1보다 작을 수 있다. `Obscomp`는 gold와 prediction observation set의 semantic Jaccard에
가깝다. `Expcom`은 matched observation 안에서 explanation chain까지 맞은 비율이고,
`Expall`은 전체 union을 분모로 하는 가장 엄격한 score다.

Official oracle smoke 10행에서 Acccat=1, Accdiag=1, Obspre=Obsrec=.8104,
Obscomp=Expcom=Expall=1을 확인해 evaluator plumbing을 검증했다.

### 5.6 Common SFT 학습 recipe

Actor prompt는 `prompt_templates/common_p0_clinical_state_readout.txt`에 고정돼 있다.

```text
<concept>{injection_char}</concept>

The vector above was extracted after a clinical presentation and a reasoning
instruction, immediately before the source model began its response. Read the
patient-specific clinical state represented by the vector.

Report only concrete clinical findings represented by the vector. Do not infer
a diagnosis, add background medical knowledge, or complete a typical disease
template. If no specific clinical finding is represented, say so in one bullet.

Return only this schema, with nothing before or after it:

<explanation>
<readout>
<observed>
- patient-specific clinical finding
</observed>
</readout>
</explanation>
```

Model/training:

| Setting | 값 |
|---|---|
| base AV | `kitft/nla-gemma3-12b-L32-av` |
| activation | CoT-P0/HS32 |
| train population | DDXPlus 4,655 originals + DiReCT 248 |
| validation | DDXPlus 50 + DiReCT 50 |
| epochs | 1; 개선 시에만 후속 seed/epoch를 열도록 사전 지정 |
| source sampling exponent | alpha=.5 |
| batch / gradient accumulation | 4 / 2, effective 8 rows/update |
| learning rate / weight decay | `2e-4` / `0` |
| optimizer | AdamW |
| gradient clipping | 1.0 |
| gradient checkpointing | on |
| LoRA rank / alpha / dropout | 16 / 32 / .05 |
| LoRA modules | q/k/v/o, gate/up/down projections |
| precision | bf16 frozen base, fp32 trainable LoRA |
| selection | source-macro validation content NLL |
| generation | greedy, max new tokens 512, batch 4 |

Alpha .5 source sampling으로 DDXPlus는 epoch당 거의 한 번씩, 작은 DiReCT는 약 1,074 exposures,
즉 case당 약 4.3회 노출된다. `<observed>` content token NLL과 XML scaffold token NLL을 따로
집계하고, 두 source의 content NLL 평균으로 checkpoint를 선택했다.

Target construction은 다음과 같다.

- 공통 output은 diagnosis와 `<answer>`가 없는 `<observed>` bullet schema다.
- DDXPlus target은 original presentation의 train-supported rendered cue를 중복 제거하고 입력
  순서를 유지하며 최대 12개 사용한다.
- DiReCT target은 physician gold deductions 중 `observation_exact_in_note=true`인 observation만
  중복 제거해 최대 12개 사용한다. Upstream DiReCT E3 builder가 `Random(seed:id)`로 순서를
  결정한 뒤 `cue_targets`에 저장했고, common builder는 그 저장 순서를 유지한다. 따라서
  DDXPlus에는 실제 presentation order가 보존되지만 DiReCT 순서는 실제 note order가 아니라
  seed 17로 동결된 deterministic order다.
- DiReCT의 초기 E3 target에는 backbone source answer가 `<answer>`로 있었지만 common SFT
  builder는 이를 제거했다. Common model에는 physician gold diagnosis도 backbone diagnosis도
  target으로 주지 않는다.
- XML scaffold가 content loss를 지배하지 않도록 bullet text token과 scaffold token을 offset
  mapping으로 분리한다.

관련 builder는 `scripts/make_direct_e3_sft_dataset.py`,
`scripts/make_common_medical_nla_sft_dataset.py`,
`scripts/make_medical_nla_v3_cue_first_targets.py`다.

---

## 6. Table 3A: DDXPlus static grounding

### 6.1 Locked-test 수치

| Method class | Method | Parse | Finding F1 | Shuffled F1 | Gap | Native value |
|---|---|---:|---:|---:|---:|---:|
| closed decoder | Frozen HS24 probe | 1.0000 | **.9562** | .7938 | **+.1624** `[.1576,.1672]` | **.7659** |
| structured monitor | Probe-guided reader | 1.0000 | **.9587** | .7938 | **+.1624** | **.7654** |
| open generator | Vanilla NLA | Not computed | Not computed | Not computed | Not computed | Not computed |
| open generator | Medical-NLA SFT-only | recipe 미정 | Not computed | Not computed | Not computed | Not computed |
| open generator | Medical-NLA final | promotion 미통과 | Not computed | Not computed | Not computed | Not computed |

Structured reader의 mean emitted claims는 validation 4.9485, locked test 4.9353이다. Probe와
reader의 finding 값이 약간 다른 이유는 probe report와 rendered output scorer의 end-to-end
normalization 차이 때문이다. Reader의 finding selection 자체는 frozen probe threshold와
수학적으로 동일하다. 표의 `+.1624`는 전체 `.9587-.7938`을 단순히 뺀 값이 아니라
same-diagnosis donor가 존재하는 4,121 paired cases에서 계산한 paired gap이다.

### 6.2 Probe-guided structured reader

이 방법은 open-ended NLA가 아니라 deterministic structured monitor다.

1. CoT-P0/HS24 activation을 frozen finding/value probe에 입력한다.
2. Validation에서 동결한 threshold 이상 finding을 모두 선택한다.
3. Probability 내림차순, evidence ID 오름차순으로 정렬한다.
4. Official train에서만 만든 evidence별 modal exact phrase를 사용해 finding을 rendering한다.
5. `<observed>` bullet list를 출력한다.

Prompt text, gold diagnosis, test label은 prediction 구성에 사용하지 않았다. Locked test 전에
probe protocol hash를 검증했다. 이 baseline은 “activation의 linearly selected clinical state를
free-generating decoder 없이 자연어 label로 rendering할 수 있는가”를 측정한다.

관련 코드: `scripts/run_ddxplus_structured_reader.py`.

---

## 7. Table 3B: DDXPlus counterfactual response

### 7.1 Locked structured-reader 결과

| Method | Original hit | Deletion phantom | Removal success | Retained preservation | Replacement hit | Old persistence | Clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Probe-guided reader | 1.0000 | **.3593** | **.6407** | **.9987** | **.1466** | **.5955** | **.0804** |
| Vanilla NLA | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed |
| Medical-NLA SFT-only | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed |
| Medical-NLA final | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed | Not computed |

분모는 deletion 4,540, value edit 539, clean switch 398이다. Validation reader 수치는 phantom
.3626, removal .6374, retention .9985, replacement .1407, old persistence .5722, clean switch
.1038이었다.

Frozen probe 자체의 locked-test deletion target probability drop은 `+.6103`, removal success는
`.6407`이었다. Value replacement `.1466`, old persistence `.5955`, clean switch `.0804`였다.

### 7.2 Metric 정의

- **Original hit**: 삭제 대상 cue가 original activation에서 threshold 이상으로 선택된 비율
- **Deletion phantom**: 그 cue를 input에서 삭제한 activation에서도 계속 선택된 비율
- **Removal success**: original hit인 case 중 deleted activation에서 threshold 아래로 내려간 비율
- **Probability drop**: `p_original(target cue) - p_deleted(target cue)`의 평균
- **Retained preservation**: intervention과 무관한 공통 cues 중 original과 derived arm 모두에서
  hit인 비율, original hit에 조건부
- **Replacement hit**: value edit 후 새 value를 예측한 비율
- **Old persistence**: value edit 후에도 이전 value를 예측한 비율
- **Clean switch**: original에서 old value가 맞았던 eligible cases 중 edit 후 new value가 맞고 old
  value는 사라진 비율

Static F1이 높아도 deletion phantom이 .3593이고 clean switch가 .0804이므로, 정적 decodability와
counterfactual responsiveness는 별개다.

---

## 8. D9a/D10: 생성형 grounding objective 개발 기록

이 절의 수치는 main result가 아니라 Appendix development gate다.

### 8.1 D9a support filtering

Changed cue를 무조건 target으로 쓰지 않고, two-fold OOF probe가 activation support를 지지할 때만
training pair로 허용했다.

- fold: `crc32(base_id) % 2`
- OOF head는 반대 fold originals에서만 학습
- minimum fold positives: 5
- donor: same fold, same diagnosis, changed cue absent, cue-count 근접, 최대 5개
- support rule: presence AND deletion delta AND donor margin
- presence grid: `.80,.90,.95,.975,.99`
- deletion/donor grids: `0,.05,.10,.20,.30,.40,.50`
- total candidate cuts: 245
- selection: null false-support <=.05 안에서 positive coverage 최대

승인 cut은 presence `.90`, deletion delta `0`, donor margin `0`이다.

| Quantity | 값 |
|---|---:|
| validation supported | 3,032 / 3,034 = **.9993** |
| validation null false-support | 112 / 2,964 = **.0378** |
| false-support 95% CI | `[.0315,.0453]` |
| official-train supported pairs | **3,104** |

Unsupported rows은 abstention target으로 바꾸지 않고 제외했다. 각 retained case는 changed cue 한
개만 target으로 갖는다. Value-edit arm은 D9a에 포함하지 않았다.

### 8.2 D10 one-claim 1x2 ranking

Original activation `h_o`, deleted activation `h_d`, changed-cue text target `y`에 대해

```text
g = NLL(y | h_d) - NLL(y | h_o)
L = SFT(y | h_o) + lambda * T * softplus(-g / T)
```

를 사용했다. Control은 `lambda=0`, ranking은 `lambda=1`, temperature `T=1`이다.

Smoke hyperparameters:

| Setting | 값 |
|---|---|
| train pairs | 3,104 |
| seeds | 17, 29, 43 |
| maximum steps | 20 |
| learning rate / weight decay | `2e-4` / 0 |
| gradient accumulation | 4 |
| LoRA rank / alpha / dropout | 16 / 32 / 0 |
| LoRA modules | q/k/v/o, gate/up/down |
| gradient checkpointing / clipping | on / 1.0 |
| optimizer | AdamW |

Validation paired arm 결과:

| Seed | Changed-gap delta | cluster 95% CI | Retained-gap delta | Specificity delta | specificity CI |
|---:|---:|---:|---:|---:|---:|
| 17 | +.0005 | [-.0006,+.0016] | +.0010 | -.0005 | [-.0020,+.0010] |
| 29 | +.0028 | [+.0017,+.0039] | -.0000 | +.0029 | [+.0015,+.0045] |
| 43 | +.0030 | [+.0015,+.0048] | -.0007 | +.0037 | [+.0017,+.0059] |

Frozen minimum effect floor `+.05`를 어느 seed도 충족하지 못했고 seed17 CI는 0을 포함했다.
따라서 D10 smoke는 FAIL이다.

### 8.3 Budget calibration

현재 마지막 남은 생성형 시험은 objective와 data를 바꾸지 않고 학습량만 늘리는 dose-response다.

- control/ranking x seeds 17/29/43 = 6 runs
- final step: 1,552, 즉 약 2 epochs
- checkpoints: 20, 194, 388, 776, 1,164, 1,552
- intermediate checkpoint는 trajectory 설명용이며 선택에 사용하지 않음
- final step만 기존 D5 gate로 판정
- 1,552 이후 자동 연장 없음

이 실험은 **pending**이다. 결과가 나오기 전 Table 2/3의 Medical-NLA final 행을 채우지 않는다.

### 8.4 중요한 provenance 확인

D10 fresh actor prompt default는 `prompt_templates/cue_position_readout.txt`일 수 있으나, 실제 run이
warm-start adapter를 사용했는지에 따라 달라질 수 있다. 논문에는 각 run의 `best.json`, 저장된
`av_prompt_template.txt`, launch log를 읽어 initialization checkpoint, prompt hash, GPU model,
CUDA/PyTorch/Transformers version을 확정한 뒤 기록한다. Wrapper default만 보고 실제 run
configuration을 단정하지 않는다.

---

## 9. Table 3C / Table 4: AR round-trip과 intervention

현재 이 표들은 **Not computed**다. 숫자를 넣을 근거가 없다.

예정 model은 `kitft/nla-gemma3-12b-L32-ar`, primary layer는 HS32다. 실행 순서는 다음과 같아야
한다.

1. Original activation을 AV text로 읽고 AR로 재구성하는 identity test
2. Matched text와 same-diagnosis shuffled text의 reconstruction FVE/cosine 비교
3. Identity gate 통과 후에만 text edit patching
4. Target logit delta, off-target KL, edited-value decoding
5. 최종 diagnosis behavior의 wrong-to-right/right-to-wrong/net correction

예정 metric:

```text
FVE = 1 - ||h - h_hat||^2 / ||h - mean(h)||^2
cosine = cosine_similarity(h, h_hat)
```

Table 4는 AR identity와 matched-over-shuffled gate를 통과할 때만 본문에 연다. Exact AR prompt,
normalization, intervention site, patch coefficient, behavioral decision rule은 아직 완전히 동결되지
않았으므로 현재 문서에서 임의의 hyperparameter를 제시하지 않는다.

---

## 10. Appendix 음성 결과

| Experiment | 핵심 수치 | 판정 |
|---|---|---|
| D14 K=5 hard-set teacher | precision .8881 | required .90 미달, FAIL |
| D16 256-d soft bottleneck | PCA cosine .999997/.999983 | compression 자체는 보존 |
| D16 Direct auxiliary delta | -.001137 / -.001476 / +.001433 | 3-seed sign 불일치, FAIL |
| D16 frozen-z finding F1 delta | -.0009 / -.0007 / -.0016 | 일관된 소폭 하락 |
| D16 frozen-z finding-gap delta | -.0050 / -.0046 / -.0058 | 일관된 하락 |
| D16 frozen-z value-accuracy delta | -.0137 / -.0096 / -.0160 | 일관된 하락 |

이 결과들은 “모든 Medical-NLA가 불가능하다”는 증거가 아니라, 해당 hard-set/soft-auxiliary
objective가 frozen gate를 통과하지 못했다는 증거다.

---

## 11. 표를 완전히 채우는 실행 순서

### Step 1. 계산 없이 바로 확정할 셀

1. Table 1B의 DDXPlus locked probe 수치
2. Table 3의 structured reader locked 수치
3. Appendix D9a, D10 smoke, D14, D16 수치

이 branch-independent 고정 셀은 canonical 문서에 이미 반영돼 있다. 다음 명령은 새 결과를
계산하거나 locked test를 읽지 않고 승인된 10개 row를 idempotent하게 materialize/검증한다.

```bash
python scripts/sync_paper_table_fixed_cells.py --write
python scripts/sync_paper_table_fixed_cells.py --check
```

`--write`는 Table 1A, DiReCT locked probe, Table 2, conditional generative row를 건드리지 않는다.
그 셀들은 아래 동결 순서를 거친 뒤 별도 batch에서만 채운다.

### Step 2. D10 최종 분기와 recipe hash 동결

1. Step 1,552 final checkpoint만 frozen D5 gate로 판정
2. 실패하면 generative final 행을 제거하는 recipe를, 성공하면 validation generation gate로
   보낼 단일 recipe를 기록
3. Input manifest, prompt, decoding, checkpoint, extractor version hash를 동결

### Step 3. Locked-label baseline batch를 한 번 실행

1. DiReCT source Direct/CoT output을 frozen 72/106 split으로 reindex
2. Table 1A parse/strict/category/semantic diagnosis와 paired McNemar 계산
3. HS24 DiReCT category/PDD probe를 test_seen 72에 적용
4. patient-group cluster CI와 exposed-label sensitivity 계산
5. 기존 Vanilla output의 prompt/decoding이 frozen recipe와 byte-level로 일치하면 겹치는 rows만
   재사용하고, 다르면 test 178건 전부 다시 생성
6. Source CoT와 Vanilla output 모두 같은 method-blind quote extractor와 official evaluator 실행
7. 72/106 panel별 metric과 cluster CI 기록

### Step 4. Medical-NLA branch

1. D10 teacher-forced gate 통과 시에만 validation generation을 실행
2. DiReCT `Obscomp>.2130`과 `Expcom>.0650`, DDXPlus generation grounding gate를 확인
3. 모두 통과하면 DDXPlus locked test와 DiReCT 72/106에 세 seed 모두 적용
4. mean +/- seed SD와 paired cluster CI 보고
5. 실패 시 final 행을 비워 두지 않고 제거하고 structured reader를 main positive result로 유지

### Step 5. AR conditional branch

Identity와 matched-over-shuffled gate를 먼저 수행한다. 통과하지 못하면 Table 3C와 Table 4를
본문에서 제거하고 appendix failure로 보고한다.

---

## 12. 제출 전 재현성 체크리스트

- [ ] Frozen split ID hash와 logical population hash를 manuscript supplement에 기록
- [ ] 모든 model repository revision 또는 local checkpoint hash 기록
- [ ] Gemma tokenizer/chat-template revision 기록
- [ ] Vanilla sidecar actor prompt dump와 SHA-256 저장
- [ ] Adapted model별 실제 `av_prompt_template.txt` hash 저장
- [ ] Codex extractor judgement artifact에서 실제 model ID와 parameters 기록
- [ ] Official Llama evaluator checkpoint hash 기록
- [ ] Python, PyTorch, Transformers, CUDA, driver version 기록
- [ ] 실제 GPU model과 visible-device mapping 기록
- [ ] 모든 random seed와 bootstrap repetitions 기록
- [ ] `best.json`, protocol JSON, input/output JSONL SHA-256 기록
- [ ] Table caption에 metric별 실제 denominator 기록
- [ ] Validation, exploratory, locked 수치를 명확히 구분
- [ ] Missing parse를 제거하지 않고 0점 처리했는지 확인
- [ ] Test 결과를 본 뒤 prompt/layer/threshold/checkpoint를 수정하지 않았는지 확인

## 현재 판정

현재 즉시 논문 본문에 확정 수치로 넣을 수 있는 핵심 positive result는 DDXPlus locked probe와
probe-guided structured reader다. DiReCT Table 1A/1B와 Table 2 baseline은 저장 artifact 재집계와
official evaluation이 남았다. 생성형 Medical-NLA final 행은 아직 promotion gate를 통과하지
않았고, AR 기반 Table 4는 실행되지 않았다. 이 경계를 유지해야 표가 빠르게 채워지면서도
validation 결과를 final result로 잘못 승격시키지 않는다.

---

## 검토 (Claude, 2026-08-29)

**[동의] 문서 전체 승인.** 특히 다음이 옳다:

- 수치 4상태 구분(Locked/Validation/Exploratory/Not computed)과 "validation을
  locked처럼 옮기지 않는다, 과거 71/100을 72/106처럼 쓰지 않는다"는 규칙.
- §2.3 말미 — Table 1A가 CPU 재집계라는 이유로 미리 열지 않고 D10 분기 후
  batch로 일괄 실행. §11 Step 3.5의 vanilla 재사용 조건(byte-level prompt/
  decoding 일치 시에만, 불일치 시 178건 전부 재생성)도 반영 확인.
- §4.3/§8.4의 재현성 결손 자백 — vanilla sidecar prompt가 git에 없다는 것,
  D10 실제 initialization은 wrapper default가 아니라 run artifact(`best.json`,
  `av_prompt_template.txt`)에서 확정해야 한다는 것, Codex extractor의 실제
  model ID는 runtime judgement metadata에서 읽어야 한다는 것. 이런 결손을
  숨기지 않고 체크리스트화한 것이 이 문서의 가치다.
- §6.1의 `+.1624`가 단순 차가 아니라 4,121 paired cases의 paired gap이라는
  명시, probe(.9562)와 reader(.9587)의 미세 차이가 scorer normalization
  차이라는 설명 — caption에 그대로 쓸 수 있는 수준.

**[보완 1] Budget run 실행 환경 provenance를 §8.4에 선기록한다.** 이번 run은
lab이 아니라 RunPod에서 실행 중이므로, §12 체크리스트의 GPU/version 항목에
해당하는 값들을 미리 적어 둔다 (완주 후 pod의 `pip list`/`best.json`으로
최종 확정):

- hardware: NVIDIA A100-SXM4-80GB x 1 (single device, sharding 없음)
- config: `configs/runpod.yaml` (`max_memory {0: 75GiB}`), 나머지
  hyperparameter는 `configs/default.yaml`과 동일
- torch `2.4.1+cu124`; transformers는 5.16.1이 torch>=2.5를 요구해 torch
  backend를 비활성화하는 문제로 `>=4.50,<5`로 동결 (`pyproject.toml`,
  커밋 `c4288ad`)
- 6 runs 전부 같은 pod에서 실행, wrapper가 `nvidia-smi` 출력을 로그에 기록

**[보완 2] Canonical 표 문서와의 동기화 항목 하나.**
`docs/paper/tables_and_figures.md`(커밋 `8800e4d`)의 Table 3 Panel B에는
`Original hit` 열이 없는데 이 문서 §7.1에는 있다(reader 1.0000). 최종 원고
반영 시 이 문서 §7.1의 열 구성(Original hit 포함)을 기준으로 맞춘다 —
Original hit 1.0000은 "removal success의 분모가 전 사례"임을 보여주는
조건부 해석 방지 장치라 두는 쪽이 맞다.

이 두 보완 외에는 수정 없이 제출 준비 문서로 사용 가능하다.
