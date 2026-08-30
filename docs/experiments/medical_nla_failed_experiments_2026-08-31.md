# Medical-NLA 실패 실험 설정과 판정 기록

> 기준일: 2026-08-31  
> 범위: Medical-NLA 생성 모델을 만들기 위해 실제로 실행했으나 validation promotion gate를 통과하지 못한 실험  
> 목적: 실패 수치뿐 아니라 데이터, activation, prompt, objective, optimizer, seed, 평가 분모와 중단 이유를 재현 가능한 형태로 보존한다.

## 1. 이 문서에서 말하는 실패

여기서 `FAIL`은 세 종류를 구분한다.

1. **모델 실패**: 동결한 validation gate를 통과하지 못해 locked test 또는 후속 generation으로 승격하지 않은 경우
2. **target/teacher 실패**: student를 학습하기 전 target 생성기의 calibration gate가 실패한 경우
3. **측정기 실패**: 양성 대조도 구분하지 못해 해당 metric 또는 checkpoint를 reward/evaluator로 인정하지 않은 경우

따라서 D22의 공개 AR 실패는 “임상 정보가 activation에 없다”는 결론이 아니다. 공개 AR cosine이 이 의료 분포에서 사례별 차이를 측정하지 못했다는 판정이다. 반대로 probe와 structured reader의 성공은 생성형 Medical-NLA의 성공으로 세지 않는다.

## 2. 공통 실험 기반

### 2.1 모델과 activation

| component | frozen setting |
|---|---|
| source backbone | `google/gemma-3-12b-it` |
| released AV | `kitft/nla-gemma3-12b-L32-av` |
| released AR diagnostic | `kitft/nla-gemma3-12b-L32-ar` |
| source position | CoT-P0: clinical presentation과 reasoning instruction을 읽고 첫 응답을 생성하기 직전 |
| NLA input layer | HS32 |
| pooling/position | last token |
| hidden dimension | 3,840 |
| dtype | bfloat16 base, float32 trainable adapter parameters |
| decoding | 별도 명시가 없으면 greedy |

Closed probe에서는 validation 선택으로 HS24를 사용했지만, 공개 AV/AR가 layer 32용이므로 모든 생성형 NLA 학습과 D22 round-trip은 HS32를 사용했다. HS24 probe 수치와 HS32 AV 학습 결과를 같은 모델 성능으로 합치지 않는다.

### 2.2 공통 LoRA와 optimizer

일반 SFT는 [`scripts/train_medical_nla_lora.py`](../../scripts/train_medical_nla_lora.py)를 사용했다.

| item | setting |
|---|---:|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| optimizer | AdamW |
| learning rate | `2e-4` |
| weight decay | `0` |
| gradient clipping | `1.0` |
| gradient checkpointing | on |

D10/D20 paired ranking은 같은 rank와 target module을 사용하지만 paired NLL 차이가 dropout noise보다 작았으므로 LoRA dropout을 `0`으로 고정하고 `model.eval()` 상태에서 gradient를 계산했다. Sentence contrastive도 warm-start adapter의 dropout을 `model.eval()`로 비활성화했다.

### 2.3 출력 prompt 두 종류

DiReCT-only SFT는 [`prompt_templates/direct_p0_evidence_readout.txt`](../../prompt_templates/direct_p0_evidence_readout.txt)를 사용했다.

```xml
<explanation>
<readout>
<observed>
- patient-specific clinical finding
</observed>
<answer>source model diagnosis</answer>
</readout>
</explanation>
```

Common/full-data/counterfactual SFT는 진단 supervision을 제거한 [`prompt_templates/common_p0_clinical_state_readout.txt`](../../prompt_templates/common_p0_clinical_state_readout.txt)를 사용했다.

```xml
<explanation>
<readout>
<observed>
- patient-specific clinical finding
</observed>
</readout>
</explanation>
```

두 번째 prompt는 “진단을 추론하거나 전형적 질환 template을 완성하지 말고 activation에 표현된 concrete finding만 출력하라”고 명시했다. 따라서 이후의 template collapse를 단순히 “prompt가 임상 finding을 요구하지 않았다”로 설명할 수 없다.

### 2.4 공통 데이터 경계

- DiReCT restricted test `test_seen=72`, `test_pdd_heldout=106`은 실패한 학습법의 선택에 사용하지 않았다.
- DDXPlus locked test는 closed probe와 frozen structured-reader 확인에는 사용했지만, 아래 생성형 학습법의 checkpoint나 threshold 선택에는 사용하지 않았다.
- 학습법 비교는 validation에서 먼저 종료했다.
- DiReCT 원문과 파생 activation은 lab server restricted 경로에만 두었고 RunPod에는 DDXPlus 파생물만 전송했다.

### 2.5 실행 하드웨어와 병렬화

| experiment family | execution |
|---|---|
| DiReCT-only SFT | lab server 62/125, 한 seed의 12B AV를 GPU 2장에 분산 |
| Common mixed SFT | lab server 62/125에서 한 seed당 2-GPU; seed17/43과 seed29를 서버 간 분리 실행 |
| Full-data/counterfactual SFT | server 125의 4x4090에서 `0,1`과 `2,3`에 서로 다른 seed를 독립 실행 |
| Sentence contrastive/D10 smoke | server 125의 4x4090에서 두 2-GPU worker 병렬, 남은 seed 순차 실행 |
| D16 | server 125의 4x4090, control을 먼저 끝내 floor를 고정한 뒤 auxiliary 실행 |
| D10 budget/D20 | RunPod A100-SXM4-80GB, 12B AV를 단일 GPU에 배치 |
| D22 public AR diagnostic | server 125의 단일 GPU에서 released AR forward |

`4x4090` 표기는 한 모델을 네 장에 분산했다는 뜻이 아니다. 12B AV 한 run은 두 장을 사용했고, 두 독립 seed를 동시에 실행해 네 장을 사용했다. A100 80GB 실행은 같은 모델을 한 장에 올려 multi-GPU 통신 차이를 제거했다.

## 3. 실패 실험 전체 요약

| experiment | 변경한 것 | 주요 모집단 | seeds/steps | primary result | 판정 |
|---|---|---|---|---|---|
| DiReCT-only SFT | physician observation + source answer CE | 248 train / 50 val | 17/29/43, 3 epochs | Obscomp `.0343/.0047/.0032` | FAIL |
| Common mixed SFT | 공통 diagnosis-free schema | 248+248 train, 50+50 val | 17/29/43, 3 epochs | DiReCT Obscomp `.0034/0/.0043` | FAIL |
| Full-data SFT | DDXPlus 전체 + source-order target | 4,655+248 train | 17/29, 1 epoch | Obscomp `.0301/.0296` vs CoT `.2130` | FAIL |
| Counterfactual sequence SFT | original/deletion/value-edit 전체 문장 CE | 4,655 DDXPlus families | 17/29, 1 epoch | seed17 contrast `.2092`, phantom `.4253`; seed29 미재현 | FAIL |
| Sentence contrastive | matched NLL < crossed NLL | source-balanced pairs | seed29 warm start, 20 steps | gap `.0013/.0022`, baseline `.0051` 미달 | FAIL |
| D10 1x2 ranking | changed claim original < deleted NLL | 3,104 train pairs / 3,032 val pairs | 17/29/43, 20 steps | changed delta `.0005/.0028/.0030` | FAIL |
| D14 K=5 OOF teacher | probe probability를 hard set target으로 변환 | 4,655 bases, 9,310 arms | 5-fold OOF | precision `.8881 < .90`; deleted count gap `18.10%` | FAIL before student |
| D16 soft bottleneck | `3840→256→3840` + train-only auxiliary head | Direct 248 + D9a 3,104 | 17/29/43, 20 steps | delta `-.001137/-.001476/+.001433` | FAIL |
| D10 budget calibration | D10의 step만 20→1,552 | 동일 3,104/3,032 pairs | 17/29/43, 2 epochs | changed `+.5558`, retained `+.5604`, specificity `-.0046` | FAIL, shortcut |
| D20 specificity anchor | D10 + retained cue CE | 동일 pairs | 17/29/43, 1,552 steps | specificity `-.0278/-.0255/-.0217` | FAIL |
| D22 public AR diagnostic | text→activation matched/shuffled cosine | 8 arms × 20 | no training | 양성 대조 gap approximately `0` | FAIL as instrument |

## 4. DiReCT-only SFT

### 4.1 질문

Physician annotation으로 AV를 직접 SFT하면 P0 activation에서 환자별 observation과 source decision을 자연어로 복원할 수 있는가?

### 4.2 데이터와 target

- builder: [`scripts/make_direct_e3_sft_dataset.py`](../../scripts/make_direct_e3_sft_dataset.py)
- runner: [`scripts/run_direct_e3_sft.sh`](../../scripts/run_direct_e3_sft.sh)
- train: 266 후보 중 gold label이 note에 직접 노출된 18행을 제외한 **248행**
- validation: 52 후보 중 같은 이유로 2행을 제외한 **50행**
- maximum observations: 12
- `<observed>`: `observation_exact_in_note=true`인 physician observation
- `<answer>`: physician gold가 아니라 동일 CoT-P0 trajectory에서 backbone이 실제 생성한 source answer
- gold PDD: 분석 metadata로만 유지

Source-wrong 행도 gold-correction target으로 바꾸지 않은 이유는 activation의 실제 source state와 gold diagnosis를 충돌시키지 않기 위해서다.

### 4.3 학습 설정

| item | value |
|---|---:|
| seeds | 17, 29, 43 |
| epochs | 3 |
| batch / grad accumulation | 1 / 8 |
| validation selection | `<observed>` content-token loss |
| max validation rows | 52, 실제 eligible 50 |
| prompt | `direct_p0_evidence_readout.txt` |

### 4.4 결과

동일 50-case validation에서 quote-constrained extraction과 official-compatible semantic evaluator를 사용했다.

| seed | extraction | Obspre | Obsrec | Obscomp | Expcom | Expall | exact duplicate rows / unique outputs |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 17 | 50/50 | .0771 | .0435 | .0343 | 0 | 0 | 43/50 / 7 |
| 29 | 50/50 | .0133 | .0047 | .0047 | 0 | 0 | 47/50 / 3 |
| 43 | 50/50 | .0200 | .0029 | .0032 | 0 | 0 | 49/50 / 1 |
| Source CoT | 50/50 | .3009 | .3903 | .2349 | .0573 | .0144 | 0/50 / 50 |

### 4.5 실패 판정

Schema와 patient-finding 형태의 문장은 생성했지만 observation alignment가 Source CoT에 크게 미달했고 seed가 바뀔수록 동일 문장 반복이 심해졌다. Rationale target이 없으므로 `Expcom/Expall=0`은 구조상 예상되지만, 핵심인 Obscomp도 낮았다. Locked evaluation과 text patching으로 승격하지 않았다.

## 5. Common mixed SFT

### 5.1 질문

데이터셋별 adapter 대신 DiReCT와 DDXPlus를 같은 diagnosis-free clinical-state schema로 학습하면 하나의 Medical-NLA가 되는가?

### 5.2 데이터와 target

- runner: [`scripts/run_common_medical_nla_pilot.sh`](../../scripts/run_common_medical_nla_pilot.sh)
- train: DiReCT 248 + DDXPlus 248, 총 496
- validation: DiReCT 50 + DDXPlus 50, 총 100
- DDXPlus 248행: 47 diagnosis strata에서 round-robin 표본
- DDXPlus arm: original만 사용
- target: diagnosis-free `<observed>` bullets
- maximum cues: 12
- DDXPlus finding order: pilot에서는 seed 고정 순서
- checkpoint: source별 content loss의 macro mean

### 5.3 학습 설정

| item | value |
|---|---:|
| seeds | 17, 29, 43 |
| epochs | 3 |
| batch / grad accumulation | 1 / 8 |
| prompt | `common_p0_clinical_state_readout.txt` |
| selection | `source_macro_content` |

### 5.4 결과

| seed | DDX cue recall | DDX precision | Direct raw schema parse | DiReCT Obscomp | extracted observations |
|---:|---:|---:|---:|---:|---:|
| 17 | .1501 | .2133 | 1.00 | .0034 | 150 |
| 29 | .1784 | .2533 | 1.00 | 0 | 150 |
| 43 | .1604 | .1520 | .08 | .0043 | 329 |

동일 cohort의 Source CoT Obscomp는 `.2399`였다. Semantic evaluator error는 0이었으므로 단순 lexical alias 문제로 보지 않았다.

### 5.5 실패 판정

DDXPlus cue 일부와 출력 schema는 학습했지만 DiReCT physician observation으로 전이되지 않았다. Seed 43은 Direct raw schema도 대부분 붕괴했다. 이 pilot은 데이터 양이 적다는 반론을 남겼으므로 다음 full-data SFT로 이어졌지만 자체 checkpoint는 폐기했다.

## 6. Full-data canonical-target SFT

### 6.1 질문

Common pilot 실패가 DDXPlus를 248행만 사용한 데이터 부족 또는 무작위 cue 순서 때문인가?

### 6.2 데이터와 sampling

- runner: [`scripts/run_common_medical_nla_full_sft.sh`](../../scripts/run_common_medical_nla_full_sft.sh)
- unique train: DDXPlus original **4,655** + DiReCT **248** = 4,903
- validation: DDXPlus 50 + DiReCT 50
- cue order: source annotation order, exact duplicate만 제거
- source-temperature alpha: `.5`
- epoch당 DDXPlus 전체를 한 번 사용하고 DiReCT는 약 1,074회, 즉 각 행 약 4.3회 노출
- target/prompt: common diagnosis-free `<observed>` schema

### 6.3 학습 설정

| item | value |
|---|---:|
| seeds | 17, 29 |
| epochs | 1 |
| batch / grad accumulation | 4 / 2 |
| effective batch | 8 |
| learning rate | `2e-4` |
| selection | source-macro content loss |

추가 epoch와 seed 43은 validation gate가 개선될 때만 실행하도록 사전 고정했고, 실패 후 실행하지 않았다.

### 6.4 결과

| method | DDX cue recall | DDX precision | current finding | phantom | removal | clean switch | DiReCT Obscomp |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed17 | .3763 | .3816 | .3389 | .2138 | .4052 | .0244 | .0301 |
| seed29 | .3506 | .3758 | .3612 | .2667 | .3232 | .0122 | .0296 |
| Source CoT | - | - | - | - | - | - | .2130 |

DiReCT extractor는 seed17에서 471개, seed29에서 228개의 인용 가능한 observation을 추출했고 parse/evaluator error는 0이었다. 그럼에도 physician annotation과의 일치는 낮았다.

### 6.5 실패 판정

데이터 확대는 DDXPlus finding recall과 deletion response를 개선했지만 DiReCT case-specific alignment를 회복하지 못했다. Exact-text census에서도 seed17은 36/50, seed29는 48/50이 중복 출력이었다. “학습 데이터가 248개라서 실패했다”는 단순 설명은 이 실험으로 기각됐고, pair specificity가 다음 병목으로 남았다.

## 7. Counterfactual sequence SFT

### 7.1 질문

Original만 학습하지 않고 cue deletion과 native value edit까지 전체 target 문장 CE로 학습하면 intervention을 따라가는가?

### 7.2 데이터와 target

- runner: [`scripts/run_ddxplus_counterfactual_grounding_4gpu_125.sh`](../../scripts/run_ddxplus_counterfactual_grounding_4gpu_125.sh)
- source: DDXPlus official train 4,655 base families
- arms: original + cue deletion + 가능한 native value edit
- target: 각 arm에 현재 존재하는 cue 전체를 source order로 출력
- maximum cues: 64, 초과 시 truncate하지 않고 hard failure
- diagnosis target 없음
- source-sampling alpha: 1.0
- train/validation base-ID overlap 검사

### 7.3 학습 설정

| item | value |
|---|---:|
| seeds | 17, 29 |
| epochs | 1 |
| batch / grad accumulation | 4 / 2 |
| checkpoint selection | content loss |
| validation diagnostic | 435 bases / 952 generated readouts |

### 7.4 결과

| seed/arm | current recall | original target hit | deleted phantom | deletion contrast | removal | clean switch |
|---|---:|---:|---:|---:|---:|---:|
| original-only 17 | .3389 | .3517 | .2138 | .1379 | .4052 | .0244 |
| counterfactual 17 | .5632 | .6345 | .4253 | .2092 | .3659 | .0488 |
| original-only 29 | .3612 | .3770 | .2667 | .1103 | .3232 | .0122 |
| counterfactual 29 | .3475 | .3770 | .2713 | .1057 | .4268 | 0 |

Seed17의 value-edit 분모 82에서 replacement hit `.0732`, old-value persistence `.4024`, clean switch `.0488`이었다.

### 7.5 실패 판정

Seed17의 deletion contrast 개선 `+.0713`은 bootstrap CI에서 0을 배제했지만 phantom도 `+.2115` 증가해 약 두 배가 됐다. Seed29에서는 contrast 개선이 재현되지 않았다. 전체 sequence CE는 바뀌지 않은 다수 cue token이 changed cue 신호를 압도하고, cue 하나를 선택적으로 억제하는 행동을 직접 규정하지 못했다.

## 8. Sentence-level matched/crossed contrastive

### 8.1 질문과 objective

같은 disease category 안에서 두 환자 `(h_i,y_i)`, `(h_j,y_j)`의 matched 합이 crossed 합보다 낮은 NLL을 갖도록 직접 최적화하면 activation-target alignment가 커지는가?

```text
g = 0.5 * [NLL(y_i|h_j) + NLL(y_j|h_i)
         - NLL(y_i|h_i) - NLL(y_j|h_j)]
L = w_sft * L_sft + lambda * T * softplus(-g/T)
```

### 8.2 설정

- trainer: [`scripts/train_medical_nla_contrastive.py`](../../scripts/train_medical_nla_contrastive.py)
- initialization: full-data SFT seed29
- pair sources: DiReCT/DDXPlus 동일 수, source와 disease stratum 내부 pair
- max pairs per source: 124
- pairs per batch: 1
- grad accumulation: 4
- optimizer steps: 20
- learning rate: `5e-5`
- temperature: `.1`
- dropout: disabled
- primary arms: `lambda=.1`, `lambda=1.0`
- 추가 사전 고정 objective screen: `(SFT=1, lambda=5)`, `(SFT=0, lambda=1)`

### 8.3 결과와 판정

| objective | symmetric gap | category-cluster 95% CI | matched win |
|---|---:|---:|---:|
| baseline full-SFT seed29 | +.0051 | `[+.0011,+.0091]` | .7333 |
| lambda=.1 | +.0013 | `[-.0006,+.0033]` | .5556 |
| lambda=1 | +.0022 | `[-.0010,+.0055]` | .5778 |
| SFT=1, lambda=5 | +.0051 | `[+.0011,+.0099]` | .5333 |
| SFT=0, lambda=1 | +.0030 | `[+.0003,+.0057]` | .6444 |

Matched NLL은 `3.9629`에서 `3.7378/3.7154`로 좋아졌지만 사례별 discrimination은 baseline을 넘지 못했다. 문장 생성 난이도를 낮추는 것과 activation을 더 읽는 것은 같지 않았다.

## 9. D10 selected changed-cue 1x2 ranking

### 9.1 D9a support filter

D10은 모든 annotation을 activation에 있다고 가정하지 않았다. Train-only OOF probe로 changed cue가 original에서 검출되고 deletion에서 반응하며 same-diagnosis cue-absent donor보다 높은 경우만 사용했다.

| item | frozen value |
|---|---:|
| presence threshold | .90 |
| deletion-delta threshold | 0 |
| donor-margin threshold | 0 |
| validation positive coverage | 3,032/3,034 = .9993 |
| validation null false support | 112/2,964 = .0378 |
| supported train pairs | 3,104 |
| claims per pair | 1 |

Unsupported cue는 abstention target으로 만들지 않고 제외했다. Value edit도 첫 smoke에서 제외했다.

### 9.2 objective와 설정

한 changed claim `y_c`를 original/deleted activation에서 비교했다.

```text
g_changed = NLL(y_c|h_deleted) - NLL(y_c|h_original)
L_rank = T * softplus((margin - g_changed)/T)
```

- trainer: [`scripts/train_ddxplus_d10_1x2.py`](../../scripts/train_ddxplus_d10_1x2.py)
- control: original-only SFT, ranking weight 0
- proposed: SFT + ranking weight 1
- seeds: 17, 29, 43
- steps: 20
- grad accumulation: 4
- temperature: 1
- margin: 0
- learning rate: `2e-4`
- LoRA dropout: 0
- retained evaluation cue: original/deleted 공통 cue 중 `SHA256(base_id || NUL || cue_text)` 최소값

### 9.3 결과와 판정

| seed | changed delta | changed CI | retained delta | specificity | specificity CI |
|---:|---:|---:|---:|---:|---:|
| 17 | +.0005 | `[-.0006,+.0016]` | +.0010 | -.0005 | `[-.0020,+.0010]` |
| 29 | +.0028 | `[+.0017,+.0039]` | -.0000 | +.0029 | `[+.0015,+.0045]` |
| 43 | +.0030 | `[+.0015,+.0048]` | -.0007 | +.0037 | `[+.0017,+.0059]` |

세 seed 모두 동결 floor `.05`보다 작고 seed17 CI는 0을 포함했다. 필수 teacher-forced gate가 실패했으므로 generation은 실행하지 않았다.

## 10. D14 K=5 OOF hard-set teacher

### 10.1 질문

자유 annotation 대신 train-only probe가 activation마다 선택한 finding set을 target으로 만들면 language decoder가 더 정렬될 수 있는가?

### 10.2 설정

- population: DDXPlus train 4,655 bases
- arms: original + cue-deleted = 9,310 rows
- labels: 91 findings
- fold: `crc32(base_id) % 5`
- 각 OOF head train 비율: 80%
- threshold: 기존 validation-selected `.5`, 변경 금지
- target order: canonical evidence ID
- validation/locked test: target materialization에 미사용
- gate 통과 전 student target을 쓰지 않음

### 10.3 결과

| reader/arm | mean claims | precision | recall | F1 | BCE |
|---|---:|---:|---:|---:|---:|
| K=5 OOF original | 5.1557 | .8881 | .9999 | .9407 | .1479 |
| full frozen original | 4.7865 | .9567 | 1.0000 | .9779 | .1120 |
| K=5 OOF deleted | 6.6644 | .5363 | .9983 | .6977 | .2258 |
| full frozen deleted | 5.6432 | .6331 | .9979 | .7747 | .1804 |

Original precision gate `.90`에 `.0119` 부족했고 deleted mean-claims relative gap은 `18.10%`로 허용치 `10%`를 넘었다. Student set-to-text model은 학습하지 않았다. 이것은 decoder 실패가 아니라 hard teacher calibration 실패다.

## 11. D16 256-dimensional soft auxiliary bottleneck

### 11.1 질문과 architecture

Hard set target을 만들지 않고 continuous activation에 soft probe supervision을 걸면 shared latent가 임상 finding별로 조직되는가?

```text
h32 -> source-balanced PCA down -> z[256]
    -> trainable up projection -> normalized AV injection

training only: z -> 91-way linear auxiliary head
inference: auxiliary head removed, projector remains
```

### 11.2 데이터와 loss

- PCA fit: DDXPlus original 4,655 + DiReCT 248, source weight 각각 .5
- PCA validation: DDXPlus 4,525 originals + DiReCT 50
- Direct rows: language SFT only
- DDXPlus rows: K=5 OOF original soft BCE + D9a selected cue paired softplus
- per step: Direct 8 rows + D9a 8 pairs
- steps: 20, source당 160 unique rows만 사용
- seeds: 17, 29, 43
- learning rate: `2e-4`
- LoRA dropout: `.05`
- temperature: 1
- gradient-parity lambda: seed17에서 1회 측정 후 세 seed 공통 `85`

Control과 auxiliary arm은 같은 row order와 budget을 사용했다. Control seed spread로 effect floor를 `max(2*range,.005)=.005`로 동결했다.

### 11.3 결과

PCA reconstruction 자체는 거의 손실이 없었다.

| population | mean cosine | retained variance |
|---|---:|---:|
| DDXPlus train | .999997 | .993513 |
| DiReCT train | .999999 | .996638 |
| DDXPlus validation | .999997 | .993229 |
| DiReCT validation | .999983 | .959699 |

| seed | control gap | auxiliary gap | delta | category-cluster 95% CI |
|---:|---:|---:|---:|---:|
| 17 | +.000953 | -.000184 | -.001137 | `[-.002652,+.000535]` |
| 29 | +.000442 | -.001034 | -.001476 | `[-.004755,+.000789]` |
| 43 | -.000571 | +.000862 | +.001433 | `[-.000769,+.003505]` |

Fresh frozen-z probe에서도 auxiliary-control delta는 finding F1 `-.0009/-.0007/-.0016`, own-shuffled gap `-.0050/-.0046/-.0058`, value accuracy `-.0137/-.0096/-.0160`, deletion drop `-.0167/-.0141/-.0151`이었다.

### 11.4 실패 판정

PCA gate 통과는 256차원 projection이 activation을 거의 그대로 보존했다는 뜻일 뿐 auxiliary objective가 case specificity를 개선했다는 뜻이 아니었다. 세 seed delta의 부호가 불일치하고 모든 CI가 0을 포함했다. Generation/Gate C는 결과를 뒤집을 수 없어 생략했다.

## 12. D10 budget calibration: 20에서 1,552 steps

### 12.1 질문과 통제

D10 실패가 objective가 아니라 20-step smoke의 작은 budget 때문인지 분리했다. **바꾼 변수는 max steps 하나뿐**이었다.

- D10과 동일한 3,104 train pairs, 3,032 validation pairs
- 동일 lambda=1, T=1, margin=0, lr=`2e-4`, seeds 17/29/43
- control/ranking 각 3 runs, 총 6 runs
- grad accumulation 4
- 776 steps = 1 epoch, 1,552 = 2 epochs
- checkpoints: 20, 194, 388, 776, 1164, 1552
- 중간 checkpoint는 report-only, 최종 1552만 판정
- RunPod A100-SXM4-80GB, DDXPlus 파생물만 사용

### 12.2 최종 결과

| seed | changed delta | retained delta | specificity |
|---:|---:|---:|---:|
| 17 | -.0177 | +.0264 | -.0442 |
| 29 | +.5618 | +.5273 | +.0345 |
| 43 | +1.1233 | +1.1273 | -.0040 |
| across-seed mean | +.5558 | +.5604 | -.0046 |

### 12.3 실패 판정

Raw changed margin은 budget에 반응했지만 retained cue margin이 거의 똑같이 커졌다. 모델은 deleted activation에서 changed claim만 억제한 것이 아니라 모든 claim NLL을 높이는 deletion detector shortcut을 학습했다. Seed17/29/43의 해도 크게 달랐다. 따라서 “20 step이라 신호가 못 자랐다”는 반론은 닫혔지만, 자란 것은 원하는 신호가 아니었다.

## 13. D20 specificity-anchored objective

### 13.1 질문과 objective

D10 shortcut의 원인은 specificity가 평가 gate에만 있고 loss에 없다는 것이었다. D20은 original/deleted에 공통으로 남은 retained claim을 두 activation에서 모두 낮은 CE로 유지하도록 직접 anchor했다.

```text
L = CE(y_changed | h_original)
  + softplus(-g_changed)
  + CE(y_retained | h_original)
  + CE(y_retained | h_deleted)
```

### 13.2 설정

- train/validation pairs: D10과 동일
- retained cue: pair에 미리 저장된 exact common cue 한 개
- ranking weight: 1
- retained-anchor weight: 1
- temperature: 1
- margin: 0
- learning rate: `2e-4`
- steps: 1,552
- seeds: 17, 29, 43
- checkpoints: 194, 388, 776, 1164, 1552; final만 판정
- control: frozen D10 1,552-step same-seed checkpoints 재사용
- hardware: RunPod A100 80GB

추가 비열등 gate는 retained gap `<= +.01`, changed/retained original NLL `<= 1.10x` same-seed control, generation mean claims `>= .90x` control로 실행 전에 동결했다.

### 13.3 결과

| seed | changed gap delta | retained gap delta | specificity delta | changed original NLL delta | retained original NLL delta |
|---:|---:|---:|---:|---:|---:|
| 17 | -.0143 | +.0135 | -.0278 | -.0756 | -.3342 |
| 29 | -.0040 | +.0215 | -.0255 | +.0576 | -.1834 |
| 43 | -.0266 | -.0049 | -.0217 | +.0622 | -.2263 |

### 13.4 실패 판정

Retained-gap은 D10의 `+.5604`에서 작은 범위로 제한돼 global deletion detector는 차단됐다. 그러나 changed-gap과 specificity가 세 seed 모두 음수가 됐다. Retained original NLL은 좋아져 optimizer와 anchor는 정상 동작했지만 target-specific changed-cue 신호는 자라지 않았다. Teacher-forced gate 실패로 generation, checkpoint 사후 선택, 연장, hyperparameter sweep을 수행하지 않았다.

## 14. D22 released AR matched-vs-shuffled diagnostic

### 14.1 질문과 설정

원 NLA의 text-to-activation reconstruction을 사용하지 않은 surrogate 실험만 실패한 것인지 확인하기 전에, released AR가 의료 분포의 사례별 차이를 측정하는지 진단했다.

- AR: `kitft/nla-gemma3-12b-L32-ar`
- activation: CoT-P0/HS32
- arm당 n=20
- control: 같은 diagnosis stratum의 다른 base ID, SHA256 결정론적 순환 donor
- metric: 동일 reconstructed vector와 own activation cosine minus shuffled activation cosine
- bootstrap: row bootstrap 95% CI
- 양성 대조: DDXPlus structured reader text, DiReCT Source CoT
- report-only: Vanilla, DiReCT-only SFT 3 seeds, full-data SFT 2 seeds

### 14.2 결과

| dataset | arm | own cosine | shuffled cosine | gap | 95% CI |
|---|---|---:|---:|---:|---:|
| DDXPlus | structured reader | .9765 | .9765 | -.0000 | `[-.0002,+.0001]` |
| DiReCT | Direct-only seed17 | .9744 | .9744 | -.0000 | `[-.0002,+.0002]` |
| DiReCT | Direct-only seed29 | .9772 | .9772 | -.0001 | `[-.0002,+.0001]` |
| DiReCT | Direct-only seed43 | .9758 | .9757 | +.0000 | `[-.0001,+.0001]` |
| DiReCT | Full-data seed17 | .9779 | .9780 | -.0001 | `[-.0002,+.0001]` |
| DiReCT | Full-data seed29 | .9777 | .9777 | +.0000 | `[-.0001,+.0002]` |
| DiReCT | Source CoT | .9835 | .9834 | +.0001 | `[-.0000,+.0002]` |
| DiReCT | Vanilla | .9962 | .9961 | +.0001 | `[+.0000,+.0002]` |

### 14.3 실패 판정

사례 특이성이 보장된 structured reader와 Source CoT 양성 대조도 own activation을 같은 진단의 다른 activation보다 구분하지 못했다. 높은 absolute cosine은 공통 방향 또는 anisotropy와 양립한다. Released AR cosine은 Medical-NLA reward나 promotion metric으로 사용할 수 없다. 이 판정은 AV-AR 방법 전체의 실패가 아니라 Medical-AR domain adaptation이 선행되어야 한다는 뜻이다.

## 15. 모델 실패로 세지 않은 무효 또는 계측 실패 실행

### 15.1 Semantic mapper v1

Mapper v1은 finding F1 1.0이었지만 `E_132` value 1-7의 별칭 처리에서 1,353개 value가 빠져 G1 native-value accuracy가 `.8765`였다. Alias/lexicon bug를 수정한 v2가 G1 value 1.0, G2 false map 0, G3 replay 1.0, G4 disagreement gate를 통과했다. 따라서 v1은 Vanilla NLA 실패 증거가 아니라 evaluator 구현 실패다.

### 15.2 SFT raw-output AI checklist

Local Llama-3-8B judge를 사용한 DiReCT 200-request 감사에서 세 번 repair 후 valid 56, invalid 144였다. Invalid는 true-without-quote 91, method population mismatch 21, non-verbatim quote 30, JSON parse 2였다. 이 AI checklist 결과는 폐기하고 전체 50-case deterministic exact-text duplicate census만 사용했다.

### 15.3 Patchscope paper-calibration v1

첫 calibration은 `foo ->` 뒤 위치를 채점해 원 논문의 final-marker 규약과 달랐다. HS32 precision@1 0인데 target log-probability lift가 `+8.0141`로 나오는 모순이 있었고, clinical 결과도 해석하지 않았다. 이것은 model negative result가 아니라 target-position mismatch로 무효화한 실행이다.

## 16. 실패 계열을 합쳐서 얻은 결론

1. **출력 형식 학습과 activation 판독은 다르다.** SFT는 parse 가능한 clinical bullets를 만들었지만 동일 문장 반복과 낮은 Obscomp를 보였다.
2. **데이터 양만의 문제는 아니다.** DDXPlus를 248에서 4,655행으로 늘리면 DDXPlus lexical recall은 개선됐지만 DiReCT case specificity는 회복되지 않았다.
3. **전체 sequence CE는 changed cue를 선택적으로 감독하지 못한다.** Counterfactual SFT는 seed17에서 recall과 contrast를 올렸지만 phantom도 두 배로 올렸다.
4. **문장 NLL 개선은 사례 특이성 개선이 아니다.** Sentence contrastive는 matched NLL을 낮췄지만 matched-vs-crossed gap은 baseline보다 작았다.
5. **Changed-only ranking은 shortcut이 있다.** Budget을 늘리자 deleted activation 전체를 억제하는 detector가 자랐다.
6. **Specificity를 loss에 넣으면 shortcut은 막히지만 목표 신호도 사라졌다.** D20은 detector를 차단했으나 changed-gap이 세 seed 모두 악화됐다.
7. **현재 released AR cosine도 바로 reward로 쓸 수 없다.** 양성 대조를 통과하지 못했으므로 geometry audit 또는 domain-adapted Medical-AR가 필요하다.

현재 가장 정확한 결론은 다음이다.

> CoT-P0 activation에는 closed probe로 읽을 수 있는 환자별 finding 정보가 있지만, 지금까지 시험한 SFT, sequence counterfactual CE, sentence contrastive, changed-cue ranking, soft bottleneck, specificity anchor는 그 정보를 안정적인 자유 자연어 설명으로 변환하지 못했다.

## 17. 재현 문서와 코드 색인

| subject | document/code |
|---|---|
| 전체 tuning 계보 | [`medical_nla_tuning_strategy_2026-08-29.md`](medical_nla_tuning_strategy_2026-08-29.md) |
| DiReCT/Common/Full SFT | [`03-medical-nla-training.md`](03-medical-nla-training.md) |
| DiReCT semantic evaluation | [`04-direct-explanation-evaluation.md`](04-direct-explanation-evaluation.md) |
| D9a/D10 support and smoke | [`08-ddxplus-d9a-selected-cue.md`](08-ddxplus-d9a-selected-cue.md) |
| D14 OOF teacher | [`discussions/2026-08-29-d14-oof-teacher.md`](discussions/2026-08-29-d14-oof-teacher.md) |
| D16 bottleneck | [`discussions/2026-08-29-soft-auxiliary-grounding.md`](discussions/2026-08-29-soft-auxiliary-grounding.md) |
| D10 budget trajectory | [`discussions/2026-08-29-program-decision-after-d16.md`](discussions/2026-08-29-program-decision-after-d16.md) |
| D20 anchor | [`discussions/2026-08-30-specificity-anchored-objective.md`](discussions/2026-08-30-specificity-anchored-objective.md) |
| D22 AR diagnostic | [`discussions/2026-08-30-d22-public-ar-diagnostic.md`](discussions/2026-08-30-d22-public-ar-diagnostic.md) |
| frozen decision ledger | [`discussions/DECISIONS.md`](discussions/DECISIONS.md) |

실제 checkpoint metadata와 private row-level scores는 server artifact가 정본이다. 이 Git 문서는 aggregate setting과 판정만 기록하며 restricted clinical text나 private activation path를 복제하지 않는다.
