# E3. Medical-NLA training

## 질문

의료 설명 supervision이 vanilla NLA를 개선하면서도 분류기 붕괴와 activation 무시를
피할 수 있는가?

## 학습 전 게이트

E2에서 P0에 decode 가능하다고 확인된 target family만 Medical-NLA의 필수 내용으로 평가한다.
Probe score를 자유 산문 target으로 그대로 복사하지는 않는다. Probe는 정보 존재와 layer를
감사하고, AV의 개별 claim faithfulness는 E5의 matched/shuffled, counterfactual, AR로 검증한다.

## 학습군

| Method | Clinical text | Reconstruction | Pair specificity |
|---|---:|---:|---:|
| Vanilla NLA | No | pretrained | No |
| SFT only | Yes | No | No |
| Reconstruction Medical-NLA | Yes | Yes | No |
| Full Medical-NLA | Yes | Yes | Yes |

현재 완료된 것은 `SFT only` 세 seed뿐이다. `train_medical_nla_lora.py`는 target token
cross-entropy만 계산하며 AR reconstruction과 pair-specificity objective는 구현하지 않았다.
따라서 Full Medical-NLA는 아래 구현 게이트를 통과하기 전에는 실행 이름으로 사용하지 않는다.

### 08-28 common-schema mixed pilot

DDXPlus locked probe에서 CoT-P0에 finding 정보가 있음을 확인했으므로, 다음 실행은 데이터셋별
adapter가 아니라 하나의 HS32 adapter를 학습하는 mixed pilot이다. DiReCT와 DDXPlus 모두
동일한 `<observed>` bullet schema를 쓰고 diagnosis `<answer>` supervision은 제거한다.

- train: DiReCT 248 + DDXPlus 248 (진단 strata round-robin 표본)
- validation: DiReCT 50 + DDXPlus 50
- activation: CoT-P0/HS32/last-token으로 고정
- DDXPlus original arm만 학습/validation에 사용
- rendered native value는 finding 문장 안에 유지하지만 value-edit response는 별도 gate로 보고한다
- source별 행 수를 동일하게 해 DDXPlus 4,655행이 DiReCT를 압도하지 않게 한다
- best epoch은 source별 content-token loss를 먼저 계산한 뒤 두 source의 macro mean으로 선택한다

이 pilot은 schema와 학습 가능성을 고정하는 development run이다. DDXPlus locked test는 이미
closed-probe 설계 판단에 사용됐으므로, mixed NLA의 최종 confirmatory 성능은 별도의 미사용
prospective holdout을 만든 뒤 한 번만 평가해야 한다.

실행 wrapper는 `scripts/run_common_medical_nla_pilot.sh`이다. 먼저
`RUN_NAME=common_medical_nla_smoke5_v1 MAX_STEPS=5`로 smoke를 통과시킨 뒤,
`RUN_NAME=common_medical_nla_pilot_v1`로 full three-seed 실행을 시작한다. 두 실행은 서로
다른 디렉터리를 사용한다.

두 서버의 storage prefix가 다르므로 복사한 activation manifest는 반드시 path remap 후
사용한다. wrapper의 `DDX_TRAIN`, `DDX_VAL`, `DIRECT` 환경변수로 서버별 canonical manifest를
명시할 수 있으며, 기본값은 해당 서버의 `DATA_ROOT` 아래 merged-v1 manifest다.

학습 후 `scripts/run_common_medical_nla_validation.sh`가 같은 100행과 같은 common prompt로
vanilla 및 각 seed의 readout을 생성한다. 이 단계의 lexical cue recall/precision은 빠른
development screen이며, 최종 explanation 점수는 method-blind semantic extraction과
DDXPlus paired counterfactual 평가로 확정한다.

### 08-28 full-data canonical-target SFT

248+248 mixed pilot의 실패만으로 SFT objective 자체를 기각할 수는 없다. Pilot은 DDXPlus
4,655행 중 248행만 사용했고, 각 환자의 target finding 순서를 activation과 무관한 RNG로
섞었다. 다음 development ablation은 구조와 prompt를 유지하고 이 두 요인만 수정한다.

- train: DDXPlus original 4,655행 + DiReCT 248행, 총 4,903행
- validation: 기존과 동일한 DDXPlus 50 + DiReCT 50
- target: diagnosis-free `<observed>` schema 유지
- finding 순서: 원본 annotation/cue 순서를 보존하고 중복만 제거
- source mixture: 모든 DDXPlus 행을 한 번 포함하고 DiReCT를 source-temperature
  `alpha=0.5`로 epoch당 약 1,074회 재생한다
- checkpoint selection: 두 source의 content loss macro mean
- 첫 실행: 1 epoch, seed 17/29; 결과가 개선될 때만 seed 43과 추가 epoch 수행

`alpha=1`이면 4,655:248의 자연 빈도로 DiReCT가 거의 사라지고, `alpha=0`이면 각 DiReCT
행을 약 19회 반복한다. `alpha=0.5`는 모든 unique row를 보존하면서 작은 source를 약 4.3회
노출하는 중간값이다. 이 실행도 SFT-only이므로 activation faithfulness의 증명이 아니라
full-corpus supervision이 case-specific readout을 회복하는지 확인하는 development gate다.

Server 125의 네 GPU는 한 모델을 네 장에 펼치지 않는다. `0,1`에서 seed 17,
`2,3`에서 seed 29를 독립적으로 병렬 실행한다. 학습이 끝나면 각 worker가 동일한 100행
readout과 고정 약 952행 DDXPlus paired grounding을 순차 수행한다.

```bash
# server 125 (/data1/heejae), GPU 0--3가 모두 비어 있어야 한다.
DATA_ROOT=/data1/heejae EPOCHS=1 RUN_VALIDATION=1 RUN_GROUNDING=1 \
  nohup bash scripts/run_common_medical_nla_full_4gpu_125.sh \
  > /data1/heejae/medical_nla/logs/common_medical_nla_full_4gpu_125.log 2>&1 &
```

전체 queue는 dataset row 수와 target style을 검증한 뒤 시작하며, 기존 incomplete adapter
directory를 덮어쓰지 않는다. Server 125에서는 DDXPlus train/validation manifest의
`/data/heejae` activation path를 `/data1/heejae`로 자동 remap하고 모든 tensor 파일의 존재를
검증한 server-local manifest를 사용한다.

#### Full-data validation 결과

Seed 17/29의 1-epoch 실행이 완료됐다. 아래는 locked test가 아닌 동일 validation screen이다.

| method | DDX cue recall | DDX cue precision | DiReCT cue recall | current finding | deletion phantom | removal success | clean switch |
|---|---:|---:|---:|---:|---:|---:|---:|
| 248+248 pilot, seed 29 | .1784 | .2533 | 0 | .1499 | .1356 | 0 | 0 |
| full data, seed 17 | .3763 | .3816 | .0216 | .3389 | .2138 | .4052 | .0244 |
| full data, seed 29 | .3506 | .3758 | .0076 | .3612 | .2667 | .3232 | .0122 |

Full-data supervision은 DDXPlus finding 복원과 cue deletion 반응을 분명히 개선했다. 그러나
value edit의 clean switch는 여전히 거의 0이고 DiReCT lexical recall도 매우 낮다. Seed 17은
source-macro validation loss가 더 낮고(`2.0303` 대 `2.1156`), phantom이 적으며 removal
success가 높으므로 primary development checkpoint로 선택한다. Seed 29는 seed sensitivity로
유지한다. 다음 gate는 두 seed의 동일 DiReCT 50행 method-blind semantic 평가이며, 이 결과
전에는 locked test나 text patching으로 승격하지 않는다.

Method-blind semantic gate도 완료됐다. Codex extractor는 CoT/seed17/seed29의 150행을 모두
parse했고 official evaluator 오류는 0이었다. Seed 17/29에서 각각 471/228개의 정확히 인용
가능한 observation을 추출했지만 `Obscomp`는 `.0301/.0296`, `Expcom`과 `Expall`은 모두
0이었다. 같은 cohort의 CoT는 `Obscomp=.2130`, `Expcom=.0650`, `Expall=.0153`이었다.
따라서 낮은 DiReCT lexical score는 약칭이나 의역으로 설명되지 않는다. Full-data SFT는
DDXPlus finding 판독과 deletion response를 개선했지만 DiReCT physician observation으로
전이되지 않았다. 추가 epoch와 seed 43은 중단하고, 이 checkpoint는 full-corpus SFT
ablation으로만 보존한다.

다음 실행은 새 학습이 아니라 activation-target alignment gate다. 동일 DiReCT validation에서
같은 disease category의 다른 환자를 deterministic donor로 배정하고, content-token NLL을
다음 세 조건에서 비교한다.

- matched: `p(y_i | h_i)`
- target shuffled: `p(y_j | h_i)`
- activation shuffled: `p(y_i | h_j)`

Target마다 고유 난이도가 다르므로 primary statistic은 두 target을 두 activation에서 모두
평가하는 대칭 2x2 cross gap이다. Donor가 disease category 내부 순환쌍이므로 category 전체를
재표집하는 cluster bootstrap을 primary CI로 사용한다. `NLL(cross)-NLL(matched)` cluster
bootstrap 95% CI가 0보다 커야 physician observation target이 activation과 사례별로
정렬됐다고 판정한다. 통과하지
못하면 DiReCT gold observation을 그대로 SFT/contrastive positive로 쓰지 않는다. 한쪽만
섞은 gap은 target-difficulty noise를 포함하므로 diagnostic으로만 남긴다. 이 gate는 validation
50행만 사용하며 locked test를 읽지 않는다.

Alignment gate는 50행 중 같은 category donor를 만들 수 있는 45행, 13개 category cluster에서
평가됐다. Seed 17의 symmetric gap은 `+.0040`이지만 category-cluster CI가
`[-.0001, +.0085]`로 0을 포함해 통과하지 못했다. Seed 29는 gap `+.0051`, CI
`[+.0011, +.0091]`, matched win rate `.7333`으로 gate를 통과했다. 따라서 physician target이
P0에 전혀 없다는 결론은 기각하지만, SFT가 사용하는 pair-specific 신호는 매우 약하다.
Seed 29를 최종 모델로 승격하는 대신 다음 objective의 development warm start/설계 근거로만
사용한다. 다음 학습은 같은 category의 `(h_i,y_i),(h_j,y_j)` matched 합이
`(h_i,y_j),(h_j,y_i)` cross 합보다 높아지도록 대칭 contrastive loss를 직접 추가한다.

첫 contrastive 실행은 full run이 아니라 server 125의 4-GPU smoke다. Full common SFT
seed 29를 동일한 warm start로 고정하고, source와 disease stratum 내부에서 서로 겹치지 않는
환자쌍을 만든다. 각 쌍은 두 matched sequence와 두 crossed sequence를 한 forward에 넣으며,
matched SFT loss에 symmetric content-NLL ranking loss를 더한다. DDXPlus와 DiReCT의 pair
수는 동일하게 맞춘다. LoRA dropout은 끈 상태에서 gradient를 계산한다. 기존 alignment gap이
약 `.005`뿐이어서 서로 다른 dropout mask가 primary 신호보다 큰 잡음을 만들 수 있기 때문이다.

Development sweep은 `lambda=.1`과 `lambda=1.0`, temperature `.1`, optimizer 20 step만
비교한다. 두 arm은 각각 2개 GPU를 사용해 병렬 실행하고, 완료 직후 동일한 DiReCT validation
50행 alignment gate를 다시 계산한다. Locked test는 읽지 않는다. 다음 단계로 승격하려면
category-cluster bootstrap CI 하한이 0보다 크고, 기존 seed 29 gap `+.0051`보다 커져야 한다.
통과 arm이 있으면 그 arm만 DiReCT semantic validation과 DDXPlus counterfactual validation에
보낸다. 둘 다 실패하면 step 수를 늘리기 전에 target construction 또는 objective를 재설계한다.

```bash
DATA_ROOT=/data1/heejae \
nohup bash scripts/run_common_medical_nla_contrastive_smoke_4gpu_125.sh \
  > /data1/heejae/medical_nla/logs/common_medical_nla_contrastive_smoke20_v1.log 2>&1 &
```

Queue log는 두 worker의 시작·종료 상태를 기록한다. 학습 곡선은
`common_medical_nla_contrastive_smoke20_v1_lambda_0p1_train.log`와
`common_medical_nla_contrastive_smoke20_v1_lambda_1p0_train.log`, alignment 결과는 같은
prefix의 `_alignment.log`에서 확인한다.

#### Mixed-pilot validation 결과

세 seed의 동일한 100행 validation 출력이 완료됐다. 아래 값은 locked test가 아닌 lexical
development screen이다. 각 source는 50행이며 cue recall/precision은 해당 source 안에서
계산했다.

| method | DDX parse | DDX cue recall | DDX cue precision | DiReCT parse | DiReCT cue recall | DiReCT cue precision |
|---|---:|---:|---:|---:|---:|---:|
| Vanilla NLA | 0 | 0 | 0 | 0 | 0 | 0 |
| Common SFT, seed 17 | 1.00 | .1501 | .2133 | 1.00 | 0 | 0 |
| Common SFT, seed 29 | 1.00 | .1784 | .2533 | 1.00 | 0 | 0 |
| Common SFT, seed 43 | 1.00 | .1604 | .1520 | .08 | 0 | 0 |

Mixed SFT는 DDXPlus cue를 일부 복원했지만 DiReCT physician observation은 세 seed 모두
lexical hit가 없었다. Seed 17/29는 출력 schema만 안정적으로 학습했고, seed 43은
DiReCT에서 schema도 붕괴했다. 따라서 이 checkpoint들은 locked test 또는 text patching으로
승격하지 않는다.

동일한 DiReCT 50행의 method-blind quote extraction과 공식 semantic evaluator도 완료됐다.
Lexical 0은 약칭이나 의역 문제가 아니었다. CoT `Obscomp=.2399`에 비해 common SFT는
seed17 `.0034`, seed29 `0`, seed43 `.0043`이었다. Seed17/29는 각각 정확히 150개
observation을 추출할 수 있는 형식으로 생성했지만 seed29는 physician observation과 의미상
일치한 claim이 0이었다. Seed43은 329개를 추출했어도 정렬 점수가 회복되지 않았다. 따라서
common SFT v1은 **schema 학습에는 성공했지만 case-specific clinical content 판독에는 실패**한
development ablation으로 동결한다.

8시간 무인 validation queue는 두 독립 작업만 실행한다. Server 125는
`run_overnight_common_direct_semantic.sh`로 Direct 50행 의미 채점을 완료한다. Server 62는
`run_overnight_common_ddx_grounding.sh`로 DDXPlus validation manifest를 base ID hash로 40등분한
뒤 고정 shard 0--3의 약 1,000행에서 original/deletion/value-edit 반응을 점수화한다. 후자는
validation diagnostic이며 locked test가 아니다. 두 작업 모두 E6 text patching은 호출하지 않는다.

Clinical text는 DiReCT의 physician deduction structure에서 만든다. Activation은 P0를
주 입력으로 한다. Source-wrong 행에서 gold physician text를 activation의 현재 결론처럼
무조건 매핑하면 misalignment가 생기므로 다음을 분리한다.

- source-correct: clinical alignment supervision 가능
- source-wrong: decision fidelity 평가 및 activation-grounding 학습에 사용
- gold diagnosis를 강제로 말하게 하는 loss와 source-state를 읽는 loss를 혼합하지 않음

세부 target은 observation reconstruction, source-decision diagnosis, physician-gold
diagnosis/rationale, activation reconstruction으로 분리하고 field별 loss mask를 사용한다.
한 note의 deduction 수가 많아도 한 환자가 과도하게 가중되지 않도록 note-level로
normalization한다. Strict PDD source-correct 수가 작을 수 있으므로 학습 전에 train의
strict/category/official semantic correct 수를 각각 기록한다.

### 08-27 SFT-only v1 target

첫 실행은 목표 충돌을 줄이기 위해 두 필드만 학습했다.

- `<observed>`: `observation_exact_in_note=true`인 physician observation만 사용한다.
- `<answer>`: physician gold가 아니라 같은 P0 trajectory에서 backbone이 실제로 생성한
  source answer를 사용한다.

따라서 source-wrong 사례도 gold-correction supervision으로 바뀌지 않는다. Gold PDD는
분석 metadata로만 남는다. Train 266과 `val_seen` 52만 읽으며 test 72+106은 dataset builder
인터페이스에 넣지 않는다. 이 중 gold label이 note에 정확히 노출된 train 18/validation 2행은
primary 학습에서 제외하므로 예상 최대 분모는 248/50이다. Validation checkpoint 선택은 고정
XML이 아니라 `<observed>`의 content-token loss를 사용한다.

두 서버에서 세 seed를 병렬 실행한다.

```bash
# server 62: seed 17 뒤에 43을 순차 실행
DATA_ROOT=/data/heejae GPUS=2,3 SEEDS="17 43" EPOCHS=3 \
  nohup bash scripts/run_direct_e3_sft.sh \
  > /data/heejae/medical_nla/logs/direct_e3_sft_seeds17_43.log 2>&1 &

# server 125: seed 29
DATA_ROOT=/data1/heejae GPUS=0,1 SEEDS="29" EPOCHS=3 \
  nohup bash scripts/run_direct_e3_sft.sh \
  > /data1/heejae/medical_nla/logs/direct_e3_sft_seed29.log 2>&1 &
```

각 서버의 builder가 만든 `summary.md`에서 train/validation 행 수, source-correct/wrong 수,
ID hash를 먼저 출력한 뒤 GPU 학습을 시작한다. 두 서버의 ID hash가 다르면 학습 결과를
합치지 않는다.

## 최종 출력 계약

Final Medical-NLA는 데이터셋별 고정 slot이나 정확히 세 개의 claim을 강제하지 않는다.

```xml
<explanation>
- zero or more concise, activation-supported clinical claims
</explanation>
```

임상 내용이 안정적으로 읽히지 않으면 abstain할 수 있어야 한다. Observation, value, diagnosis,
relation 분류는 method-blind post-hoc extractor가 수행한다. 현재 SFT-only v1의
`<observed>/<answer>` schema는 warm-start ablation이며 최종 자유 판독 계약과 구분한다.

## 필수 통제

- Patient-disjoint split
- confirmatory PDD-heldout 12개는 train에서 완전 제외
- 3 random seeds
- 동일 LoRA rank/target modules/token budget
- Early stopping은 val_seen
- 진단명 제거 또는 masking sensitivity

## 중단 기준

Seen 점수만 높고 PDD-heldout, hard shuffle gap, cue counterfactual이 낮으면 분류기 또는
문구 암기로 판정한다. 이 경우 모델 크기나 epoch를 늘리기 전에 objective를 수정한다.

## Full objective 구현 게이트

설명 text가 discrete이므로 AR MSE를 SFT CE에 단순 가산할 수 없다. 이 작업은 현재 밤샘
실행 큐에서 제외한다. SFT-only가 E4를 개선하지만 E5 grounding에 실패할 때 공개 NLA 방식에 가까운
RL/GRPO 또는 AR/clinical/pair score로 후보 설명을 순위화한 offline preference optimization
중 하나를 먼저 구현한다. 다음 smoke가 모두 통과해야 full run을 시작한다.

- AR reconstruction reward가 matched text를 shuffled text보다 높게 평가
- zero/mean activation이 matched activation보다 높은 reward를 받지 않음
- 한 optimizer step에서 AV LoRA parameter가 실제로 갱신
- metadata에 objective weight, AR checkpoint, prompt, seed 기록
- HS32 사용. 다른 hidden-state index면 layer-matched AV/AR 필요
