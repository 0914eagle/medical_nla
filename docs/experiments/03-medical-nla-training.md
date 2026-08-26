# E3. Medical-NLA training

## 질문

의료 설명 supervision이 vanilla NLA를 개선하면서도 분류기 붕괴와 activation 무시를
피할 수 있는가?

## 학습군

| Method | Clinical text | Reconstruction | Pair specificity |
|---|---:|---:|---:|
| Vanilla NLA | No | pretrained | No |
| SFT only | Yes | No | No |
| Full Medical-NLA | Yes | Yes | Yes |

현재 주 실험으로 실행하는 것은 `SFT only`뿐이다. `train_medical_nla_lora.py`는 target token
cross-entropy만 계산하며 AR reconstruction과 pair-specificity objective는 구현하지 않았다.
따라서 Full Medical-NLA는 아래 구현 게이트를 통과하기 전에는 실행 이름으로 사용하지 않는다.

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

첫 실행은 목표 충돌을 피하기 위해 두 필드만 학습한다.

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
