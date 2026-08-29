# D14 probe-distilled OOF teacher

## 질문

Training-only finding probe가 activation-conditioned set target을 만들 수 있으며,
하나의 Medical-NLA decoder가 이를 자연어로 distill할 수 있는가?

최종 inference는 다음 형태여야 한다.

```text
raw CoT-P0/HS32 activation -> one Medical-NLA decoder -> <observed> findings
```

Probe, DDXPlus ontology, prompt cue는 inference 입력으로 사용하지 않는다.

## K=2 materialization

- 모집단: official DDXPlus train `4,655` base
- rows: original + cue-deleted `9,310`
- labels: `91`
- complete-pair coverage: `1.0000`
- threshold: validation-selected `.5`
- target order: canonical evidence-ID
- validation/locked test read: `no/no`

초기 분포:

| metric | value |
|---|---:|
| original mean claims | 6.0745 |
| deleted mean claims | 8.3590 |
| original/deleted Jaccard | .5440 |
| changed original hit | 1.0000 |
| deleted phantom | .5101 |
| removal | .4899 |
| untouched preservation | .9986 |
| same-diagnosis specificity ceiling | +.1958 |

## Calibration audit

| reader | arm | mean claims | precision | recall | F1 | BCE |
|---|---|---:|---:|---:|---:|---:|
| K=2 OOF | original | 6.0745 | .7538 | .9999 | .8595 | .2271 |
| K=2 OOF | deleted | 8.3590 | .4276 | .9985 | .5988 | .3004 |
| full-data frozen | original | 4.7865 | .9567 | 1.0000 | .9779 | .1120 |
| full-data frozen | deleted | 5.6432 | .6331 | .9979 | .7747 | .1804 |

Deletion 후 newly added labels는 평균 `3.5091`개였다. 총 `16,335`개 중
deleted input에 없는 label은 `16,333`개(`.9999`)였고 threshold 초과 margin
중앙값은 `.1077`이었다. 따라서 원인은 다음 둘로 분리한다.

1. **K=2 cross-fit calibration shift**: 절반 train head가 full-data threshold `.5`에서
   과다 선택한다.
2. **Counterfactual OOD/pattern completion**: full-data probe도 deletion에서 claims
   `4.7865→5.6432`, phantom `.3968`을 보인다.

입력 cue 일치는 calibration 진단이지 activation content의 절대 정답이 아니다.

## 승인된 K=5 one-shot

- fold: `crc32(base_id) % 5`
- 각 OOF head train 비율: 80%
- threshold: `.5` 유지
- hyperparameter/epoch: validation-selected 값 유지
- 추가 K, threshold, epoch sweep: 금지
- validation/locked test: 미사용
- student target: gate 검토 전 미생성

Gate는 [`DECISIONS.md`](DECISIONS.md)의 D15를 따른다.

## 실행 및 산출물

Server 125:

```bash
PROBE=/data1/heejae/medical_nla/results/ddxplus_finding_value_probe_val_v1/finding_value_hs32.pt

DATA_ROOT=/data1/heejae \
GPU=0 \
PROBE_ARTIFACT="$PROBE" \
nohup bash scripts/run_ddxplus_oof_teacher_k5_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_oof_teacher_k5_v2.log 2>&1 &
```

```text
/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/
  oof_finding_teacher_hs32_k5_v2/
    private_teacher_scores.jsonl       # expected 9,310
    report.json
    summary.md
    calibration_audit_v1/
      report.json
      summary.md
      private_label_prevalence.jsonl   # expected 91
```

## 판정

현재 상태: **K=5 실행/검토 대기**.

- PASS: P2-P4 target/gate를 별도 사람 승인한 뒤 student smoke 설계 문서를 새로 만든다.
- FAIL: 추가 K/threshold sweep 없이 hard-set text distillation을 중단하고 새 주제
  문서에서 probabilistic/alternative objective를 논의한다.

## 열린 항목

1. K=5의 D15 일곱 gate 통과 여부
2. 통과 시 original-only와 original+deleted target contract
3. 통과 시 P2-P4 `.80/.05/.02` 최종 승인
