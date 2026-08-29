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

## K=5 결과

산출물은 `9,310` teacher rows와 `91` label-prevalence rows로 완성됐다.
Student target, validation, locked test는 사용하지 않았다.

| reader | arm | mean claims | precision | recall | F1 | BCE |
|---|---|---:|---:|---:|---:|---:|
| K=2 OOF | original | 6.0745 | .7538 | .9999 | .8595 | .2271 |
| K=5 OOF | original | 5.1557 | .8881 | .9999 | .9407 | .1479 |
| full-data frozen | original | 4.7865 | .9567 | 1.0000 | .9779 | .1120 |
| K=2 OOF | deleted | 8.3590 | .4276 | .9985 | .5988 | .3004 |
| K=5 OOF | deleted | 6.6644 | .5363 | .9983 | .6977 | .2258 |
| full-data frozen | deleted | 5.6432 | .6331 | .9979 | .7747 | .1804 |

K=5는 K=2보다 original precision을 `.7538→.8881`, original F1을
`.8595→.9407`, OOF/full original Jaccard를 `.8535→.9437`로 개선했다.
Deletion에서도 phantom은 `.5101→.4263`, newly added claims는
`3.5091→2.3787`로 감소했다.

그러나 deletion 후 새로 선택된 label `11,073`개 중 `11,071`개(`.9998`)는
deleted input에 없었다. Threshold 초과 margin 중앙값도 `.1429`이므로 단순히
`.5` 경계에 걸친 수치 오차로 볼 수 없다.

## 동결 Gate 판정

| criterion | observed | gate | result |
|---|---:|---:|---|
| original cue precision | .8881 | >= .90 | **FAIL** |
| original cue recall | .9999 | >= .98 | PASS |
| original mean-claims relative gap | 7.71% | <= 10% | PASS |
| OOF/full original Jaccard | .9437 | >= .90 | PASS |
| deleted mean-claims relative gap | 18.10% | <= 10% | **FAIL** |
| deleted phantom absolute gap | .0295 | <= .05 | PASS |
| minimum fold original precision | .8711 | >= .85 | PASS |

전체 gate는 AND이므로 최종 판정은 **FAIL**이다. Original precision은 기준에
`.0119` 부족하고, deleted mean-claims gap은 허용치보다 `8.10%p` 크다.

## 최종 판정

1. D14의 hard-set OOF teacher target은 만들지 않는다.
2. P2-P4 승인과 student set-to-text smoke는 진행하지 않는다.
3. D15에 따라 K, threshold, epoch를 추가 탐색하지 않는다.
4. 이 실패는 activation 판독 자체의 불가능을 뜻하지 않는다. Hard threshold로
   만든 OOF claim set이 frozen calibration contract를 충족하지 못했다는 판정이다.
5. 다음 learned method는 별도 문서에서 soft/probabilistic auxiliary grounding을
   논의한다.

현재 상태: **resolved / FAIL**.
