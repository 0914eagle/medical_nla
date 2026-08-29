# DDXPlus HS32 full-label OOF teacher

## 목적

D14의 첫 단계다. Official DDXPlus train의 CoT-P0/HS32 activation에서
validation-selected finding probe 설정을 이용해 **out-of-fold(OOF) 91-label
확률**을 물질화한다. Probe는 이 단계와 이후 target 생성에서 training-only
teacher로만 사용한다. 최종 Medical-NLA inference에는 probe나 DDXPlus ontology를
입력하지 않는다.

이 단계에서 하는 일은 두 가지뿐이다.

1. `4,655` base의 original과 cue-deleted activation을 동일한 OOF head로 채점한다.
2. 합계 `9,310`행의 teacher set 분포와 read-only 대조 지표를 보고한다.

자연어 `<observed>` target, student dataset, adapter, P2-P4 gate는 만들거나
동결하지 않는다. DDXPlus validation과 locked test도 읽지 않는다.

## 교차적합 규약

| 항목 | 고정값 |
|---|---|
| activation | CoT-P0 / HS32 |
| 모집단 | official train `4,655` base |
| arms | original + 기존 1-cue deletion |
| fold | `crc32(base_id) % 2` |
| OOF head | 반대 fold의 original activation으로만 학습 |
| 표준화 | 반대 fold original의 mean/std만 사용 |
| ontology | validation-selected HS32 artifact의 finding labels |
| hyperparameter/epoch/threshold | 같은 HS32 artifact에서 상속 |
| pair rule | 같은 base의 original/deleted는 같은 OOF head 사용 |
| selected set order | canonical evidence-ID 오름차순 |

Probability vector는 artifact의 `finding_labels` 배열 순서로 저장한다. 선택된
ID는 probability 순위와 무관하게 canonical evidence-ID 순서로 저장한다. 따라서
student가 teacher의 임의적인 confidence 순서까지 예측하도록 만들지 않는다.

## 산출물

기본 출력은 server 125의 다음 경로다.

```text
/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/
  oof_finding_teacher_hs32_v1/
    private_teacher_scores.jsonl
    report.json
    summary.md
```

`private_teacher_scores.jsonl`의 각 행에는 다음만 들어간다.

- row/base ID, original 또는 cue-deleted variant, diagnosis, OOF fold
- HS32/P0 metadata
- 전체 91-label probability vector
- threshold 이상인 evidence ID의 canonical-order set

Prompt, activation path, cue text, 자연어 target은 넣지 않는다. `report.json`은
입력 파일 SHA256과 teacher JSONL SHA256을 기록한다.

## Read-only report

Student 학습 전에 다음을 확인한다.

- original/deleted 행 수와 complete-pair coverage
- original/deleted row당 selected finding 수와 empty-set 수
- original/deleted teacher-set Jaccard
- changed cue original hit, deleted phantom, conditional removal
- untouched finding original hit와 preservation
- same-fold/same-diagnosis teacher-set hard control
- fold별 zero/low-positive label과 diagnosis별 selected-label coverage

Same-diagnosis control의 matched teacher self-F1은 정의상 `1.0`이다. 따라서
`1 - shuffled teacher-set F1`은 **target-set specificity ceiling**이지 student
성능이 아니다. 이후 P4에서는 student matched output과 shuffled activation
output을 teacher target에 대해 직접 비교해야 한다.

Deletion phantom은 해당 intervention 뒤에도 finding이 decode된 비율이다. 남은
cue들만으로 해당 finding을 추론할 수 있으므로, phantom만으로 representation
failure라고 단정하지 않는다. 이 값은 student가 모방할 teacher의 intervention
response 범위를 정한다.

## Server 125 실행

```bash
cd /home/eagle0914/medical_nla
git pull origin main

PROBE=/data1/heejae/medical_nla/results/ddxplus_finding_value_probe_val_v1/finding_value_hs32.pt

DATA_ROOT=/data1/heejae \
GPU=0 \
PROBE_ARTIFACT="$PROBE" \
nohup bash scripts/run_ddxplus_oof_teacher_125.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_oof_teacher_hs32_v1.log 2>&1 &
```

진행 확인:

```bash
tail -f /data1/heejae/medical_nla/logs/ddxplus_oof_teacher_hs32_v1.log
```

완료 확인:

```bash
OUT=/data1/heejae/medical_nla/data/ddxplus_counterfactual_train_v1/oof_finding_teacher_hs32_v1

wc -l "$OUT/private_teacher_scores.jsonl"
cat "$OUT/summary.md"
```

정상 행 수는 `9,310`이다. 이 report를 검토하기 전에는 target builder나 student
smoke를 실행하지 않는다.
