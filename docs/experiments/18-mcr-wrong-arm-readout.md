# 18 — MCR wrong-arm 내부 판독과 교정 사다리

**상태**: 🔶 activation/readout 생성 완료; arm-aware 재채점과 r5 비교 대기

이 실험은 MedCaseReasoning(MCR)의 wrong-note 실행에서 마지막 prompt token의
activation을 읽어, 판독이 **그 실행에서 모델이 실제로 내린 결론**을 사례별로
따르는지 확인하고 그 내용을 교정에 되먹이는 실험이다. DDXPlus의 고정 49-class
probe는 MCR의 열린 진단 어휘에 그대로 정의할 수 없으므로, 이 실험은 자연어
결론 판독이 열린 어휘에서 제공하는 추가 범위를 시험한다.

## 1. 생성이 끝난 산출물

| 산출물 | 경로 | 상태 |
|---|---|---|
| position rows | `$DATA/mcr_hint_position_rows.jsonl` | ✅ |
| L32 activation | `$ART/activations/mcr_hint_positions_L32/` | ✅ |
| conclusion readout | `$ART/results/readout_mcr_hint_final_L32.jsonl` | ✅ 3,086행 |

3,086행은 하나의 모집단이 아니다.

```text
none arm  × final position = 1,543
wrong arm × final position = 1,543
```

따라서 파일 전체를 한 번에 채점하면 no-note 상태와 wrong-note 상태를 평균한
값이 된다. 두 상태는 서로 다른 답을 가질 수 있으므로 pooling하지 않는다.

## 2. 폐기한 첫 채점과 원인

첫 실행은 readout 3,086행을 모두 `mcr_source_answers_{train,test}`의 no-note
답과 `base_id`만으로 조인했다. 그 결과:

- none readout × none answer는 맞는 조인이지만,
- wrong readout × none answer는 **다른 상태의 답**과 비교됐다.

따라서 그 실행의 다음 수치는 인용하지 않는다.

```text
all rows vs model .6361
deranged .0029
gap +.6332
source-correct 2,964/3,086
```

값이 높거나 통제가 낮은 것과 무관하게, 측정 대상이 섞였으므로 무효다. 이
오류는 `a21875e`에서 조인 키를 `(base_id, hint_variant)`로 바꾸고
`--variant`/`--by-variant`를 추가해 막았다.

## 3. 즉시 실행할 CPU 감사

wrong readout의 충실성 대상은 **wrong arm에서 모델이 실제로 낸 답**이다. 그
답은 개입 답 파일에 있으므로 `mcr_source_answers`가 아니라
`mcr_hint_answers_full_rescored.jsonl`을 사용한다.

```bash
python scripts/score_readout_against_model.py \
  --readouts "$ART/results/readout_mcr_hint_final_L32.jsonl" \
  --answers "$ART/results/mcr_hint_answers_full_rescored.jsonl" \
  --variant wrong \
  2>&1 | tee "$ART/reports/mcr_wrong_readout_faithfulness.txt"

python scripts/analyze_readout_grounding.py \
  --readouts "$ART/results/readout_mcr_hint_final_L32.jsonl" \
  --by-variant \
  2>&1 | tee "$ART/reports/mcr_readout_grounding_by_variant.txt"
```

### 필수 확인값

1. wrong 평가행이 1,543이고 `unjoined=0`인가.
2. wrong arm의 `vs model`, `vs gold`, `deranged`, case-specific gap은 얼마인가.
3. wrong arm에서 실제로 모델이 틀린 행이 몇 개인가.
4. 그 subset에서 readout이 `model only / gold only / both / neither` 중 어디에
   속하는가.
5. 근거 필드의 own-vs-other gap과 반복률이 none/wrong에서 각각 얼마인가.

이 재실행 전에는 “wrong-note 판독이 model answer를 따른다”거나 “gold를
추측한다”는 문장을 쓰지 않는다. 이전의 96% source-correct도 잘못된 no-note
조인의 산물이므로 corrected wrong-arm 결과를 미리 예측하는 근거가 아니다.

## 4. 판정 관문

| 결과 | 판정과 다음 단계 |
|---|---|
| wrong `vs model`이 deranged보다 명확히 높음 | 결론 필드는 r5 입력 후보로 통과 |
| wrong `vs model`과 deranged가 비슷함 | 사례별 결론 판독 실패; MCR r5 중단 |
| source-wrong에서 model-only가 gold-only보다 높음 | faithful narrator 해석 지지 |
| 둘이 비슷함 | case-specific signal은 있어도 model-vs-gold 방향은 미결 |
| 근거 gap이 작거나 반복률이 높음 | supporting cues는 실패로 유지; 결론과 분리 |

Derangement 통과는 “사례를 읽는다”는 뜻이지 “정확하다”거나 “임상적으로
타당하다”는 뜻이 아니다.

## 5. 교정 사다리 실행 순서

### 5.1 r3/r4 — 결론 판독과 독립

```bash
nohup env CUDA_VISIBLE_DEVICES=0,1 RUNGS="3 4" \
  bash scripts/run_mcr_ladder.sh \
  > "$ART/logs/mcr_ladder_r3_r4_launcher.log" 2>&1 &
```

- r3: 첫 답을 보여주고 단순 재고 요청
- r4: r3 + 원래 findings 재제시

### 5.2 r5 — arm-aware 결론 판독이 관문을 통과한 뒤

현재 구현의 r5는 `internal conclusion`과 `encoded findings`를 함께 넣는다.
그러나 첫 grounding 감사에서는 cue containment gap이 작고 반복 문장이 많았다.
따라서 최종 비교는 다음 둘로 나눈다.

| 조건 | 제공 내용 | 질문 |
|---|---|---|
| r5-conclusion | internal conclusion만 | 사례별 결론 자체가 교정에 유용한가 |
| r5-full | conclusion + encoded findings | 현재 판독 전체를 넣었을 때 순효과가 있는가 |

`r5-conclusion`을 만들기 위한 builder flag는 아직 구현 전이다. 이 통제 없이
기존 r5만 실행하면 결과는 “신뢰 가능한 결론 + 대부분 비접지일 수 있는 의료
산문”의 합성 효과이며, grounded explanation의 효과로 해석할 수 없다.

기존 full r5 실행 형식은 다음과 같다.

```bash
nohup env \
  CUDA_VISIBLE_DEVICES=0,1 \
  RUNGS="5" \
  READOUTS="$ART/results/readout_mcr_hint_final_L32.jsonl" \
  bash scripts/run_mcr_ladder.sh \
  > "$ART/logs/mcr_ladder_r5_launcher.log" 2>&1 &
```

### 5.3 분석

```bash
python scripts/analyze_correction_ladder.py \
  --rungs \
    "$ART/results/mcr_ladder_r3.jsonl" \
    "$ART/results/mcr_ladder_r4.jsonl" \
    "$ART/results/mcr_ladder_r5.jsonl" \
  --common-ids
```

주 비교는 r5-conclusion 대 r4이며, full r5는 근거 산문이 추가된 민감도
분석이다. MCR에서는 DDXPlus 49-class r6를 직접 만들지 않는다.

## 6. 논문에서 가능한 주장

- **현재 가능**: MCR에서 wrong-note 행동 효과가 재현됐다.
- **CPU 관문 통과 후 가능**: wrong-arm 마지막 토큰의 자연어 결론 판독이 해당
  사례의 wrong-arm 모델 답과 사례별로 대응한다.
- **r5 성공 후 가능**: 열린 진단 어휘에서도 내부 결론을 되먹이는 것이 단순
  재고/소견 재제시보다 moved 사례 회복에 유용하다.
- **계속 불가능**: supporting-cue grounding이 검증됐다; 자연어 형식 자체가
  교정 원인이다; 현재 판독이 임상의용 설명이다.

