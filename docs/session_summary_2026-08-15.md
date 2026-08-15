# 세션 정리 2026-08-15: 통제 실험 결과와 방향 재정립

이 문서는 2026-08-13~15 세션에서 잰 수치와 의논한 결론의 통합 기록이다.
상세 실험 기록: `docs/results_2026-08-13_ood_and_probe_controls.md`,
배경: `docs/session_handoff_2026-08-01.md`.

## 1. 수치 총정리

### 기존 결과 (handoff 기준, 재현 확인됨)

| 항목 | 수치 |
|---|---|
| Linear probe (all-cue format, layer 32) | acc1 99.17%, acc5 100% |
| Medical-AV v2 (e3_b2) answer_hit | 86.96% (920/1058) |
| Medical-AV v2 mean_cue_recall | 0.7994 |
| source-correct 케이스에서 AV 정답률 | 97.07% (265/273) |
| source-wrong 케이스에서 AV 정답률 | 83.44% (655/785) |
| source/AV 불일치 → source 오답률 | 98.90% |
| source/AV 불일치 AUROC / AP | 0.9427 / 0.9708 |
| source confidence baseline AUROC | 0.67~0.70 |
| NLA-only sampling 불안정성 AUROC | ~0.56 |

### 실험 A: Probe 불일치 대조군 (2026-08-13)

v2 test셋(n=1058, 오답률 74.2%)에서 source/probe 불일치를 source/NLA
불일치와 비교. probe 예측은 probe 자체 test split과의 교집합 152행.

| 신호 | n | AUROC | AP |
|---|---:|---:|---:|
| source_nla_disagree | 1058 | 0.9427 | 0.9708 |
| source_probe_disagree | 152 | 1.0000 | 1.0000 |
| probe_low_top1_prob | 152 | 0.6102 | 0.8231 |

Paired 비교 (둘 다 정의된 152행):

- nla_disagree_auroc = 0.9282
- probe_disagree_auroc = 1.0000
- **nla_minus_probe_auroc = -0.0718 (probe 승)**
- probe binary rule: tp/fp/tn/fn = 118/0/34/0 (완벽)

주의: paired n=152로 작음. 엄밀 버전은 probe를 v2 train행만으로 재학습해
1058행 전체 예측 필요 (방향이 바뀔 가능성은 낮음 — probe가 99%인 이상).

### 실험 B: Diagnosis-heldout OOD (2026-08-13)

분할 `medical_nla_diagnosis_heldout_v1`: train 18클래스 / heldout 8클래스
(seed 17). train 884 / val 189 (source-correct만) / test_seen 727 /
test_heldout 800. 어댑터 `medical_nla_diagnosis_heldout_v1_lora_e3`
(3 epoch, val_loss 0.0396 — 학습 자체는 정상).

| pool | n | answer_hit | mean_cue_recall | mean_output_cue_recall |
|---|---:|---:|---:|---:|
| test_seen | 727 | 0.9037 | 0.7690 | 0.7690 |
| test_heldout | 800 | **0.0000** | 0.3066 | 0.3066 |

- heldout 답의 **98.75% (790/800)가 train 18개 클래스 이름** (분류기 붕괴)
- 붕괴는 불균일: `urti→Bronchitis` 98% (cue_recall 0.71, 임상적 인접) ~
  `sle→Scombroid` 96%, `pulmonary_neoplasm→Anemia` 100% (무의미)
- 운영 메모: 최초 test_seen readout이 동시 실행으로 중복(1419행, 유니크
  727) → dedup 후 재채점한 수치가 위 표

### 실험 C: Vanilla AV 대조군 (2026-08-15)

같은 heldout 800행, 어댑터 없이. 전 지표 0.0000, cjk_fraction 0.0
(injection 정상). 출력은 형식 서술("Structured medical Q&A format
signals a clinical diagnosis response ...")뿐, 진단 시도 없음. 얕은 cue
반향은 간혹 있으나(예: "chest pain, fever, GI symptoms") DDXPlus cue
원문과 어휘가 달라 매칭 안 됨 → 0.0은 "진단 통합 없음"으로 읽되
"내용 인지 0"으로 과장하지 말 것. 2026-07 specificity 파일럿(형식 위치
진단 언급 3/100)과 일치.

## 2. 종합 결론

```text
같은 활성값 (layer 32, all-cue format 위치):
  linear probe : 99%  → 진단 정보는 선형적으로 존재
  vanilla AV   : 형식 서술만, 임상 내용 통합 0
  LoRA AV      : train 클래스 라벨로 collapse (클래스 수준 암기)
  오류 탐지    : probe가 NLA와 동등 이상 → NLA 고유 기여 아님
```

**살아남은 주장**
- layer-32 활성값에 강한 선형 진단 정보가 있다 (probe 99.17%)
- source 오답의 다수는 정보 부재가 아니라 디코딩 실패다
- 불일치 신호는 오류를 탐지한다 — 단, probe로도 되므로 NLA 기여가 아님

**죽은 주장 (레시피 수정 전까지)**
- "Medical-AV가 활성값을 의미적으로 읽는다" (heldout 0%)
- "answer_hit 86.96%는 readout의 증거다" (in-distribution 분류의 증거임)

**v1의 정체**: 케이스 암기는 아니고(**seen 새 케이스 90.4%** 일반화)
**클래스 수준 암기**다. 활성값 클러스터 18개 → 외운 라벨 + 외운 전형
cue 텍스트. heldout cue_recall 0.31은 인접 seen 클래스의 암기된 전형
cue가 우연히 겹친 것으로 해석.

## 3. 의논한 핵심 논점들

### 3.1 목표 계층 (오답노트에 매몰되지 않기)

```text
궁극 목표: 의료 LLM이 왜 틀리는지를 내부 상태로부터 이해하고,
           오류를 예측·교정하는 것 (교수님 3축: 설명/진단/교정)
핵심 수단: 활성값을 "근거 있게" 자연어로 판독하는 도구
응용 예시: 오답노트, layer-wise 궤적, 오류 경보, 재고 유도 교정
```

오답노트는 응용의 한 형태일 뿐 목표가 아니다. 현재 병목은 "판독 도구가
진짜냐" 하나.

### 3.2 우리 기여의 정의 (probe/SAE 대비)

- probe: "클래스 7, 확률 0.93" — 미리 정한 라벨, '왜' 없음
- SAE: "feature #4821 활성" — feature 해석은 사람 몫, layer별 사전
- NLA: 조합·불확실성·갈등을 열린 어휘의 문장으로 서술 — 유일하게 "서술" 가능

단, "자연어라 직관적"만으론 부족. **활성값에 근거함(grounded)**까지
가야 NLA만의 기여다. 그럴듯함과 진짜는 다르다 (v1이 그 함정이었음).

### 3.3 왜 cue-first 타깃인가

"읽는다" = 벡터 없이는 못 맞히는 내용을 출력한다.

```text
진단명: 26지선다 → 클러스터 암기로 뚫림 → 읽기의 증거 못 됨
cue 조합: 케이스마다 다른 3~12개 조합 → class memorization만으로는
          맞히기 어려운 고엔트로피 supervision
```

단, cue도 완전한 암기 방지는 아님 — "cluster로 seen class 판별 → 그
클래스의 전형 cue 뿌리기" 우회가 남아 있고(v1이 실제로 간 길), 그래서
recall 단독이 아니라 precision + mismatched + counterfactual의 4-기준이
필요하다. cue는 목적이 아니라 activation-grounded readout을 검증하기
위한 supervision이자, 오답노트류 응용의 '근거' 칸 재료.

### 3.4 마지막 토큰에 cue가 있다는 가정인가? — 아니다, 가설이다

- 가능성 A: 마지막 토큰에 cue들이 개별적으로 잔존 → cue-first 학습 성공
- 가능성 B: 모델이 cue를 결론으로 압축, 개별 cue 소실 → 아무리 학습해도 실패

probe 99%는 A/B를 구분 못 한다 (B여도 진단 분류는 됨). 실험이 이걸
판정하며, B로 나오면 cue 정보가 확실한 위치(cue 토큰 자리, §7.2 entity
실험)나 더 이른 layer로 이동. "통합 지점에 근거가 남는가, 결론만
남는가"는 그 자체로 발견이자 layer-wise 서사의 첫 데이터.

### 3.5 AV-only 유지와 AR의 역할

- 다음 런도 학습은 AV만. full NLA(joint)로 점프하지 않는 이유:
  충실하게 만들 대상(의미 있는 readout)이 먼저 존재해야 함.
  reconstruction-only는 "형식 서술" 수렴 위험 (§21.1, vanilla가 그 상태).
- AR은 다음 런부터 **채점관**으로 투입 (post-hoc), 학습은 조건부.
- 끝까지 AV-only + AR-심판으로 남아도 목표는 성립 — full NLA는 목표가
  아니라 grounding이 안 지켜질 때 꺼내는 수단.

### 3.6 AR cos-sim 교란 (중요 수정, 사용자 지적)

AV를 파인튜닝해 출력 문체가 바뀌면, vanilla 분포로 학습된 frozen AR의
cos-sim은 (a) 근거 없음과 (b) 문체 이탈을 구분 못 한다. 절대값 판정 불가.

보정:
1. **matched vs mismatched 대조**: score(내 텍스트, 내 활성값) vs
   score(내 텍스트, 남의 활성값). 짝 구분 못 하면 AR 채점 자체를 무효 처리
2. 타깃 문체를 vanilla 설명문에 가깝게 (분포 이탈 축소)
3. **faithfulness 주 증거를 intervention으로 승격**: cue-제거
   counterfactual, activation patching은 AR·문체와 무관하게 성립
4. AR을 진지하게 쓰려면 우리 문체로 AR 적응이 필요할 수 있음 →
   AR 학습 시점이 예상보다 앞당겨질 수 있음

### 3.7 Layer-wise 아키텍처 (heldout 관문 통과 후의 이야기)

- layer별 독립 NLA: 판독기간 차이가 궤적과 섞임. baseline으로만 필요
- 단일 모델 + layer 토큰만: layer별 활성 통계(norm/스케일) 불일치를
  프롬프트가 해결 못 함 → 실패 확률 높음
- **공유 디코더 + layer별 경량 프로젝션 (+ layer 토큰): 추천.**
  분포 문제는 프로젝션이, 의미 공간은 공유 디코더가 분담
- 순서: layer별 probe → 정보 있는 2~3 layer 독립 LoRA(baseline) →
  공유+프로젝션 비교. **단, v1이 분류기로 판명됐으므로 판독 도구가
  생기기 전에는 layer-wise 확장 금지** (layer별 분류기 모음이 될 뿐)

## 4. 다음 런 스펙 (medical_nla_v3, cue-first)

입력: 지금과 동일 (layer-32 활성값 1개 주입).

출력 타깃 (2026-08-15 수정: 첫 v3는 assessment 없음):

```text
<explanation>
<readout>
<observed>
- moderate fever
- sore throat
- nasal congestion
- cough
</observed>
</readout>
</explanation>
```

- `<observed>` = 유일한 타깃·채점 대상. 케이스별 cue 원문(순서 셔플, max 12)
- **진단명은 타깃에서 완전 제거** — assessment에 진단명을 넣으면 `<answer>`
  슬롯을 없앤 의미가 반감되는 label shortcut 재유입 (`--include-assessment`
  플래그로 후속 버전에서만 복원)
- 문체는 3.6에 따라 vanilla 설명문 쪽으로 조정 검토

판정 — 4-기준 관문 (2026-08-15 확정):

```text
1. heldout cue recall     — 읽는 양. 암기 수준 0.31을 유의미하게 초과해야 함
2. heldout cue precision  — 뿌리기 방지. recall만 높고 precision 낮으면
                            "cue를 많이 뿌리는 모델"일 뿐 (채점기에 추가됨)
3. mismatched 하락        — score(내 텍스트, 내 활성값) vs (내 텍스트, 남의 활성값).
                            hard negative는 반드시 "같은 진단의 다른 케이스" 포함
                            (cross-class만 쓰면 분류기도 통과함). AR 절대값 사용 금지
4. counterfactual         — cue 제거 후: 제거 cue가 readout에서 소실(민감성)
                            + 유지 cue는 보존(특이성) 둘 다 성립해야 함.
                            진단 변화는 1차 판정에서 제외 (프롬프트 교란과 얽힘)

1·2는 v3 readout 채점에서 바로 나옴. 3·4 도구는 1·2가 암기 수준을
넘었을 때만 제작 (실패 시 위치/layer 이동이 먼저).
```

cue-first의 위상 (합의된 표현): cue는 최종 목표가 아니라 activation-grounded
readout을 검증하기 위한 고엔트로피 supervision이다. 진단명은 class
memorization으로 맞힐 수 있지만, case-specific cue 조합은 activation을 읽지
않으면 맞히기 어렵다. v3의 목적은 진단 정확도 향상이 아니라 "activation에
남은 clinical content를 읽을 수 있는가"의 검증이며, 실패해도 "layer-32
format 위치는 진단 클래스 신호는 강하나 개별 임상 근거를 보존하지 않는다"는
layer-wise 연구의 출발점이 된다.

병행 가능: correction 실험(§24.5)은 v1 분류기-readout으로도 유효
(in-distribution 정확도 높음) — "교정" 축의 보험.

## 5. 산출물 위치

코드 (branch `claude/work-start-997ix6`):

- `scripts/make_medical_nla_diagnosis_heldout_splits.py` — OOD 분할
- `scripts/summarize_diagnosis_heldout_readouts.py` — seen/heldout 비교 +
  classifier-collapse 체크
- `scripts/make_error_prediction_table.py` — `source_probe_answer_agree` 추가
- `scripts/evaluate_error_prediction.py` — probe 대조 신호 + paired 비교
- `scripts/score_medical_nla_v2_readouts.py` — `output_cue_recall` 추가
- `EXPERIMENTS.md` §8~9 — 서버 runbook

서버 결과물:

- `/data1/heejae/medical_nla/train/medical_nla_diagnosis_heldout_v1/` — 분할
- `/data1/heejae/medical_nla/adapters/medical_nla_diagnosis_heldout_v1_lora_e3`
- `/data1/heejae/medical_nla/results/ddxplus_medical_nla_diagnosis_heldout_v1_*`
- `/data1/heejae/medical_nla/results/error_prediction_probe_control_v1_*`
