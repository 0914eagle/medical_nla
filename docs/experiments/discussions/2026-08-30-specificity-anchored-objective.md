# Specificity-anchored 2-target x 2-activation objective (D20 동결 규약)

## 질문

D10 budget calibration이 확정한 결함 — specificity가 평가 gate에만 있고 loss에
없어서 deleted-activation detector 퇴화 해가 자랐다 — 를 loss 수준에서 직접
차단하면, 생성형 판독이 changed cue를 **선택적으로** 반영하는가?

이것은 사람이 승인한 **마지막 생성형 시도**다. 종료 규칙(하단)이 함께
사전 등록된다.

## 배경 정정 두 개 (사람 검토 반영)

1. **SFT 학습량**: "SFT가 20-step이라 실패했다"는 반론은 성립하지 않는다
   (full-data SFT 전 데이터 학습, CF sequence SFT 높은 recall 도달, D10
   control이 single-claim SFT 1,552 steps 완주). 단 정확한 결론은 "epoch을
   더 줘도 절대 안 된다"가 아니라: **단순 CE/SFT는 충분한 예산에서 형식과
   평균적 의료 문장은 학습하지만 activation-case alignment를 자동으로
   학습하지 못했다.**
2. **Claude의 초기 anchor 항은 결함**: `|NLL(retained|h_del) −
   NLL(retained|h_orig)|`는 두 NLL을 함께 올려도 0이 된다("둘 다 못 읽기"
   편법). 차이가 아니라 절대 수준을 앵커해야 한다.

## Objective (동결 대상)

```text
L =
    CE(y_changed  | h_original)                       # 삭제 전에는 changed cue를 읽는다
  + alpha * CE(y_retained | h_original)               # 원본에서 retained cue를 읽는다
  + alpha * CE(y_retained | h_deleted)                # 삭제본에서도 retained cue를 계속 읽는다
  + lambda * T * softplus(
        (m - [ NLL(y_changed | h_deleted)
             - NLL(y_changed | h_original) ]) / T )   # 삭제 후에는 changed cue만 어려워진다
```

- **Detector 차단 기제**: h_del에서 전 문장을 억제하면
  `CE(y_retained | h_deleted)`가 커져 손해를 본다. Changed는 꺼리고
  retained는 유지하려면 어떤 cue가 삭제됐는지 **사례별로 읽어야만 한다.**
- **"둘 다 못 읽기" 차단 기제**: retained 항이 차이가 아니라 CE(절대
  수준)이므로 NLL을 함께 올리는 해가 벌점을 받는다.
- Hyperparameter 동결: **alpha=1, lambda=1, T=1, margin은 기존 D10과 동일.**
  Sweep 금지.

구현상 `margin=0`이다. Changed original의 첫 항은 기존 D10과 동일한 전체
supervised-token CE를 유지한다. 새 retained 두 항은 XML scaffold를 세 번
중복 학습하는 것을 피하기 위해 **claim content token CE**로 구현한다. Ranking도
기존처럼 content-token NLL을 쓴다. 이는 실행 전에 동결하며 scaffold CE를 넣는
별도 arm은 만들지 않는다.

## 데이터·비용

- **신규 데이터 구축 불필요**: D9a approved pairs는 R11에서 이미 retained
  control cue(`SHA256(base_id || NUL || cue_text)` 최소값, exact common cue)
  를 포함한다. Trainer에 CE 두 항 추가만 필요.
- Arms: original-only SFT와 unanchored ranking은 **기존 1,552-step
  checkpoint 재사용.** 신규 학습은 specificity-anchored arm seeds 17/29/43
  **3 runs뿐** (run당 ~43분 @ A100 80GB 실측 기준).
- Budget: **1,552 steps 고정** — "또 budget이 부족했다" 논쟁 차단.
  Checkpoint {194, 388, 776, 1164, 1552} report-only, 조기 선택·연장 금지.

## 편법 잔여분과 평가 분리

한 사례의 retained anchor가 하나이므로 "그 anchor만 보존하고 나머지 억제"
편법이 남는다. 평가에서 세 층을 분리해 잡는다:

| 층 | 내용 |
|---|---|
| Training anchor | 사전 고정 retained cue (학습에 사용) |
| Validation specificity cue | validation 사례의 retained cue — train/validation 사례가 분리돼 있어 동일 선택 규칙을 써도 사례 누출 없음 |
| Generation retention | **삭제되지 않은 모든 finding**의 보존율 — anchor-only 편법의 최종 검출기 |

## 동결 gate

### Teacher-forced (validation 3,032쌍, step 1,552에서만 판정)

1. Changed-gap delta(vs original-only control) `>= .05`, 세 seed 부호 일치
2. Changed-gap cluster CI 모두 `> 0`
3. Specificity delta 세 seed `> 0`
4. Specificity cluster CI 모두 `> 0`
5. Retained gap이 control 대비 **비열등** — 허용 폭은 아래 "수치 동결 필요"
6. Original arm의 changed/retained NLL이 control 대비 **악화 금지** — 허용
   폭은 아래 "수치 동결 필요"

### Generation (동일 checkpoint)

1. Original changed-cue hit 유지
2. Deleted phantom 감소
3. **전체** untouched-finding retention 유지 (anchor만이 아니라)
4. 출력 전체를 비우는 abstention/collapse 금지 — mean claims가 control 대비
   허용 폭 이내
5. Unsupported finding 증가 금지

### 비열등 수치 동결 결과 (사람 승인 완료)

5·6번과 generation 4번은 동일 seed의 frozen D10 control에 짝지어 비교한다.
2026-08-30 사람 승인으로 다음 수치를 동결했다. 실행 후 조정은 무효다.

| gate | 동결 기준 |
|---|---:|
| retained-gap anchored minus control | `<= +.01` |
| changed original content NLL | `<= 1.10 x` same-seed control |
| retained original content NLL | `<= 1.10 x` same-seed control |
| mean generated claims | `>= .90 x` same-seed control |

### Control calibration과 기각한 규칙

처음에는 D16과 같은 아래 규칙을 검토했다.

```text
allowance = max(2 x three-seed D10-control range, absolute floor)
```

실측 control은 다음과 같았다.

| seed | retained gap | changed original NLL | retained original NLL | mean claims |
|---:|---:|---:|---:|---:|
| 17 | -.027570 | .736734 | 1.034897 | 2.0 |
| 29 | -.032321 | .632297 | .895790 | 1.0 |
| 43 | -.000709 | .588965 | .895378 | 1.0 |

따라서 2x-range 규칙은 retained gap `+.063224`, changed NLL `+.295537`,
retained NLL `+.279039`, claim relative drop `1.50`을 허용한다. 특히 마지막
수치는 claim이 전부 사라져도 통과시키므로 비열등 gate로 기능하지 않는다.
이 규칙은 **기각**하고 위의 same-seed 상대 기준을 승인했다.

실행 시 적용되는 seed별 절대 상한은 다음과 같다.

| seed | changed original NLL max | retained original NLL max | mean claims min |
|---:|---:|---:|---:|
| 17 | .810407 | 1.138387 | 1.8 |
| 29 | .695527 | .985369 | .9 |
| 43 | .647862 | .984915 | .9 |

Calibration generation은 validation 40-shard 중 `0/1/2/3` 합집합 952행,
`max_new_tokens=128`, `batch_size=4`, greedy decoding, adapter actor prompt를
사용했다. Locked test는 읽지 않았다. 입력 hash와 기준은
`configs/ddxplus_d20_gate_protocol.json`에 동결한다. 승인 시점 protocol
SHA256은 `3f85371f1185e3d463d1acb25a16351392b14dcb927761145dc9f5c671c09eeb`다.

```bash
cd /home/eagle0914/medical_nla
git pull origin main

nohup env \
  REPO_DIR=/home/eagle0914/medical_nla \
  GPU_A=0 GPU_B=1 \
  bash scripts/run_ddxplus_d20_control_calibration_runpod.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d20_control_calibration_v1.log 2>&1 &
```

한 GPU pod에서는 `GPU_A=0 GPU_B=0`으로 실행한다. 완료 후 확인할 파일은
`/data1/heejae/restricted/direct/e4/ddxplus_d20_control_calibration_v1/summary.md`다.
이 단계는 control 생성과 숫자 추천만 하며 D20 학습을 시작하지 않는다.

승인 protocol이 main에 들어간 뒤 D20 본 실행은 다음 한 명령이다.

```bash
nohup env \
  REPO_DIR=/home/eagle0914/medical_nla \
  GPU_A=0 GPU_B=1 \
  bash scripts/run_ddxplus_d20_specificity_anchor_runpod.sh \
  > /data1/heejae/medical_nla/logs/ddxplus_d20_specificity_anchor1552_v1.log 2>&1 &
```

Wrapper는 anchored seed 17/29를 병렬 실행한 뒤 seed 43을 실행한다. 기존 D10
control checkpoint와 score를 재사용하며 새 control/ranking arm은 학습하지 않는다.
Checkpoint `194/388/776/1164`는 report-only이고 `1552`만 판정 가능하다.

## 종료 규칙 (사전 등록)

1. 이 실험은 **한 번**이다. 실패 시 alpha/lambda/margin/step sweep 금지.
2. 실패 시 offline preference로 **자동 이동하지 않는다** — 별도 사람 결정
   없이는 생성형 시도를 추가하지 않고, 정직한 결론 조항(프로그램 결정
   문서 선택지 A)으로 간다.
3. 성공 시에만 validation generation gate → (통과 시) Gate C / locked
   순서로 진행하며, 각 단계는 기존 동결 규칙을 따른다.
4. 실패의 논문 가치: "평가 gate뿐 아니라 loss 수준의 직접적 specificity
   anchor로도 생성형 판독이 안정화되지 않았다" — 기존 7건보다 강한 음성
   결과.

## 판정

현재 상태: **실행 완료 — 최종 판정 FAIL (아래 결과 절).** 실행 위치는 D10과
동일 조건 유지를 위해 RunPod A100 80GB를
사용했다(기존 checkpoint 재사용 호환).

## 결과와 판정 (2026-08-30, RunPod A100 80GB 실행 완료)

3 seed × 1,552 steps 완주. 판정 지점 step 1,552의 same-seed paired delta
(anchored − frozen D10 control, validation 3,032쌍):

| seed | changed gap | retained gap | specificity | changed orig NLL | retained orig NLL |
|---:|---:|---:|---:|---:|---:|
| 17 | −.0143 | +.0135 | −.0278 | −.0756 | −.3342 |
| 29 | −.0040 | +.0215 | −.0255 | +.0576 | −.1834 |
| 43 | −.0266 | −.0049 | −.0217 | +.0622 | −.2263 |

동결 gate 판정: changed-gap ≥ +.05 **fail**(3 seed 모두 음수), changed
cluster CI **fail**, specificity 부호 일치 **fail**(3 seed 모두 음수),
specificity CI **fail**, retained-gap 비열등(≤ +.01) **fail**(seed 17/29가
+.0135/+.0215로 초과), changed-original NLL 비열등 **fail**(seed 43이
same-seed 절대 상한 초과), retained-original NLL 비열등 **pass**.
**Teacher-forced 최종 판정: FAIL.** Wrapper는 사전 등록대로
`[stop] FAIL: no generation, extension, checkpoint selection, or sweep`으로
종료했고 generation gate는 열지 않았다.

### 해석 세 가지

1. **Anchor의 detector 차단은 작동했다.** Retained-gap delta는 전 구간
   |값| ≤ .0225로 유지됐다 — budget run의 동일 step retained-gap delta
   +.5604와 대비된다. h_del 전 문장 억제 해는 이번 손실 아래에서 자라지
   않았다. 설계된 차단 기제가 목적대로 동작했다는 뜻이다.
2. **편법을 막자 남는 학습 가능 신호가 없었다.** Changed-gap delta는 어느
   dose에서도 +.05 근처에 가지 않았다(step 194–776 ±.009 이내, step 1164
   일시적 +.0225/+.0360도 seed 17이 −.0197로 부호 불일치·CI 0 포함, step
   1,552에서 3 seed 모두 음수로 역전). 이는 budget run의 changed-gap 성장
   (+.5558)이 **전부 detector 편법이었다**는 것의 독립적 확인이다: 편법
   경로를 잠그자 성장 자체가 사라졌다.
3. **모델이 학습을 안 한 것이 아니라 anchor만 학습했다.** Retained original
   NLL은 −.13~−.33으로 크게 개선됐다(retained CE 항의 직접 효과). 최적화는
   정상 동작했고, 사례별 changed-cue 선택이라는 목표 신호만 잡히지 않았다.

### 사전 반론 기재 — "step 1164에서 멈췄어야 한다"

Step 1164의 양수 transient는 (a) seed 17이 음수로 부호 불일치, (b) cluster
CI가 0을 포함, (c) 388 step 뒤 3 seed 모두 음수로 역전이라는 세 가지
이유로 신호가 아니다. Checkpoint 선택 금지는 실행 전 동결됐고, 이 역전
자체가 그 규칙의 근거를 사후 입증한다.

### 종료 규칙 적용

사전 등록 종료 규칙에 따라: alpha/lambda/margin/step **sweep 없음**,
offline preference로 **자동 이동 없음**. 프로그램은 정직한 결론 조항
(프로그램 결정 문서 선택지 A)으로 간다. 별도 생성형 시도는 사람의 새
결정 없이 추가하지 않는다.

논문 기록 가치: 평가 gate 우회(detector)를 loss 수준에서 차단한 상태에서도
생성형 판독의 사례별 선택 신호가 나타나지 않았다 — 기존 7건 실패에 더해
"편법 제거 후 신호 부재"를 보인 **여덟 번째이자 가장 강한 음성 결과**이며,
budget run 해석(성장 = 전부 편법)의 독립 확인을 겸한다.

### Ledger 행 제안 (사람 승인 대기)

- **D19**: D10 budget calibration(1,552 steps) FAIL — detector 퇴화 해 확정,
  1×2 unanchored 계열 종결.
- **D21**: D20 specificity-anchored objective FAIL — detector 차단 성공,
  changed-gap 신호 부재, 생성형 시도 종료(선택지 A), Medical-NLA 행은
  사전 규칙에 따라 논문 주표에서 제외.

두 행이 승인되면 decision record와 recipe hash를 동결하고 DiReCT locked
batch를 연다.

## 예상 반론 사전 기재 — "데이터 다양성 부족이 원인이다"

부분 인정: DDXPlus는 템플릿 조립이라 cue 표현 다양성이 낮고, 관찰된 실패
양상(전형 문장 생성)이 템플릿 암기와 겹친다. 그러나 세 개의 통제된 대조가
다양성을 binding constraint로 보기 어렵게 만든다:

1. **같은 데이터, closed decoder 성공**: 선형 probe가 동일 activation·동일
   데이터에서 사례 특이성(own-shuffled gap +.1624, deletion delta .79)을
   학습했다. 데이터를 고정하고 decoder만 바꿨을 때 성패가 갈린다.
2. **다양성 최소의 reader가 최고 성적**: structured reader는 evidence당
   표현 1개로 F1 .9587 — 과제의 핵심(내용 선택)은 언어 다양성을 요구하지
   않으며, 실패한 것은 표현이 아니라 선택이다.
3. **양쪽 코너 모두 실패**: 양 많음/다양성 낮음(DDXPlus)과 다양성 높음/양
   적음(DiReCT 248행, Obscomp .03) 모두에서 CE 계열이 실패했다. Detector
   편법은 데이터 다양성과 무관한 최적화 병리이며, 동일 데이터에서 seed별
   해 발산은 데이터가 해를 제약하지 않았다는 신호다.

인정하는 한계: 양과 다양성을 동시에 키운 (activation, 충실한 서술) 쌍
corpus는 시험되지 않았고 존재하지도 않는다 — 그 부재 자체가 이 분야의
병목이라는 것이 논문의 논점이다. 최종 주장은 "이 데이터 regime + 이
objective들 + 이 backbone에서"로 조건화하고 limitations에 선제 기재한다.
