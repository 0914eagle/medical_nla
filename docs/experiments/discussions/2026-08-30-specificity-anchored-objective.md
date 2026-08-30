# Specificity-anchored 2-target x 2-activation objective (D20 사전 등록 초안)

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

### 수치 동결 필요 (실행 전 확정, Codex 제안 요청)

5·6번의 비열등 허용 폭과 generation 4번의 claims 허용 폭은 **정성 문구로
남기면 사후 판단이 된다.** 기존 관례대로 control arm의 seed spread에 앵커한
수치(예: spread 상한 또는 고정 절대값)를 실행 전에 이 문서에 기입하고
동결한다. 실행 후 조정은 무효.

### 수치 산출 규칙과 실행 잠금

Codex 구현은 허용 폭 추천에 D16과 같은 규칙을 사용한다.

```text
allowance = max(2 x three-seed D10-control range, absolute floor)
```

| gate | absolute floor |
|---|---:|
| retained-gap delta upper bound | `.01` |
| changed-original content-NLL delta upper bound | `.05` |
| retained-original content-NLL delta upper bound | `.05` |
| mean-claim relative drop upper bound | `.10` |

마지막 값은 D10 control 세 seed를 동일한 validation pilot에서 greedy 생성한 뒤
`mean claims`의 seed range를 평균으로 나눈 relative range에 적용한다. Pilot은
기존 D16에서 동결한 validation 40-shard 중 shard `0/1/2/3`의 합집합이며,
`max_new_tokens=128`, `batch_size=4`, adapter에 기록된 actor prompt를 사용한다.
Locked test는 읽지 않는다.

이 표의 floor는 **최종 effective allowance가 아니다.** RunPod control artifact가
저장소에 없으므로 다음 read-only queue가 실제 세 seed 값, range, 추천 allowance,
입력 SHA256을 만든다. 그 출력 숫자를 이 문서와
`configs/experiments/ddxplus_d20_gate_protocol.json`에 사람 승인과 함께 커밋하기
전에는 trainer wrapper가 hard-fail한다.

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

현재 상태: **사람 방향 승인(제안 채택) / trainer·RunPod wrapper 구현 완료 /
control-spread 수치 동결 대기.** 비열등 허용 폭 수치가 기입·동결되면
실행을 연다. 실행 위치는 D10과 동일 조건 유지를 위해 RunPod A100 80GB를
우선한다(기존 checkpoint 재사용 호환).

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
