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

현재 상태: **사람 방향 승인(제안 채택) / 수치 동결 대기 / Codex 구현 검토
대기.** 비열등 허용 폭 수치가 기입·동결되고 Codex가 trainer 수정을 검토하면
실행을 연다. 실행 위치는 D10과 동일 조건 유지를 위해 RunPod A100 80GB를
우선한다(기존 checkpoint 재사용 호환).
