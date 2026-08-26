# Experimental design and population audit

기준일: 2026-08-26. 이 문서는 현재 논문 설계를 코드와 대조한 감사 기록이다.
과거 wrong-note pilot은 설계 근거로 참고할 수 있지만 현재 표의 모집단이나 결과로
재사용하지 않는다.

## 1. 결론부터

연구 방향 자체는 유지할 수 있다.

1. DiReCT에서 CoT, vanilla NLA, Medical-NLA의 임상 정보 복원을 비교한다.
2. DDXPlus에서 판독 문장이 해당 activation에 사례 특이적으로 근거하는지 검증한다.
3. 앞의 두 관문을 통과한 판독만 text patching에 사용한다.

다만 현재 DiReCT 171행 결과는 이미 P0/P1/P2와 vanilla AV 분석에 사용됐다. 따라서 이
171행을 앞으로의 최종 confirmatory test라고 부르면 안 된다. 현재 결과는
`exploratory pilot`으로 동결했다. Downstream Medical-NLA용 새 split은
266/52/72/106으로 동결했지만, 과거 source artifact와의 overlap을 집계하기 전에는
`dataset-level untouched`라고 부르지 않는다.

## 2. 현재 모집단

| 단계 | n | 정의 | 사용 가능 범위 |
|---|---:|---|---|
| Raw DiReCT | 511 | restricted release 전체 | dataset audit |
| Eligible | 496 | label conflict 10, patient parse 실패 4, duplicate copy 1 제외 | split universe |
| Train | 263 | patient-disjoint seen-PDD | 학습 |
| Validation | 62 | patient-disjoint seen-PDD | layer, regularization, threshold 선택 |
| Exploratory test-seen | 71 | 현재 이미 분석한 seen-PDD 사례 | 탐색 결과만 |
| Exploratory PDD-heldout | 100 | 5개 PDD, 현재 이미 분석함 | 탐색 결과만 |

현재 held-out PDD는 HFrEF 29, NSTEMI 28, Low-risk PE 21, HFpEF 19,
Non-Allergic Asthma 3이다. HFrEF와 HFpEF는 같은 환자 연결 성분이어서 함께 holdout됐다.

### Split 코드가 보장하는 것

- 동일 patient group은 여러 split에 들어가지 않는다.
- 동일 input digest의 duplicate copy는 하나만 남긴다.
- unresolved label과 patient ID를 파싱하지 못한 행은 제외한다.
- held-out PDD connected component는 train에 나타나지 않는다.
- seen PDD는 최소 한 번 train에 나타난다.

### 아직 보장하지 않는 것

- 현재 171행이 사람의 설계 변경으로부터 untouched라는 보장은 이미 사라졌다.
- PDD/category 분포가 train, validation, test에서 동일하다는 보장은 없다.
- 작은 held-out PDD의 per-class 추정치는 안정적이지 않다.
- source-correct subset을 학습 모집단으로 사용할 경우 충분한 표본이 된다는 보장은 없다.
- raw note에서 gold PDD/root 정규화 구문이 직접 등장한 비율은 28/511(5.48%)다. Eligible 및
  새 confirmatory split별 비율은 split 생성 summary에서 동결한다.

## 3. Confirmatory protocol

현재 171행은 파일과 수치를 그대로 보존하되 논문에서 pilot로 표시한다. 다음 권장안 A를
선택해 E3 학습 전에 ID hash와 설정을 기록했다.

### 선택안 A: 새 label-heldout downstream-confirmatory split

현재 pilot에서 holdout하지 않은 PDD connected component를 새 confirmatory holdout으로
선택한다. 새 split으로 모든 모델을 처음부터 학습하고 다음을 금지한다.

- test output 원문 열람
- test 기반 prompt, layer, epoch, threshold 변경
- parse 실패 행 삭제
- source-correct 여부를 이용한 test eligibility 변경

장점은 주표가 단순하다는 점이고, 단점은 DiReCT가 작아 train이 더 줄 수 있다는 점이다.

동결 결과는 train 266, val-seen 52, test-seen 72, PDD-heldout 106이다. Held-out은
12 PDD, 10 disease categories이며 pilot-heldout 5 PDD component와 겹치지 않는다.
Logical population SHA-256은
`7d0a89a880fa868959099b7146c369cccaac5e7701d7ce5d8f01356ecfb68894`다.
Split별 exact gold-label-in-note는 18/266, 2/52, 3/72, 5/106이다.
목표 20%는 99.2행이지만 PDD component와 category coverage를 쪼갤 수 없어 106행
(21.4%)에서 멈췄다. 이는 결과를 본 사후 표본 조정이 아니라 split 알고리즘의 원자 단위
제약이다.

단, 새 106행이 과거 pilot split의 train/validation/seen test에 포함됐을 수 있고, source
output이 이미 materialize됐을 수 있다. 이는 E3 이후 downstream 선택을 지금부터 동결하는
용도에는 사용할 수 있지만, 데이터셋 수준의 pristine external test라는 주장은 막는다.
`audit_direct_confirmatory_exposure.py`로 실제 artifact overlap을 기록한다.

### 보류안 B: nested patient/PDD group evaluation

외부 fold는 patient group 또는 PDD connected component로 만들고, 각 외부 fold 안에서
validation을 다시 나눠 hyperparameter를 선택한다. 평균과 fold-level interval을 보고한다.
표본을 더 효율적으로 쓰지만 GPU 비용과 구현 복잡도가 커진다.

어느 안을 택하든 현재 171행에서 이미 관찰한 결과를 최종 test 결과와 섞지 않는다.

## 4. 표별 모집단과 분모

### Table 1A: backbone behavior

- 동일 case ID의 Direct와 CoT만 paired comparison한다.
- strict PDD, disease category, official semantic diagnosis를 분리한다.
- parse 실패는 분모에서 빼지 않고 failure로 센 뒤 parse coverage를 함께 적는다.
- Direct와 CoT는 instruction과 assistant prefill이 다르므로 같은 생성 방식이라고 부르지 않는다.
- paired interval과 검정은 row가 아니라 `patient_group`을 cluster 단위로 resample한다.

### Table 1B: internal readout capability

- 같은 CoT-P0 activation과 같은 case ID를 output head, probe, NLA에 제공한다.
- supervised PDD probe는 학습에 없던 PDD를 출력할 수 없으므로 PDD-heldout 칸은 `N/A`다.
- category probe는 held-out PDD의 category가 train에 존재할 때만 평가할 수 있다.
- probe의 open-text explanation은 `N/A`이며 0점으로 평균하지 않는다.
- source answer fidelity와 physician gold alignment를 서로 다른 열로 둔다.

### Table 2: DiReCT clinical alignment

- 모든 방법이 동일한 confirmatory case ID를 사용한다.
- free text를 official schema로 바꾸는 claim extractor는 모든 방법에 동일하게 적용한다.
- extractor는 method 이름, gold annotation, 원 임상 note를 보지 않고 method output만 본다.
- 추출 실패를 삭제하지 않고 coverage와 함께 failure로 처리한다.
- official Llama-3-8B matcher는 semantic matching 도구이지 독립 faithfulness judge가 아니다.
- 동일 환자의 반복 note를 독립 표본으로 세지 않도록 CI는 patient-cluster bootstrap을 쓴다.

### Table 3: DDXPlus activation grounding

- own pair와 donor pair는 같은 diagnosis, 비슷한 cue count/길이로 맞춘다.
- donor는 source answer나 같은 evidence value까지 같은 행이면 안 된다.
- cue deletion/edit은 같은 base case의 paired counterfactual로 만든다.
- 평균 점수만 아니라 paired effect와 bootstrap CI를 보고한다.
- zero, mean, shuffled activation은 NLA language prior 바닥을 측정한다.

### Table 4: text patching

- Table 3을 통과한 method만 주 실험에 넣는다.
- identity round-trip이 원 answer와 non-target distribution을 먼저 보존해야 한다.
- patch 대상 case를 gold correctness로 사후 선택하지 않는다.
- oracle selection은 현실적 policy가 아니라 upper bound로 명시한다.
- wrong-to-right와 right-to-wrong, net correction, intervention rate를 모두 보고한다.

## 5. 학습 target alignment

DiReCT의 physician deduction을 P0 activation에 곧바로 붙이면 source-wrong 행에서
`activation의 현재 상태 -> gold 결론`이 되어 판독기가 아니라 context-to-gold solver를
학습할 수 있다. target field와 loss를 분리해야 한다.

| Target | 사용 행 | 의미 |
|---|---|---|
| Observation reconstruction | 모든 train 행 | P0가 note에서 보존한 관찰 복원 |
| Source-decision diagnosis | source answer가 파싱된 모든 행 | activation이 향한 실제 모델 결론 판독 |
| Gold diagnosis/rationale | source-correct 행 또는 별도 auxiliary | physician alignment, state fidelity와 구분 |
| Reconstruction/pair loss | 모든 유효 activation | activation 무시 방지 |

source-correct만으로 전체 임상 target을 학습하면 strict PDD 기준 표본이 매우 작아질 수
있다. 먼저 train의 strict PDD/category/official semantic correct 수를 집계하고, field별
loss mask와 case weighting을 고정한다. 한 note의 deduction 수가 많다고 그 환자의 loss가
과도하게 커지지 않도록 note-level normalization을 사용한다.

## 6. P0/P1/P2와 layer 명칭

- P0는 단순 note 끝이 아니라 CoT instruction까지 포함한 user prompt의 마지막 토큰이다.
- P1/P2는 실제 CoT response를 teacher-force한 상태다.
- P1은 모델 answer alias가 이미 reasoning에 나온 행을 제외하면 현재 pilot에서 15행뿐이다.
- P2는 answer-exposed positive control이며 주 faithfulness 위치가 아니다.
- 추출 코드는 `outputs.hidden_states[16/24/32]`를 저장한다.

공개 AV sidecar는 `extraction_layer_index: 32`를 명시하므로 현재 index 32 입력은 공개
checkpoint와 맞는다. 다만 논문에서는 `block L32`보다 `hidden-state extraction index 32`
또는 `HS32`라고 써서 embedding을 포함하는 tuple convention을 명확히 한다.

공개 AV와 AR은 index 32용이다. HS16/HS24를 같은 checkpoint에 넣는 결과는 layer 정보량과
decoder distribution shift가 섞인 sensitivity analysis다. Primary Medical-NLA와 round-trip,
patching은 HS32로 고정한다. 다른 index를 primary로 선택하려면 해당 index용 AV와 AR를 같은
recipe로 다시 학습해야 한다.

## 7. Vanilla NLA baseline의 공정성

공개 AV의 기본 prompt는 일반 activation을 2-3개 snippet으로 설명하라고 요청한다.
CoT는 임상 추론과 최종 진단을 명시적으로 요청한다. 따라서 default vanilla AV가 진단명을
말하지 않은 결과만으로 activation에 진단 정보가 없다고 결론 내릴 수 없다.

최소 baseline은 다음과 같다.

1. 공개 default AV prompt
2. validation에서 고정한 task-aligned generic AV suffix, medical label supervision 없음
3. zero/mean/shuffled activation under the same AV prompt
4. output head와 linear probe

Medical-NLA의 이득은 2번보다 높아야 의료 supervision의 기여로 해석할 수 있다.

## 8. 현재 결과의 안전한 해석

- Direct와 CoT의 strict PDD/category 차이는 유의하지 않았다. 이것은 설명 품질 비교가 아니다.
- P0 vanilla AV의 진단 phrase recovery 0은 default AV task failure다. P0 정보 부재의 증거가 아니다.
- P1 전체의 높은 source-answer mention은 대부분 CoT 문자열 누출이다.
- P2 own-donor gap은 AV가 answer-exposed state에서 사례별 답 신호를 일부 읽는 positive control이다.
- Table 2 official metrics가 높아도 Table 3을 통과하기 전에는 faithful activation reader라고 부르지 않는다.

## 9. 실행 전 필수 체크리스트

- [x] downstream-confirmatory protocol과 ID hash 동결
- [ ] confirmatory heldout과 과거 materialized artifact overlap 집계
- [ ] train/val/test patient 및 PDD 교집합 0 재검사
- [ ] DiReCT raw leakage 28/511 확인 완료. Confirmatory split별 비율과 sensitivity cohort 동결
- [ ] 각 표의 expected IDs 파일 생성
- [ ] 모든 method output의 join rate 100% 또는 missing-as-failure 적용
- [ ] Direct/CoT/NLA 실제 prompt와 generation config 저장
- [ ] hidden-state extraction index와 NLA sidecar index 일치 검사
- [ ] common claim extractor의 oracle, empty, adversarial smoke
- [ ] source-correct/wrong을 subgroup으로만 사용하고 primary denominator를 바꾸지 않음
- [ ] layer, prompt suffix, LoRA 설정은 validation에서만 선택
- [ ] confirmatory output을 보기 전에 analysis script와 table schema 동결

## 10. 구현 감사: E3 full objective는 아직 없다

현재 `scripts/train_medical_nla_lora.py`가 최적화하는 것은 target text의 next-token
cross-entropy뿐이다. Validation도 전체/content/scaffold token CE이며 다음 항목은 없다.

- AR forward 또는 activation reconstruction MSE
- matched activation과 shuffled activation의 pair-specificity objective
- AV와 AR의 joint update
- reconstruction reward를 위한 policy-gradient, GRPO 또는 preference optimization

따라서 현재 실행 가능한 학습군은 `Medical-NLA, SFT only` 하나다.
`reconstruction-only`와 `full Medical-NLA`를 같은 스크립트의 옵션처럼 실행하면 안 된다.

원 NLA의 설명은 discrete text이므로 AR MSE를 일반적인 미분 loss처럼 AV token CE에 바로
더할 수 없다. Full objective를 구현하려면 다음 중 하나를 먼저 선택해야 한다.

1. 공개 방식에 가까운 RL/GRPO: frozen AR reconstruction을 AV reward로 사용하고 KL을 둔다.
2. Offline preference optimization: activation마다 여러 설명을 생성하고 AR reconstruction,
   clinical alignment, pair-specificity로 순위를 매긴 뒤 DPO류로 AV LoRA를 학습한다.
3. 이번 논문에서는 SFT-only를 제안법으로 제한하고 reconstruction/pair 검증은 평가 관문으로만
   사용한다. 이 경우 `full NLA fine-tuning`이라는 표현과 full ablation row를 삭제한다.

E3 GPU 학습은 이 결정을 코드, smoke test, metadata에 반영하기 전에는 시작하지 않는다.

## 11. DDXPlus의 학습/평가 역할도 아직 동결해야 한다

DDXPlus counterfactual을 Medical-NLA의 grounding objective에 사용하면서 같은 사례 또는
같은 evidence value 조합으로 Table 3을 계산하면 activation grounding의 독립 검증이 아니다.
E3 전에 다음 두 설정을 구분한다.

1. `DiReCT-only adaptation`: DiReCT train/validation만으로 학습하고 DDXPlus는 cross-corpus
   grounding 평가로만 사용한다.
2. `DiReCT + DDX grounding adaptation`: DDXPlus train split의 counterfactual을 objective에
   사용하되, DDXPlus test는 base case, cue/value 조합과 donor pool까지 분리한다.

Primary claim이 범용 의료 판독이면 1번을 먼저 보고하고 2번은 supervision-available
ablation으로 둔다. 2번을 primary로 쓸 경우에는 `DDXPlus에서 검증됐다`가 아니라
`held-out DDXPlus counterfactual에 일반화됐다`고 제한해 쓴다. 어느 설정이든 task-aligned
prompt, reward weight, shuffle 난이도는 DDXPlus test가 아니라 train/validation에서만 고정한다.
