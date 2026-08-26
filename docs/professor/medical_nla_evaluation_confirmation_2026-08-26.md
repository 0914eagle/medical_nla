# Medical-NLA 평가 설계 (2026-08-26)

현재 논문 전체 정본은 [`../paper/README.md`](../paper/README.md), 실행 상태는
[`../paper/experiment_status.md`](../paper/experiment_status.md)를 따른다. 이 문서는
평가 설계를 상세히 설명하는 교수님 보고용 문서다.

## 0. 핵심 결정

현재 제안은 다음 한 문장이다.

> 의료 LLM의 CoT와 activation 기반 자연어 판독을 동일한 전문가 주석 기준으로
> 평가하고, Medical-NLA 판독이 임상적으로 필요한 관찰과 관찰-진단 연결을
> 보존하는지, 실제 activation에 사례 특이적으로 연결되는지, 자연어 편집을 다시
> 내부 상태로 되돌릴 수 있는지를 단계적으로 검증한다.

---

## 1. 연구 질문과 평가 층

### 대전제

> 의료 LLM의 설명을 믿으려면 (1) 의사가 중요하다고 본 관찰과 추론에 임상적으로
> 정렬되어야 하고, 동시에 (2) 그 설명이 현재 사례의 실제 activation에 의존해야
> 한다. CoT는 첫 조건을 일부 충족할 수 있으나 둘째를 보장하지 않고, probe는 닫힌
> label을 정확히 읽을 수 있으나 사전에 정의하지 않은 사례 고유 내용을 열린 자연어로
> 서술하는 도구가 아니다. 본 연구는 두 조건을 함께 검사할 수 있는 Medical-NLA를
> 만들고 검증한다.

논문 한 문장은 다음 범위로 고정한다.

> **We investigate whether medically adapted natural-language activation readouts can
> complement closed-label probes by producing clinically aligned, activation-grounded,
> and causally testable explanations that are more faithful to a model's internal
> diagnostic state than its generated chain of thought.**

이는 아직 결론문이 아니라 검증할 연구 명제다. Table 2만 성공하면 `clinically
aligned`, Table 3까지 성공해야 `activation-grounded`, Table 4까지 성공해야
`causally useful`을 순서대로 주장한다.

### 먼저 고정할 용어 구분

DiReCT는 faithfulness를 보장하지 않는다. DiReCT가 제공하는 것은 생성 설명과
의사 주석 reasoning 사이의 **expert-reference alignment**다. 설명이 backbone의
실제 내부 계산을 반영하는지는 DiReCT 점수만으로 알 수 없으며, DDXPlus의
짝 깨기·증거 반사실과 AV-AR 검증으로 별도 판정한다.

| 이 문서의 용어 | 의미 | 주 근거 |
|---|---|---|
| Clinical alignment | 의사 annotation과 관찰·관계·진단이 일치 | DiReCT `Obs*`, `Exp*` |
| Decision fidelity | 판독 진단이 gold가 아니라 backbone의 실제 결론을 반영 | source-wrong 사례 |
| Activation grounding | 판독 내용이 해당 activation 짝과 반사실 변화에 의존 | DDXPlus Table 3 |
| Causal utility | 판독 편집을 내부로 되돌렸을 때 목표 상태가 선택적으로 변함 | DDXPlus Table 4 |

### 가설 H1. CoT의 임상적 그럴듯함과 내부 상태 충실성은 같은 것이 아니다

같은 source run에서 CoT reasoning과 answer-boundary activation 판독은 서로 다른
정보를 보존할 수 있다. Medical-NLA가 CoT보다 우월하다는 주장은 모든 설명 점수의
일괄 우월성이 아니라, CoT에 나타나지 않은 source-decision 또는 사례 특이적
정보를 activation-grounded 판독이 추가로 복원할 때 성립한다.

### 가설 H2. 의료 적응은 vanilla NLA보다 임상 설명을 개선할 수 있지만 SFT만으로는 부족하다

진단명·전문가 설명을 직접 생성하도록 SFT한 AV는 seen-class 분류기로 붕괴할 수
있다. 따라서 `Medical-AV SFT only`를 ablation으로 두고, 본 방법은 clinical
supervision과 AV-AR reconstruction/contrastive grounding을 함께 사용한다.

### 가설 H3. 개선된 설명은 독립적인 activation 검증을 통과해야 한다

DiReCT `Obs*`/`Exp*`가 높더라도 matched-shuffled gap, evidence counterfactual,
identity round-trip을 통과하지 못하면 faithful Medical-NLA로 인정하지 않는다.
검증을 통과한 뒤에만 dataset-native text edit의 표상·행동 제어를 평가한다.

### RQ1. 설명 품질

> CoT, vanilla NLA, Medical-NLA 중 어떤 방법이 전문가가 표시한 임상 관찰과
> 관찰-진단 연결을 가장 잘 복원하는가?

이 질문은 설명의 **임상적 내용**을 평가한다. Activation 충실성을 아직 뜻하지 않는다.

### RQ2. Activation 충실성

> Medical-NLA 설명은 언어화 모델이 의료 지식으로 지어낸 문장이 아니라, 현재
> 사례의 activation에 실제로 의존하는가?

이 질문은 matched-vs-shuffled, activation swap, evidence counterfactual,
AV-AR round-trip으로 평가한다.

### RQ3. 자연어 개입

> 검증된 판독의 dataset-native claim을 편집하고 AR로 되돌렸을 때, 대응하는 내부
> 속성과 출력 likelihood가 선택적으로 변하는가?

이 질문은 먼저 표상 제어를 평가하고, 그다음 행동 제어를 평가한다. 반사실 정답
라벨이 없는 경우에는 “새 답이 임상적으로 정답”이라고 주장하지 않는다.

---

## 2. 데이터셋 역할

| 데이터셋 | 원래 제공하는 정답 | 본 연구 역할 | 하지 않을 주장 |
|---|---|---|---|
| DDXPlus | pathology, evidence ID/value, differential | 구조화 정보 복원, cue 제거·교체, activation swap, text patching | 자연스러운 임상 설명 품질의 최종 증거 |
| DiReCT | 의사 observation, rationale, diagnosis entailment tree | CoT 대 NLA의 주 설명 평가 | 모델 activation 자체의 ground truth |
| MedCaseReasoning | 실제 case text, final diagnosis, diagnostic reasoning | 긴 꼬리 진단과 자연 텍스트 OOD 전이 | gold evidence span이 있는 것으로 취급 |
| MedicalBench | concept label, 전문가 evidence sentence, justification | evidence grounding 보조 검증 | 대규모 training corpus |

### 제안하는 데이터 분담

1. **DDXPlus train/validation**: 구조화 clinical supervision, reconstruction 학습,
   controlled counterfactual과 patching 개발.
2. **DiReCT PDD-disjoint split**: train/validation에서 Medical-NLA를 적응하고,
   held-out PDD test에서 `Obs*`/`Exp*`를 평가. 공식 benchmark 전체 점수와 섞지 않고
   custom split임을 명시한다.
3. **MCR test**: Medical-NLA 학습에 사용하지 않는 자연 임상 텍스트 외부 OOD.
   Gold observation tree가 없으므로 DiReCT metric 대신 진단/source-decision 복원,
   matched-shuffled, 소규모 expert/reference judge만 보고한다.

DiReCT를 완전히 external-only로 둘지, PDD-disjoint supervised split으로 사용할지는
교수님 확인 사항이다. 본 방법의 의료 supervision을 보여주려면 후자가 현실적이며,
MCR이 독립 external test 역할을 맡는다.

### 왜 DDXPlus만으로 끝내지 않는가

DDXPlus에는 자연 임상 문장이 없다. Evidence ID와 value를 우리가 영어로 렌더링한다.
따라서 DDXPlus에서 높은 evidence recovery가 나와도 “임상적으로 좋은 설명”이라고
말할 수 없다. 대신 변수의 존재와 값이 정확히 알려져 있으므로 통제와 patching에는
가장 적합하다.

### 왜 DiReCT인가

DiReCT는 의사가 관찰, 관찰에서 진단으로 가는 rationale, 최종 진단을 주석했다.
논문이 이미 다음 지표를 정의했다.

- `Accdiag`: 최종 진단 정확도
- `Obspre`: 생성 설명의 관찰 precision
- `Obsrec`: gold 관찰 recall
- `Obscomp`: 예측 관찰과 gold 관찰의 Jaccard completeness
- `Expcom`: 공통 관찰에서 rationale과 연결 진단까지 맞춘 비율
- `Expall`: 전체 gold/predicted 관찰을 기준으로 rationale과 진단 흐름까지 맞춘 비율

따라서 `Evidence F1`, `Relation F1`을 새로 임의 정의하지 않고 기존 의료 설명
벤치마크의 평가 단위를 가져온다.

단, 이 지표는 일반적인 자유문 설명에 바로 붙일 수 있는 범용 metric이 아니다.
DiReCT의 예측 단위는 `(observation, rationale, diagnosis)` 구조이며, `Obs*`와
`Exp*`는 예측 구조와 의사 annotation 구조 사이의 대응을 전제로 한다. 따라서
사용 조건은 다음과 같다.

1. DiReCT 원 데이터와 annotation schema를 사용한다.
2. CoT와 NLA 출력을 동일한 `(observation, rationale, diagnosis)` claim 집합으로
   정규화한다.
3. 공식 Llama-3-8B semantic matcher와 공식 통계 코드를 먼저 재현한다.
4. 자유문에서 claim 구조를 추출하는 단계는 별도 오차원으로 감사한다.
5. `Expcom/Expall`은 **expert-reference reasoning alignment**로 부르고,
   activation faithfulness로 부르지 않는다.

DDXPlus나 MCR 결과에 같은 이름을 그대로 붙이지 않는다. 두 데이터셋에는 DiReCT와
동일한 physician observation-rationale-diagnosis annotation이 없기 때문이다.

참고:

- DiReCT: <https://proceedings.neurips.cc/paper_files/paper/2024/file/892850bf793e03b5c410dfd9425b94c8-Paper-Datasets_and_Benchmarks_Track.pdf>
- MedicalBench: <https://physionet.org/content/mimic-iv-ext-medicalbench/1.0.0/>
- MedThink-Bench: <https://www.nature.com/articles/s41746-025-02208-7>

---

## 3. CoT와 NLA를 무엇으로 비교하는가

CoT는 activation을 복원하지 않는다. CoT와 NLA의 공통 비교 단위는 **생성된 설명
텍스트가 가진 임상 정보**다.

동일 사례에서 다음 산출물을 만든다.

```text
Case
  ├─ source model의 <reasoning>...</reasoning>     = CoT 설명
  ├─ vanilla NLA(activation)                       = vanilla 판독
  └─ Medical-NLA(activation)                       = 의료 판독
```

세 설명을 각각 DiReCT gold observation/rationale과 비교한다. Method 이름은 judge에게
보이지 않는다.

### CoT 답 누출 방지

CoT의 `<answer>`를 포함하면 diagnosis recovery가 자명해진다. 따라서 주 분석은
`<reasoning>`만 사용한다.

1. `<answer>` 블록 제거
2. 자연스러운 reasoning을 유지하고, reasoning 안에 source final diagnosis alias가
   실제로 나왔는지 행별 flag 저장
3. 설명 품질은 reasoning 전체와 diagnosis-alias 마스킹 버전을 모두 평가
4. CoT와 NLA의 주 비교는 CoT 생성 전 **P0 activation**을 사용
5. P1 source-decision 판독은 reasoning에 final-answer alias가 없는 subset에서만 보조 분석
6. leakage 포함 전체 P1 결과는 상한(upper-bound) 분석으로만 보고

Reasoning에 진단명이 등장하는 것 자체도 설명 품질의 일부이므로 `Obs*`, `Exp*`
주 분석에서는 원문을 유지한다. 그러나 P1 activation의 source-decision recovery는
이미 읽은 진단명 문자열을 기억하는 것일 수 있으므로 leakage-free subset을 별도로
고정한다. 텍스트 마스킹만 하고 같은 P1 activation을 평가하는 것으로는 activation에
이미 들어간 문자열 누출을 제거할 수 없다.

### 이 비교가 말하는 것과 말하지 않는 것

- 높은 `Obs*`, `Exp*`: 전문가 reasoning과 임상적으로 잘 정렬된 설명
- 높은 source-decision recovery: 모델의 실제 결론과 잘 정렬된 설명
- 이것만으로 activation 충실성이 증명되지는 않음
- Activation 충실성은 §6의 별도 통제로 검증

---

## 4. Activation 위치: 한 위치가 아니라 두 질문에 두 위치

### P0. Final prompt token: 조기 상태

```text
[clinical prompt ... What is the diagnosis?] <P0>  → 첫 생성 토큰
```

마지막 prompt token의 hidden state는 아직 모델의 reasoning이나 answer 문자열을 보지
않았다. “출력 전에 내부 상태에서 무엇을 읽을 수 있는가?”의 주 위치다.

용도:

- 조기 source-answer/diagnosis prediction
- output leakage 없는 내부 정보 판독
- 실제 사전 경보 가능성

### P1. Answer boundary token: reasoning 이후 trajectory 분석

Source 출력 형식을 다음처럼 고정한다.

```text
<reasoning>
...
</reasoning>
<answer>
DIAGNOSIS
</answer>
```

전체 reasoning prefix와 `<answer>` marker까지 teacher-force로 다시 통과시킨 뒤,
`<answer>` marker의 마지막 subtoken activation을 저장한다. 이 hidden state가 첫
진단명 token을 예측한다.

```text
reasoning + <answer> <P1> → DIAGNOSIS
```

P1은 아직 진단명 token 자체를 보지 않았지만, 동일 run의 CoT가 끝난 뒤 형성된
의사결정 상태다. 따라서 P0에서 읽힌 상태가 reasoning 이후 어떻게 바뀌는지 추적하는
위치로는 유용하지만, CoT와 독립적인 내부 설명의 주 입력은 아니다.

단, reasoning에서 최종 진단명을 먼저 말한 행은 P1 activation에도 그 문자열이
이미 들어가 있다. 따라서 P1의 주 source-decision 분석은 `final diagnosis alias not
present in reasoning`인 행으로 제한하고, 전체 P1은 누출 상한으로만 보고한다.

10행 E1 smoke에서 이 flag가 `8/10`이었다. 표본이 작아 최종 비율은 아니지만, P1을
주 비교 위치로 쓰면 대부분의 행에서 이미 생성된 진단 문자열을 다시 읽을 위험이
있다는 설계 문제를 확인하기에는 충분했다. 그러므로 Table 2의 NLA 계열은 P0를 주
입력으로 고정하고, P1은 leakage-free subset과 P0->P1 trajectory에만 사용한다.

### P2. Post-answer token: 양성 통제만

진단명 생성 후 activation은 이미 answer token identity를 포함한다. 높은 진단
복원률이 자명할 수 있으므로 주 결과로 사용하지 않는다. 파이프라인이 답 정보를
읽을 수 있는지 확인하는 positive control과 layer trajectory에만 사용한다.

### Cue-token 위치: 국소 evidence 분석

Gold 또는 원문 span이 있는 데이터셋에서는 각 observation span의 마지막 subtoken
activation을 별도로 저장한다. 이는 전체 설명의 주 입력이 아니라 다음을 확인하는
국소 분석이다.

- 해당 observation이 해당 위치에서 읽히는가
- cue 제거/교체 후 판독이 함께 변하는가
- P0/P1에서 누락된 세부정보가 원래 cue 위치에는 남아 있는가

### 주 분석 사전등록

| 질문 | 주 위치 | 보조 위치 |
|---|---|---|
| 출력 전 조기 판독 | P0 final prompt token | P1, P2 |
| CoT 대 NLA 설명 비교 | P0 final prompt token | leakage-free P1 |
| evidence locality | cue span final subtoken | P0/P1 |
| positive leakage control | P2 post-answer | 없음 |

한 위치의 결과를 다른 질문에 가져다 쓰지 않는다.

---

## 5. Layer 선택

기존 파일럿에서 cue readout은 HS16/HS24/HS32에 따라 달랐다. 따라서 test 결과를 보고
가장 좋은 index만 고르면 안 된다. 공개 AV/AR는 `extraction_layer_index=32`용이므로
primary Medical-NLA, round-trip, patching은 HS32로 고정한다. HS16/HS24는 같은 decoder의
distribution shift가 섞인 sensitivity다.

권장안:

1. HS16, HS24, HS32를 모두 추출
2. HS32를 primary로 고정
3. HS16/HS24는 probe와 appendix sensitivity로 보고
4. 다른 index를 primary로 쓰려면 해당 index용 AV와 AR를 같은 recipe로 학습

초기 구현은 layer별 독립 LoRA/reader로 시작한다. 하나의 layer-conditioned NLA는
세 독립 reader가 모두 작동한 뒤의 확장 실험으로 둔다.

---

## 6. LLM-as-a-judge 사용 원칙

### 사용한다

자연어 paraphrase 때문에 exact string match만으로 observation/rationale 대응을
판정할 수 없다. DiReCT도 LLM으로 predicted observation과 gold observation의 의미
대응을 만들고, 일부 표본을 의사 판정으로 감사했다.

### 단, judge의 역할을 제한한다

Judge 입력:

```text
Gold expert observations and rationale
Candidate explanation
Fixed matching rubric
```

Judge 출력:

```text
predicted observation ↔ gold observation 대응
rationale relation match 여부
근거 없는 predicted claim 목록
```

Judge가 새로운 임상 정답이나 중요한 evidence를 스스로 만들게 하지 않는다.

### 주 평가 프로토콜

1. Method 이름 제거 및 순서 무작위화
2. temperature 0, judge prompt/version 고정
3. DiReCT 공식 evaluator를 먼저 그대로 재현한 뒤 `Obs*`, `Exp*` 계산
4. exact/ontology matcher 결과도 함께 저장
5. 무작위 100건을 두 명 이상의 임상의가 독립 판정
6. judge-human agreement, Cohen's kappa 또는 precision/recall 보고
7. 가능하면 두 judge model로 민감도 분석

Reference-free “설명이 좋아 보이는가?” 점수는 주 결과로 쓰지 않는다. MedThink-Bench도
expert rationale을 함께 준 reference-based judge가 전문가 평가와 더 잘 정렬됨을
보고했다.

---

## 7. 비교 모델과 학습 목적식

표에 등장하는 이름부터 고정한다. 현재까지 만든 LoRA reader와 앞으로 만들 full
NLA를 같은 이름으로 부르면, reconstruction을 사용하지 않은 SFT 결과가 NLA 전체의
성능처럼 보인다.

| 이름 | AV 학습 | AR 학습 | activation-grounding 강제 | 논문에서의 역할 |
|---|---|---|---|---|
| **Linear probe** | 해당 없음 | 해당 없음 | 지도 label 분류 | 닫힌 진단 공간의 강한 기준선 |
| **CoT reasoning** | 해당 없음 | 해당 없음 | 없음 | 모델이 스스로 말한 설명 기준선 |
| **Vanilla NLA** | 공개 NLA 체크포인트 | 공개 NLA 체크포인트 | 일반도메인 reconstruction | 의료 적응 전 기준선 |
| **Medical-AV, SFT only** | 의료 target에 next-token CE | 고정 또는 미사용 | 없음 | 분류기/문구 암기 붕괴를 드러내는 ablation |
| **Medical-NLA** | 의료 warm-start 뒤 reconstruction reward | 의료 텍스트에서 activation regression | 있음 | 제안 방법 |

원 NLA는 `h -> AV -> z -> AR -> h_hat` 구조다. AV는 activation `h`를 자연어 `z`로
바꾸고, AR은 `z`만 보고 activation `h_hat`을 복원한다. 원 논문의 기본 목적은

```text
L_recon = E[ || h - AR(z) ||_2^2 ],   z ~ AV(. | h)
```

이며 reconstruction 품질은 다음 `FVE`로 보고한다.

```text
FVE = 1 - MSE(h, h_hat) / MSE(h, mean_train_activation)
```

`FVE=0`은 train activation 평균만 예측한 수준, `FVE=1`은 완전 복원이다. 이 정의는
NLA 원 논문에서 가져온다. 원 NLA는 AR을 MSE regression으로, AV를 AR reconstruction
reward를 사용하는 RL로 공동 학습하며, 설명 유창성을 보존하기 위해 초기 AV에 대한
KL penalty를 사용한다.

본 연구의 현실적인 학습은 두 단계다.

1. **Medical warm-start**: `(h, physician-structured text)`로 AV를 CE-SFT하고,
   `(physician-structured text, h)`로 AR을 MSE 학습한다.
2. **Grounded joint training**: sampled AV 설명을 AR이 복원하게 하고, AV에는
   reconstruction reward와 임상 claim 보존 reward를 주되 초기 모델 KL을 유지한다.

개념적으로 AV reward는 다음 항을 가진다.

```text
R_AV = -log MSE(h, AR(z))
       + lambda_clinical * ClinicalMatch(z, expert_claims)
       + lambda_pair * PairSpecificity(z, h)
       - beta * KL(AV || AV_warmstart)
```

`ClinicalMatch`는 DiReCT train split의 구조화 claim에만 사용하고, test gold를 학습에
쓰지 않는다. `PairSpecificity`는 맞는 activation-description 짝이 diagnosis와 cue
수를 맞춘 shuffled 짝보다 높은 점수를 갖도록 하는 항이다. 이 결합 목적식은 제안안이며,
각 항의 제거 실험이 필요하다. 현재 `train_medical_nla_lora.py`는 SFT CE만 구현하며
아래 full objective는 아직 코드로 구현되지 않았다. 구현된 뒤에만 실험군으로 부른다.
특히 SFT-only와 full Medical-NLA의 차이가 Table 2뿐
아니라 Table 3에서도 나타나야 reconstruction이 실제 역할을 했다고 말할 수 있다.

참고: NLA 구조, MSE/FVE, AV-RL/AR-regression 정의는
<https://transformer-circuits.pub/2026/nla/index.html>에서 가져온다.

---

## 8. 최종 표 설계: 각 표가 답하는 질문, 열의 출처와 계산법

### Table 1. Backbone behavior and internal readout capability

이 표는 “probe보다 NLA가 진단을 더 잘 맞힌다”를 주장하기 위한 표가 아니다. 서로 다른
분모와 출력 공간을 한 점수로 합치지 않기 위해 두 panel로 분리한다.

#### Panel A. Backbone diagnostic behavior on identical case IDs

| Method | n | Parse coverage | Strict PDD | Disease category | Official semantic diagnosis |
|---|---:|---:|---:|---:|---:|
| Direct, answer-prefilled | TBD | TBD | TBD | TBD | TBD |
| Source CoT | TBD | TBD | TBD | TBD | TBD |

#### Panel B. CoT-P0 internal readout on identical activations

| Method | Coverage | Seen-PDD gold | Held-out-PDD gold | Category gold | Source-decision fidelity | Open evidence | Trained task head | Eval ontology |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Output-head candidate score | TBD | TBD | TBD | TBD | TBD | N/A | no | yes |
| Linear PDD probe | TBD | TBD | N/A | TBD | TBD | N/A | yes | yes |
| Vanilla NLA, default prompt | TBD | TBD | TBD | TBD | TBD | TBD | no | no |
| Vanilla NLA, task-aligned prompt | TBD | TBD | TBD | TBD | TBD | TBD | no | no |
| Medical-NLA | TBD | TBD | TBD | TBD | TBD | TBD | no | train text only |

- **Strict PDD/category/official semantic**은 서로 다른 난이도이므로 합치지 않는다.
- **Source-decision fidelity**는 gold가 아니라 backbone이 실제 생성한 answer를 판독했는지다.
  Source-wrong subgroup에서 gold match와 나란히 보고해 state reading과 context 재풀이를 구분한다.
- **Output-head candidate score**는 P0 뒤에 각 사전등록 PDD 문자열을 teacher-force하고 label
  token 평균 log probability로 순위를 매긴다. 별도 head는 없지만 평가 ontology를 받으므로
  열린 zero-shot 생성이 아니다.
- **PDD probe의 held-out PDD**는 output node가 없으므로 0이 아니라 N/A다. Category가 train에
  존재할 때 category probe는 별도로 평가할 수 있다.
- **Open evidence**는 observation/rationale free text다. Probe에 없는 능력이므로 N/A이고,
  필요하면 appendix에서 ontology와 head 수를 명시한 multi-label probe를 따로 평가한다.
- 두 panel 모두 parse/extraction 실패를 삭제하지 않고 coverage와 함께 failure로 센다.
- Accuracy와 방법 간 차이는 동일 case ID의 paired bootstrap 95% CI를 사용한다.

---

### Table 2. DiReCT clinical explanation quality

이 표가 교수님이 요청한 **“정답을 얼마나 잘 맞추고, 설명을 얼마나 잘하는가”**의
주 표다. DiReCT의 511개 clinical note에는 의사가 `(observation, rationale,
diagnosis)`를 주석했다. 아래 metric은 새로 만든 것이 아니라 DiReCT 논문 Section 3.5의
정의를 사용한다.

| Method | Text source | n | Extraction coverage | Accdiag | Obspre | Obsrec | Obscomp | Expcom | Expall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CoT reasoning | generated reasoning | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA | P0 activation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, SFT only | P0 activation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA, full objective | P0 activation | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Full objective 행은 AR reconstruction 또는 preference/RL objective가 실제 구현되고 smoke를
통과한 경우에만 유지한다. 현재 학습 코드는 SFT-only다. 모든 method 출력은 동일한 claim
extractor로 official schema에 맞추고, extractor는 원 note, gold annotation, method 이름을
보지 않는다. Extraction 실패는 행 삭제가 아니라 failure이며 coverage를 함께 보고한다.

기호를 먼저 고정한다.

```text
O      = 의사가 주석한 gold observation 집합
O_hat  = 방법이 생성한 predicted observation 집합
M      = semantic matcher가 greedy하게 만든 O와 O_hat의 일대일 대응
m      = 대응된 observation 중 rationale와 연결 diagnosis까지 맞은 개수
```

각 열의 정확한 뜻:

- **Accdiag**: 공식 코드의 `acc_diag`. 예측 chain의 마지막 진단 문자열과 gold chain의
  마지막 진단 문자열을 첫 글자 대문자화 후 exact match한다. 출력은 평가 전에 공식
  PDD vocabulary로 canonicalize해야 한다.
- **Obspre = |M| / (|O_hat| + 1)**: 공식 코드의 `comp_pre`. 모델이 관찰이라고 말한 것
  중 의사 관찰과 의미적으로 대응된 정도다. 구현에는 `+1` smoothing이 있어 완전
  일치해도 1.0이 되지 않는다.
- **Obsrec = |M| / (|O| + 1)**: 공식 코드의 `comp_re`. 의사가 표시한 관찰 중 모델이
  복원한 정도이며 이 열에도 `+1` smoothing이 있다.
- **Obscomp = |M| / |O union O_hat|**: observation 집합의 semantic Jaccard다.
  precision과 recall을 하나의 completeness 값으로 묶는다.
- **Expcom = m / |M|**: 이미 observation이 서로 대응된 쌍 안에서 rationale와 그
  rationale가 향하는 diagnosis node까지 맞은 비율. 즉 **관찰을 찾은 뒤의 관계 품질**이다.
- **Expall = m / |O union O_hat|**: observation 누락, 불필요한 관찰, 잘못된 rationale,
  잘못된 diagnosis 연결을 모두 포함한 end-to-end 설명 점수다. 주 설명 metric으로 삼는다.

DiReCT는 자연어 observation 대응과 rationale match를 exact string이 아니라
Llama-3-8B semantic matcher로 판정하고 temperature 0을 사용했다. 공식 `evaluation.py`는
최종 점수 계산기가 아니라 observation/rationale의 `Yes` 매칭 결과를 `_eval` JSON으로
만드는 1단계이며, `statistics.py`가 위 산식을 집계한다. Prediction schema는
`{observation: [rationale, note_section, diagnosis], ..., "chain": [...]}`다.

구현 감사에서 세 가지 민감도 검사가 필요함을 확인했다. 첫째, observation matching은
gold 순서대로 첫 `Yes` prediction을 선택하는 greedy matching이라 dict 순서에 의존할 수
있다. 둘째, judge 응답이 정확히 `"Yes"`일 때만 match로 인정된다. 셋째, 평가 중 예외는
건너뛰고 누락된 eval 파일은 통계 단계에서 모든 metric 0으로 처리된다. 따라서 공식
점수를 그대로 재현한 결과와 함께 prediction 순서 permutation, 응답 정규화, maximum
bipartite matching 민감도 결과를 별도로 낸다. Method 이름을 가린 뒤 최소 100건을
임상의 두 명이 감사한다. `Expcom/Expall`은 DiReCT 논문이 “faithfulness”라고 부르지만,
본 논문에서는 혼동을 막기 위해 **expert-reference reasoning alignment**라고 부른다.
이것만으로 activation faithfulness가 증명되지는 않는다.

공식 출처:

- 논문과 metric 정의: <https://proceedings.neurips.cc/paper_files/paper/2024/file/892850bf793e03b5c410dfd9425b94c8-Paper-Datasets_and_Benchmarks_Track.pdf>
- evaluator 구현: <https://github.com/wbw520/DiReCT>

**CoT의 공정한 입력**: `<answer>`를 제거한 같은 source run의 reasoning만 넣는다.
**NLA의 공정한 주 입력**: reasoning이나 answer가 생성되기 전 final-prompt-token인 P0
activation이다. CoT와 NLA가 같은 임상 note에서 출발하되, NLA가 CoT 문자열을 입력으로
재사용하지 않게 한다. 첫 diagnosis token 직전 P1은 reasoning 이후 상태의 보조 분석이며,
reasoning에 diagnosis alias가 이미 적힌 행을 제외한 결과와 전체 leakage upper bound를
나란히 보고한다.

**이 표가 답하지 않는 것**: 높은 `Expall`은 의사 annotation과 비슷하다는 뜻이지,
그 문장이 해당 activation에서 읽혔다는 뜻이 아니다. 그 반박을 Table 3가 담당한다.

---

### Table 3. Case specificity and activation grounding on controlled DDXPlus cases

이 표의 용어는 DiReCT 공식 metric이 아니라 **본 연구가 activation 의존성을 검증하기
위해 사전등록하는 operational metric**이다. 따라서 논문에서 출처를 “ours”로
명시하고 산식·shuffle 제약·분모를 함께 공개한다.

| Method | Information channel | Own-case evidence F1 | Shuffled evidence F1 | Case gap | Removed-cue deletion | Untouched-cue retention | Round-trip FVE |
|---|---|---:|---:|---:|---:|---:|---:|
| CoT reasoning | input text | TBD | TBD | TBD | TBD | TBD | N/A |
| Vanilla NLA | activation | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-AV, SFT only | activation | TBD | TBD | TBD | TBD | TBD | N/A 또는 고정 AR |
| Medical-NLA | activation | TBD | TBD | TBD | TBD | TBD | TBD |

각 열의 정의:

- **Own-case evidence F1**: activation `h_i`에서 생성한 claim `z_i`를 같은 환자의
  native evidence set `E_i`와 비교한 micro-F1.
- **Shuffled evidence F1**: 같은 `z_i`를 다른 환자 `E_j`와 비교한 F1. `j`는 같은
  diagnosis, 같은 arm, 비슷한 cue count에서 뽑아 질환명·길이 shortcut을 막는다.
- **Case gap = mean(F1_own - F1_shuffled)**: 양수이고 paired CI가 0을 배제해야 사례
  고유 정보가 있다고 판정한다. 전체 평균 둘을 따로 빼지 않고 행별 차이를 bootstrap한다.
- **Removed-cue deletion**: 원 판독에서 cue `e`가 읽힌 행 중, 입력 prompt에서 `e`를
  제거하고 activation을 재추출했을 때 판독에서도 `e`가 사라진 비율.
- **Untouched-cue retention**: cue `e`를 제거한 뒤에도 나머지 gold cues가 유지된 비율.
  삭제 성공만 보면 모든 내용을 지운 reader가 높은 점수를 받으므로 반드시 함께 본다.
- **Round-trip FVE**: AV 문장만 받은 AR이 원 activation을 얼마나 복원하는지다.
  산식은 §7의 NLA FVE를 그대로 쓴다.

CoT 행의 case gap과 cue-removal 값은 **input-text specificity**다. CoT는 prompt를 직접
보므로 이 값이 높아도 activation grounding이라고 부르지 않는다. 반대로 NLA 행은 AV가
원문을 받지 않고 activation만 받는 설정에서 계산하므로 case gap과 반사실 추적이
activation 의존성의 증거가 된다. 이 공통 열은 “설명 텍스트가 사례 변화에 얼마나
민감한가”를 CoT와 NLA에 동일하게 묻고, FVE 열은 NLA에만 가능한 추가 검증이다.

추가 필수 control:

1. activation 대신 mean activation을 넣은 바닥
2. patient 간 activation swap
3. text만 유지하고 activation을 shuffled한 AV-SFT loss/control
4. diagnosis가 같은 환자끼리의 hard shuffle
5. 원 cue가 AV 출력에 애초에 없었던 행을 deletion 성공으로 세지 않는 조건부 분모

**통과 기준**은 test 전에 고정한다. 최소 조건은 NLA에서 `(a) case gap의 95% CI > 0`,
`(b) removed-cue deletion이 matched control보다 높음`, `(c) untouched retention이
사전등록한 하한 이상`, `(d) Medical-NLA FVE가 mean baseline보다 높음`이다.

---

### Table 4. Text-mediated intervention

이 표는 설명의 “유용성”을 가장 강하게 시험하지만, Table 3 grounding을 통과한 뒤에만
실행한다. 자연어를 prompt에 추가하는 것과 activation을 자연어 bottleneck을 통해
수정하는 것을 구분해야 한다.

| Intervention | No-op top-1 preservation | No-op KL | Edited-value decoding | Target logit delta | Off-target KL | Target behavior rate |
|---|---:|---:|---:|---:|---:|---:|
| Plain-text prompt edit | N/A | N/A | TBD | TBD | TBD | TBD |
| Raw activation patch | TBD | TBD | TBD | TBD | TBD | TBD |
| Vanilla NLA text patch | TBD | TBD | TBD | TBD | TBD | TBD |
| Medical-NLA text patch | TBD | TBD | TBD | TBD | TBD | TBD |

- **Plain-text prompt edit**: 같은 문장을 user prompt에 직접 넣는 baseline이다. 이것이
  없으면 효과가 NLA patching 때문인지 단순 힌트 때문인지 구분할 수 없다.
- **Raw activation patch**: 실제 counterfactual prompt에서 추출한 activation을 원 run에
  직접 patch하는 oracle/상한이다.
- **NLA text patch**: `h -> AV text -> native-value edit -> AR -> h_edit` 후 같은
  layer/position에 `h_edit`를 주입한다.
- **No-op top-1 preservation**: 텍스트를 편집하지 않고 AV-AR round-trip만 했을 때
  원 top-1 진단이 유지된 비율.
- **No-op KL**: 원 output distribution과 no-op patch distribution의 KL divergence.
  top-1이 같아도 분포 전체가 크게 훼손되는 문제를 잡는다.
- **Edited-value decoding**: 독립적으로 동결한 attribute probe/decoder가 patch된
  activation에서 목표 native value를 읽는 비율. 학습에 쓰인 reader로 자기 자신을
  채점하지 않는다.
- **Target logit delta**: 사전 지정한 target diagnosis 또는 attribute logit의
  `patched - original` 차이. 방향은 데이터의 differential/native relation으로 미리 정한다.
- **Off-target KL**: target pair를 제외한 diagnosis distribution이 얼마나 변했는지다.
  작을수록 선택적인 개입이다.
- **Target behavior rate**: 실제 생성 답이 목표 방향으로 바뀐 비율. 반사실 gold가 없는
  경우 이를 accuracy improvement라고 부르지 않는다.

주 분석은 case-level paired bootstrap CI를 쓰고, edit type별로 `present/absent`, location,
severity를 분리한다. 많은 edit를 한 평균 하나로 합치면 쉬운 present/absent가 어려운
relation edit를 가릴 수 있다.

---

## 9. 본문 Figure 설계와 Table과의 역할 분리

원칙은 간단하다. **Table은 전체 test set의 정확한 수치·분모·CI를 보고하고, Figure는
동일 평균을 다시 그리지 않고 방법·실패 형태·paired distribution·인과 경로를 보여준다.**
따라서 이전에 제안한 `Expall` 대 grounding-gap 네 점 scatter는 Table 2·3의 재배치에
불과해 삭제한다.

### Figure 1. Medical-NLA 학습·평가 개요

```text
Clinical note ──> target medical LLM ──> CoT ──> diagnosis
                         │
                         ├─ P0: final-prompt activation
                         └─ P1: answer-boundary activation
                                      │
                                      v
                              AV ──> natural-language claims
                                      │
                ┌─────────────────────┼─────────────────────┐
                v                     v                     v
        DiReCT clinical        DDXPlus grounding      AR reconstruction
        reference match       / counterfactual       / text patching
```

그림 안에서 `clinical alignment != activation grounding`을 두 개의 별도 gate로
표시한다. 이 그림은 숫자를 담지 않고 논문의 전체 논리를 한 번에 설명한다.

### Figure 2. 한 DiReCT 사례의 설명 비교

Table 2의 평균을 막대그래프로 반복하지 않는다. 동일 clinical note 한 건에 대해
다음을 열로 배치한다.

```text
Physician gold tree | CoT | Vanilla NLA | SFT-only | Medical-NLA
```

- 초록: gold observation/rationale/diagnosis edge와 대응
- 회색: gold에는 있으나 누락
- 빨강: note 또는 gold tree에 근거 없는 claim
- 주황: observation은 맞지만 rationale나 diagnosis edge가 틀림

사례는 test 결과를 본 뒤 가장 예쁜 예시를 고르지 않는다. 본문 정성 그림은 validation
사례에서 고른다. 예: `CoT와 Medical-NLA의 Expall 차이가 중앙값에 가장 가까우면서 gold
observation 수가 전체 중앙값에 가까운 사례`. 성공 사례 하나와 failure 사례 하나를
두 panel로 두고, test 사례를 꼭 쓸 경우에는 method output을 보기 전에 note 길이와 gold
observation 수만으로 ID를 사전등록한다.

### Figure 3. Activation-grounding counterfactual

두 panel을 권장한다.

1. **사례 panel**: 원 prompt/activation/readout과 cue 하나를 제거한
   prompt/CoT 및 activation/NLA readout을 나란히 보여준다. 삭제 cue와 유지 cue를
   색으로 표시해 같은 반사실에 두 설명 채널이 어떻게 반응하는지 비교한다.
2. **분포 panel**: 모든 test pair의 `F1_own - F1_shuffled`, removed-cue deletion,
   untouched retention을 CoT/vanilla/SFT-only/full별 violin/ECDF로 보여준다.

Table 3는 평균과 CI를 제공하고, Figure 3는 소수 outlier가 평균을 만든 것이 아니라
paired distribution 전체가 이동했는지를 보여준다.

### Figure 4. Text-mediated patching

Table 4가 통과했을 때만 본문에 넣는다.

1. 왼쪽: `h -> AV -> original text -> one native edit -> AR -> h_edit -> patch` 경로
2. 가운데: no-op, prompt edit, raw patch, vanilla NLA, Medical-NLA의 target-logit delta
3. 오른쪽: target-logit delta 대 off-target KL 산점도 또는 Pareto frontier

좋은 방법은 오른쪽 아래가 아니라 **target shift가 크고 off-target distortion이 작은
영역**에 있어야 한다. 행동 성공 사례만 보여주지 않고 no-op 실패와 과도한 분포 왜곡도
같이 보인다.

### Appendix figures

| Figure | 내용 | 부록으로 보내는 이유 |
|---|---|---|
| A1 | P0/P1/cue-token x L16/L24/L32 heatmap | 위치 선택 ablation이지 주 결론이 아님 |
| A2 | diagnosis/cue별 held-out 성능 | 본문 평균의 이질성 감사 |
| A3 | AV CE, AR MSE, FVE, shuffled-control 학습 곡선 | 최적화 안정성·overfit 감사 |
| A4 | hallucination/omission/wrong-relation failure taxonomy | 정성 실패 분석 |
| A5 | MCR external-OOD 사례 | gold reasoning tree가 없어 주 설명 Figure와 분리 |
| A6 | seed와 layer sensitivity | test layer 선택·분산 감사 |

---

## 10. Text patching 정의

자유로운 산문 전체를 임의로 편집하지 않는다. DDXPlus의 native evidence ID/value와
정렬된 claim만 편집한다.

```text
Finding: dyspnea | status: present
Finding: leg swelling | location: left calf
Finding: chest pain | severity: severe
```

허용 편집 예:

- `present -> absent`, 단 데이터에 명시적 negative value가 있는 변수만
- `left calf -> right calf`, evidence dictionary의 허용 value 안에서만
- `mild -> severe`, 순서형 value가 데이터에 정의된 경우만

평가를 두 단계로 분리한다.

1. **표상 제어**: held-out probe/decoder가 편집된 native value를 읽는가
2. **행동 제어**: 관련 진단 logit이 사전 지정한 방향으로 움직이는가

반사실 gold diagnosis가 없으면 output change를 임상 정답률로 부르지 않는다.

---

## 11. 필요한 실험, 산출물과 중단 기준

새 실험은 아래 의존관계를 따른다.

```text
E0 data/evaluator audit
  -> E1 source CoT + activation extraction
      -> E2 vanilla/probe baseline
          -> E3 SFT-only, full objective는 구현된 경우에만 추가
              -> E4 DiReCT explanation evaluation
              -> E5 DDXPlus grounding controls
                  -> E6 text patching (E5 통과 시에만)
          -> E7 MCR external OOD
```

| ID | 실험 | 주 데이터 | 계산 자원 | 만드는 표/그림 | 선행 조건 |
|---|---|---|---|---|---|
| E0 | schema, split, evaluator 재현 | DiReCT | CPU + judge GPU/API | 평가 감사 부록 | 없음 |
| E1 | 동일 source run의 CoT와 P0/P1/P2 activation 추출 | DiReCT | target-model GPU | Figure 1 입력 | E0 |
| E2 | output head, probe, vanilla NLA | DDXPlus + DiReCT | probe CPU/GPU + AV GPU | Table 1 | E1 |
| E3 | SFT-only와 full AV/AR 학습, 3 seeds | DDXPlus + DiReCT train | 다중 GPU | Tables 1--3, Fig. A3 | E0--E2 |
| E4 | expert-reference 설명 평가 | DiReCT heldout | judge GPU/API | Table 2, Figure 2 | E3 |
| E5 | hard shuffle, cue removal, swap, round-trip | DDXPlus heldout | target+AV+AR GPU | Table 3, Figure 3 | E3 |
| E6 | dataset-native no-op/edit patching | DDXPlus heldout | target+AV+AR GPU | Table 4, Figure 4 | E5 통과 |
| E7 | external OOD 판독 | MCR | target+AV GPU | Appendix A5 | E3 |

### E0. 데이터와 evaluator 감사

1. DiReCT의 511 note, 25 disease category, 공식 data list의 61 PDD 분포를
   로컬에서 재집계한다. 경로 기반 최초 감사에서 나온 62는 3-depth 경로의 PDD를
   annotation root로 추정한 값이므로 정본 수치로 사용하지 않는다.
2. 동일 환자, note, PDD alias가 train/test에 겹치지 않는지 확인한다. 최종 manifest의
   469개 환자 그룹 중 14개 그룹(37행)이 여러 resolved PDD에 걸치므로, 이 환자들이
   연결한 PDD는 하나의 connected component로 묶어 함께 분할한다.
3. seed 17 pilot은 511행 중 label conflict 10, unparsed patient 4, duplicate copy 1행을
   제외한 496행으로 고정했다. 분할은 train 263 / val-seen 62 / test-seen 71 /
   test-PDD-heldout 100이며 patient 및 held-out PDD leakage가 없다. Held-out은 4개
   component, 5개 PDD다. 이 한 split의 심폐계 편중과 3행짜리 PDD 때문에 최종 보고는
   connected-component 복수 seed 또는 group K-fold 및 PDD macro 결과를 추가한다.
   이 171개 test case는 위치와 vanilla AV 설계에 이미 사용했으므로 pilot로만 남긴다.
   Pilot-heldout 5 PDD component를 금지한 downstream-confirmatory split은
   266/52/72/106으로 동결했고, heldout은 12 PDD와 10 categories다. 단, 같은 496행에서
   과거 source output이 생성됐을 수 있으므로 artifact overlap 감사 전에는
   `dataset-level untouched`라는 표현을 쓰지 않는다.
4. DiReCT official prediction schema `(observation: [rationale, note section, diagnosis])`를
   그대로 serialize할 수 있는지 sample에서 확인한다.
5. 공개 baseline output 또는 sample에 official evaluator를 실행해 paper 범위와
   재현되는지 확인한다.
6. 공개 data list에는 amended case 73개가 있지만 restricted release는 디렉터리
   구조만 유지한 채 파일명을 모두 바꿨다. 따라서 amendment flag를 경로로 개별
   restricted note에 붙이지 않는다. Content-based 조인이 일대일로 검증될 경우에만
   원 note span이 있는 사례의 sensitivity 결과를 내며, 그렇지 않으면 이 한계를
   명시한다. 의사가 plausible observation을 보충한 사례를 text-grounding과 같은
   것으로 세면 안 된다.
7. 폴더 PDD와 annotation chain root가 다른 43건 중 공백·개행·복수형은 공식 PDD
   vocabulary로 해결했다. 해결되지 않은 10건은 모두 `STEMI` 폴더와 `NSTE-ACS`
   annotation root가 충돌하므로 자동 보정하지 않고 primary split에서 제외한다.
8. restricted KG 24개와 공식 공개 KG 25개를 비교한 결과 공통 24개 중 7개만
   canonical JSON hash가 같고 17개는 내용이 달랐다. 따라서 공개 `Gastritis` KG를
   restricted release에 섞지 않는다. Sample annotation만 필요한 주 설명 평가는 KG
   없이 수행하고, KG 의존 실험은 restricted 24개와 Gastritis 제외를 명시한다.

**완료 조건**: split manifest, schema validator, evaluator version/prompt hash, sample
reproduction report가 모두 저장된다.

### E1. 동일 실행의 CoT와 activation 추출

Source model, tokenizer, chat template, decoding을 고정한다. 각 case에서 한 source run을
만들고 그 transcript를 teacher-force하여 P0/P1/P2 activation을 추출한다. CoT와 P1이
다른 run에서 나오면 sampling 차이가 내부-출력 차이로 섞이므로 허용하지 않는다.

저장 필드:

```text
case_id, split, source_prompt, generated_reasoning, generated_answer,
source_correct, diagnosis_alias_in_reasoning, P0/P1/P2 token indices,
layer, activation_path, generation seed, decoding config
```

**완료 조건**: 모든 activation path가 존재하고, token index를 원문에 역표시한 100건
감사에서 P0/P1/P2 정의 오류가 없다. `diagnosis_alias_in_reasoning`과
`gold_alias_in_reasoning`을 전수 집계하며, P1의 누출 없는 유효 표본 수도 함께 고정한다.

10행 smoke에서는 strict canonical PDD hit가 `0/10`, disease-category hit가 `6/10`,
model-answer alias의 reasoning 선행 등장이 `8/10`, gold PDD alias의 선행 등장이
`1/10`이었다. 이는 activation 추출 실패가 아니라 open-ended source answer와 세부 PDD
ontology가 어긋나는 문제다. Full run에서는 공식 exact `Accdiag`, category match,
blinded semantic match를 분리해 보고하고, source-wrong activation을 gold target과
자동 정렬된 것으로 간주하지 않는다.

### E2. 기준선과 capability boundary

동일 activation과 split에서 output-head likelihood, linear probe, vanilla NLA를 평가한다.
Probe hyperparameter는 validation에서 선택하고 test diagnosis별로 macro/micro accuracy를
보고한다. NLA의 진단명은 exact alias와 blinded semantic matcher 두 방식으로 채점한다.

**핵심 반증 조건**:

- Vanilla NLA가 이미 Table 2·3에서 충분하면 의료 adaptation 기여는 축소한다.
- Output head가 probe와 동일하면 “숨은 정보”가 아니라 출력층에 이미 드러난 정보일 수 있다.
- Source-wrong subset에서 NLA가 model answer보다 gold를 더 자주 말하면 faithful reader보다
  context solver일 가능성을 우선 조사한다.

### E3. 의료 적응과 ablation

계획상 세 모델을 같은 train IDs와 token budget으로 학습한다. 현재 실행 가능한 것은
첫 번째 SFT-only뿐이다.

1. `Medical-AV SFT only`
2. `Medical-NLA` without clinical reward: reconstruction만 의료 activation에 계속 학습 (미구현)
3. `Medical-NLA full`: reconstruction + clinical match + pair specificity (미구현)

각 모델은 최소 3 seeds를 사용하고 best checkpoint는 validation metric으로만 선택한다.
Training curve에는 AV CE/reward, AR MSE, FVE, real-pair 대 shuffled-pair gap을 기록한다.
Parameter 수, LoRA rank, 학습 token 수, GPU 시간도 표에 붙인다.

**성공 조건**: full 모델이 SFT-only보다 heldout 설명 점수와 activation-grounding 중
최소 하나만이 아니라 **둘 다** 개선한다. 설명만 좋아지고 Table 3 gap이 줄면 임상 문구
생성기로 이동한 것이므로 방법 성공으로 판정하지 않는다.

### E4. DiReCT 설명 품질과 CoT 비교

각 method 출력을 동일 parser로 claim 집합으로 만들고 official matcher로 Table 2를
계산한다. Parser 실패율과 빈 출력률을 별도 열 또는 각주로 보고한다. 구조화된 NLA만
잘 parse되고 자유문 CoT가 실패하는 불공정성을 막기 위해 다음 두 분석을 낸다.

1. 모든 방법에 동일 LLM claim extractor를 적용한 주 분석
2. 사람이 확인한 100건의 직접 semantic match 민감도 분석

방법 간 `Expall`, `Obscomp` 차이는 동일 case의 paired bootstrap CI로 검정한다.

**H1 지지 조건**: Medical-NLA가 CoT보다 단순히 `Accdiag`만 높은 것이 아니라
`Obscomp/Expall`에서 개선되고, 그 개선된 행들이 E5 grounding도 통과한다. Table 2만
이기면 “더 좋은 임상 설명 생성기”이지 “내부에 더 충실한 설명기”는 아니다.

### E5. Activation-grounding 통제

DDXPlus native evidence에서 다음 paired set을 만든다.

```text
original prompt / original activation
cue-e removed prompt / counterfactual activation
same-diagnosis other-patient activation
mean activation
```

원문 cue가 tokenizer에서 차지하는 span과 삭제 후 prompt의 형식 동일성을 검사한다.
삭제 때문에 문법이 깨지지 않도록 template slot 단위로 제거한다. Hard shuffle은 같은
diagnosis와 cue-count bin 안에서 derangement permutation을 고정한다.

**H3 grounding 통과 조건**: Table 3의 case gap CI가 0보다 크고, cue deletion과
untouched retention이 함께 통과하며, full NLA FVE가 mean baseline보다 높다. 하나라도
실패하면 “clinically aligned readout”까지만 주장하고 “activation-grounded”는 철회한다.

### E6. Text-mediated patching

E5를 통과한 layer/position 하나와 edit type 하나로 pilot을 시작한다. 먼저 no-op
round-trip 200건, 그다음 present/absent edit 200건을 실행한다. No-op preservation이
낮으면 relation/location edit로 확장하지 않는다.

비교군은 plain-text prompt edit, raw activation patch, vanilla NLA, Medical-NLA다.
Target behavior보다 먼저 internal target logit과 off-target KL을 확인한다.

**중단 조건**: no-op top-1 preservation이 사전등록 하한보다 낮거나 raw patch도 target
방향을 만들지 못하면 text patching을 본문 기여에서 제외한다. Raw patch는 되지만
text patch가 안 되면 정보는 존재하나 AV-AR bottleneck이 보존하지 못한 것으로 결론낸다.

### E7. MCR external OOD

MCR은 학습과 checkpoint 선택에 쓰지 않는다. Gold evidence tree가 없으므로 Table 2의
`Obs*`/`Exp*`를 계산하지 않는다. Source-answer fidelity, matched-versus-deranged
diagnosis match, parse rate, 반복 문구율을 보고하고, 50--100건을 임상의 또는
reference-based matcher로 정성 감사한다.

**성공 조건**: 절대 source-answer fidelity가 바닥보다 높고 derangement gap의 CI가
0을 배제해야 한다. 답 필드만 통과하고 supporting-cue field가 실패하면 두 필드를
분리해 그대로 보고한다.

---

## 12. 교수님께 그대로 물을 질문

1. 설명 품질의 주 벤치마크로 DiReCT의 task·구조화 출력·공식 evaluator 전체를
   재현하고, 그 조건에서만 `Obs*`, `Exp*`를 사용하는 것에 동의하시는가?
2. DDXPlus는 임상 설명 benchmark가 아니라 controlled patching testbed로 한정해도
   되는가?
3. CoT 비교는 같은 note에서 생성한 reasoning과, 그 reasoning 생성 전 P0 activation
   판독을 비교하는 방식이 적절한가?
4. P0를 주 설명 판독, P1을 reasoning 이후 trajectory와 leakage-free 보조 분석,
   P2를 positive leakage control로 사전등록하는 것에 동의하시는가?
5. LLM judge는 expert-reference semantic matcher로만 쓰고, 100건 임상의 감사를
   붙이는 수준이면 충분한가?
6. 본문 기여를 explanation quality + activation grounding까지로 두고, text patching은
   identity round-trip 통과 후 확장 결과로 두는가? 아니면 patching을 필수 주 기여로
   요구하는가?
7. DiReCT는 PDD-disjoint supervised split으로 사용하고 MCR을 external OOD로 둘 것인가,
   아니면 DiReCT도 학습에서 완전히 제외한 external-only benchmark로 둘 것인가?

---

## 13. 현재 기존 결과의 위치

- Probe all-cue 진단 판독, layer-position sweep, cue/diagnosis-heldout, classifier
  collapse는 새 연구의 문제 정의와 baseline으로 재사용한다.
- Wrong-note 행동·궤적·탐지·교정 결과는 인공 개입의 타당성 문제가 있으므로 새
  연구의 주 근거에서 제외하고 controlled stress-test appendix 후보로 보존한다.
- 구 1,747 cohort, mixed-arm MCR readout, noncanonical scorer 수치는 인용하지 않는다.
- 새 Table 1–4는 동일 split, 동일 activation, 동일 scorer로 다시 계산한다.
