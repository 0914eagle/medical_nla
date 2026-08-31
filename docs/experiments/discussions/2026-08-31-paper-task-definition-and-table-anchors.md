# 논문 task 정의와 외부 표 anchor

## 출발점: 교수님 피드백

현재 DiReCT/DDXPlus 결과는 Medical-NLA 개발 과정과 내부 진단에는 유용하지만, 그 자체가
논문의 외부 task를 정의하지는 않는다. 다음 세 질문을 먼저 닫아야 한다.

1. 이 논문이 푸는 **task**는 무엇인가? 현재 결과를 어느 공개 task의 표에 놓을 수 있는가?
2. Medical-NLA는 CoT, text-only explanation, probe, SAE 등과 비교해 무엇이 더 좋은가?
3. 2025--2026 의료 explanation/interpretability 논문과 DiReCT를 사용하는 연구는 어떤
   benchmark와 표를 쓰며, 우리는 무엇을 재현 또는 확장할 수 있는가?

이 문서는 답을 동결하는 결정 원장이 아니다. 주장, task, 표의 관계를 먼저 분리하고, 이후
사람 승인을 받아 benchmark와 final recipe를 고정하기 위한 discussion이다.

## 먼저 구분할 것: 주장과 task

| 구분 | 이 논문에서의 내용 |
|---|---|
| 문제 | 의료 LLM의 hidden activation에는 환자별 임상 상태가 있을 수 있지만, 사용자는 그것을 직접 읽을 수 없다. Visible CoT는 그 상태를 완전히 또는 충실하게 드러낸다는 보장이 없다. |
| 방법 | Medical-NLA: fixed target medical LLM의 activation을 받아 자연어 clinical-state text로 변환하는 activation-conditioned verbalizer. |
| 논문 주장 | Medical-NLA가 activation의 환자별 임상 정보를 자연어로 **faithfully verbalize**하고, 그 text가 visible CoT보다 임상적으로 더 유용한 explanation evidence가 될 수 있다. |
| task | 위 주장을 판별할 수 있도록 고정한 입력, 출력, reference, metric, control의 묶음. |
| 표 | 각 task에서 어떤 주장 성분을 검증했는지 보고하는 결과 형식. |

따라서 "Medical-NLA가 내부 신호를 자연어로 바꾼다"는 방법 설명이고,
"다른 LLM output보다 더 많은 설명을 제공한다"는 검증할 가설이다. 둘 자체가 task는 아니다.

## 제안하는 중심 주장

논문 중심 문장은 다음으로 좁힌다.

> **Medical-NLA is a faithful natural-language interface for medical LLM activations: it
> verbalizes patient-specific hidden clinical state into explanations that are clinically
> informative beyond visible chain-of-thought.**

이 문장은 두 검증 조항을 갖는다.

1. **Clinically informative:** Medical-NLA가 제공한 정보로 만든 explanation이 CoT 또는
   text-only explanation보다 전문가 reference에 더 잘 정렬되는가?
2. **Faithful to activation:** 그 설명이 그럴듯한 의료 상식이 아니라 해당 환자의 activation을
   실제로 반영하는가?

"Medical-NLA가 모델을 faithful하게 만든다"고 주장하지 않는다. Target backbone을 바꾸거나
더 정답으로 만드는 것이 아니라, 이미 존재하는 내부 상태를 읽는 interface를 만드는 것이다.

## 중심 task: Medical Activation Verbalization

### 공통 입출력 계약

임상 사례를 `X`, 고정된 target medical LLM의 P0 activation을 `h_P0(X)`, Medical-NLA
출력을 `Z`라고 둔다.

```text
X --target medical LLM, P0--> h_P0(X) --Medical-NLA--> Z
```

- `P0`는 **최종 rationale prompt가 아니라 activation 추출 위치/protocol**이다.
- P0는 사례를 읽은 target model의 마지막 prompt-token state이며, source answer나 source CoT
  token을 activation에 넣지 않는 조기 상태로 고정한다.
- `Z`는 diagnosis-free patient-state text 또는 finding/value claim set이다.
- 각 benchmark에서 source model, P0 prompt, layer, token position은 validation에서 한 번
  정하고 test에서는 변경하지 않는다. 현재 DDXPlus/DiReCT 축적물의 P0와 HS24/HS32는
  개발 자산이지, 새 benchmark test에서 layer를 고르는 근거가 아니다.

이 하나의 mapping이 논문의 중심 task다. 아래 Table 1과 Table 2는 같은 mapping의 서로
다른 성공 조건을 평가한다.

## Table 1: activation-augmented clinical rationale generation

### 질문

`Z`가 실제로 최종 의료 explanation을 더 좋게 만드는가? 이것이 외부 의료 explanation
benchmark에 놓을 수 있는 task다.

### 기존 표 anchor

가장 직접적인 peer-reviewed anchor는 Chen et al., **"Benchmarking Large Language Models on
Answering and Explaining Challenging Medical Questions"** (NAACL 2025)다.

- 공식 논문: <https://aclanthology.org/2025.naacl-long.182/>
- benchmark: JAMA Clinical Challenge와 Medbullets
- 원 task: clinical case/question과 정답을 바탕으로 rationale을 생성하고 expert explanation과
  비교
- Table 3 계열 metric: ROUGE-L, BERTScore, BLEURT, CTC relevance/preservation/consistency,
  G-Eval relevance/coherence/consistency, BARTScore
- 논문 자체도 automatic metric과 human judgment의 불일치를 보고하므로, 재현 시에는
  공개 평가 protocol과 별도로 제한된 expert/clinician audit을 명시해야 한다.

이 표의 원래 입출력은 `X, Y* -> R`이다. Medical-NLA row는 다음으로 확장한다.

```text
X, Y*, h_P0(X) -> Z -> R
```

여기서 `Y*`는 benchmark의 **explanation-only setting**에서 주어진 target answer다. 따라서
이 task의 answer accuracy는 Medical-NLA의 점수가 아니며, "정답을 안 뒤 얼마나 좋은
rationale을 쓰는가"를 분리해 보는 setting이다.

### 권장 실행 구조

```text
case X
  └─ frozen target model + P0 -> h_P0(X)
       └─ Medical-NLA -> patient-state text Z

case X + benchmark answer Y* + Z
  └─ frozen rationale actor -> final clinical rationale R
```

동일한 rationale actor, prompt template, temperature, maximum tokens, answer `Y*`를 모든 행에
고정한다. 바뀌는 것은 actor에 주는 추가 evidence뿐이다.

| Table 1 row | rationale actor의 추가 evidence |
|---|---|
| Text-only | 없음: `X + Y*` |
| Source CoT | target model의 visible CoT |
| Vanilla NLA | frozen vanilla activation verbalization `Z_vanilla` |
| Medical-AV SFT | supervised activation verbalization `Z_sft` |
| Medical-NLA | proposed `Z_medical` |
| Shuffled Medical-NLA control | 같은 stratum의 다른 사례 activation에서 만든 `Z_shuffle` |

같은-backbone text-only row와 shuffled-activation row가 없으면, 더 좋은 base model 또는 더 긴
prompt가 이긴 것인지 activation evidence가 이긴 것인지 분리할 수 없다.

### Table 1이 말하는 것과 말하지 않는 것

- 높은 explanation metric: `Z`가 rationale 생성에 임상적으로 유용한 evidence일 수 있다.
- 낮은 metric: Medical-NLA가 좋은 natural-language interface라는 주장을 지지하지 못한다.
- 높은 Table 1 점수만으로 activation faithfulness는 증명되지 않는다. CoT나 상식 문장이
  충분히 좋은 rationale을 만들 수 있기 때문이다.
- 현재 DiReCT `Obs*`/`Exp*` 표는 유용한 development evidence이지만, 이 외부 task Table 1을
  대체하지 않는다. DiReCT는 physician observation/rationale tree라는 다른 schema를 쓴다.

## Table 2: patient-specific activation faithfulness

### 질문

`Z`가 해당 환자의 `h_P0(X)`를 읽는가, 아니면 진단군의 전형적 문장을 생성하는가?

### 기존 표 anchor와 한계

완전히 동일한 **의료** 표는 현재 확인하지 못했다. 가장 가까운 두 2026 연구는 evaluation
원리를 제공하지만, 의료 benchmark를 제공하지는 않는다.

| 연구 | 원 task | 가져올 수 있는 것 | 그대로 가져올 수 없는 것 |
|---|---|---|---|
| [PRISM](https://arxiv.org/abs/2606.09563), 2026 preprint | activation에서 active instruction set 복원 | set recovery, coverage, hallucination, text-only/activation decoder 비교 표 형식 | instruction label과 의료 finding/value label은 다름; 의료 결과를 재실행해야 함 |
| [CHIVE](https://arxiv.org/abs/2608.16747), 2026 preprint | tool output이 counterfactual behavior 예측을 개선하는지 | transcript-only, NLA, activation oracle, SAE 비교와 counterfactual utility 원리 | 실제 환자 observation/finding ground truth가 없음 |

따라서 Table 2는 PRISM의 **set retrieval**과 CHIVE의 **counterfactual control**을 DDXPlus의
structured evidence에 적용한 새 의료 task로 명시해야 한다. 기존 논문의 숫자에 Medical-NLA
행 하나를 붙이는 표가 아니다.

### 공통 평가 단위

모든 방법의 출력을 공통 `(evidence_id, value_id)` claim set으로 정규화한다.

| 방법군 | 공통 claim set으로의 변환 |
|---|---|
| Linear probe | threshold를 넘긴 label/value를 직접 claim으로 사용 |
| SAE | train-only feature-to-finding mapping을 거친 claim set |
| Patchscope | fixed parser/mapper가 continuation에서 읽은 claim set |
| Probe-guided structured reader | probe claim set을 train-only lexicon으로 결정론적으로 렌더링 |
| Vanilla NLA / Medical-NLA | frozen method-blind mapper가 open text를 ontology claim으로 정규화 |

Structured reader는 독립 decoder가 아니라 **closed probe + deterministic renderer**다. 이 행은
"probe score를 자연어 형식으로 보여줄 수 있는가"의 upper-bound/monitor control이며,
open-ended Medical-NLA와 같은 학습 방법으로 부르지 않는다.

### 필요한 지표

| panel | 지표 | 판별하는 실패 |
|---|---|---|
| Static recovery | finding coverage/F1, unsupported-claim rate, conditional value accuracy | gold finding을 안 읽거나 환각하는가 |
| Same-diagnosis control | own activation vs 같은 진단 다른 환자 activation의 recovery gap | 진단 전형 문장만 말하는가 |
| Cue deletion | deleted claim removal, deletion phantom, retained-finding preservation | activation에서 삭제 cue만 선택적으로 잊는가 |
| Value edit | replacement hit, old-value persistence, clean value switch | value를 실제로 업데이트하는가 |

Table 2의 성공 기준은 "probe보다 모든 metric에서 이김"이 아니다. Probe는 정해진 ontology의
닫힌 분류기로 설계되어 static label F1에서 우세할 수 있다. Medical-NLA의 필요 주장은
**열린 자연어를 생성하면서도** 최소한 same-diagnosis control과 counterfactual specificity를
통과하고, 공통 claim metric에서 probe/SAE류와 비교 가능한 수준의 fidelity를 보인다는 것이다.

### 현재 결과의 위치

현재 frozen HS24 probe와 structured reader의 DDXPlus locked 결과는 Table 2의 control/
upper-bound 자산이다. 예를 들어 structured reader는 finding F1 `.9587`, deletion removal
`.6407`, retained preservation `.9987`, clean value switch `.0804`를 기록했다. 이것은
Medical-NLA 성공 결과가 아니다. Vanilla NLA semantic row는 ontology claim `0`으로 확인되어
같은 task에서 음성 baseline으로만 사용 가능하다.

현재 D10/D20 SFT/ranking/anchor 실패는 Table 2의 Medical-NLA 행으로 test를 열지 않았으며,
main-table comparison result로 바꾸어 적지 않는다. 이것들은 method-development appendix의
promotion-failure evidence다.

## 선택적 Table 3: explanation의 downstream utility

Table 1은 explanation quality, Table 2는 activation faithfulness다. "설명을 다른 clinical
solver에게 주었을 때 실제 의사결정이 좋아지는가"까지 주장하려면 세 번째 task가 필요하다.

```text
case X + additional evidence E -> independent solver -> diagnosis / answer A
```

`E`를 없음, visible CoT, source CoT, probe labels, shuffled Medical-NLA, own Medical-NLA로
바꾸고 answer accuracy와 expert reasoning-step coverage를 비교한다. 이 task의 가장 중요한
규칙은 `E`와 target model activation을 **gold answer를 보기 전에** 생성하는 것이다.

MedThink-Bench (500 high-difficulty questions, expert step references)는 explanation step
coverage의 candidate anchor다: <https://www.nature.com/articles/s41746-025-02208-7>. 다만
현재는 Medical-NLA method도, benchmark conversion도, solver protocol도 동결되지 않았다.
따라서 Table 3은 main claim에 필요하다면 새로 설계할 task이지, 현재 표를 빈칸으로 유지할
이유는 아니다. Table 1 + Table 2만으로도 "clinically informative and activation-faithful
verbalization"이라는 더 좁은 논문 주장은 가능하다.

## DiReCT의 올바른 역할과 citation audit

DiReCT는 2024 NeurIPS Datasets and Benchmarks Track benchmark이며, physician observation,
rationale, diagnosis tree를 제공한다.

- 원 논문: <https://proceedings.neurips.cc/paper_files/paper/2024/file/892850bf793e03b5c410dfd9425b94c8-Paper-Datasets_and_Benchmarks_Track.pdf>
- 현재 우리 사용: PDD-disjoint development/locked clinical-alignment audit.
- 강점: `Obs*`/`Exp*`가 자유문 clinical explanation을 observation/rationale structure와
  비교할 수 있게 한다.
- 한계: activation ground truth나 counterfactual activation pair를 주지 않으며,
  ChallengeClinicalQA와 같은 공개 explanation table의 직접 대체물이 아니다.

"DiReCT를 citation한 2025--2026 논문이 어떤 표를 썼는가"는 별도 literature audit으로
완료해야 한다. 그 audit은 논문마다 (a) DiReCT 전체 benchmark인지 custom split인지,
(b) `Accdiag`만 쓰는지 `Obs*`/`Exp*`도 쓰는지, (c) target answer exposure가 있는지,
(d) original evaluator인지 새 judge인지를 표로 기록한다. 현재 이 citation census가 완료되기
전에는 "DiReCT 관행이 이 표 구조를 요구한다"고 쓰지 않는다.

## 2025--2026 literature/benchmark audit: 현재 결론

| 필요 | 가장 직접적인 근거 | 논문 설계에 주는 결론 |
|---|---|---|
| 의료 rationale quality의 기존 표 | ChallengeClinicalQA, NAACL 2025 | Table 1은 동일 benchmark/protocol을 재현하고 Medical-NLA-assisted row를 추가하는 경로가 가장 보수적이다. |
| 의료 reasoning-step completeness | MedThink-Bench, 2026 | Table 1의 보조 지표 또는 Table 3 utility task 후보. |
| activation-to-language set fidelity | PRISM, 2026 preprint | Table 2의 coverage/hallucination/set-retrieval schema를 제공하지만 medical row는 새로 측정해야 한다. |
| activation tool의 utility/counterfactual test | CHIVE, 2026 preprint | Table 3 또는 Table 2 counterfactual panel의 control 철학을 제공한다. |
| 의료 hidden-state intervention | SAE/probing 의료 연구들 | 의료 activation을 분류/steer하는 baseline은 있으나, probe+SAE+Patchscope+NLA를 같은 환자별 자연어 verbalization task에서 비교한 2025--2026 표는 현재 확인하지 못했다. |

따라서 교수님 요구를 엄밀히 충족하는 범위는 다음과 같다.

1. **Table 1:** 기존 peer-reviewed 의료 explanation task를 재현해 Medical-NLA row를 추가할 수 있다.
2. **Table 2:** 기존 activation literature의 평가 원리를 가져오되, 의료 activation verbalization
   benchmark로 새로 정의하고 모든 baseline을 같은 data/backbone에서 재실행해야 한다.
3. **Table 3:** downstream utility까지 주장하려면 별도 solver task를 구축해야 하며, 현재 결과만으로는
   만들 수 없다.

## 논문이 피해야 할 주장

1. 현재 DDXPlus probe F1을 Medical-NLA의 explanation quality로 부르지 않는다.
2. DiReCT `Obs*`/`Exp*`를 activation faithfulness로 부르지 않는다.
3. Gold answer를 rationale actor에 준 Table 1에서 Medical-NLA가 diagnosis accuracy를 개선했다고
   주장하지 않는다.
4. Published model numbers와 Medical-NLA row를 같은 표에 놓더라도 model, prompt, evaluator가
   다르면 직접 SOTA 비교라고 쓰지 않는다. 동일 Gemma text-only/CoT control이 primary다.
5. 현재 promotion을 통과하지 못한 SFT/ranking checkpoint에 locked-test Medical-NLA 행을 만들지 않는다.

## 다음 결정과 실행 순서

### Decision A: 논문 범위

사람이 다음 중 하나를 선택해야 한다.

| 선택 | 주장 | 필요한 main results |
|---|---|---|
| A. Two-table core paper | activation verbalization의 clinical quality + faithfulness | Table 1 ChallengeClinicalQA 계열 + Table 2 DDXPlus/DiReCT faithfulness |
| B. Three-task paper | 위 두 주장 + independent solver utility | A + Table 3 MedThink/medical QA decision-support |
| C. Development/negative-results paper | 현재 objective failure와 probe-reader boundary | 현재 DiReCT/DDXPlus 표 중심, Medical-NLA 성공 주장은 하지 않음 |

현재 사용자 목표인 "Medical-NLA를 성공시켜 논문 방법으로 제시"는 A가 최소 범위이며,
B는 더 강한 주장 대신 새로운 benchmark work가 필요하다.

### Decision B: Table 1 actor contract

Table 1 실행 전 다음을 frozen protocol로 기록한다.

1. Target medical LLM, P0 prompt, layer, token position.
2. Medical-NLA output schema `Z`와 max length.
3. Rationale actor model, system/user prompt, decoding params.
4. `Y*`를 주는 explanation-only setting인지, `Y*` 없이 answer+rationale를 내는 utility setting인지.
5. Same-backbone text-only, CoT, shuffled activation controls.
6. Published benchmark evaluator 재현 범위와 human/clinician audit 범위.

### Decision C: Table 2 baseline feasibility

Probe, structured reader, vanilla NLA는 이미 일부 자산이 있다. SAE, Patchscope, LatentQA/
Activation-Oracle 계열은 같은 target backbone, same P0, same ontology mapper 아래에서 새로
실행할 수 있을 때만 main-table row가 된다. 구현 또는 checkpoint가 없으면 빈 행으로 두지 않고
관련 work/appendix comparison으로 내린다.

## 판정

현재 논문의 task를 단순히 "Medical-NLA"로 두면 안 된다. 권장되는 중심 task는
**Medical Activation Verbalization**이며, 최소 두 가지 관측 가능한 성공 조건은
**activation-augmented clinical rationale quality**와 **patient-specific activation faithfulness**다.

Table 1은 ChallengeClinicalQA라는 기존 의료 explanation table을 확장하는 경로로 설계할 수 있다.
Table 2는 기존 의료 표를 복사하는 것이 아니라 PRISM/CHIVE의 activation evaluation 원리를
DDXPlus/DiReCT에 맞게 구현한 새로운 medical faithfulness benchmark로 명시해야 한다.

이 문서가 승인되기 전에는 기존 DiReCT locked Table 1A/1B/2를 새 논문의 main table이라고
부르지 않는다. 그것들은 current baseline/development evidence이며, 새 task protocol을 결정한
뒤 재사용 범위를 명시적으로 정한다.
