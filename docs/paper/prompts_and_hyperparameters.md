# Prompt와 하이퍼파라미터 정본

## Source CoT prompt

DiReCT 원문 note 앞에는 다음 role/header가 붙는다.

```text
You are an expert physician. A patient presents as follows:

{note_text}
```

그 뒤에 다음 CoT instruction을 붙인다. Instruction은 presentation 뒤에 오며 P0는 이
전체 user prompt의 마지막 토큰이다.

```text
Work through this case as a natural reasoning process.

Think about:
- What the key clinical findings suggest
- Which diagnoses fit the presentation and which do not
- Whether your conclusion holds up under scrutiny

You MUST end your response with exactly "The answer is <diagnosis>."
```

답은 정규식으로 마지막 `The answer is <diagnosis>.`를 파싱한다. 전체 CoT에서 gold
문자열을 찾는 방식은 rule-out된 진단도 정답으로 셀 수 있으므로 사용하지 않는다.

## Direct baseline prompt

같은 presentation 뒤에 다음 instruction만 바꾼다.

```text
What is the single most likely diagnosis?

Give the diagnosis only. Do not explain your reasoning.

You MUST end your response with exactly "The answer is <diagnosis>."
```

Direct arm은 assistant turn을 `The answer is`에서 prefill한다. 그렇지 않으면 모델이
지시를 무시하고 CoT를 생성해 Direct-vs-CoT 대비가 사라진다.

## E1 generation

| 항목 | 값 |
|---|---|
| Model | `google/gemma-3-12b-it` |
| Dtype | bfloat16 |
| Device map | auto, 22 GiB per visible GPU |
| Decoding | greedy, `do_sample=false` |
| CoT max new tokens | 2048 |
| Direct prefilled max new tokens | 64 |
| Forced answer | E1에서 비활성화 |
| CoT batch size | 1 |
| Direct batch size | 4 |
| Seed | 17 |

`configs/default.yaml`의 일반 generation default 256은 E1 CoT에 적용되지 않는다.
`run_source_answers.py`가 condition별 기본값으로 CoT 2048을 사용한다.

두 arm 모두 Hugging Face tokenizer의 checkpoint chat template에 user message 하나를 넣고
`add_generation_prompt=true`로 렌더한다. Direct는 렌더된 assistant turn 뒤에
`The answer is`를 prefill하고, 저장할 때 같은 cue를 response 앞에 복원한다. CoT는 prefill
없이 자유 생성한다. `temperature`와 `top_p`는 전달하지 않으며 sampling도 사용하지 않는다.
Padding은 left padding이고 pad token이 없으면 EOS를 pad token으로 사용한다.

## Activation extraction

| 항목 | 값 |
|---|---|
| Hidden-state extraction indices | 16, 24, 32 |
| Hidden dimension | 3840 |
| P0 | prompt final token |
| P1 | last `The answer is` marker의 last subtoken |
| P2 | parsed diagnosis의 last subtoken |
| P1/P2 transcript | 실제 생성 response를 teacher-force |

P1/P2는 같은 source run의 정확한 transcript를 teacher-force하므로 별도 생성 run이 아니다.
다만 CoT 본문에 answer alias가 먼저 등장한 행은 P1의 clean pre-answer 분석에서 제외한다.
P0는 note 끝이 아니라 CoT instruction을 포함한 전체 user prompt의 마지막 토큰이다.
공개 AV sidecar는 `extraction_layer_index: 32`를 명시하므로 index 32 입력은 checkpoint와
호환된다. 논문에서는 embedding을 포함하는 hidden-state tuple convention을 캡션에 적는다.

## 보고할 추가 하이퍼파라미터

E3가 확정되면 LoRA rank/alpha/dropout, target modules, optimizer, learning rate, scheduler,
effective batch size, max sequence length, epochs, early stopping, objective weights, seed 세 개를
이 문서에 추가한다. Test 결과를 본 뒤 값을 바꾸지 않는다.
