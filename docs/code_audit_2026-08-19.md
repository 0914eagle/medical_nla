# 파일럿 코드 감사 — 논문용 재실행 전 점검

파일럿은 경향성만 보면 됐기 때문에 넘어간 부분들이 있다. 본실험은 표에 그대로
올라가므로 아래를 먼저 정리하고 시작한다. **A는 고치지 않으면 결과 해석 자체가
무효**가 되는 것들이고, B는 논문에 실으려면 정의·고정이 필요한 것, C는 남아 있는
교란 요인이다.

각 항목에 재실행 비용을 표시했다: **[재추출]** = 활성화를 다시 뽑아야 함,
**[재학습]** = LoRA만 다시 돌리면 됨, **[재채점]** = 저장된 출력에서 다시 계산하면 됨.

---

## A. 결과 타당성을 깨는 문제

### A1. 활성화를 뽑은 프롬프트와 정답 라벨을 만든 프롬프트가 다르다 **[재추출]**

가장 심각하다. 활성화 추출은 케이스 프롬프트를 그대로 쓴다:

```
A patient presents with {cues}. What diagnosis is most likely?
```
(`scripts/make_ddxplus_cue_count_cases.py:38-39`)

그런데 소스 모델 정답 라벨은 이걸 한 번 감싼 다른 프롬프트로 만든다:

```python
def source_answer_prompt(prompt: str) -> str:
    return (
        "Answer the clinical question with the most likely diagnosis or syndrome "
        "first, in one short sentence. Then give one brief reason.\n\n"
        f"Question: {prompt}"
    )
```
(`scripts/run_source_model_answers.py:85-90`)

프롬프트가 다르면 내부 상태도 다르다. 즉 **우리가 뽑은 활성화는 정답/오답 라벨이
붙은 그 실행의 상태가 아니다.** "이 활성화에서 읽은 내용이 모델이 틀릴지를
예측한다"는 주장(Table 1의 오류 예측 열, 가설 3 전체)이 정면으로 무너진다.

조치: 케이스 프롬프트를 하나로 확정하고, 추출·정답 생성·26-way likelihood가
**모두 동일한 문자열**을 쓰게 한다. 출력 형식 지시가 필요하면 그 지시를 케이스
프롬프트 자체에 넣고 그 상태에서 추출한다.

### A2. 2번째 에폭부터 dropout이 꺼진 채로 학습된다 **[재학습]**

`evaluate()`가 `model.eval()`을 걸고(`scripts/train_medical_nla_lora.py:115`)
끝나고 `model.train()`으로 되돌리지 않는다. 학습 루프는 이후 에폭에서 그대로
진행되므로 **에폭 1만 dropout이 켜져 있고 에폭 2·3은 꺼진 상태**로 학습된다.

여기에 에폭 수까지 실험마다 다르다 — v4(L32)는 `--epochs 3`, v5(L16/L24)는
`--epochs 2` (`EXPERIMENTS.md` §11 vs §12). 문서에는 "Same recipe per layer"라고
적혀 있지만 사실이 아니다. 결과적으로 레이어 궤적 "L16 0.34 / L24 0.73 / L32 0.56"은
**레이어 차이 + 에폭 차이 + dropout 적용 비율 차이**가 섞인 숫자다. Figure 2가
이 위에 서 있다.

조치: `evaluate()` 뒤에 `model.train()` 복구, 전 레이어·전 데이터셋 에폭 통일.

### A3. 부정어가 사라져도 "읽었다"로 채점된다 **[재채점]**

소프트 지표는 불용어를 뺀 내용 토큰 재현율이고 임계값이 0.5다
(`scripts/summarize_cue_position_readouts.py:53-65`, `evaluate_cue_counterfactuals.py:161`).
불용어 목록에 부정어가 없는 건 맞지만, 임계값이 낮아서 부정어를 잃어도 통과한다:

| | 내용 토큰 | 재현율 | 0.5 임계 |
|---|---|---|---|
| gold `no pain at rest` | {no, pain, rest} | — | — |
| readout `pain at rest` | {pain, rest} | 2/3 = 0.67 | **통과** |

임상적으로 소견이 뒤집혔는데 성공으로 집계된다. 반사실 실험의 "추적률"과
"phantom rate"가 전부 이 지표 위에 있다.

조치: 부정어(no/not/without/denies/negative/absent 등)를 **필수 토큰**으로
지정해 하나라도 빠지면 재현 실패로 처리하거나, 별도의 극성 일치 검사를 추가한다.

### A4. cue precision이 양방향 포함 때문에 부풀려져 있다 **[재채점]**

```python
def matched_cue_items(items, cues):
    return [item for item in items
            if any(contains_term(item, cue) or contains_term(cue, item) for cue in cues)]
```
(`scripts/score_medical_nla_v2_readouts.py:91-96`)

뒤쪽 `contains_term(cue, item)`이 문제다. 모델이 `pain` 한 단어만 내놔도 gold
`chest pain even at rest` 안에 포함되므로 **매칭으로 집계**된다. precision은
"cue를 남발하지 않았다"를 보증하는 지표인데, 정확히 그 남발을 잡지 못한다.

조치: precision은 단방향(`contains_term(item, cue)`)으로만 판정하거나, 최소
내용 토큰 수 조건을 건다.

### A5. 케이스 내 상관을 무시하고 있다 — 신뢰구간이 아예 없다 **[재채점]**

DDXPlus 16,410행은 4,900 케이스에서 나온다. 같은 케이스의 3~4행은 같은 프롬프트,
같은 환자, 같은 진단이라 독립 관측이 아니다. 그런데 모든 요약이 점추정만 찍는다
(`bootstrap`/`ci` 문자열이 코드 전체에 없음).

조치: 모든 headline 수치에 **케이스 단위 클러스터 부트스트랩** 신뢰구간을 붙인다.
AUROC도 마찬가지다(`scripts/evaluate_error_prediction.py:42`는 점추정만 반환).

---

## B. 논문에 실으려면 정의·고정이 필요한 것

### B1. 이름이 같은 서로 다른 지표가 두 개 돌아간다 **[재채점]**

| 지표 | 구현 | 성격 |
|---|---|---|
| `cue_recall` (readout 표) | `contains_term` — **다어절은 부분문자열 완전 일치** | 패러프레이즈 = 실패 |
| "read rate" (반사실 표) | `content_tokens` 재현율 ≥ 0.5 | 어순·활용 무시, 부분 일치 허용 |

`contains_term`은 다어절이면 경계 없는 부분문자열 검사라 `pain at rest`가
`pain at restaurant`에도 걸린다(`scripts/score_specificity_outputs.py:74-79`).
표를 나란히 놓으면 독자는 같은 것으로 읽는다.

조치: 지표 이름을 분리하고(예: `exact_cue_match` / `soft_cue_recall@τ`), 논문에
정의를 병기한다. 어느 표가 어느 지표인지 명시.

### B2. 임계값 0.5가 검증된 적이 없다 **[재채점]**

DDXPlus cue는 내용 토큰이 3~5개 수준이라, 0.5는 절반만 맞아도 통과다.
전체 faithfulness 결론이 이 한 숫자에 걸려 있다.

조치: τ를 0.3~0.9로 쓸어서 민감도 곡선을 부록에 넣고, 본문 수치는 사전 등록한
하나의 τ로 보고한다.

### B3. 26-way likelihood의 랭킹 필드를 사후에 고를 위험 **[재채점]**

`scripts/score_nla_diagnosis_logprobs.py`는 `logprob_sum`, `logprob_mean`,
`first_token_logprob`과 각각의 calibrated 버전까지 6개를 저장하고
`--rank-field`로 고른다(기본 `logprob_mean`). 결과를 보고 고르면 p-hacking이다.

조치: 본문 지표를 `calibrated_logprob_mean` 하나로 **사전 고정**하고 나머지는
부록 강건성 표로만 쓴다.

### B4. AV 프롬프트가 실험마다 다르고 타깃 스키마와도 안 맞는다 **[재학습]**

| 실행 | 프롬프트 | 지도 타깃 |
|---|---|---|
| v2/v3 | `prompt_templates/medical_nla_v2_readout.txt` (`<answer>`+`<supporting_cues>` 요구) | 동일 |
| v4/v5 | **플래그 없음 → 체크포인트 sidecar 기본값** | `<explanation><readout><observed>- cue` |

v4는 프롬프트가 요구하는 스키마와 학습 타깃 스키마가 아예 다르다. LoRA가 형식을
외워서 동작할 뿐이다. 또 `prompts/medical_actor_prompt_template.txt`는 "Describe
the medical **diagnosis**…"로 시작하는데, cue-position 실험의 타깃에는 진단이
없으므로 방향이 어긋나 있다.

조치: cue-position 전용 템플릿 1개를 만들어 DDX·MCR·전 레이어에 동일 적용,
타깃 스키마와 정확히 일치시키고 부록에 verbatim 게재. (교수님이 물어본 LLM 판정
프롬프트도 같은 기준으로 공개.)

### B5. best checkpoint 선택이 없고, val loss가 앞 128행으로만 계산된다 **[재학습]**

- `val_loss`를 출력만 하고 쓰지 않는다. 저장되는 건 마지막 에폭 어댑터뿐
  (`scripts/train_medical_nla_lora.py:296-310`).
- `evaluate()`가 `rows[:max_eval_rows]`로 자른다(`:117`). 무작위 표본이 아니라
  **파일 앞쪽 케이스들**이다. 기본 128행.

조치: val loss 기준 best 저장, val 표본은 무작위 추출 + 표본 수 명시.

### B6. seed 1회 **[재학습]**

전부 seed 17 단일 실행이다. 표에 점추정 하나만 올라간다.

조치: 최소 3 seed, mean ± sd 보고.

---

## C. 남아 있는 교란 요인

### C1. `cue_count_all`은 cue 순서를 섞지 않는다 **[재추출]**

```python
selected = list(cues) if cue_count is None else rng.sample(cues, selected_count)
```
(`scripts/make_ddxplus_cue_count_cases.py:91`)

개수를 지정한 변형은 무작위 표집이지만, 우리가 실제로 쓰는 `all` 변형은 DDXPlus
증거 순서 그대로다. 증거 순서는 문진 구조를 따르므로 **cue 인덱스와 cue 종류가
상관**된다(앞쪽 = 주호소, 뒤쪽 = 병력 등). cue 위치별 readout을 비교할 때
"위치 효과"와 "cue 종류 효과"가 분리되지 않는다.

조치: 케이스별 결정적 셔플을 넣거나, 분석에서 cue 종류를 통제 변수로 넣는다.

### C2. 케이스 프롬프트에 임상 프레이밍도 출력 형식 제약도 없다 **[재추출]**

`A patient presents with ... What diagnosis is most likely?` 한 줄이 전부다.
소스 모델은 자유 서술로 답하고, 정답 여부는 별칭 부분문자열 매칭으로 판정한다.
논문에 실을 정확도 숫자로는 약하다. A1과 함께 한 번에 확정해야 한다.

### C6. 음성 소견이 프롬프트에서 통째로 빠져 있었다 — 수정함 **[재추출]**

DDXPlus는 문장이 아니라 `(질문 id, 답 값)`을 저장하므로 cue 문구를 우리가 만든다.
긍정 답은 조동사를 떼면 되지만(`Do you have a cough?` → `a cough`), 부정 답은
조동사를 이미 떼어낸 뒤라 부정을 붙일 자리가 없어서 값을 뒤에 이어붙였다:

```
Have you traveled out of the country in the last 4 weeks? + no
  → "traveled out of the country in the last 4 weeks no"
```

프롬프트에 들어가면 문법이 깨질 뿐 아니라 **긍정으로 읽힌다**. v2-beta는 이런
항목을 통째로 제외하는 것으로 대응했고, 그 결과 DDXPlus 프롬프트에는 **양성
소견만** 남았다. 즉 음성이 임상적으로 불필요해서가 아니라 렌더링이 깨져서 빠진
것이다.

측정 결과 evidence 항목의 **10.6%**가 명시적 음성이다(20,000 환자 기준
`neg` 40,739 / `pos_valued` 179,146 / `bare(=yes)` 163,004). 환자당 약 2개꼴.

수정: 조동사를 떼지 말고 부정형으로 되살린다(`Have you` → `has not`).
`render_negative_phrase()`가 7개 조동사를 매핑하고, 인식하지 못하는 문장 형태는
비문을 내놓느니 제외한다. `--negative-cues` 플래그로 켜며 기본값은 꺼짐이라
파일럿 동작이 그대로 재현된다. cue마다 `cue_polarity`가 기록된다.

**결론: 켜지 않는다.** 10.6%는 등장 횟수였고, 코퍼스 전체에서 재보니 서로 다른
음성 소견은 **1종**이었다 — `has not traveled out of the country in the last 4
weeks`가 4,900 케이스 중 92.4%에 등장한다. 거의 상수인 cue는 진단 정보를 담지
않으면서, 조건 없이 뱉기만 해도 판독 점수를 주는 암기 경로가 된다. cue-heldout
설계 전체가 막으려던 바로 그 경로다. 빈도 상한 필터도 불필요하다: 그 하나를
빼면 최빈 cue가 `shortness of breath` 45.5%, `a cough` 29.1%로, 823종에 걸쳐
정상적인 증상 분포다. 50%에서 자르면 그 음성 하나만 잘린다.

부정 표현의 판독은 **MedCaseReasoning에서 측정한다.** 실제 산문이라
`no fever at any point`, `denied any history of trauma`처럼 다양하고 케이스마다
다르다. 결과적으로 두 데이터셋의 역할 분담이 선명해진다 — DDX는 span이 확정된
정밀 계측, MCR은 부정을 포함한 실제 임상 텍스트.

살릴 여지가 있는 음성이 3개 남아 있다(`the pain does not radiate to another
location`, `the lesion is not larger than 1cm`, `their lesions do not peel
off`). 현재는 `negative_value_unrenderable`로 빠진다. 나머지 8개는 부모 소견이
없어서 생긴 척도 0 값이라 중복이다. 필요해지면 그때 3개만 추가한다.

- **암묵적 음성은 여전히 불가능하다.** 이진 문항에 "아니오"로 답한 항목은
  `EVIDENCES` 목록에 아예 등장하지 않는다.
- **antecedent 유지는 이 결정과 무관하다.** `--no-prefer-symptoms`의 근거였던
  "음성이 antecedent에 산다"는 사라졌지만, 흡연력·COPD·당뇨·수술력·가족력 자체가
  다양하고 진단적이므로 유지한다.

### C7. wh-질문의 값 렌더링도 같은 방식으로 어색하다

같은 "값을 뒤에 붙인다" 규칙이 wh-질문에서도 어색한 문구를 만든다:

```
Where is the pain located? + chest  →  "where is the pain located chest"
```

`strip_question_to_phrase`는 `do you`류만 제거하므로 `where is`가 남는다.
음성 문제만큼 치명적이지는 않지만(의미가 뒤집히지는 않는다) 프롬프트 품질
문제이고, 프롬프트를 다시 확정하는 김에 같이 볼 항목이다. 현재는 기존 동작을
유지했고 테스트로 고정해 두었다.

### C3. `make_prompt(cues, *, diagnosis_options=False)`의 인자가 미구현이다

함수 본문이 `diagnosis_options`를 전혀 쓰지 않는다
(`scripts/make_ddxplus_cue_count_cases.py:38-39`). 객관식 변형을 만들려다 만
흔적으로 보인다. 쓸 거면 구현하고, 안 쓸 거면 지운다.

### C4. LoRA 파라미터가 bf16이다 **[재학습]**

베이스를 bf16으로 로드하므로 어댑터도 bf16으로 생성되고, AdamW 상태도 bf16이다.
표준 관행은 어댑터만 fp32로 올리는 것이다. 학습 안정성에 실제로 영향이 있다.

조치: `get_peft_model` 직후 학습 대상 파라미터를 fp32로 캐스팅.

### C5. 요약의 예시 표가 앞 100개 고정이다 **[재채점]**

`rows[:100]`(`scripts/score_medical_nla_v2_readouts.py:222`). 체리피킹은 아니지만
대표 표본도 아니다. 부록 예시는 무작위 추출 + seed 기록으로 바꾼다.

---

## 정리 — 무엇을 먼저 결정해야 하나

**추출 전에 확정해야 하는 것** (바꾸면 뽑아둔 활성화가 전부 무효):

1. 케이스 프롬프트 문자열 (A1, C2)
2. 음성 소견 포함 여부 (C6) — 구현은 끝났고 `--negative-cues`로 켜면 된다
3. wh-질문 렌더링 (C7)
4. cue 순서 정책 (C1)
5. 레이어 집합·위치 집합

**추출 후 학습 전에 확정** (재학습만 하면 됨): A2, B4, B5, B6, C4

**저장된 출력만 있으면 언제든 재계산** (재채점): A3, A4, A5, B1, B2, B3, C5

A1이 가장 급하다. 다른 모든 항목은 재추출 없이 되돌릴 수 있지만, A1은 추출
프롬프트 자체를 바꾸는 문제라 지금 결정하지 않으면 나중에 전부 다시 뽑아야 한다.
