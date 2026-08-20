# 소스 모델 정답 라벨 — 무엇을 고쳤고, 그 숫자를 어떻게 읽어야 하나

활성화를 뽑기 전에 정답 라벨부터 만든 기록이다. 라벨은 3축 중 **교정(오류 예측)**
축 전체가 딛고 서는 것이라, 여기가 틀리면 뒤의 모든 수치가 무의미해진다.

프롬프트가 이미 고정돼 있어서 GPU 작업 중 유일하게 추출을 기다리지 않아도 되는
잡이었고, 실제로 먼저 돌린 값을 했다 — 첫 실행에서 **832건 중 0건 파싱**이 나왔다.

---

## A. 첫 실행이 드러낸 결함 네 개

MCR 테스트 832건, direct 조건, `max_new_tokens=128`. 결과 `answer_parse_rate: 0.0`.

`scripts/inspect_source_answers.py`가 원인을 셋으로 갈라준다: 예산 소진 / 형식
미준수 / 정규식 미스. 답은 명확했다 — 응답 87%가 문장 중간에서 끊겼고 마감 문구
`The answer is`는 **한 건도** 등장하지 않았다.

### A1. direct 조건이 direct가 아니었다

응답 본문:

```
Okay, let's break down this case and arrive at the most likely diagnosis.
Here's my reasoning, considering the patient's history, presentation, and
response to initial treatment.

**Key Findings & Considerations:**
```

"단일 최유력 진단이 무엇인가"라고 물었는데 Gemma 3 12B IT가 스스로 추론을 펼친다.
**예산을 올리는 것은 해결이 아니다.** 128 → 512로 올리면 파싱은 되지만 그 순간
direct 팔이 두 번째 CoT 팔이 되고, 논문이 재는 direct/CoT 대비가 존재하지 않게 된다.

고친 방법 두 가지:

1. **어시스턴트 턴을 `The answer is`에서 시작**시킨다. 추론할 자리를 없애는 것이지
   하지 말라고 부탁하는 것이 아니다. Turpin et al.의 no-CoT 팔, Lanham et al.의
   forced answer와 같은 장치다.
2. 지시문에 `Give the diagnosis only. Do not explain your reasoning.`를 추가한다.
   1번만으로는 부족했다 — 프리필된 턴이 산문으로 이어지면서
   `The answer is relatively straightforward given the constellation of findings.`
   같은 문장이 나오고, 파서가 그걸 진단명으로 돌려준다. "하나만 고르라"는 말은
   무엇을 고를지를 정하지, 답 칸에 에세이를 쓰지 말라고 말하지 않는다.

**활성화는 손해 보지 않는다.** 인과 마스킹 아래에서 뒤에 붙은 토큰은 앞 토큰의
은닉 상태를 바꿀 수 없다. 지시문이 제시부 **뒤에** 있도록 설계한 것이 여기서 값을
한다 — cue 위치도, format 위치(케이스 프롬프트의 마지막 토큰)도 프리필 유무와
무관하게 동일하고, 추출 하나가 두 조건을 계속 커버한다.

CoT 팔은 예산 2048로 올리고, 그래도 답 없이 소진한 체인은 **자기가 쓴 전체 추론을
그대로 돌려받고** 답만 요구받는다. 잘린 체인에 강제하는 것은 early-answering
*개입*이므로 라벨로 쓰면 안 된다. 강제된 행은 `answer_forced`로 표시된다.

결과: **파싱률 1.000.**

### A2. MCR의 정답 라벨이 진단명이 아니었다

`final_diagnosis` 필드에 형식이 섞여 있다.

```
Posterior ischemic optic neuropathy   ← 산문
Scleroderma_renal_crisis              ← 언더스코어
AmeloblasticFibroma                   ← 붙여쓰기
```

두 곳이 망가진다. 채점은 명백하다 — 모델이 `scleroderma renal crisis`라고 정확히
답해도 0점이다. **더 심각한 쪽은 진단 누출 필터다.** "제시부에 정답 진단명이 이미
적혀 있는 케이스는 버린다"는 규칙의 구현은 라벨 문자열을 제시부에서 찾는다.
라벨이 `Scleroderma_renal_crisis`면 제시부에 "scleroderma renal crisis"라고 버젓이
적혀 있어도 못 찾는다 → 안 버린다 → **답이 적힌 케이스가 코퍼스에 남는다.**
그리고 감사 스크립트가 같은 코드를 써서 "0건, 깨끗함"이라고 통과시켰다.

라벨을 공백 토큰 단위로 복원하고(산문 라벨은 그대로 통과), 원본 형태를 별칭으로
보관하고, 바꾼 라벨 전부를 TSV로 덤프해 검토 가능하게 했다.

측정된 효과:

| | 이전 | 이후 |
|---|---|---|
| train 누출 탈락 | 903 (6.9%) | **1,003 (7.7%)** |
| test 누출 탈락 | ~57 | **68 (7.6%)** |
| train 채택 | 11,888 | 11,799 |
| test 채택 | 832 | 821 |

train 100건 + test 11건이 답을 적어놓고 통과하던 오염 케이스였다. 실재했지만 규모는
0.8%p로, 처음 경고했던 어조보다 작다.

이 숫자는 두 단계로 나온 것이다. 라벨 복원만 했을 때 train 탈락이 903 → 996이었고,
그 뒤 정규화에 소유격 제거와 영국식 철자 접기를 넣자 1,003이 됐다. 나머지 7건은
제시부가 `Whipple's disease`라고 쓰고 라벨이 `Whipple disease`인 식으로, 같은 이름을
다른 표기로 적어놓아 문자열 비교를 빠져나가던 것들이다. 표기 접기를 채점을 위해
넣었는데 누출 필터가 같은 함수를 쓰는 덕에 함께 고쳐졌다 — 두 서버에서 같은 입력에
다른 개수가 나온 이유이기도 하다.

### A3. 채점이 표기 차이를 흡수하지 못했다

`**Erythema Multiforme**`(마크다운), `Guillain-Barré`/`Guillain-Barre`(악센트),
`Whipple's`/`Whipple’s`(어퍼스트로피 글리프), `Eagle's`/`Eagle`(소유격),
`oedema`/`edema`(영국식). 전부 진단에 대한 이견이 아니라 표기다. 정규화에서 접는다.

소유격은 특히 조용히 실패했다 — 비영숫자를 공백으로 바꾸는 처리가 `Eagle's`를
`eagle s`로 만들어 떠도는 `s` 토큰이 매치를 막고 있었다.

영국식 철자는 목록을 명시했다. `ae`/`oe`를 통째로 접으면 `aerosol`까지 망가진다.

### A4. 프리필 예산 32가 긴 라벨을 잘랐다

`...atypical peripheral retinal degeneration (PR` — 매치 하나를 잃는다. 64로.

---

## B. DDXPlus: 라벨이 분류기용으로 쓰여 있다

`URTI`, `PSVT`는 모델이 풀어 쓴 답과 **한 단어도 공유하지 않는다**. `Larygospasm`은
DDXPlus 자체의 오타다(`n` 누락). `Pancreatic neoplasm`은 아무도 입에 올리지 않는
표현이다.

라벨 집합이 닫혀 있고 49개뿐이므로 이것은 판단 규칙이 아니라 **유한한 글쓰기**다 —
MCR의 자유 텍스트 라벨 6,934개에는 절대 쓸 수 없는 수단이고, 여기서만 허용된다.

**넣은 것**: 약어 확장, 라벨 오타 교정, 같은 병의 다른 이름.
**안 넣은 것**: 인접·상위 질환. `Heart failure`는 급성 폐부종의 63건 오답이지만
넣지 않았다 — 다른 DDXPlus 클래스이고, 더 중요하게는 **`heart failure`가 이
프롬프트들의 cue 문자열 자체(E_106)**라서 인정하면 "프롬프트에 적힌 병력을 되읊은
것"에 점수를 주게 된다. `Allergic rhinitis`(부비동염 92건)도 같은 이유로 제외.

감사가 **어떤 라벨과도 매칭되지 않는 별칭 키를 hard violation으로 잡는다.** 기억으로
쓴 항목이 커버리지인 척 앉아 있을 수 없다.

측정된 효과: **0.292 → 0.3724 (+394건).** 그중 거의 4분의 1이 라벨 하나다 —
`Pulmonary neoplasm`이 0/100에서 **99/100**으로 갔다. 모델은 매번 "lung cancer"라고
정확히 답하고 있었고 전부 0점이었다.

> 여기서 내 추정이 두 번 틀렸다. 처음엔 `Larygospasm` 오타가 ~100건을 먹는다고
> 했지만 실제로는 4건이었고(모델은 68번 "Asthma"라고 답한다), 전체 회수량을
> "~1%"로 잡았지만 실제는 8.0%p였다. 둘 다 표본 8개의 오답 분포에서 전체를
> 추정한 결과다. 전수를 보기 전에는 규모를 말하지 말 것.

정답률 0인 진단은 13개 → 5개. 남은 다섯은 답변 분포상 전부 진짜 실패다:
`Acute otitis media` → "Viral upper respiratory infection" (67/100),
`Allergic sinusitis` → "Allergic rhinitis" (92), `Localized edema` →
"Nephrotic Syndrome" (70). **모델이 진단마다 하나의 오답으로 거의 결정론적으로
무너진다.**

---

## C. 확정된 수치

| | 값 |
|---|---|
| DDXPlus direct 정확도 | **0.3724** (4,900건) |
| DDXPlus 파싱률 | 1.000 |
| 답이 케이스 감별진단 안에 | 34.5% (1,691건) |
| — 그중 상위 3위 내 | **90.1%**, 평균 순위 1.8 |
| MCR direct 정확도 | **0.1340** (821건), 수기 보정 후 **≈0.164** |
| 진단명만으로 오류 예측 | **0.850** |

MCR의 보정치는 추정이 아니다. 엄격 규칙과 토큰 겹침이 어긋난 **55건 전수를 손으로
읽었다**: 25건이 같은 병의 다른 표기(`Essential thrombocytosis`/`thrombocythemia`,
`lichen spinulosus`/`spinulosum`, `membranous glomerulonephritis`/`membranous
nephropathy`), 22건이 진짜 오답 — 여럿은 케이스의 핵심 감별이다
(`deep soft tissue leiomyoma` vs `Soft tissue sarcoma`는 양성 대 악성,
`arrhythmogenic **left** ventricular cardiomyopathy` vs `ARVC`는 좌우) — 8건이 판단.

**임계값 지표는 채택하지 않았다.** token-F1 0.75에 정답(`Recurrent Guillain Barre
Syndrome`/`Relapsing Guillain-Barré syndrome`)과 오답(`Leydig cell tumor of the
ovary`/`Sertoli-Leydig cell tumor`)이 나란히 있고 0.50에서도 마찬가지다. 어떤 컷도
둘을 가르지 못한다. 겹침 점수는 행마다 기록만 하고 정확도에는 쓰지 않는다.

---

## D. 이 숫자가 "Gemma의 실력"인가 — 선행 연구와의 비교

**주의**: 이 컨테이너의 egress 프록시가 arxiv·ACL·PMC·OpenReview·HuggingFace를
모두 차단해 **원문 표를 열지 못했다.** 아래는 검색 결과 요약에서 온 값이므로 논문에
싣기 전 원문 확인이 필요하다.

### MedCaseReasoning

보고된 값: **OpenAI o3 64.5%**(10-shot), **DeepSeek R1 48.0%**. 판정은
**Qwen2.5-32B-Instruct를 LLM 심판**으로 쓴다.

우리 값 0.164와의 격차는 세 축이 겹친 결과이고, 셋 다 저쪽에 유리하다:

| 축 | 선행 연구 | 우리 |
|---|---|---|
| 모델 | 프런티어 **추론 모델** (o3, R1) | Gemma-3-12B-IT |
| 추론 | CoT | **프리필된 direct — 추론 토큰 0개** |
| 예시 | 10-shot | 0-shot |
| 판정 | LLM 심판 | 엄격 문자열 포함 |

즉 **0.164 대 48%는 같은 것을 재고 있지 않다.** 우리 direct 팔은 정의상 추론을
금지한 조건이고, 비교 가능한 값은 아직 돌리지 않은 **CoT 팔**이다. 그리고 LLM 심판은
우리가 손으로 센 25건 같은 동의어를 대부분 정답 처리하므로, 판정 방식만으로도
수 %p가 벌어진다.

### DDXPlus

보고된 값: MEDDxAgent(ACL 2025) 기준 GPT-4o가 **에이전트 3회 반복 후 0.86**,
Llama3.3 70B가 **Diagnosis Accuracy 87.53%**. 그러나 같은 논문에서 **zero-shot
single-turn(n=0)** GPT-4o는 **0.18–0.27**이다.

우리 0.3724는 그 사이에 있다. 이유가 설명된다 — MEDDxAgent의 n=0은 에이전트가 아직
질문을 하지 않은 시점이라 소견이 거의 없고, 우리 프롬프트는 **증거 전체**(케이스당
평균 6.79 cue)를 처음부터 준다. 반대로 0.86/0.87은 반복 질의·검색·후보 제시가 붙은
값이다.

**결론: 0.3724과 0.164는 이 설정에서 이상하지 않다.** 12B 비추론 모델에게,
후보 목록 없이, 0-shot으로, 문자열 일치로 채점한 값이다. 파싱률 1.000과
"답의 90.1%가 정답 감별진단 상위 3위 안"이라는 사실이 모델이 붕괴한 게 아님을
따로 뒷받침한다.

논문에 쓸 때 반드시 함께 적을 것: **후보 목록을 주지 않았다**는 점. DDXPlus는 원래
49지선다 분류 벤치마크이고 우리는 개방형으로 물었다. 이건 결함이 아니라 선택이며
(MCR과 대칭을 유지하기 위한), 그렇게 밝혀야 비교가 오독되지 않는다.

---

## E. Likelihood는 어떻게 재는가

자유 생성은 **샘플된 것 하나**만 보고한다. "확신에 차서 틀렸다"와 "헷갈리면서
틀렸다"를 구분할 수 없다. 오류 예측 축은 정확히 그 구분을 필요로 하므로, 닫힌 후보
집합 위의 **우도**를 따로 잰다. `scripts/score_source_diagnosis_logprobs.py`.

작동 방식: 케이스 프롬프트 + 어시스턴트 턴 시작 문자열(`The answer is`)을 접두로
두고, 49개 후보 각각을 그 이어쓰기로 채점한다. 후보당 한 번의 forward, 배치로 처리.

**후보당 기록되는 네 점수** — 서로 다른 질문에 답하므로 랭킹 기준은 CLI로 고른다:

| 점수 | 무엇을 재나 | 약점 |
|---|---|---|
| `logprob_sum` | 후보 전체의 로그확률 합 | 짧은 이름이 유리 |
| `logprob_mean` | 토큰당 평균 | 길이는 제거되지만 밋밋한 토큰만 가진 후보가 유리 |
| `first_token_logprob` | 첫 토큰 | **greedy 디코더가 실제로 한 결정에 가장 가까움** |
| `calibrated_*` | 위 값에서 **내용 없는 프롬프트 아래 같은 후보의 점수를 뺀 것** | 보정 프롬프트를 골라야 함 |

보정이 중요하다. 빼지 않으면 흔한 질환이 증거가 아니라 **빈도** 때문에 희귀 질환을
이긴다. `--calibration-prompt`로 내용 없는 프롬프트를 주고
`--rank-field calibrated_first_token_logprob`을 쓰면 이름 자체의 사전확률이 상쇄된다.

여기서 파생되는 **불확실성 특징**이 오류 예측의 no-NLA 기준선이다:
`top1_prob`, `top1_top2_prob_margin`, `candidate_entropy`, `candidate_entropy_norm`,
그리고 정답의 `gold_rank`. 활성화 프로브는 이 특징들을 이겨야 의미가 있다.

### 이 스크립트에 남아 있던 결함 두 개 (수정함)

1. **프롬프트를 한 번 더 감싸고 있었다** —
   `"Answer the clinical question...\n\nQuestion: {prompt}"`. 이건 라벨과 활성화가
   서로 다른 forward를 서술하게 만든, 파일럿에서 이미 잡았던 바로 그 결함이다.
   이제 케이스 프롬프트를 **쓰인 그대로** 쓴다.
2. **이어쓰기 접두가 `"The most likely diagnosis is"`였다.** 답변 생성은
   `The answer is`에서 이어졌으므로, 다른 도입부 뒤에서 후보를 채점하면 **모델이 이어
   쓰라고 요구받은 적 없는 문맥**에서의 순위를 재는 것이 된다. 기본값을 `ANSWER_CUE`로
   맞췄다.

### 돌리는 법

```bash
python scripts/score_source_diagnosis_logprobs.py \
  --config configs/default.yaml \
  --input $ART/data/ddxplus_cue_count_cases.jsonl \
  --output-jsonl $ART/results/ddxplus_source_logprobs.jsonl \
  --summary-md $ART/results/ddxplus_source_logprobs.md \
  --rank-field first_token_logprob \
  --candidate-batch-size 16
```

후보 목록을 `--candidates-jsonl`로 주지 않으면 입력의 `diagnosis_name`에서 모은다 —
DDXPlus는 49개가 그대로 모인다. MCR은 라벨이 6,934개라 **닫힌 집합이 아니므로 이
측정이 성립하지 않는다.** 우도 축은 DDXPlus 전용이고, 그 비대칭 자체가 두 코퍼스를
같이 쓰는 이유의 일부다.

---

## F. 오류 예측 축이 넘어야 할 선

진단명만 보고 그 진단의 다수결 결과를 답하는 예측기가 **0.850**을 낸다. 무조건
"틀림"이 0.628. **활성화 프로브가 0.80을 내면 아무것도 보여준 게 아니다.**

"양쪽 결과가 모두 있는 진단만" 남기는 것으로는 부족했고, 측정으로 확인됐다 —
정확히 상수인 진단만 뺐더니 기준선이 0.833으로 거의 그대로였다. 가장 어려운 15개 중
9개가 0.01–0.07 구간이고, 100번 중 1번 맞히는 진단은 라벨 예측기에게 99%를 그냥 준다.

기준을 **소수 결과의 비율**로 바꿨다 (`--min-minority-rate`, 기본 0.10). 남는 것은
소스 모델이 진짜로 일관되지 않은 진단들 — "이 **개별 케이스**가 틀릴 것인가"에
라벨이 이미 주지 않는 답이 존재하는 유일한 집합이다.

이건 평가 집합을 **프로브가 아니라 소스 모델의 성질로** 고르는 것이다. 정당하지만
반드시 명시해야 하고, 기준선은 항상 같은 부분집합에서 다시 계산한다.

---

## G. 다음

- [ ] CoT 팔 실행 — MedCaseReasoning 선행 값과 비교 가능한 유일한 조건
- [ ] Likelihood 베이스라인 실행 (DDXPlus 전용)
- [ ] 활성화 추출 (`docs`가 아니라 `EXPERIMENTS.md` §0c)
- [ ] 선행 연구 수치 **원문 확인** — 이 문서 D절은 검색 요약 기반이다
