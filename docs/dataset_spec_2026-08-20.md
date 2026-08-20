# 데이터셋 명세 — 원본, 변환, 속성, 논문에서의 쓰임

`data_construction_log_2026-08-20.md`가 **무엇이 틀렸고 왜 바꿨나**를 다룬다면,
이 문서는 **지금 데이터가 무엇이고 논문의 어느 자리에 들어가나**를 다룬다.
협업자에게 넘기거나 논문 부록으로 옮길 수 있는 형태로 썼다.

---

## 1. 원본 데이터

### DDXPlus

Tchango et al., NeurIPS 2022 Datasets & Benchmarks. CC BY 4.0.
`aai530-group6/ddxplus` (HF)에서 받는다. 규칙 기반 시뮬레이터로 생성한 합성
환자 코퍼스이며, 파일 두 개로 구성된다.

**환자 CSV** — 한 행이 환자 한 명:

| 컬럼 | 내용 | 우리가 쓰나 |
|---|---|---|
| `PATHOLOGY` | 정답 병리 (49종) | ✅ 진단 라벨 |
| `EVIDENCES` | 문진 항목 id 목록 | ✅ cue 원천 |
| `AGE` · `SEX` | 인구학 정보 | ✅ 프롬프트 머리말 (cue 아님) |
| `DIFFERENTIAL_DIAGNOSIS` | 확률 붙은 감별진단 목록 | ✅ 필드로 보존 (§7-2) |
| `INITIAL_EVIDENCE` | 주호소에 해당하는 첫 항목 | ❌ 미사용 |

`EVIDENCES`의 형태는 두 가지다:

```
E_DYSPNEA          값 없이 id만  →  이진 문항에 "예"
E_TRAVEL_@_N       id_@_값       →  다지선다 문항과 선택지
```

**중요**: "아니오"로 답한 이진 문항은 **목록에 등장하지 않는다.** 즉 암묵적
음성은 데이터에 기록되어 있지 않고, 만들어낼 수도 없다.

**문진 사전** (`release_evidences.json`) — id가 무슨 질문이고 값이 무슨 뜻인지:

```json
"E_TRAVEL": {
  "question_en": "Have you traveled out of the country in the last 4 weeks?",
  "value_meaning": {"N": {"en": "no"}, "Y": {"en": "yes"}},
  "is_antecedent": true
}
```

**핵심 성질**: DDXPlus에는 **문장이 없다.** 영어 cue는 우리가 두 파일을 합쳐
만든다. 그래서 cue 텍스트는 데이터셋이 아니라 우리 책임이며, 구축 기록의 A1·A2가
전부 이 지점에서 나왔다.

### MedCaseReasoning

Stanford, CC BY 4.0. `zou-lab/MedCaseReasoning` (HF). 공개 case report에서
추출한 실제 임상 텍스트 14,489건.

| 필드 | 내용 | 우리가 쓰나 |
|---|---|---|
| `case_prompt` | 환자 제시부 (진단 이전) | ✅ 프롬프트이자 cue 원천 |
| `final_diagnosis` | 최종 진단 | ✅ 진단 라벨 |
| `pmcid` | 원 논문 식별자 | ✅ 케이스 id |
| `diagnostic_reasoning` | 임상의 추론 서술 | ❌ **미사용** (아래 §7) |

**핵심 성질**: MedCaseReasoning에는 **문장만 있다.** 어느 부분이 진단 근거인지는
주석되어 있지 않다 — `diagnostic_reasoning`의 인용문 중 `case_prompt`에서
발견되는 것은 1.7%뿐이다(구축 기록 B1). 그래서 cue는 우리가 제시부를 잘라 만들며,
자르기만 하므로 텍스트는 임상의가 쓴 그대로다.

### 두 원본의 대칭

| | DDXPlus | MedCaseReasoning |
|---|---|---|
| 원본에 문장이 | **없음** (id + 값) | **있음** (임상의가 씀) |
| 우리가 하는 일 | **만든다** | **자른다** |
| cue 텍스트의 저자 | 우리 (문진 사전 기반) | 임상의 |
| 실패 모드 | 변환이 사실을 훼손 | 자르는 위치가 어색 |
| span 확정 방식 | 조립했으므로 | 오려냈으므로 |

**span이 확정된다는 결론은 같고 경로가 다르다.** 이것이 두 데이터셋을 함께 쓰는
방법론적 근거다 — 텍스트 성격이 정반대인데 계측 조건은 동일하다.

---

## 2. 변환 사슬

```
[DDXPlus]
  train.csv + release_evidences.json
    │  make_ddxplus_cue_count_cases.py
    │    · (문항, 값) → 영어 cue  ← 결정이 가장 많이 개입하는 지점
    │    · 진단당 100 케이스 표집
    ↓
  ddxplus_cue_count_cases.jsonl        4,900 케이스
    │  make_ddxplus_cue_position_rows.py
    │    · 케이스 하나를 cue별 행으로 분해 (케이스당 최대 4)
    ↓
  ddxplus_cuepos_rows.jsonl            18,646 행

[MedCaseReasoning]
  mcr_{train,test}.jsonl
    │  make_clinical_span_cases.py
    │    · case_prompt를 절 단위로 분할 → cue
    │    · 진단명 언급 케이스 제외, 독자용 질문 제거
    ↓
  mcr_cases_{train,test}.jsonl         11,888 / 832 케이스
    │  make_ddxplus_cue_position_rows.py   (같은 스크립트)
    ↓
  mcr_cuepos_rows_{train,test}.jsonl   47,122 / 3,319 행
```

각 단계 뒤에 감사 게이트가 있다 — `audit_ddxplus_cue_rendering.py`,
`audit_clinical_span_cases.py`. 하드 위반이 하나라도 있으면 exit 1이다.

프롬프트 문자열은 `src/case_prompts.py` 한 곳에서 나온다. 두 데이터셋이 같은
지시문을 쓰고, 지시문이 제시부 **뒤에** 오므로 직답 조건과 CoT 조건이 cue 위치
활성화를 공유한다.

---

## 3. 산출물과 필드

### 케이스 파일

한 행이 한 환자/한 케이스. 소스 모델이 답하는 단위이자 프롬프트의 단위.

| 필드 | 의미 |
|---|---|
| `id` · `base_id` · `case_id` | 식별자 (cue-position 행이 `base_id`로 되돌아옴) |
| `prompt` | **직답 조건** 프롬프트 (활성화 추출·정답 생성이 쓰는 바로 그 문자열) |
| `prompt_cot` | **CoT 조건** 프롬프트. `prompt`와 제시부가 바이트 동일 |
| `presentation` | (MCR) 지시문 없는 제시부 원문 |
| `diagnosis_name` · `diagnosis_id` · `diagnosis_aliases` | 정답 진단 |
| `cue_targets` | cue 문자열 목록. 각각 `prompt`의 정확한 부분문자열 |
| `cue_count` · `available_cue_count` | 선택된 수 / 가용 수 |
| `cue_types` | (DDX) `symptom` 또는 `antecedent` |
| `cue_polarities` | (DDX) `positive` / `negative` |
| `cue_is_boilerplate` | (MCR) 정형 문구 여부 → **층화 보고용** |
| `cue_evidence_ids` · `cue_value_ids` · `cue_value_labels` | (DDX) 원본 추적 |
| `cue_merged_value_counts` | (DDX) 다중값 병합 개수 (1이면 병합 없음) |
| `clean_cues` · `negative_cues` · `prefer_symptoms` | **생성 설정이 행마다 기록됨** |
| `excluded_cue_count` | 이 케이스에서 걸러진 cue 수 |

마지막에서 두 번째가 중요하다. 코퍼스를 나중에 받은 사람이 **명령을 제대로
쳤는지에 의존하지 않고** 이 데이터가 무엇인지 알 수 있다.

### cue-position 행

한 행이 (케이스, cue) 하나. 활성화 추출의 단위.

| 필드 | 의미 |
|---|---|
| `id` | `{base_id}__cuepos{NN}` |
| `case_id` · `base_id` | 케이스로 되돌아가는 키 |
| `prompt` | 케이스 프롬프트와 **바이트 동일** (18,646행 전수 확인) |
| `cue_text` · `target_text` | 이 행이 겨냥하는 cue |
| `cue_index` | 케이스 내 순번 |
| `position_mode` | `target_text` |
| `target_text_strategy` | `last_subtoken` |
| `target_text_occurrence` | 같은 문자열이 반복될 때 몇 번째인지 |

---

## 4. 현재 코퍼스 속성

| | DDXPlus | MCR train | MCR test |
|---|---|---|---|
| 케이스 | 4,900 | 11,888 | 832 |
| cue-position 행 | 18,646 | 47,122 | 3,319 |
| 진단 종수 | 49 (각 100) | 6,000+ | — |
| cue 수 (평균 / p90 / max) | 6.79 / 10 / 21 | 9.94 / 16 / 39 | 12.08 / 19 / 32 |
| cue 단어 수 (평균) | 7.28 | 8.77 | 8.70 |
| 프롬프트 토큰 (p50 / max) | 115 / 250 | 322 / 910 | 378 / 797 |
| 1회만 등장하는 cue | — | 97.8% | 99.2% |
| 최빈 cue 비율 | — | 1.35% | 1.56% |
| boilerplate cue | — | 9.8% | 9.3% |
| 중첩 · 비verbatim | 0 · 0 | 0 · 0 | 0 · 0 |

**논거로 쓰이는 성질 네 가지**

1. **cue 길이가 붙어 있다** (7.28 vs 8.77). 두 데이터셋 비교가 cue 길이로
   교란되지 않는다.
2. **cue 다양성이 정반대다.** DDXPlus는 문진표에서 나오므로 같은 문자열이
   반복되고, unseen cue 풀을 인위적으로 떼어내야 한다. MCR은 97.8%가 한 번만
   등장해 unseen이 저절로 생긴다(측정된 test unseen 비율 95.9%).
3. **MCR은 진단 라벨이 6,000종을 넘는다.** 닫힌 라벨 집합을 전제하는 도구(선형
   probe, 26-way likelihood)가 **정의되지 않는다.** H2의 경험적 근거다.
4. **프롬프트가 전부 sliding window(1024) 안에 든다.** local attention 층에서도
   cue가 제시부 전체를 본다.

---

## 5. 논문 항목별 쓰임

`paper_tables_final_2026-08-19.md`의 항목과 데이터의 대응.

| 논문 항목 | 쓰는 산출물 | 어느 데이터셋 |
|---|---|---|
| **Table 1** 설명의 출처 비교 (main) | 케이스(①④열) + cue-position(②③열) | DDX · MCR 양쪽 |
| **Figure 1** 설계 ablation | cue-position + 진단명 타깃 변형 | DDX |
| **Figure 2** 판독 궤적 | cue-position, 13개 레이어 | DDX (+ MCR 재추출 시 양쪽) |
| 인라인 실물 — 증거의 네 보존 형태 | cue-position 판독 출력 | DDX · MCR |
| **Table 2** 충실성 검사 상세 | 반사실 행 (아직 미생성) | DDX(재구성) · MCR(span 치환) |
| **Table 3** 오류 해부학 | 케이스 + 소스 정답 라벨 | DDX · MCR |
| **Figure 3** 오답노트 사다리 | 케이스, 자연 분포 split | DDX |
| **Figure 4** 실시간 개입 | 케이스, 재시도 조건 | DDX · MCR |
| **Table 4** 임상 타당성 | 판독 출력 표본 + 임상의 평가 | DDX · MCR |

**아직 만들지 않은 것**: 반사실 행(Table 2). split이 있어야 test 케이스 풀이
정해지므로 활성화 추출 뒤에 만든다.

### 두 조건 프롬프트의 쓰임

| 조건 | 필드 | 쓰이는 곳 |
|---|---|---|
| 직답 | `prompt` | 본선 — 판독, 오류 예측, 26-way likelihood, 정확도 |
| CoT | `prompt_cot` | 가설 1 — 절단(early answering), 개입 언급 여부 |

두 조건이 제시부를 공유하므로 **cue 위치 활성화는 한 번만 추출한다.** format
위치(프롬프트 마지막 토큰)만 조건별로 다르다.

---

## 6. 두 데이터셋의 역할 분담

| | DDXPlus | MedCaseReasoning |
|---|---|---|
| 역할 | **정밀 계측** | **실제 임상 텍스트 이식** |
| 강점 | cue를 조립하므로 개입 쌍을 문자열 일치까지 검증 가능(construction-exact). 진단 49종 균형 표집 | 실제 임상 문체. unseen cue가 자연히 95.9%. 부정 표현이 다양 |
| 약점 | 합성 텍스트. cue 수가 질환과 얽힘. 명시적 음성이 사실상 1종 | 개입이 span 치환뿐(재조립 불가). 진단 6,000종이라 닫힌 라벨 도구 사용 불가 |
| 담당 주장 | 인과·개입, 레이어 궤적, 오류 예측 | 일반화, 부정 판독, 열린 과제 |

**부정 표현의 판독은 MCR에서만 측정한다.** DDXPlus의 명시적 음성이 실질 1종이라
그쪽에서는 잴 수 없다(구축 기록 B2).

---

## 7. 원본에서 아직 안 쓰는 것 — 검토 필요

세 가지가 남아 있고, 셋 다 지금 결정하는 편이 낫다.

### 7-1. DDXPlus의 `AGE` · `SEX` — 넣기로 했다 ✅

우리 DDXPlus 프롬프트에는 인구학 정보가 없었다. 그런데 MCR 제시부는 거의 항상
`A 58-year-old woman presented with...`로 시작한다. 나이·성별은 진단적이고(croup은
소아, 성별 특이 질환 다수), 시뮬레이터도 이를 조건으로 환자를 생성한다. 빼면
**데이터가 의도한 것보다 어려운 과제**가 되어, 소스 모델 정확도가 낮은 것이
모델 탓이 아니라 우리 가공 탓이 된다.

**조치**: 머리말에 넣는다. MCR 첫 문장과 평행한 형태다.

```
You are an expert physician. A 3-year-old boy presents with the following findings:
- a cough
```

**cue로는 만들지 않는다.** 우리가 재는 것은 소견의 판독이고 인구학은 그것을 읽는
문맥이다. 목록에 넣으면 cue가 되어 버린다.

남는 비대칭: MCR은 첫 절에 인구학이 묶여 들어가 cue가 되는 경우가 있다. 한계
절에 적을 소소한 차이다.

컬럼이 실제로 있는지는 생성 로그에서 확인된다 — 없으면 `A patient`로 물러난다.

### 7-2. DDXPlus의 `DIFFERENTIAL_DIAGNOSIS` — 필드로 보존한다 ✅

DDXPlus는 정답 병리 하나뿐 아니라 **확률이 붙은 감별진단 목록**을 제공한다.
케이스 행에 `differential_diagnosis`로 보존한다(프롬프트에는 넣지 않는다 —
정답을 흘리게 된다). 이걸 쓰면:

- 26-way likelihood를 **단일 정답 맞춤이 아니라 순위 상관**으로 평가할 수 있다
- 오류 해부에서 "얼마나 가까운 오답인가"를 gold 기준으로 잴 수 있다 —
  감별 2순위를 고른 것과 무관한 질환을 고른 것은 다른 실패다
- Table 3(오류 해부학)의 유형 분류가 임의 기준이 아니라 데이터 기반이 된다

**추출과 무관하다** (프롬프트를 안 바꾼다). 나중에 케이스 파일에 필드로 추가하면
된다. 우선순위는 높다고 본다.

### 7-3. MCR의 `diagnostic_reasoning` — 참조 CoT

임상의가 쓴 추론 서술이다. cue 정의로는 못 쓴다는 게 밝혀졌지만(1.7%), **가설
1의 참조점**으로는 쓸 수 있다:

- 모델 CoT와 임상의 추론을 같은 케이스에서 비교
- Table 4(임상 타당성)에서 "사람은 무엇을 근거로 들었나"의 기준선
- 모델 CoT가 언급한 소견 집합 vs 임상의가 언급한 집합

역시 추출과 무관하다.

---

## 8. 재현

**받기** (재생성하지 않는다):

```bash
export HF_HOME=/data/heejae/hf_cache
python -c "
from huggingface_hub import snapshot_download
snapshot_download('0914eagle/medical-nla-cases', repo_type='dataset',
                  local_dir='/data/heejae/medical_nla/data', allow_patterns=['data/*'])
"
```

`data/`에 JSONL(파이프라인이 읽는 원본), `parquet/`에 같은 행(뷰어용). CSV는
없다 — 리스트 필드와 줄바꿈이 든 프롬프트를 CSV가 문자열로 뭉갠다.

**다시 만들기**: `EXPERIMENTS.md`의 "0. MedCaseReasoning ingestion"과
"0b. DDXPlus rebuild" 절. 생성 명령과 감사 명령이 함께 있고, 감사가 exit 1이면
재빌드를 막는다.

**설정 확인**: 받은 케이스 행의 `clean_cues` / `negative_cues` /
`prefer_symptoms`를 보면 그 코퍼스가 무엇인지 알 수 있다. 논문 수치를 낸 설정은
`clean_cues=True`, `negative_cues=False`, `prefer_symptoms=False`다.
