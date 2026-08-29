# DDXPlus open-text semantic mapper/scorer protocol

현재 통제 상태: **구현 승인 / validation gate 대기 / sealed locked generation 허용 / semantic scoring 금지**. 2026-08-30
사람 결정으로 G4의 사람 감사를 제거하고 아래의 독립 AI concordance gate로 대체했다.

## 질문

Free-generating 방법의 open text 출력(당장은 Vanilla NLA locked 10,028행, 이후
Medical-NLA가 promotion을 통과하면 동일 적용)을 **91 evidence ID + native
value**로 매핑해, probe/structured reader와 같은 metric family(finding F1,
shuffled gap, deletion phantom/removal/retention, value replacement/
persistence/clean switch)를 계산할 채점기를 무엇으로 동결할 것인가?

현재 lexical pilot scorer는 약칭·의역을 놓쳐 paper primary로 쓸 수 없다는
것이 기록돼 있고(실행 계획 문서), 이 protocol이 동결되기 전에는 10,028행
generation을 시작하지 않는다(Lane A 순서).

## 설계 원칙

1. **Method-blind**: mapper 입력에는 방법의 출력 텍스트와 candidate ontology만
   들어간다. Case ID, gold, diagnosis, split, method 이름은 주지 않는다
   (DiReCT extractor와 동일 원칙).
2. **단일 계측기**: open-generator 행들끼리는 반드시 같은 mapper로 채점한다.
   Probe/reader 행은 기존 확정값을 유지하고, 계측기 차이는 caption에 명시한다.
3. **결정론 우선, 의미 판정은 잔여분만**: 결정 가능한 것은 LLM에 묻지 않는다.
4. **동결 후 불변**: lexicon/prompt/model/코드의 hash를 기록하고, locked
   출력 채점 후에는 어떤 구성요소도 바꾸지 않는다.

## 제안 파이프라인 (3-stage)

### Stage 0 — claim 분리 (결정론)

- `<observed>` bullet(`- `로 시작하는 행)을 claim 단위로 사용한다.
- Bullet이 없는 자유 산문이면 문장 단위로 분리한다(마침표/개행 기준, 기존
  verbosity 감사와 동일 규칙). 빈 출력은 claim 0개로 처리하고 분모에서
  제거하지 않는다.

### Stage 1 — 결정론적 lexical match

- **Frozen alias table**에 대한 정규화 문자열 매칭: (a) official train에서
  만든 evidence별 modal exact phrase(structured reader lexicon 재사용),
  (b) DDXPlus 배포 메타데이터의 evidence question/name 문자열, (c) 사전
  고정한 표기 변형(대소문자, 관사, 단·복수).
- Hit이면 해당 evidence ID로 확정하고 Stage 2에 보내지 않는다.
- Multi-value evidence(6종)는 value 문자열 매칭 표를 같은 방식으로 둔다.

### Stage 2 — LLM semantic mapper (잔여 claim만)

- 입력: claim 텍스트 하나 + 91 evidence 후보 목록(이름/설명) + `none` 옵션.
- 출력 계약(DiReCT extractor와 동형): JSON only —
  `{"evidence_id": <id 또는 null>, "value": <native value 또는 null>,
  "supporting_quote": <claim 원문 내 연속 인용>}`.
- 후처리 validator가 quote의 원문 존재를 검증하고, 실패 시 null 처리.
- Backend는 DiReCT extractor와 동일 계열(Codex CLI). 실제 model ID는 runtime
  judgement metadata에서 읽어 provenance에 기록한다(추정 금지 — 수치 원장
  §5.3 규칙).
- Temperature 0, claim별 판정은 `SHA256(claim_text)` key로 캐시해 재실행
  결정론을 보장한다.

### 집계 규칙

- Claim → evidence ID는 다대일 허용, 사례 수준에서는 set으로 dedupe.
- Stage 1 확정이 Stage 2에 우선하며, 한 claim은 최대 한 evidence ID에만
  매핑된다.
- 이렇게 만든 per-case predicted evidence set/value를 **structured reader와
  같은 metric 코드**에 넣는다 — metric 정의를 재구현하지 않는다.

## 동결 전 검증 gate (전부 validation/fixture, locked 채점 전)

| # | Gate | 기준 | 근거 |
|---|---|---|---|
| G1 | Reader round-trip | Validation structured-reader 출력에서 evidence micro F1과 native-value accuracy 각각 **≥ .98** | Canonical reader도 못 읽으면 mapper가 병목 |
| G2 | Negative control | Validation same-diagnosis cue-absent donor에서 target false-map **≤ .05** | D9a false-support 규칙과 동일 논리 |
| G3 | 결정론 | Stage 0/1과 frozen-cache replay byte-identical; Stage 2 cold agreement 별도 보고 | 재현성과 LLM 변동성 분리 |
| G4 | 독립 AI concordance | Stage 2 validation decision 100건에서 독립 auditor와 evidence/value 불일치 **≤ .05** | 단일 mapper의 자기확인 방지 |

G1~G4는 validation 자료(reader validation 출력, 필요 시 validation 50행
소규모 vanilla 표본 — validation은 열려 있음)와 synthetic fixture로만
수행한다. 어느 gate든 실패하면 locked 채점을 열지 않고 protocol을 재설계
한다(단, locked 출력을 본 뒤의 재설계는 금지).

## Hash/provenance 동결 목록

- alias table 파일 SHA-256
- Stage 2 prompt 전문 SHA-256
- mapper backend 실제 model ID (runtime metadata에서)
- claim 분리/집계 코드 버전 (commit hash)
- G1-G4 결과 report와 입력 artifact SHA-256

## 비용 추정 (참고치, 보장 아님)

Vanilla 행당 claim ~5개, Stage 1이 절반을 소화한다고 가정하면 Stage 2는
10,028 x ~2.5 ≈ 25,000 판정. 기존 judge run 단가 기준 소액이며 캐시로 중복
claim은 1회만 판정된다.

## 열린 항목 (합의 필요)

1. **Reader 행의 계측기 일관성 열**: main은 기존 확정값을 유지하되, mapper로
   reader를 재채점한 값을 appendix 일관성 검증으로 병기할지 (제안: 병기 —
   G1이 그 수치를 이미 만들어 준다).
2. Stage 1 alias table에 DDXPlus 메타데이터 외 수동 동의어를 넣을지 —
   넣는다면 작성 시점을 G1-G4 이전으로 제한(locked 출력을 보고 추가 금지).
3. Value 정규화 표의 범위 (단위 표기, 숫자 서식).
4. Bullet 없는 산문의 문장 분리 규칙 세부(약어 마침표 처리).

## 판정

이 초안 단계의 판정은 아래 에이전트 검토와 최종 사람 결정으로 대체됐다. 현재 통제 상태는
문서 머리말과 마지막 절을 따른다. G1-G4 통과 + hash 동결 전에는 Vanilla 10,028행 generation과
채점을 시작하지 않는다.

## Codex 검토 (2026-08-30)

### 총평

3-stage 구조, method blindness, open-generator 사이의 단일 계측기, quote validator,
locked scoring 전 hash 동결에는 동의한다. 다만 아래 차단 항목을 먼저 계약에 반영해야
한다. 현재 초안을 그대로 구현하는 것은 승인하지 않는다.

### 차단 수정 1 — G1은 validation reader만 사용

G1 표의 `Structured reader locked 출력`은 74--77행의 validation-only 원칙과 충돌한다.
G1 입력은 **DDXPlus validation structured-reader output만** 사용한다. Locked-test reader
output이나 locked 4,543-case label은 mapper prompt, alias 수정, threshold 선택, gate 판정에
사용하지 않는다. Locked reader를 mapper로 재채점한 값은 protocol 동결 뒤 appendix
measurement로만 만들 수 있다.

### 차단 수정 2 — claim 하나에서 여러 evidence를 허용

한 bullet이나 문장에는 `fever, cough, and dyspnea`처럼 여러 finding이 함께 나올 수 있다.
따라서 `한 claim은 최대 한 evidence ID` 규칙은 열린 자연어 방법의 recall을 구조적으로
낮춘다. Stage 1은 boundary-safe alias hit를 모두 모으고, Stage 2 계약은 다음처럼 배열을
반환한다.

```json
{
  "mappings": [
    {
      "evidence_id": "one exact candidate ID",
      "value_id": "one evidence-specific native value ID or null",
      "supporting_quote": "an exact contiguous quote from the claim"
    }
  ]
}
```

동일 claim/evidence 중복만 제거하고 서로 다른 evidence는 모두 보존한다. Candidate 밖 ID,
해당 evidence의 native-value enum 밖 value, 원문에 없는 quote는 해당 mapping만 null/drop한다.

### 차단 수정 3 — assertion과 value를 분리해 검증

Mapper는 환자에게 명시적으로 진술된 finding만 매핑한다. 일반 의학 지식, 권고, 감별진단의
가능성, 조건문을 patient finding으로 만들지 않는다. `no fever`처럼 부정된 finding은 fever
evidence ID로 매핑할 수 있지만 value는 해당 evidence의 명시적 native negative value로만
매핑한다. Evidence 이름 hit만으로 value를 추론하지 않는다.

Value 정규화는 release metadata와 official-train-derived lexicon에 존재하는 native value ID,
고정 단위/숫자 표기 변환으로 제한한다. Fuzzy numeric matching이나 locked/validation output을
보고 추가한 value alias는 금지한다.

### 차단 수정 4 — cache key를 protocol-bound로 변경

`SHA256(claim_text)`만 사용하면 prompt, ontology, alias, model이 바뀌어도 오래된 판정을
재사용할 수 있다. Cache key는 최소한 다음 canonical payload의 SHA-256이어야 한다.

```text
claim text + ontology hash + alias-table hash + prompt hash + backend model ID
```

Raw request, raw response, validated mapping, backend/model metadata를 함께 보존한다. G3는 두
항목으로 나눈다.

1. Stage 0/1과 frozen-cache replay는 byte-identical이어야 한다.
2. Stage 2의 cold duplicate sample은 별도 agreement diagnostic으로 보고한다.

Cache를 읽은 같은 실행이 같다는 사실만으로 LLM 자체의 결정론을 주장하지 않는다.

### 차단 수정 5 — G2 분모와 false-map을 명시

G2는 validation structured-reader donor output을 사용한다. Donor는 target evidence가
`selected_claims`에 없고 같은 diagnosis인 사례로 고정한다. Mapper에는 donor text와 ontology만
주고 target ID, case ID, diagnosis, donor 관계는 주지 않는다. False map은 mapper 결과에 그
target evidence가 나타난 pair 수를 전체 eligible donor pair 수로 나눈다. Donor text가 실제로
target을 렌더링한 행은 control에서 제외하며 제외 수와 coverage를 보고한다.

### 차단 수정 6 — 기존 metric 코드를 공유 모듈로 사용

Mapper output adapter는 `run_ddxplus_structured_reader.py`와 같은 `selected_claims` schema를
만든다. `evidence_id`와 `value_id`를 채운 뒤 finding F1, hard shuffle, deletion, retention,
value-edit metric은 기존 `evaluate_readouts` 구현을 공용 모듈로 옮겨 양쪽이 import한다.
공식을 복사한 두 번째 scorer를 만들지 않는다.

### 차단 수정 7 — Stage 2를 claim별 subprocess 25,000회로 실행하지 않음

현재 비용 추정은 candidate ontology를 매 claim마다 반복하고 Codex CLI process를 약 25,000번
띄우는 overhead를 반영하지 않았다. 먼저 validation에서 Stage 0/1 잔여 unique claim 수와
token 수를 dry-run report로 만든다. Stage 2는 opaque claim SHA를 붙인 **고정 크기 batch**로
요청하고, 응답을 검증한 뒤 claim별 cache로 분해한다. Batch size는 validation runtime smoke로
한 번 고정하고 locked run에서 바꾸지 않는다. 실패 batch만 동일 frozen request로 재개한다.

### G1--G4 수정 계약

| Gate | 수정된 입력과 판정 |
|---|---|
| G1 | validation structured reader만 사용; selected evidence-set micro F1 >= .98 및 native-value accuracy를 함께 보고 |
| G2 | validation same-diagnosis cue-absent donor pair; target false-map <= .05, eligible coverage와 Wilson CI 보고 |
| G3 | deterministic stages/cache replay byte-identical; cold Stage-2 duplicate agreement는 별도 수치 |
| G4 | Stage-2 validation decision 100건을 다른 model ID의 독립 AI가 blind 재매핑; evidence/value 불일치를 분리하고 각각 <= .05 |

G4는 사람 정확도 감사가 아니라 AI 간 concordance다. 논문에서도 `human audit` 또는
`human-validated`라고 부르지 않는다.

### 열린 항목에 대한 Codex 권고

1. Reader mapper 재채점은 protocol 동결 후 appendix consistency로 병기한다.
2. 수동 동의어는 사용하지 않는 것을 기본으로 한다. 필요하면 공개 출처와 작성 시점을 기록하고
   G1--G4 전에 한 번만 동결한다.
3. Value alias는 release metadata, official train lexicon, 고정 숫자/단위 변환으로 제한한다.
4. Bullet은 행 단위로 유지하고 산문은 versioned abbreviation list를 가진 결정론적 sentence
   splitter를 사용한다. 다중 evidence mapping을 허용하므로 conjunction을 임의 분해하지 않는다.

### 판정

상기 일곱 수정은 구현 계약으로 유지한다. G4의 최종 대체 계약은 문서 마지막의 사람 결정이
우선한다. D18 이후 generation과 scoring gate는 분리한다. Frozen HS32 population, actor prompt,
checkpoint, greedy decoding을 기록한 generation protocol이 있으면 10,028행을 생성해 hash 봉인할
수 있다. G1--G4와 mapper hash-freeze receipt 전에는 봉인된 출력의 semantic mapping, 본문 열람,
결과 집계 및 이를 이용한 방법 수정이 금지된다.

## Claude 응답 (2026-08-30)

**[동의] 일곱 수정 전부 수용한다.** 특히 세 개는 내 초안의 실수다:

1. **G1의 locked reader 사용은 내 문서 내 자기모순이었다** — 74-77행에서
   "전부 validation"이라 해놓고 G1 표에는 locked 출력을 적었다. Validation
   reader output으로 정정하고, locked reader 재채점은 동결 후 appendix로만.
2. **claim당 evidence 1개 제한은 open generator의 recall을 구조적으로 깎는
   설계 결함이었다** — "fever, cough, and dyspnea"가 한 bullet에 오는 게
   자연어의 기본형이다. 측정 대상인 방법에게 불리한 계측기를 만들 뻔했다.
   Array 계약 + 동일 claim/evidence 중복만 제거로 수정. 사례 수준 집계는
   모든 mapping의 union이다.
3. **`SHA256(claim_text)` cache key는 stale 판정 재사용 통로였다** — prompt/
   ontology/model이 바뀌어도 캐시가 살아남는다. Protocol-bound canonical
   payload key로 수정하고, G3의 분해("cache replay 결정론"과 "LLM cold
   duplicate agreement"는 다른 주장)도 받아들인다 — 내 초안은 이 둘을
   섞어서 캐시 재현성을 LLM 결정론처럼 보이게 했다.

수정 3(assertion/value 분리, 부정 finding은 명시적 negative native value로만),
5(G2 분모·coverage·Wilson CI 명세), 6(`evaluate_readouts` 공용 모듈화, 복사
scorer 금지), 7(고정 크기 batch + dry-run report 선행)도 그대로 채택한다.
이 응답 당시에는 G4를 실제 사람 감사로 제안했으나, 문서 마지막의 2026-08-30 사람 결정으로
독립 AI concordance gate로 대체됐다.

**[추가 제안 2개] (수정이 아니라 세부 고정)**

- 수정 7의 batch 구성도 결정론이어야 한다: **잔여 claim을 claim-SHA 오름차순
  으로 정렬해 순서대로 고정 크기 batch에 담는다.** 그래야 "동일 frozen
  request로 재개"가 재현 가능하다.
- G4의 표본 100건 추출 규칙을 사전 고정한다: claim-SHA 정렬 순서에서
  `seed 17` 결정론적 표본 추출. 결과를 본 뒤 표본을 다시 뽑는 경로를 막는다.

## 최종 판정

Codex 일곱 수정 + Claude 추가 세부 2개를 통합한 위 protocol로 **에이전트
합의 완료**. 이후 G4는 문서 마지막의 사람 결정으로 대체됐다. 구현 순서는 (1) alias
table/prompt/adapter 구현 + fixture test, (2) validation dry-run report
(잔여 claim 수·token 수·batch size 고정), (3) G1-G4, (4) hash 동결,
(5) Vanilla 10,028행 generation + 채점이다.

## Codex 재확인 (2026-08-30)

일곱 수정의 수용과 두 결정론 세부에 동의한다. Claim-SHA 정렬 뒤 고정 batch를 만드는 규칙과
seed 17 표본 추출은 재개 가능성과 결과 확인 후 재표집하는 경로를 모두 막는다.

구현 계약에는 아래 두 수치 기준을 포함한다.

1. **G1 value threshold**: validation structured-reader round-trip에서 evidence-set micro F1뿐
   아니라, reader가 native value를 렌더링한 eligible claim의 conditional native-value accuracy도
   **>= .98**이어야 한다. 단순 보고만 하고 통과 조건에서 빼지 않는다.
2. **G4 value coverage**: 100건은 unique Stage-2 decision에서 method-blind로 뽑되, native value를
   반환한 decision을 최대 30건 우선 포함한다. 해당 stratum이 30건보다 작으면 전부 포함하고
   부족분은 non-value mapped, null 순서로 채운다. 각 stratum 안에서는 claim-SHA 정렬 후 seed 17로
   표본을 고정한다. Evidence decision 불일치율은 전체 100건에서 <= .05, conditional value
   불일치율은 value-audit 분모에서 <= .05여야 한다. Value-audit 분모가 20 미만이면 value mapper
   gate는 판정 불가이며, value를 paper primary로 채점하기 전에 별도 validation 표본 계약을 다시
   승인한다.

이 두 항목을 포함해 구현한다. Mapper 구현은 validation artifact만 읽고 G1--G4와 dry-run
report가 모두 나온 뒤 별도 hash-freeze receipt를 만든다. 그 receipt는 locked **scoring** wrapper의
필수 입력이다. D18의 generation-only wrapper는 mapper receipt를 읽지 않지만 생성 즉시 출력
SHA-256과 exact population receipt를 만들고 semantic scorer를 호출하지 않는다.

## 사람 결정 — G4 AI-only 대체 및 구현 승인 (2026-08-30)

사람 100건 감사를 제거한다. G4는 다음 독립 AI concordance 계약으로 대체하며, 나머지 합의된
protocol의 validation 구현을 승인한다.

1. Primary mapper와 auditor는 모두 method-blind하게 동일한 frozen claim/ontology를 받지만 서로의
   출력은 받지 않는다.
2. Runtime에서 얻은 `primary_model_id`와 `auditor_model_id`는 모두 비어 있지 않아야 하고 서로
   달라야 한다. 둘 중 어느 것도 backbone 계열(`gemma`, `nla-gemma`)이면 gate를 실행하지 않는다.
3. Auditor는 primary와 같은 JSON schema 및 quote/value validator를 사용해 claim을 처음부터
   독립적으로 재매핑한다. 불일치를 제3 AI로 수정하거나 다수결로 덮지 않는다.
4. 표본은 unique Stage-2 validation decision에서 뽑는다. Native value를 반환한 decision을 최대
   30건 우선 포함하고, 부족분은 non-value mapped, null 순으로 채운다. 각 stratum 안에서
   claim-SHA 정렬 후 seed 17로 고정한다.
5. Evidence disagreement는 두 validated evidence-ID set이 다른 decision 수 / 100이다. 통과 기준은
   `<= .05`다. Conditional value disagreement는 value stratum에서 동일 evidence에 대해 native
   value ID가 다른 decision 비율이며 통과 기준은 `<= .05`다. Value 분모가 20 미만이면 value
   mapper gate는 판정 불가다.
6. G4는 **AI 간 일치도**만 보이며 실제 임상 정확도의 사람 검증이 아니다. 이 한계와 두 실제
   model ID, 표본 분모, 불일치율을 appendix/limitations에 보고한다.

이 결정은 validation mapper 구현과 G1--G4 실행을 승인한다. 이 문장의 generation 차단은 이후
D18로 좁혀졌다. Prompt/model/decoding/HS32 population을 동결한 generation-only queue는 먼저
실행할 수 있지만, 네 gate 통과와 protocol/alias/prompt/model/scorer receipt 없이는 봉인된
10,028행을 semantic 채점할 수 없다.

## 사람 결정 - generation/scoring 분리 (D18, 2026-08-30)

장시간 Vanilla generation을 mapper 구현과 병렬화한다. 허용되는 순서는 아래로 고정한다.

1. DDXPlus locked CoT-P0 10,028행에서 public AV가 요구하는 HS32 activation을 만든다. 이는
   HS24 probe layer 선택을 바꾸지 않으며 Vanilla L32 checkpoint와의 차원 계약을 맞추는 작업이다.
2. 실제 sidecar actor prompt를 dump하고 manifest, config, model ID,
   `do_sample=false`, `max_new_tokens=512`, batch size를 generation protocol에 hash 동결한다.
3. 두 base-ID-complete shard를 생성하고 exact `4,543 original + 4,543 deletion + 942 value edit`을
   검증한 뒤 readout SHA-256 receipt를 만든다.
4. 이 시점에는 output text를 열람하거나 mapper/scorer를 실행하지 않는다.
5. 별도 validation-only G1-G4 mapper receipt가 생기면 동일 sealed readout을 재생성 없이 채점한다.

구현 entry point는 `run_ddxplus_vanilla_hs32_locked_activations_4gpu.sh`,
`run_ddxplus_vanilla_locked_generation_4gpu.sh`,
`score_ddxplus_vanilla_locked_from_seal.sh` 세 개다.
