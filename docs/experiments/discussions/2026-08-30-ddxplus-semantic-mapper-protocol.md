# DDXPlus open-text semantic mapper/scorer protocol (초안)

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
| G1 | Reader round-trip | Structured reader locked 출력을 mapper에 넣었을 때 reader 자신의 selected set과 micro F1 **≥ .98** | Reader 출력은 canonical phrase라 mapper가 이걸 못 읽으면 mapper가 병목 |
| G2 | Negative control | Cue-absent donor 사례 텍스트에서 그 absent cue로의 false-map rate **≤ .05** | D9a false-support 규칙과 동일 논리 |
| G3 | 결정론 | 동일 입력 재실행 시 출력 byte-identical (Stage 1 전체, Stage 2는 캐시 경유) | 재현성 |
| G4 | 사람 표본 감사 | Stage 2 매핑 100건 무작위 표본의 사람 불일치 **≤ .05** | LLM mapper 신뢰 확보 |

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

현재 상태: **초안 / 에이전트 검토 대기**. Codex 검토 후 합의되면 사람 승인을
받아 구현을 열고, G1-G4 통과 + hash 동결 후에만 Vanilla 10,028행 generation과
채점을 시작한다.
