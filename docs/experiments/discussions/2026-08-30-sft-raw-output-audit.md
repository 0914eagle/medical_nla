# SFT 계열 원문 출력 감사

## 질문

집계 점수만으로 구분하기 어려운 SFT 실패 양상을 기존 validation 출력에서 직접 확인한다.
이 감사는 다음 질문에 답한다.

1. DiReCT SFT 출력이 실제 physician observation을 말하는가, 질환 전형이나 형식 문구만
   생성하는가?
2. 기존 quote-constrained extractor가 출력에 명시된 reference-aligned claim을 놓쳤는가?
3. DDXPlus counterfactual SFT가 삭제된 cue를 계속 말하거나 수정 전 값을 유지하는가?
4. 개선처럼 보이는 값이 finding을 많이 말한 결과인지 선택적 counterfactual response인지?

이 감사는 exploratory다. 기존 DiReCT semantic FAIL, DDXPlus lexical grounding 값 또는 locked
결과를 변경하거나 재소송하지 않는다.

## 승인된 규약

### DiReCT 감사 A

- 모집단: `sft_val.jsonl`과 모든 비교 방법의 출력·기존 semantic artifact가 공통으로 존재하는
  validation case ID의 교집합.
- 교집합이 50이면 표본 추출 없이 50건 전수 사용한다.
- 교집합이 50보다 크면 seed 17의 SHA256 stable-hash 순서 상위 50건을 사용한다.
- 교집합이 50 미만이면 실행을 중단한다. 다른 사례를 임의로 보충하지 않는다.
- 비교 방법:
  - Source CoT
  - Vanilla NLA
  - DiReCT-only SFT seeds 17/29/43
  - Full-data SFT seeds 17/29
- Common mixed pilot은 full-data SFT가 계승한 개발 ablation이므로 핵심 원문 감사에서 제외한다.
- 비교 기준: SFT target에 사용된 exact-note-grounded physician observations.
- 기존 extractor raw response, accepted exact quotes, official semantic evaluation JSON을 같은
  private bundle에 결합한다.

출력별 사전 고정 checklist:

1. physician observation과 정렬된 patient-specific finding이 있는가?
2. 환자 사례가 아니라 질환 전형 설명에 머무르는가?
3. task/format boilerplate에 머무르는가?
4. reference가 지지하지 않는 patient-specific claim을 만드는가?
5. 기존 extractor가 명시된 reference-aligned claim을 놓쳤는가?

각 `true` 판정은 해당 출력의 exact contiguous quote를 요구한다. 방법명은 judge request에서
opaque ID로 치환한다. 24 GiB GPU의 local judge attention memory와 방법 간 상호영향을 줄이기
위해 한 request에는 최대 두 방법만 넣으며, 50건 x ceil(7/2) = 200개 request를 고정한다.
사례 간 상투 반복은 exact duplicate count와 각 출력의 다른 사례 대비 maximum word-set
Jaccard 중앙값으로 별도 계산한다.

### DDXPlus 감사 B

한 cohort로 합치면 value-bearing 사례에 조건화되므로 두 모집단을 분리한다.

1. **Deletion cohort 50건**: 네 방법에 original/deletion 출력이 모두 존재하는 base case에서
   diagnosis 층화 후 stable hash로 선택한다.
2. **Value-edit cohort 50건**: 네 방법에 original/value-edited 출력이 모두 존재하는 base
   case에서 changed evidence ID 층화 후 stable hash로 선택한다.

비교 방법:

- Original-only full SFT seeds 17/29
- Counterfactual SFT seeds 17/29

Deletion checklist는 original target mention, deleted phantom, untouched finding retention,
unsupported patient claim이다. Value-edit checklist는 old value original mention, replacement
mention, old-value persistence, clean switch, untouched finding retention, unsupported patient
claim이다. 모든 positive 판정에 exact quote를 요구한다.

## 데이터 경계

- DiReCT physician observation과 생성 원문은 MIMIC 파생 제한 자료다.
- DiReCT bundle과 judge request/response는 server 125의
  `/data1/heejae/restricted/direct/e4/` 아래에만 둔다.
- DiReCT 원문은 Codex/OpenAI API 등 외부 judge에 보내지 않는다. 서버 내부
  `Meta-Llama-3-8B-Instruct`만 사용한다.
- DDXPlus는 public-derived지만 취급 실수를 막기 위해 동일하게 local bundle과 aggregate
  summary를 분리한다.
- Git에 올릴 수 있는 것은 원문 없는 protocol, source SHA, checklist별 건수·비율과 반복성
  통계뿐이다. `private_bundle.jsonl`, requests, judgements, private adjudications는 금지한다.

## 구현 및 실행

- 공통 builder/finalizer: `scripts/audit_sft_family_raw_outputs.py`
- DiReCT server-125 wrapper: `scripts/run_direct_sft_raw_audit50_125.sh`
- DDXPlus server-125 wrapper: `scripts/run_ddxplus_sft_raw_audit50_125.sh`

두 wrapper는 `MODE=prepare|run|finalize|all`을 지원한다. `prepare`가 모집단 교집합,
결정론적 표본, source SHA와 private request를 먼저 고정한다. `run`은 고정 request만 판정하고,
`finalize`는 exact quote와 request population을 검증한 뒤 aggregate-only summary를 쓴다.

## 결과

DiReCT local Llama checklist는 200개 request 중 최초 34개만 frozen JSON/exact-quote
계약을 통과했고, 세 차례 deterministic retry 뒤에도 56 valid / 144 invalid였다. Invalid는
`true_without_quote=91`, `method_population_mismatch=21`, `non_verbatim_quote=30`,
`json_parse_error=2`였다. 따라서 valid 56개만 선택하면 selection bias가 생기므로 AI checklist
계측기는 폐기한다.

DiReCT 최종 원문 감사 집계는 이미 동결된 exact-quote extractor와 official evaluator를
50건 전수에 다시 적용하지 않고 저장된 artifact에서 결정론적으로 계산한다. 보고 항목은
extractable-observation row, physician-reference matched row, unmatched-only row, no-extractable
row, observation count와 cross-case repetition이다. `unmatched`는 available physician reference와
매칭되지 않았다는 뜻이며 의학적 허위라고 단정하지 않는다. 별도 AI 없이는 확인할 수 없는
extractor miss와 disease-template classification은 `not assessed`로 남긴다.

## 판정

어떤 결과도 기존 frozen score를 변경하지 않는다. extractor miss가 발견되면 누락 건수만
exploratory annotation으로 기록한다. 재채점은 별도 승인 없이는 수행하지 않는다.
