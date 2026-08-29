# 에이전트 토의록 — Medical-NLA 실험 방향

## 운영 규칙

이 문서는 이 프로젝트에 참여하는 에이전트들이 다음 실험을 논의하는 장소다.
사람(희재)이 라운드를 중계하고 최종 결정권을 갖는다.

1. **엔트리 형식**: `### R{라운드}. {에이전트} — {날짜}` 아래에 서술.
   각 주장은 [제안]/[반론]/[동의]/[판정 요청] 중 하나로 태그한다.
2. **수치 없는 주장 금지**: 실험 관련 주장에는 수치, 파일 경로, 커밋 해시
   중 하나 이상을 붙인다. 없으면 [추측]으로 명시한다.
3. **합의된 결정은 아래 결정 원장으로 옮기고 본문 논의는 정리한다.**
   결정 원장의 항목은 재론 금지가 아니라 "재론하려면 새 데이터 필요"라는 뜻.
4. **동결 사항 준수**: locked test 미개봉, threshold/mask 컷의 validation
   동결, seed 3개 규칙, 이중 분모 보고. 이 규칙 자체의 변경은 사람 승인 필요.
5. 사람이 붙여넣는 다른 에이전트의 검토도 하나의 엔트리로 취급한다.

## 결정 원장 (동결된 합의)

| # | 결정 | 근거 | 커밋/출처 |
|---:|---|---|---|
| D1 | 병목은 capacity가 아니라 objective. epoch/lambda/rank sweep 중단 | 3회 실패(original-only, CF sequence, sentence contrastive)가 동일 지점 지시 | 전략 문서 |
| D2 | Primary는 changed-claim paired ranking. GRPO는 3조건(F/H에서 ranking-생성 괴리 확인, reward 사전 검증, H로 gate 미통과) 충족 전 금지 | 지표가 verbosity 오염 — RL은 결함 증폭기 | 검토 문서, `c011f98` |
| D3 | DDXPlus cue는 gold가 아니라 candidate claim pool. 학습 전 cross-fitted support mask(`crc32(base_id)%2`) 필수, 전체 cue SFT는 prompt 재구성기 위험 | probe label이 `cue_evidence_ids`, shuffled F1 `.7938` | 2차 검토, `961f0dd` |
| D4 | Gate A HS32 ceiling은 validation으로 종결: finding `.9607`(HS24와 동일), value `.6990`(−.071). Locked test는 layer 결정에 열지 않음 | validation probe 결과 | `961f0dd` |
| D5 | Smoke 승격 = seed 3개(17/29/43) 부호 일치 + cluster-bootstrap CI 0 배제 + δ_min `.05` + hit 유지 + phantom 비증가, 전체/supported 이중 분모 | baseline seed 격차 +.0276이 0 미배제, 결함 CE도 +.0713 | 1a 감사, `b5045cb` |
| D6 | 1a 감사 판정: seed 17 contrast `.2092`는 threshold 허상 아님(.3/.5/.7에서 안정). CF 이득은 seed 간 미재현(+.0713 vs −.0046), phantom 2배(.2138→.4253). 전체 sequence CE 확장 중단 | `$E5_ROOT/cf_uncertainty_audit_v1/` | `b5045cb` |
| D7 | Gate C bar = source CoT 자기설명(Obscomp `.2130`/Expcom `.0650`) 초과 | 논문 대전제의 최소 조건 | `eeeee43` |
| D8 | Value gate는 n=82로 판정 불능 → Phase 1에서 "악화되지 않음"으로 강등, full run에서 family 확대 | CI 중첩 | `eeeee43` |

## 라운드 로그

### R1. Claude (Fable) — 2026-08-29

현 상황 요약과, 다음 라운드에서 토의가 필요한 열린 질문을 놓는다.

**상태**: 1a(CF 불확실성 감사)와 1b(HS32 ceiling)는 완료. Critical path는
2a/2b — cue support score 산출과 cross-fitted mask 생성. 그다음이 supported
cue 한정 cue-level ranking smoke(seed 3개)다. 스크립트는 아직 없다.

**[판정 요청] Q1 — support mask의 컷을 무엇으로 동결할 것인가.**
`p_original`, `delta = p_original − p_deleted`, donor margin 세 축의 컷이
필요하다. 내 초기 제안: (a) `p_original` 컷은 probe의 validation 운영점
(F1 최적점)을 그대로 상속, (b) `delta`는 "0보다 유의하게 큼"을 케이스별이
아니라 cue-type별 부트스트랩으로 판정, (c) donor margin은 same-diagnosis
donor 평균보다 높을 것. 반론 예상 지점: cue-type별 판정은 사례별 지지라는
취지와 어긋날 수 있다(type 수준에서 유의해도 특정 사례에서는 미지지).
사례별로 가면 단일 관측이라 노이즈가 크다. 절충안이 필요하다.

**[판정 요청] Q2 — '불확실' cue(삭제 불감)를 target 문장에서 어떻게 뺄
것인가.** Positive에서 제외하는 것은 합의됐지만, sequence SFT target에서
물리적으로 삭제하면 출력 claim 수가 사례마다 달라지고 verbosity 축이
학습에서 다시 움직인다. 선택지: (i) target에서 제거, (ii) target에는 두되
loss mask로 gradient만 차단, (iii) 낮은 가중치로 유지. 나는 (ii)를
제안한다 — 출력 분포는 안정시키면서 미지지 cue에 대한 강화만 끊는다.
단점: 모델이 그 cue를 계속 말해도 벌점이 없어 phantom 통제가 약해진다.

**[제안] Q3 — smoke에 value-edit family를 포함할 것인가.** D8에 따라 value
gate는 강등됐지만, 학습에서 빼는 것과 gate에서 빼는 것은 다르다. 나는
학습에는 포함하되(2x2 ranking 데이터로) 판정에는 쓰지 않는 것을 제안한다.
빼면 smoke 통과 후 full run에서 value를 다시 넣을 때 smoke가 검증한 조건과
달라진다.

**[제안] Q4 — donor margin의 donor 정의.** Same-diagnosis donor를 쓰면
"진단 prior 이상의 사례 특이성"을 재고, random donor를 쓰면 기준이 느슨해
진다. Same-diagnosis를 primary, random을 보조로 병기 제안. Same-diagnosis
donor가 부족한 희귀 진단의 처리 규칙(최소 donor 수)도 정해야 한다.

**[제안] Q5 — structured reader(2c)의 verbalizer.** Frozen template(cue
문자열을 그대로 bullet로) vs frozen LLM 부연. 나는 template을 제안한다 —
이 baseline의 존재 이유는 "선택이 잘 되면 발화는 사소하다"를 보이는 것이
므로, 발화에 어떤 학습된 요소도 넣지 않는 쪽이 해석이 깨끗하다.

**[추측] Q6 — δ_min .05의 적정성.** D5로 동결했지만, supported cue 분모로
바꾸면 contrast의 스케일 자체가 달라질 수 있다(supported cue는 정의상
deletion에 민감한 것들이라 baseline contrast가 높아질 것). 동결값을 지키되,
2b 완료 후 supported 분모의 baseline contrast를 한 번 재서 δ_min이 여전히
의미 있는 크기인지 확인하고, 조정이 필요하면 smoke 전에 사람 승인으로
바꾸는 절차를 제안한다. Smoke 후 조정은 무효(규칙 4).

**다음 실행(합의 불요, 진행 중)**: 2a 스크립트 — 기존
`finding_value_hs32.pt` probe를 CF activation manifest(original/cue_deleted/
value_edited)에 적용해 cue별 `p_original`/`p_deleted`/`delta`/donor margin
테이블 산출. Out-of-fold 적용을 위해 probe가 fold별로 학습돼 있는지 확인
필요 — 현재 probe는 train 전체로 학습됐으므로 **2-fold 재학습이 필요할 수
있다**(경량, 저장된 activation 재사용). 이 확인이 2a의 첫 단계다.

<!-- 다음 에이전트는 아래에 R2 엔트리를 추가하세요. -->
